"""T1.5 / T3.5 / T3.6 · 图组装：节点 + 条件边 + Send 动态扇出 + 反思循环。

流程（P3 形态）：

    START → intent_router ─┬─ direct  → direct_responder → END
                           └─ research → planner ─(Send 扇出 2N 路)→ scout_web ⤵
                                                                     scout_local ⤵
                                                     evidence_judge（判定+缺口+矛盾）
                                                          ↓
                                                       reflect ─有缺口且未超轮且未熔断→ 再扇出 2N 路
                                                          ↓ 否则
                                                       analyst → auditor（引用审计+降级）→ END

关键点：
  - `scout_web` / `scout_local` 是 Send 的目标节点，入参是**单个子问题包**（不是全量 state）；
    两者返回的 `evidence` 经 `operator.add` 归并回主 state——这就是 L2-1 的落点。
  - 反思循环用 `reflect_targets` 驱动：`reflect` 是唯一写入者，路由函数是纯读，
    所以"补检哪几路、为什么"在 state/trace 里都可回查。
  - `Send` 一律 `from langgraph.types`（1.x 的 `langgraph.graph` 不再导出，P-1 实测）。
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..agents import (
    analyst,
    auditor,
    direct_responder,
    evidence_judge,
    intent_router,
    planner,
    reflect,
    scout_local,
    scout_web,
)
from ..agents.base import NodeContext, bind
from ..agents.planner import fan_out_queries
from ..budget.account import BudgetConfig
from ..config import DATA_DIR, Settings
from ..llm.gateway import LLMGateway
from ..logging import get_logger
from ..quality.citation_auditor import make_citation_auditor
from ..retrieval.local_search import make_local_client as build_local_client
from ..retrieval.mock_search import FixtureSearchClient
from ..retrieval.ports import SearchClient
from ..retrieval.rerank import DashScopeReranker, LexicalReranker
from ..retrieval.tavily_client import TavilyClient
from ..trace.events import TraceWriter
from .state import INITIAL_STATE, GraphState

log = get_logger(__name__)


def make_search_client(settings: Settings) -> SearchClient:
    if settings.search_mode == "tavily":
        assert settings.tavily_api_key  # config 已校验
        log.info("[graph] 检索走 tavily（含录制/回放缓存）")
        return TavilyClient(settings.tavily_api_key, DATA_DIR / "cache")
    log.info("[graph] 检索走离线 fixture 回放")
    return FixtureSearchClient(settings.fixture_dir / "web")


def make_reranker(settings: Settings) -> Any:
    if settings.llm_mode == "dashscope" and settings.dashscope_api_key:
        log.warning(
            "[graph] rerank 走 DashScope qwen3.7-text-rerank——**本分支尚未实跑**（无 key 验证），"
            "失败时 LocalSearchClient 会自动退回融合排序"
        )
        return DashScopeReranker(settings.dashscope_api_key)
    log.info("[graph] rerank 走离线词典兜底 LexicalReranker（**不是** rerank 模型）")
    return LexicalReranker()


#: embedding 维度探测缓存：维度是"换模型必须重建索引"的判据，必须来自真实返回值而非硬编码
_DIM_CACHE: dict[str, int] = {}


def _resolve_dim(settings: Settings, gateway: LLMGateway) -> int:
    key = "mock" if settings.llm_mode == "mock" else settings.model_embed
    if key not in _DIM_CACHE:
        vectors, _, _ = gateway.embed(["dimension probe"], task="dim_probe")
        _DIM_CACHE[key] = len(vectors[0]) if vectors else 0
        log.info(f"[graph] embedding 维度探测：{key} -> {_DIM_CACHE[key]} 维")
    return _DIM_CACHE[key]


def make_local_search_client(
    settings: Settings, gateway: LLMGateway
) -> SearchClient | None:
    """装配本地知识库检索（T3.4）。返回 None 表示未启用——节点会如实留痕，不假装检索过。"""
    if not settings.local_enabled:
        log.info("[graph] 本地知识库关闭（ATTEST_LOCAL_ENABLED=0）")
        return None

    use_chroma = settings.local_store == "chroma"
    has_index = (settings.local_chroma_dir / "chroma.sqlite3").exists()
    has_docs = settings.local_docs_dir.exists()
    if use_chroma and not has_index:
        if not has_docs:
            log.info("[graph] 指定 chroma 但既无索引也无文档目录，本地检索关闭")
            return None
        log.warning(
            "[graph] 指定 chroma 但索引不存在（先跑 scripts/ingest_local.py），本次退回内存向量库"
        )
        use_chroma = False
    if not use_chroma and not has_docs and not has_index:
        log.info("[graph] 未找到本地文档目录，本地检索关闭")
        return None

    embedder = "mock-hashing" if settings.llm_mode == "mock" else settings.model_embed
    return build_local_client(
        docs_dir=settings.local_docs_dir,
        embed_fn=lambda texts: gateway.embed(list(texts), task="embed_local")[0],
        embedder=embedder,
        dim=_resolve_dim(settings, gateway),
        use_chroma=use_chroma,
        chroma_dir=settings.local_chroma_dir if use_chroma else None,
        reranker=make_reranker(settings),
        chunk_chars=settings.local_chunk_chars,
        chunk_overlap=settings.local_chunk_overlap,
    )


def build_context(
    settings: Settings,
    *,
    search: SearchClient | None = None,
    local: SearchClient | None = None,
    trace: TraceWriter | None = None,
    run_id: str = "local",
) -> NodeContext:
    trace = trace or TraceWriter(path=settings.trace_dir / "trace.jsonl", run_id=run_id)
    gateway = LLMGateway(settings=settings, trace=trace)
    return NodeContext(
        settings=settings,
        gateway=gateway,
        search=search or make_search_client(settings),
        local=local if local is not None else make_local_search_client(settings, gateway),
        auditor=make_citation_auditor(settings, gateway),
        trace=trace,
        budget=BudgetConfig(
            total_cny=settings.budget_total_cny,
            total_tokens=settings.budget_total_tokens,
            warn_ratio=settings.fuse_warn_ratio,
            hard_ratio=settings.fuse_hard_ratio,
        ),
    )


# ------------------------------------------------------------------ 路由函数


def route_after_intent(state: GraphState) -> str:
    return "direct_responder" if state.get("route") == "direct" else "planner"


SCOUT_NODES = ("scout_web", "scout_local")


def _seen_keys(state: GraphState) -> list[str]:
    """已登记证据的来源键（url 优先，退标题）。

    补检轮次把它下发给 scout，让 scout **跳过已见过的来源**——这是"同一来源被重复编号"
    的根因修复：编号格式含轮次，若补检把同一文档再抓一次，它就会拿到 `WEB2-x-y`，
    参考列表里同一篇资料出现两遍。晚做不如现在做，补检一上线这个坑就必然出现。
    """
    return [e.url or e.title for e in (state.get("evidence") or [])]


def fanout_sub_questions(state: GraphState) -> list[Send]:
    """首轮扇出：每个子问题同时下发给 web 与 local 两个 scout（FR-07 动态并行）。"""
    payloads = fan_out_queries(state)
    seen = _seen_keys(state)
    sends = [Send(node, {**p, "seen_keys": seen}) for p in payloads for node in SCOUT_NODES]
    log.info(f"[graph] Send 扇出 {len(payloads)} 个子问题 × {len(SCOUT_NODES)} 源 = {len(sends)} 路")
    return sends


def route_after_reflect(state: GraphState) -> str | list[Send]:
    """反思路由：有补检目标就再扇出（双源），否则成文。轮数上限/熔断已由 reflect 判定。"""
    targets = state.get("reflect_targets") or []
    if not targets:
        return "analyst"
    seen = _seen_keys(state)
    sends = [Send(node, {**p, "seen_keys": seen}) for p in targets for node in SCOUT_NODES]
    log.info(
        f"[graph] 补检第 {state.get('round')} 轮：{len(targets)} 个子问题 × {len(SCOUT_NODES)} 源 "
        f"= {len(sends)} 路（已跳过 {len(seen)} 个已见来源）"
    )
    return sends


# ------------------------------------------------------------------ 组装


def build_graph(ctx: NodeContext, *, checkpointer: Any = None):
    g = StateGraph(GraphState)

    g.add_node("intent_router", bind(intent_router.run, name="intent_router", ctx=ctx))
    g.add_node("direct_responder", bind(direct_responder.run, name="direct_responder", ctx=ctx))
    g.add_node("planner", bind(planner.run, name="planner", ctx=ctx))
    g.add_node("scout_web", bind(scout_web.run, name="scout_web", ctx=ctx))
    g.add_node("scout_local", bind(scout_local.run, name="scout_local", ctx=ctx))
    g.add_node("evidence_judge", bind(evidence_judge.run, name="evidence_judge", ctx=ctx))
    g.add_node("reflect", bind(reflect.run, name="reflect", ctx=ctx))
    g.add_node("analyst", bind(analyst.run, name="analyst", ctx=ctx))
    g.add_node("auditor", bind(auditor.run, name="auditor", ctx=ctx))

    g.add_edge(START, "intent_router")
    g.add_conditional_edges("intent_router", route_after_intent, ["direct_responder", "planner"])
    g.add_conditional_edges("planner", fanout_sub_questions, list(SCOUT_NODES))
    g.add_edge("scout_web", "evidence_judge")
    g.add_edge("scout_local", "evidence_judge")
    g.add_edge("evidence_judge", "reflect")
    g.add_conditional_edges(
        "reflect", route_after_reflect, [*SCOUT_NODES, "analyst"]
    )
    g.add_edge("analyst", "auditor")
    g.add_edge("auditor", END)
    g.add_edge("direct_responder", END)

    return g.compile(checkpointer=checkpointer)


def initial_state(query: str, thread_id: str = "local") -> dict[str, Any]:
    return {"query": query, "thread_id": thread_id, **INITIAL_STATE}


def graph_node_sequence() -> list[str]:
    """图里必须出现的节点（冒烟断言用）。"""
    return [
        "intent_router",
        "direct_responder",
        "planner",
        "scout_web",
        "scout_local",
        "evidence_judge",
        "reflect",
        "analyst",
        "auditor",
    ]
