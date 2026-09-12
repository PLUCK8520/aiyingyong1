"""T1.5 / T3.5 / T3.6 / T5.1 / T5.4 / T7.4 · 图组装：节点 + 条件边 + Send 动态扇出 + 反思循环 + 人工确认。

流程（P7 形态）：

    START → memory_loader → intent_router ─┬─ direct  → direct_responder → END
                                           └─ research → planner ⏸（静态断点 interrupt_before）
                                                        → human_confirm（放行/编辑后继续）
                                                        → (Send 扇出 2N 路) scout_web ⤵
                                                                             scout_local ⤵
                                                             evidence_judge（判定+缺口+矛盾）
                                                                  ↓
                                                               reflect ─有缺口且未超轮且未熔断→ 再扇出 2N 路
                                                                  ↓ 否则
                                                               analyst → auditor（引用审计+降级）
                                                                       → memory_writer（T7.4 沉淀 supported 结论）→ END

关键点：
  - `scout_web` / `scout_local` 是 Send 的目标节点，入参是**单个子问题包**（不是全量 state）；
    两者返回的 `evidence` 经 `operator.add` 归并回主 state——这就是 L2-1 的落点。
  - `scout_local` 自 T7.4 起**双路检索**：用户资料库（docs）+ 历史研究结论（research_memory）。
  - 反思循环用 `reflect_targets` 驱动：`reflect` 是唯一写入者，路由函数是纯读，
    所以"补检哪几路、为什么"在 state/trace 里都可回查。
  - `Send` 一律 `from langgraph.types`（1.x 的 `langgraph.graph` 不再导出，P-1 实测）。
  - **T5.4 人工确认 = 编译期静态断点**（2026-09-11 重写）：
    `build_graph(..., interrupt_before=["human_confirm"])`。断点由 `should_interrupt()`
    在**节点执行前**判定，不依赖 `interrupt()` 的 scratchpad 时序。
    ⚠️ 09-10 那版用「节点内 `interrupt()`」，会因 `RESUME` 残留写 + 版本单调递增
    被**静默放行**（详见 `agents/human_confirm.py` 模块文档串），已废弃。
    恢复方式：`invoke(None, cfg)` 停在断点 / `invoke(Command(resume=...), cfg)` 放行。
  - **T5.1 检查点**：`build_graph(ctx, checkpointer=...)`；不传则退化为无状态单跑（P1~P4 行为）。
  - **T7.4 研究闭环**：`memory_writer` 是终点前的最后一跳，**必须在 auditor 之后**——
    审计是唯一给出 `verdict` 的环节，沉淀只能等它判完（防自我投毒）。
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
    human_confirm,
    intent_router,
    memory_loader,
    memory_writer,
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
from ..memory.profile import ProfileStore
from ..memory.research_memory import ResearchMemoryStore
from ..quality.citation_auditor import make_citation_auditor
from ..retrieval.local_search import make_local_client as build_local_client
from ..retrieval.mock_search import FixtureSearchClient
from ..retrieval.ports import SearchClient
from ..retrieval.rerank import DashScopeReranker, LexicalReranker, SiliconFlowReranker
from ..retrieval.stores import ChromaVectorStore, NumpyVectorStore
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
    if settings.llm_mode == "siliconflow" and settings.siliconflow_api_key:
        log.info(
            f"[graph] rerank 走硅基流动 {settings.model_rerank}（官方标免费，实测 0.3s 排序正确）"
        )
        return SiliconFlowReranker(
            settings.siliconflow_api_key,
            base_url=settings.siliconflow_base_url,
            model=settings.model_rerank,
        )
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
    try:
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
    except Exception as exc:  # noqa: BLE001
        # 降级不停机（D4）：建索引要调 embedding，而**不是每个厂商都提供 embedding 接口**
        # （例：Kimi/Moonshot 只有对话模型）。这种情况不该让整个应用起不来——
        # 本地知识库退化为"未启用"，如实留痕，web 检索与成文照常。
        log.warning(
            f"[graph] 本地知识库装配失败，本次禁用本地检索（web 检索与成文不受影响）："
            f"{type(exc).__name__}: {exc}\n"
            f"  常见原因：当前厂商（{settings.llm_mode}）不提供 embedding 接口，"
            f"或模型名 {settings.model_embed!r} 在该厂商不存在。"
        )
        return None


def make_research_memory(
    settings: Settings, gateway: LLMGateway
) -> ResearchMemoryStore | None:
    """装配研究结论记忆（T7.4）。返回 None 表示未启用——`scout_local` 不检索历史结论、
    `memory_writer` 如实留痕跳过，主流程零改动。

    ⚠️ **每个 run 一个新 store**：与本地知识库（`make_local_search_client`）的**进程级缓存**
    语义不同——
      - docs 索引是只读的，缓存安全；
      - research_memory 是**读写的**，`_ids`/`_metas` 是内存副本。若跨 run 共用同一实例，
        评测中"跑 A 断言的写会污染跑 B 断言的读"，对照组也就不可比了。故此处**不缓存**。
    """
    if not settings.research_memory_enabled:
        log.info("[graph] 研究闭环关闭（ATTEST_RESEARCH_MEMORY_ENABLED=0）")
        return None

    embedder = "mock-hashing" if settings.llm_mode == "mock" else settings.model_embed
    dim = _resolve_dim(settings, gateway)
    # 独立 collection：docs 是用户资料、research_memory 是我们自己的结论，必须分得清"谁说的"。
    store = ChromaVectorStore(
        path=settings.research_memory_dir,
        collection="research_memory",
        embedder=embedder,
        dim=dim,
    )
    if store.count():
        log.info(f"[graph] 研究闭环已启用：载入历史沉淀 {store.count()} 条")
    return ResearchMemoryStore(
        store=store,
        embed_fn=lambda texts: gateway.embed(list(texts), task="embed_memory")[0],
        enabled=True,
    )


def build_context(
    settings: Settings,
    *,
    search: SearchClient | None = None,
    local: SearchClient | None = None,
    trace: TraceWriter | None = None,
    run_id: str = "local",
    memory: Any = "__auto__",
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
        # T5.2：画像存储。关闭时传 None，`memory_loader` 会如实留痕并返回空画像。
        profile=ProfileStore(settings.profile_db, enabled=settings.profile_enabled)
        if settings.profile_enabled
        else None,
        # T7.4：研究结论记忆。显式传 None/实例可覆盖（测试要隔离存储）；默认自动装配。
        memory=make_research_memory(settings, gateway) if memory == "__auto__" else memory,
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


def build_graph(
    ctx: NodeContext,
    *,
    checkpointer: Any = None,
    interrupt_before: Any = None,
):
    """组装并编译图。

    Args:
        ctx: 节点上下文（依赖注入）。
        checkpointer: 检查点后端；不传 = 无状态单跑（P1~P4 行为）。
        interrupt_before: 编译期静态断点节点名列表。
            `None`（默认）→ 读 `ctx.settings.human_confirm_enabled`：
                True  → `["human_confirm"]`（T5.4 人工确认大纲）；
                False → 不断点，`human_confirm` 会走 `auto_approval` 直接放行。
            显式传空列表 `[]` → 强制不断点（测试用）。
    """
    if interrupt_before is None:
        enabled = bool(getattr(ctx.settings, "human_confirm_enabled", False))
        interrupt_before = ["human_confirm"] if enabled else []
        if enabled:
            log.info("[graph] 人工确认已开启：编译期静态断点 interrupt_before=['human_confirm']")

    g = StateGraph(GraphState)

    g.add_node("memory_loader", bind(memory_loader.run, name="memory_loader", ctx=ctx))
    g.add_node("intent_router", bind(intent_router.run, name="intent_router", ctx=ctx))
    g.add_node("direct_responder", bind(direct_responder.run, name="direct_responder", ctx=ctx))
    g.add_node("planner", bind(planner.run, name="planner", ctx=ctx))
    g.add_node("human_confirm", bind(human_confirm.run, name="human_confirm", ctx=ctx))
    g.add_node("scout_web", bind(scout_web.run, name="scout_web", ctx=ctx))
    g.add_node("scout_local", bind(scout_local.run, name="scout_local", ctx=ctx))
    g.add_node("evidence_judge", bind(evidence_judge.run, name="evidence_judge", ctx=ctx))
    g.add_node("reflect", bind(reflect.run, name="reflect", ctx=ctx))
    g.add_node("analyst", bind(analyst.run, name="analyst", ctx=ctx))
    g.add_node("auditor", bind(auditor.run, name="auditor", ctx=ctx))
    # T7.4：沉淀节点放在审计**之后**——审计是唯一能给出 verdict 的环节，
    # 沉淀必须等它判完才知道哪些结论可信（防自我投毒的时序前提）。
    g.add_node("memory_writer", bind(memory_writer.run, name="memory_writer", ctx=ctx))

    g.add_edge(START, "memory_loader")
    g.add_edge("memory_loader", "intent_router")
    g.add_conditional_edges("intent_router", route_after_intent, ["direct_responder", "planner"])
    g.add_edge("planner", "human_confirm")
    # T5.4：确认后进入首轮扇出。**必须是条件边**——`fanout_sub_questions` 返回 `Send` 列表
    # 做动态并行，而普通 `add_edge` 不支持 Send（只有条件边的路径函数能产出 Send）。
    g.add_conditional_edges("human_confirm", fanout_sub_questions, list(SCOUT_NODES))
    g.add_edge("scout_web", "evidence_judge")
    g.add_edge("scout_local", "evidence_judge")
    g.add_edge("evidence_judge", "reflect")
    g.add_conditional_edges(
        "reflect", route_after_reflect, [*SCOUT_NODES, "analyst"]
    )
    g.add_edge("analyst", "auditor")
    g.add_edge("auditor", "memory_writer")
    g.add_edge("memory_writer", END)
    g.add_edge("direct_responder", END)

    return g.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


def initial_state(query: str, thread_id: str = "local") -> dict[str, Any]:
    return {"query": query, "thread_id": thread_id, **INITIAL_STATE}


def graph_node_sequence() -> list[str]:
    """图里必须出现的节点（冒烟断言用）。"""
    return [
        "memory_loader",
        "intent_router",
        "direct_responder",
        "planner",
        "human_confirm",
        "scout_web",
        "scout_local",
        "evidence_judge",
        "reflect",
        "analyst",
        "auditor",
        "memory_writer",
    ]
