"""T1.5 · 图组装：节点 + 条件边 + Send 扇出。

流程（P2 形态，单轮）：
    START → intent_router ─┬─ direct  → direct_responder → END
                           └─ research → planner ─(Send 扇出 N 路)→ scout_web ⤵
                                         evidence_judge → analyst → END
（Reflect 补检循环、审计、熔断在 P3/P4 接入，此处预留位置，不提前实现。）

关键：`scout_web` 是 Send 的目标节点，它的入参是**单个子问题包**（不是全量 state）；
它的返回 `evidence` 经 `operator.add` 归并回主 state——这就是 L2-1 的落点。
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..agents import analyst, direct_responder, evidence_judge, intent_router, planner, scout_web
from ..agents.base import NodeContext, bind
from ..agents.planner import fan_out_queries
from ..budget.account import BudgetConfig
from ..config import DATA_DIR, Settings
from ..llm.gateway import LLMGateway
from ..logging import get_logger
from ..retrieval.mock_search import FixtureSearchClient
from ..retrieval.ports import SearchClient
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


def build_context(
    settings: Settings,
    *,
    search: SearchClient | None = None,
    trace: TraceWriter | None = None,
    run_id: str = "local",
) -> NodeContext:
    trace = trace or TraceWriter(path=settings.trace_dir / "trace.jsonl", run_id=run_id)
    return NodeContext(
        settings=settings,
        gateway=LLMGateway(settings=settings, trace=trace),
        search=search or make_search_client(settings),
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


def fanout_sub_questions(state: GraphState) -> list[Send]:
    payloads = fan_out_queries(state)
    log.info(f"[graph] Send 扇出 {len(payloads)} 路子问题检索")
    return [Send("scout_web", p) for p in payloads]


# ------------------------------------------------------------------ 组装


def build_graph(ctx: NodeContext, *, checkpointer: Any = None):
    g = StateGraph(GraphState)

    g.add_node("intent_router", bind(intent_router.run, name="intent_router", ctx=ctx))
    g.add_node("direct_responder", bind(direct_responder.run, name="direct_responder", ctx=ctx))
    g.add_node("planner", bind(planner.run, name="planner", ctx=ctx))
    g.add_node("scout_web", bind(scout_web.run, name="scout_web", ctx=ctx))
    g.add_node("evidence_judge", bind(evidence_judge.run, name="evidence_judge", ctx=ctx))
    g.add_node("analyst", bind(analyst.run, name="analyst", ctx=ctx))

    g.add_edge(START, "intent_router")
    g.add_conditional_edges(
        "intent_router", route_after_intent, ["direct_responder", "planner"]
    )
    g.add_conditional_edges("planner", fanout_sub_questions, ["scout_web"])
    g.add_edge("scout_web", "evidence_judge")
    g.add_edge("evidence_judge", "analyst")
    g.add_edge("analyst", END)
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
        "evidence_judge",
        "analyst",
    ]
