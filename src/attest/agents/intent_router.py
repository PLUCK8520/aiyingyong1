"""T1.2 · 意图二分流：direct / research（flash 模型，温度 0）。"""

from __future__ import annotations

from typing import Any

from ..llm.prompts import build_intent_messages
from ..logging import get_logger
from ..schemas import RouteDecision
from .base import NodeContext, accumulate, budget_after, report_budget

log = get_logger(__name__)
NODE = "intent_router"
TAG = "intent"


def run(state: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    query = state.get("query", "")
    log.node(TAG, NODE, "开始", query=query)

    resp = ctx.gateway.chat(
        build_intent_messages(query),
        task="intent",
        response_model=RouteDecision,
        temperature=0.0,
    )
    decision: RouteDecision = resp.parsed  # type: ignore[assignment]

    cost, toks = budget_after(state, resp)
    report_budget(ctx, NODE, cost, toks)
    log.node(TAG, NODE, "完成", route=decision.route, reason=decision.reason, attempts=resp.attempts)

    return {
        "route": decision.route,
        "route_reason": decision.reason,
        **accumulate(resp),
    }
