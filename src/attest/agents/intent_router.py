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
    fuse = ctx.budget_snapshot(state).fuse_level
    log.node(TAG, NODE, "开始", query=query, fuse_level=fuse)

    try:
        resp = ctx.gateway.chat(
            build_intent_messages(query),
            task="intent",
            response_model=RouteDecision,
            temperature=0.0,
            fuse_level=fuse,  # T4.3：熔断 L1 起，轻任务由路由器切便宜档
        )
        decision: RouteDecision = resp.parsed  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001 - 分流失败不杀进程（报告必须产出）
        # 降级走 **research**：这条路径会真实检索、判别、审计、出报告；
        # 而 direct 只是"不检索直接答"。若用户其实想问的是 direct 类问题，
        # 走 research 只是多花些成本，**不会给出错误答案**——宁可多做，不可漏做。
        ctx.trace.emit(
            "intent_failed",
            node=NODE,
            query=query,
            error=f"{type(exc).__name__}: {exc}",
        )
        log.error(
            f"[intent] 分流调用失败（已重试耗尽），降级走 research：{type(exc).__name__}: {exc}"
        )
        log.node(TAG, NODE, "降级完成", route="research", degraded=True)
        return {"route": "research", "route_reason": "意图分流调用失败，降级走完整调研路径"}

    cost, toks = budget_after(state, resp)
    report_budget(ctx, NODE, cost, toks)
    log.node(TAG, NODE, "完成", route=decision.route, reason=decision.reason, attempts=resp.attempts)

    return {
        "route": decision.route,
        "route_reason": decision.reason,
        **accumulate(resp),
    }
