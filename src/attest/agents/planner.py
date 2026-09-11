"""T1.4 · 任务规划：把问题拆成可独立检索的子问题 + 报告大纲。

结构化输出校验失败由网关自动重试（≤2 次），此处不再重复兜底逻辑。
"""

from __future__ import annotations

from typing import Any

from ..llm.prompts import build_planner_messages
from ..logging import get_logger
from ..schemas import Plan
from .base import NodeContext, accumulate, budget_after, report_budget

log = get_logger(__name__)
NODE = "planner"
TAG = "planner"


def run(state: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    query = state.get("query", "")
    log.node(TAG, NODE, "开始", query=query)

    # T5.2：把用户画像注入系统提示词（空画像则为空串，不注入空段落）
    from .memory_loader import profile_block as _profile_block

    resp = ctx.gateway.chat(
        build_planner_messages(query, profile_block=_profile_block(state)),
        task="planner",
        response_model=Plan,
        temperature=0.2,
    )
    plan: Plan = resp.parsed  # type: ignore[assignment]

    cost, toks = budget_after(state, resp)
    report_budget(ctx, NODE, cost, toks)
    log.node(
        TAG,
        NODE,
        "完成",
        objective=plan.objective,
        sub_questions=len(plan.sub_questions),
        outlines=len(plan.outlines),
        attempts=resp.attempts,
    )

    return {"plan": plan.model_dump(), **accumulate(resp)}


def fan_out_queries(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Send 映射函数：每个子问题一个检索任务（序号从 1 开始，决定引用编号）。"""
    plan = state.get("plan") or {}
    subs: list[str] = plan.get("sub_questions") or []
    rnd = int(state.get("round", 1) or 1)
    return [
        {
            "objective": plan.get("objective", ""),
            "sub_question": sq,
            "subq_no": idx,
            "round": rnd,
        }
        for idx, sq in enumerate(subs, start=1)
    ]
