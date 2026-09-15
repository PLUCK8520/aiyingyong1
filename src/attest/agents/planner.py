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

    try:
        resp = ctx.gateway.chat(
            build_planner_messages(query, profile_block=_profile_block(state)),
            task="planner",
            response_model=Plan,
            temperature=0.2,
        )
        plan: Plan = resp.parsed  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001 - 规划失败不杀进程（报告必须产出，见下）
        # 降级为**最小可用 plan**：用原问题当唯一子问题。
        # 为什么不直接失败：规划失败时链路还没开始搜索，但"最小 plan"能让**真实检索照常发生**——
        # 出来的证据是真的、引用是真的，只是覆盖面窄（只有一个分支）。
        # 为什么不伪造多个子问题：凭空拆出来的子问题很可能与真实资料错配，
        # 那正是 T7.10 拒编闸门要防的"看起来完整、实则张冠李戴"。宁可少，不可假。
        ctx.trace.emit(
            "planner_failed",
            node=NODE,
            query=query,
            error=f"{type(exc).__name__}: {exc}",
        )
        log.error(
            f"[planner] 规划调用失败（已重试耗尽），降级为最小可用计划（单子问题）："
            f"{type(exc).__name__}: {exc}"
        )
        plan = Plan(
            objective=query or "调研",
            sub_questions=[query or "本次调研"],
            outlines=["调研发现"],
            requires_data=True,
        )
        log.node(TAG, NODE, "降级完成", sub_questions=1, outlines=1, degraded=True)
        return {"plan": plan.model_dump()}

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
