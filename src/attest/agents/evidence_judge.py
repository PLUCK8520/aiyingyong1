"""T2.4 · 证据判别：逐条评相关度/可信度，并指出缺口（供 Reflect 补检）。"""

from __future__ import annotations

from typing import Any

from ..llm.prompts import build_judge_messages
from ..logging import get_logger
from ..schemas import JudgeResult
from .base import NodeContext, accumulate, budget_after, report_budget

log = get_logger(__name__)
NODE = "evidence_judge"
TAG = "judge"


def run(state: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    plan = state.get("plan") or {}
    objective = plan.get("objective", "")
    sub_questions: list[str] = plan.get("sub_questions") or []
    evidence = list(state.get("evidence") or [])

    if not evidence:
        log.node(TAG, NODE, "跳过", reason="本轮无证据")
        return {"judgments": [], "gaps": ["本轮未检索到任何证据，建议放宽查询词或补充信源"]}

    log.node(TAG, NODE, "开始", evidence=len(evidence), sub_questions=len(sub_questions))

    resp = ctx.gateway.chat(
        build_judge_messages(objective, sub_questions, evidence),
        task="judge",
        response_model=JudgeResult,
        temperature=0.0,
    )
    result: JudgeResult = resp.parsed  # type: ignore[assignment]

    cost, toks = budget_after(state, resp)
    report_budget(ctx, NODE, cost, toks)
    low = [j for j in result.judgments if j.relevance <= 2]
    log.node(
        TAG,
        NODE,
        "完成",
        judgments=len(result.judgments),
        low_relevance=len(low),
        gaps=len(result.gaps),
    )

    return {
        "judgments": result.judgments,
        "gaps": result.gaps,
        **accumulate(resp),
    }
