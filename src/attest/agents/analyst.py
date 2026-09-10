"""T2.5 · Analyst：按大纲分章撰写，内联引用编号；T2.6 后处理在同节点内完成。

一个刻意的设计：**低相关证据先被筛掉再交给撰稿**。
判别节点给出的 relevance ≤ 2 的证据不进正文——这让"证据判别"这一步是**有后果的**，
而不是流水线上的装饰。若筛完为空，则退回全量并留痕（不静默降级）。
"""

from __future__ import annotations

from typing import Any

from ..llm.prompts import build_analyst_messages
from ..logging import get_logger
from ..quality.citation_check import finalize
from ..retrieval.citations import CitationIndex
from ..retrieval.ports import Evidence
from .base import NodeContext, accumulate, budget_after, report_budget

log = get_logger(__name__)
NODE = "analyst"
TAG = "analyst"

MIN_RELEVANCE = 3


def _select_evidence(state: dict[str, Any]) -> tuple[list[Evidence], dict[str, Any]]:
    evidence: list[Evidence] = list(state.get("evidence") or [])
    judgments = list(state.get("judgments") or [])
    if not judgments:
        return evidence, {"filtered": False, "reason": "无判别结果，使用全量证据"}

    keep = {j.citation_id for j in judgments if j.relevance >= MIN_RELEVANCE}
    selected = [e for e in evidence if e.citation_id in keep]
    if not selected:
        return evidence, {"filtered": False, "reason": "筛选后为空，退回全量（已留痕）"}
    return selected, {"filtered": True, "dropped": len(evidence) - len(selected)}


def run(state: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    plan = state.get("plan") or {}
    objective = plan.get("objective", "调研报告")
    outlines: list[str] = plan.get("outlines") or []
    evidence = list(state.get("evidence") or [])

    selected, filter_info = _select_evidence(state)
    ctx.trace.emit("evidence_selected", node=NODE, selected=len(selected), **filter_info)
    log.node(TAG, NODE, "开始", evidence=len(selected), outlines=len(outlines), **filter_info)

    resp = ctx.gateway.chat(
        build_analyst_messages(objective, outlines, selected),
        task="analyst",
        temperature=0.3,
    )

    # 索引基于**全量证据**建：筛选只影响"写什么"，不影响"能不能回查"
    index = CitationIndex.from_evidence(evidence)
    final_report, refs, check = finalize(resp.text, index)

    if not check["pass"]:
        ctx.trace.emit("citation_check_failed", node=NODE, **check)
        log.warning(
            f"[analyst] 引用完整性校验未通过：unresolved={check['unresolved']} "
            f"referenced={check['referenced']}"
        )

    cost, toks = budget_after(state, resp)
    report_budget(ctx, NODE, cost, toks)
    log.node(
        TAG,
        NODE,
        "完成",
        chars=len(final_report),
        referenced=check["referenced"],
        unresolved=len(check["unresolved"]),
        cited_ratio=check["cited_ratio"],
    )

    return {
        "report": final_report,
        "reference_list": refs,
        "citation_check": check,
        **accumulate(resp),
    }
