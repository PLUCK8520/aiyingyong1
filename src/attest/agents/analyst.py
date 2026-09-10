"""T2.5 · Analyst：按大纲分章撰写，内联引用编号；T2.6 后处理在同节点内完成。

一个刻意的设计：**低相关证据先被筛掉再交给撰稿**。
判别节点给出的 relevance ≤ 2 的证据不进正文——这让"证据判别"这一步是**有后果的**，
而不是流水线上的装饰。若筛完为空，则退回全量并留痕（不静默降级）。
"""

from __future__ import annotations

from typing import Any

from ..budget.account import FUSE_HARD
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


def truncate_topk(
    evidence: list[Evidence], judgments: list[Any], k: int
) -> list[Evidence]:
    """T4.3 / 熔断 L2 · 上下文截断：按相关性降序取前 k 条（内部保持原有相对顺序）。

    为什么是"按相关性"而不是"按顺序砍尾巴"：证据经 reducer 累加，顺序≈检索先后，
    与重要度无关；reflect 补检来的证据排在后面，砍尾会把补检成果整段丢掉。
    无判别信息（judgments 为空）时相关性一律记 0，退化为保留前 k 条——**已留痕**。
    """
    if k <= 0 or len(evidence) <= k:
        return list(evidence)
    relevance = {j.citation_id: int(getattr(j, "relevance", 0) or 0) for j in judgments}
    ranked = sorted(range(len(evidence)), key=lambda i: (-relevance.get(evidence[i].citation_id, 0), i))
    return [evidence[i] for i in sorted(ranked[:k])]


def run(state: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    plan = state.get("plan") or {}
    objective = plan.get("objective", "调研报告")
    outlines: list[str] = plan.get("outlines") or []
    evidence = list(state.get("evidence") or [])
    conflicts = list(state.get("conflicts") or [])
    fuse = ctx.budget_snapshot(state).fuse_level

    selected, filter_info = _select_evidence(state)

    # T4.3 / L2：预算硬熔断 → 上下文截断（证据按相关性取 top-k）。**不改证据本身**，
    # 只影响"这一次写正文时送进上下文多少"；引用索引仍基于全量证据（回查能力不受影响）。
    trunc_info: dict[str, Any] = {}
    if fuse >= FUSE_HARD and len(selected) > ctx.settings.fuse_ctx_top_k:
        before = len(selected)
        selected = truncate_topk(selected, list(state.get("judgments") or []), ctx.settings.fuse_ctx_top_k)
        trunc_info = {"truncated": True, "before": before, "after": len(selected)}
        ctx.trace.emit(
            "context_truncated",
            node=NODE,
            fuse_level=fuse,
            top_k=ctx.settings.fuse_ctx_top_k,
            **trunc_info,
        )

    ctx.trace.emit(
        "evidence_selected",
        node=NODE,
        selected=len(selected),
        conflicts=len(conflicts),
        fuse_level=fuse,
        **filter_info,
        **trunc_info,
    )
    log.node(
        TAG,
        NODE,
        "开始",
        evidence=len(selected),
        outlines=len(outlines),
        conflicts=len(conflicts),
        fuse_level=fuse,
        **filter_info,
        **trunc_info,
    )

    resp = ctx.gateway.chat(
        build_analyst_messages(objective, outlines, selected, conflicts),
        task="analyst",
        temperature=0.3,
        fuse_level=fuse,
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
        # T4.4：记录正文是用哪个档位写的，供 trace / 成本面板复盘"降级后报告由谁产出"
        "model_tier": ctx.gateway.router.tier("analyst", fuse_level=fuse),
        **accumulate(resp),
    }
