"""T7.4 · 沉淀节点：把本次报告里**已核验为 supported** 的结论写进 research_memory。

**位置**：`auditor → memory_writer → END`。放在审计**之后**是刻意的——
审计是唯一能给出 `verdict` 的环节，沉淀必须等它判完才知道哪些结论可信。

**为什么独立成节点而不是塞进 auditor**：
  1. 职责单一：auditor 判"引用是否成立"，本节点判"哪些结论值得记住"；
  2. 可关断：`settings.research_memory_enabled=False` 时本节点如实留痕跳过，
     主流程零改动（与 `memory_loader` 的关闭语义一致）；
  3. 评测可见：沉淀了多少、拒收了多少都进 trace，便于验证"防自我投毒真的生效"。

与其它节点一致：签名 `(state, ctx) -> increment`，由 `bind()` 包成 LangGraph 节点。
"""

from __future__ import annotations

from typing import Any

from ..logging import get_logger
from ..memory.research_memory import distill_from_report
from .base import NodeContext

log = get_logger(__name__)
NODE = "memory_writer"
TAG = "memory_writer"


def run(state: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    if not ctx.settings.research_memory_enabled:
        ctx.trace.emit("memory_write", node=NODE, skipped="research_memory 未启用")
        log.node(TAG, NODE, "跳过", reason="research_memory 未启用")
        return {}

    mem = getattr(ctx, "memory", None)
    if mem is None:
        ctx.trace.emit("memory_write", node=NODE, skipped="未装配 research_memory")
        log.node(TAG, NODE, "跳过", reason="未装配 research_memory")
        return {}

    report_id = str(state.get("thread_id") or "unknown")
    audit_items = list(state.get("audit_items") or [])

    # T7.10：拒编页没有任何 supported 结论，沉淀无从谈起。显式跳过（而不是"跑一遍写 0 条"），
    # 让"为什么没沉淀"在 trace 里可归因。
    suff = state.get("evidence_sufficiency") or {}
    if suff and suff.get("sufficient") is False and not audit_items:
        ctx.trace.emit("memory_write", node=NODE, skipped="证据不足（拒编页无结论可沉淀）")
        log.node(TAG, NODE, "跳过", reason="拒编页无结论可沉淀")
        return {}

    candidates = distill_from_report(report_id=report_id, audit_items=audit_items)
    written, rejected = mem.add(candidates)

    ctx.trace.emit(
        "memory_write",
        node=NODE,
        report_id=report_id,
        candidates=len(candidates),
        written=written,
        rejected=rejected,
        total=mem.count(),
    )
    log.node(
        TAG,
        NODE,
        "完成",
        report_id=report_id,
        candidates=len(candidates),
        written=written,
        rejected=rejected,
    )
    # 本节点只做副作用（写向量库），不改 state——返回空 increment 是合法且清晰的。
    return {}
