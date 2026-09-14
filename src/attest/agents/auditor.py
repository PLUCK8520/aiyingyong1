"""T4.1 / T4.2 · 引用审计节点：报告成文后的**核验闸门**。

为什么单独成节点而不是塞进 analyst：
  - analyst 的职责是"写"，auditor 的职责是"查"——两者失败模式不同，混在一起会让
    "报告为什么被降级"无法单独观测；
  - 引用审计是项目的头号亮点（L1-1），值得在 trace 里有一个独立、成对的节点。

两条降级动作（T4.2，均在**不静默删证据**的前提下做）：

  a. 逐句降级：unsupported 的句子去掉引用编号并标注「（未证实）」——让读者看见
     "这里原本有个结论、但它没通过核验"，而不是被无声抹掉；
  b. 章节重写（T4.2b）：某章节 unsupported 占比 > 阈值 → 用证据**只重写这一节**，
     限 `audit_max_rewrites` 次；重写后仍不达标，就落到 (a) 的逐句标注兜底。

T4.3 熔断联动（只读派生值 `fuse_level`，本节点不决定级别）：
  - L2：报告头加降级横幅，让"这份报告是在降级状态下产出的"成为报告自身的一部分。

参考资料一律用**同一 `CitationIndex`** 重渲染，保证正文编号与参考列表永远对齐。
"""

from __future__ import annotations

from typing import Any

from ..budget.account import FUSE_HARD
from ..llm.prompts import build_rewrite_messages, parse_rewrite_output
from ..logging import get_logger
from ..quality.audit import (
    citations_in_section,
    degrade_report,
    replace_section,
    rewrite_targets,
    section_body,
    section_failure_ratios,
)
from ..quality.citation_check import REF_HEADING, finalize
from ..retrieval.citations import CitationIndex, extract_ids
from ..retrieval.ports import Evidence
from ..schemas import AuditItem
from .base import NodeContext

log = get_logger(__name__)
NODE = "auditor"
TAG = "auditor"

#: L2 降级横幅。内容刻意写清"三件事各自发生了什么"，避免读者把降级误当正常产出。
FUSE_BANNER = (
    "> ⚠️ **降级提示**：本次运行触发预算硬熔断（L2）。已停补检、证据上下文按相关性截断、"
    "引用审计仅跑初筛。报告仍如实产出，但覆盖度可能不完整。\n"
)

#: T8.6 正文零引用时的横幅。与降级横幅**同机制、同位置**（一级标题之后），
#: 因为"这份报告没通过回查"和"这份报告是降级产出的"属于同一类信息：
#: 都是**告诉读者该怎么看待下面的内容**，不能只写在日志里。
#:
#: 为什么必须有这道确定性兜底：实测（2026-09-13）模型可能整篇不写引用编号。
#: 此时审计器**没有任何对象可审**（`total=0`），摘要里 `unsupported_ratio=0.0`
#: 会被前端与读者读成"全部通过"——而实际上正文里可能同时存在
#: 证据根本没支持过的内容（实测出现过凭空生成的"财政补贴 / 社会资本"）。
#: 也就是说：**引用缺失会让整个核验网失效，且失效的表现是"看起来干净"**。
UNCHECKED_BANNER = (
    "> ⚠️ **本报告未通过逐句回查**：正文没有生成任何引用编号，"
    "因此「每一句事实性结论都能追溯到来源」这一保证对本次报告**不成立**。"
    "下面的具体数字与表述请勿直接引用；建议补充更聚焦的资料后重跑。\n"
)


def _insert_after_title(report: str, banner: str) -> str:
    """把横幅插到一级标题之后（无标题则置顶）。调用方负责幂等判定。"""
    lines = report.splitlines()
    idx = next((i for i, ln in enumerate(lines) if ln.startswith("# ")), -1)
    block = ["", banner.rstrip()]
    if idx == -1:
        return "\n".join([banner.rstrip(), ""] + lines)
    return "\n".join(lines[: idx + 1] + block + lines[idx + 1 :])


def _with_fuse_banner(report: str, fuse: int) -> str:
    """L2 时把降级横幅插到一级标题之后（无标题则置顶）。幂等——重复调用不叠加。"""
    if fuse < FUSE_HARD or "降级提示" in report:
        return report
    return _insert_after_title(report, FUSE_BANNER)


def _with_unchecked_banner(report: str, unchecked: bool) -> str:
    """正文零引用时插入"未通过回查"横幅。幂等。"""
    if not unchecked or "未通过逐句回查" in report:
        return report
    return _insert_after_title(report, UNCHECKED_BANNER)


def _llm_rewrite(
    ctx: NodeContext,
    objective: str,
    title: str,
    body: str,
    evidence: list[Evidence],
    fuse: int,
) -> tuple[str, Any]:
    """重写单节。返回 (新正文, 响应)；调用失败返回 ("", None)——**失败不中断链路**。"""
    try:
        resp = ctx.gateway.chat(
            build_rewrite_messages(objective, title, body, evidence),
            task="analyst_rewrite",
            temperature=0.2,
            fuse_level=fuse,
        )
    except Exception as exc:  # noqa: BLE001 - 重写是补救动作，失败就退回逐句标注
        log.warning(f"[auditor] 章节「{title}」重写调用失败：{type(exc).__name__}: {exc}")
        return "", None
    return parse_rewrite_output(resp.text), resp


def _rewrite_loop(
    report: str,
    items: list[AuditItem],
    evidence: list[Evidence],
    *,
    objective: str,
    ctx: NodeContext,
    fuse: int,
) -> tuple[str, list[AuditItem], list[dict[str, Any]], float, int]:
    """T4.2b：失败率超阈值的章节 → 重写 → 复检，最多 `audit_max_rewrites` 轮。

    **重写者是注入点**（走 `ctx.gateway`），单测用桩网关即可断言"重写→复检"序列，
    不需要真实模型——与 P3 的离线兜底范式一致。
    """
    max_rewrites = int(ctx.settings.audit_max_rewrites)
    threshold = float(ctx.settings.audit_rewrite_ratio)
    ev_by_id = {e.citation_id: e for e in evidence}
    history: list[dict[str, Any]] = []
    inc_cost = 0.0
    inc_toks = 0
    done = 0

    while done < max_rewrites:
        targets = rewrite_targets(items, threshold)
        if not targets:
            break
        changed = False
        for title, ratio in targets.items():
            old_body = section_body(report, title)
            sec_ev = [ev_by_id[c] for c in citations_in_section(report, title) if c in ev_by_id]
            if not old_body or not sec_ev:
                history.append(
                    {
                        "section": title,
                        "ratio": ratio,
                        "action": "skip",
                        "reason": "章节无正文或该节无可用证据，退回逐句标注",
                    }
                )
                continue
            new_body, resp = _llm_rewrite(ctx, objective, title, old_body, sec_ev, fuse)
            if resp is not None:
                inc_cost += resp.cost_cny
                inc_toks += resp.total_tokens
            new_report = replace_section(report, title, new_body) if new_body else report
            if new_report == report:
                history.append({"section": title, "ratio": ratio, "action": "no_change"})
                continue
            report = new_report
            changed = True
            history.append({"section": title, "ratio_before": ratio, "action": "rewritten"})
        if not changed:
            break
        done += 1
        # 重写后**必须复检**：不复检就无法回答"重写到底有没有把不支撑的句子清掉"。
        if ctx.auditor is not None:
            items = ctx.auditor.audit(report, evidence, objective=objective, fuse_level=fuse)

    final_ratios = section_failure_ratios(items)
    for h in history:
        if h.get("action") == "rewritten":
            h["ratio_after"] = final_ratios.get(h["section"], 0.0)
    return report, items, history, inc_cost, inc_toks


def run(state: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    report = state.get("report") or ""
    auditor = ctx.auditor
    if not report or auditor is None:
        ctx.trace.emit("audit_skipped", node=NODE, reason="审计关闭或无报告")
        return {}

    # T7.10：证据不足时的报告是**拒编页**（不含任何带引用的结论），审计没有对象。
    # 跑一遍只会得到"0 条判定"，还会把 citation_check 覆盖成"未通过"（报告里没有引用编号）——
    # 那是噪音，如实跳过并留痕才是正确语义。
    suff = state.get("evidence_sufficiency") or {}
    if suff and suff.get("sufficient") is False:
        ctx.trace.emit(
            "audit_skipped",
            node=NODE,
            reason="证据不足，报告为拒编页，无引用可审",
            n_evidence=suff.get("n_evidence"),
        )
        log.node(TAG, NODE, "跳过", reason="拒编页无引用可审")
        return {}

    evidence: list[Evidence] = list(state.get("evidence") or [])
    plan = state.get("plan") or {}
    objective = plan.get("objective", "调研报告")
    fuse = ctx.budget_snapshot(state).fuse_level

    log.node(TAG, NODE, "开始", evidence=len(evidence), fuse_level=fuse)
    items = auditor.audit(report, evidence, objective=objective, fuse_level=fuse)

    # ---- T4.2b：章节重写（在逐句降级之前做，重写成功的句子就没必要再标注）----
    report, items, rewrite_hist, rw_cost, rw_toks = _rewrite_loop(
        report, items, evidence, objective=objective, ctx=ctx, fuse=fuse
    )
    if rewrite_hist:
        ctx.trace.emit("section_rewrite", node=NODE, history=rewrite_hist, rounds=len(rewrite_hist))

    # ---- T4.2a：逐句降级 ----
    # ⚠️ `unchecked` 必须用**降级前**的正文判定，且先切掉参考资料章节（它本身列编号）。
    # 否则"审计把不支撑的编号降级成（未证实）"会被误读成"模型没写编号"——
    # 两条完全不同的故障会显示成同一句横幅，排查时会被带偏。
    unchecked = bool(evidence) and not extract_ids((report or "").split(REF_HEADING, 1)[0])
    degraded, info = degrade_report(report, items)
    body = degraded.split(REF_HEADING, 1)[0].rstrip()
    body = _with_unchecked_banner(body, unchecked)
    body = _with_fuse_banner(body, fuse)
    index = CitationIndex.from_evidence(evidence)
    final_report, refs, check = finalize(body, index)

    counts = {"supported": 0, "partial": 0, "unsupported": 0}
    for it in items:
        counts[it.verdict] = counts.get(it.verdict, 0) + 1
    total = len(items)
    summary: dict[str, Any] = {
        "auditor": auditor.name,
        "total": total,
        **counts,
        "unsupported_ratio": round(counts["unsupported"] / total, 4) if total else 0.0,
        "degraded_sentences": info["degraded"],
        "removed_citations": info["removed_citations"],
        "sections": info["sections"],
        "rewrites": sum(1 for h in rewrite_hist if h.get("action") == "rewritten"),
        "rewrite_history": rewrite_hist,
        "fuse_level": fuse,
        "section_failure_ratios": section_failure_ratios(items),
        # T8.6：审计"没有结论"要能与"审计通过"区分开。
        # 零引用时 total=0，只给数字的话**任何读者都会把它读成干净**——
        # 这正是「凭空内容随报告交付」的通道，必须显式命名。
        "inconclusive": unchecked,
        "inconclusive_reason": (
            "正文未包含任何引用编号，审计没有可核验的对象——**本次审计没有结论，不等于通过**。"
            if unchecked
            else ""
        ),
    }

    if unchecked:
        ctx.trace.emit(
            "audit_inconclusive",
            node=NODE,
            reason="正文零引用，无可审计对象",
            evidence=len(evidence),
        )

    ctx.trace.emit(
        "citation_audit",
        node=NODE,
        **{k: v for k, v in summary.items() if k != "section_failure_ratios"},
    )
    log.node(
        TAG,
        NODE,
        "完成",
        total=total,
        supported=counts["supported"],
        partial=counts["partial"],
        unsupported=counts["unsupported"],
        degraded=info["degraded"],
        rewrites=summary["rewrites"],
        fuse_level=fuse,
        inconclusive=unchecked,
    )

    return {
        "report": final_report,
        "reference_list": refs,
        "audit_items": list(items),
        "audit_summary": summary,
        "citation_check": check,
        "rewrite_count": summary["rewrites"],
        "cost_incurred": rw_cost,
        "tokens_incurred": rw_toks,
    }
