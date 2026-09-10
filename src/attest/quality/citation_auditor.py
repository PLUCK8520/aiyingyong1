"""T4.1 · 引用审计器：报告拆句 → 按 citation_id 分组批量核验 → supported / partial / unsupported。

两个实现、接口一致（对齐 `make_reranker` 的离线兜底范式）：

  - `RuleCitationAuditor`：规则版（实词覆盖 + 数值硬否决）。零成本、零随机性，离线默认。
  - `LLMCitationAuditor`：LLM 版（FR-17 分组批量核验 +「flash 初筛 → 强模型复核」两段式）。
    调用失败或结构化输出校验不过时**退回规则版**——审计闸门不能因为模型抽风而放行。

与 T3.10 的关系：T3.10 先在 20 条样本对上把**规则版**单点跑出准确率（91.3%，≥70% 闸门达标），
P4 这里换的是"谁来判"，**判据口径与评测脚本都不动**（`eval/audit_accuracy.py` 换个实现即可复测）。
这就是"先立尺子、再换零件"。

记账不变式：规则版不是 LLM 调用；LLM 版确实经 `LLMGateway.chat`，所以"绕过网关=漏账"仍成立。
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..budget.account import FUSE_HARD
from ..config import Settings
from ..llm.gateway import LLMGateway
from ..llm.prompts import build_audit_messages
from ..logging import get_logger
from ..retrieval.ports import Evidence
from ..schemas import AuditItem, AuditResult
from .audit import (
    extract_claims_sectioned,
    sentence_citation_map,
    union_evidence_text,
    verify_claim,
)

log = get_logger(__name__)


def _evidence_by_id(evidence: Sequence[Evidence]) -> dict[str, str]:
    return {e.citation_id: e.content for e in evidence}


@runtime_checkable
class CitationAuditor(Protocol):
    name: str

    def audit(
        self,
        report: str,
        evidence: Sequence[Evidence],
        *,
        objective: str = "",
        fuse_level: int = 0,
    ) -> list[AuditItem]: ...


class RuleCitationAuditor:
    """规则版：实词覆盖 + 数值一致性硬否决（判据见 `audit.verify_claim`）。

    `fuse_level` 对规则版**无影响**：它零成本、零调用，本就是最省的一档，
    没有"只跑初筛"可降。签名保持一致只为满足 Protocol（调用方无需分支）。
    """

    name = "rule"

    def audit(
        self,
        report: str,
        evidence: Sequence[Evidence],
        *,
        objective: str = "",
        fuse_level: int = 0,
    ) -> list[AuditItem]:
        ev = _evidence_by_id(evidence)
        claims = extract_claims_sectioned(report, ev.keys())
        # 判定用**该句所引来源的并集**：一句话可能同时引用两个来源（「争议与分歧」里很常见），
        # 按单条来源核对数值必然缺数 → 假阳性降级。实测踩过。
        unions = {
            sent: union_evidence_text(cids, ev)
            for sent, cids in sentence_citation_map(claims).items()
        }
        out: list[AuditItem] = []
        for sent, cid, section in claims:
            verdict, reason = verify_claim(sent, unions.get(sent, ev.get(cid, "")))
            out.append(
                AuditItem(
                    citation_id=cid, verdict=verdict, reason=reason, sentence=sent, section=section
                )
            )
        return out


class LLMCitationAuditor:
    """LLM 版：flash 初筛全部 → 对**非 supported**（降级候选）用强模型复核（FR-17）。

    两段式的成本动机：supported 占多数，且"判错"的代价小（无非少降级一条）；
    真正高风险的是"要动手降级"的那些，所以只对它们上强模型。
    """

    name = "llm"

    def __init__(self, settings: Settings, gateway: LLMGateway) -> None:
        self.settings = settings
        self.gateway = gateway
        self._rule = RuleCitationAuditor()

    def audit(
        self,
        report: str,
        evidence: Sequence[Evidence],
        *,
        objective: str = "",
        fuse_level: int = 0,
    ) -> list[AuditItem]:
        ev = _evidence_by_id(evidence)
        claims = extract_claims_sectioned(report, ev.keys())
        if not claims:
            return []
        section_of = {(sent, cid): section for sent, cid, section in claims}
        sent_cids = sentence_citation_map(claims)
        unions = {s: union_evidence_text(cids, ev) for s, cids in sent_cids.items()}

        # 按 citation_id 分组（FR-17 的成本约束所在）；组内证据取"该组句子所引来源的并集"，
        # 与规则版同口径——否则多源句子会被误判。
        grouped: dict[str, list[str]] = {}
        for sent, cid, _ in claims:
            bucket = grouped.setdefault(cid, [])
            if sent not in bucket:
                bucket.append(sent)
        groups: list[tuple[str, str, list[str]]] = []
        for cid, sents in grouped.items():
            cids: list[str] = [cid]
            for s in sents:
                for c in sent_cids.get(s, []):
                    if c not in cids:
                        cids.append(c)
            groups.append((cid, union_evidence_text(cids, ev), sents))

        screen = self._call(groups, objective, "audit_fast", fuse_level)
        if screen is None:
            return self._rule.audit(report, evidence, objective=objective)

        items = self._materialize(screen, grouped, unions, ev, section_of)

        flagged = [it for it in items if it.verdict != "supported"]
        # T4.3 / 熔断 L2：「审计只跑初筛」——跳过强模型复核这一趟调用。
        # 代价是复核兜底没了，所以降级候选一律**保守处理**（保留初筛裁决，不因复核而翻案）；
        # 收益是省掉本报告最贵的一次调用（audit_strong 走 qwen3.8-max）。
        if flagged and fuse_level >= FUSE_HARD:
            log.warning(
                f"[audit] 熔断 L{fuse_level}：审计只跑初筛，跳过强模型复核（{len(flagged)} 条降级候选）"
            )
        elif flagged:
            recheck = self._call(
                [
                    (it.citation_id, unions.get(it.sentence, ev.get(it.citation_id, "")), [it.sentence])
                    for it in flagged
                ],
                objective,
                "audit_strong",
                fuse_level,
            )
            if recheck:
                override = {(it.citation_id, it.sentence): it for it in recheck}
                items = [
                    self._with_section(override.get((it.citation_id, it.sentence), it), section_of)
                    for it in items
                ]
        return items

    # ---------------- 内部 ----------------

    def _call(
        self,
        groups: Sequence[tuple[str, str, Sequence[str]]],
        objective: str,
        task: str,
        fuse_level: int = 0,
    ) -> list[AuditItem] | None:
        try:
            resp = self.gateway.chat(
                build_audit_messages(objective, groups),
                task=task,
                response_model=AuditResult,
                temperature=0.0,
                fuse_level=fuse_level,
            )
        except Exception as exc:  # noqa: BLE001 - 审计失败不允许静默放行，退回规则版
            log.warning(f"[audit] {task} 调用/解析失败，退回规则版：{type(exc).__name__}: {exc}")
            return None
        parsed = resp.parsed
        return list(getattr(parsed, "items", []) or [])

    def _materialize(
        self,
        raw: Sequence[AuditItem],
        grouped: dict[str, list[str]],
        unions: dict[str, str],
        ev: dict[str, str],
        section_of: dict[tuple[str, str], str],
    ) -> list[AuditItem]:
        """把模型返回对齐到"应判定集合"：**漏判的句子由规则版补**，保证不落空。"""
        got = {(it.citation_id, it.sentence): it for it in raw}
        out: list[AuditItem] = []
        for cid, sentences in grouped.items():
            for sent in sentences:
                it = got.get((cid, sent))
                if it is None:
                    verdict, reason = verify_claim(sent, unions.get(sent, ev.get(cid, "")))
                    it = AuditItem(
                        citation_id=cid,
                        verdict=verdict,
                        reason=f"（模型漏判，规则版补）{reason}",
                        sentence=sent,
                    )
                out.append(self._with_section(it, section_of))
        return out

    @staticmethod
    def _with_section(
        item: AuditItem, section_of: dict[tuple[str, str], str]
    ) -> AuditItem:
        if item.section:
            return item
        return item.model_copy(
            update={"section": section_of.get((item.sentence, item.citation_id), "")}
        )


def make_citation_auditor(settings: Settings, gateway: LLMGateway) -> CitationAuditor | None:
    """按模式选审计器；返回 None 表示审计关闭（节点会如实留痕，不假装审计过）。"""
    if not settings.audit_enabled:
        log.info("[audit] 引用审计已关闭（ATTEST_AUDIT_ENABLED=0）")
        return None
    if settings.llm_mode != "dashscope":
        log.info("[audit] 引用审计走规则版（离线；与 T3.10 同口径，91.3% 基线）")
        return RuleCitationAuditor()
    log.info("[audit] 引用审计走 LLM 版（flash 初筛 + 强模型复核）——**本分支尚未实跑**")
    return LLMCitationAuditor(settings, gateway)
