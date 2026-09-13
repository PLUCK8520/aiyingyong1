"""T4.1 / T4.2 · 引用审计与降级的单测。

覆盖三块：
  1. 分节抽取 + 按 citation_id 分组（FR-17 的成本约束所在）；
  2. 规则版审计器（数值硬否决）与 LLM 版两段式（用桩网关验证调用序列与回退）；
  3. 降级动作：unsupported 去编号 + 标注「未证实」，且 supported 不受影响。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from attest.agents import auditor as auditor_node
from attest.quality.audit import (
    degrade_report,
    extract_claims_sectioned,
    group_claims_by_citation,
    section_failure_ratios,
    verify_claim,
)
from attest.quality.citation_auditor import LLMCitationAuditor, RuleCitationAuditor
from attest.retrieval.ports import Evidence
from attest.schemas import AuditItem, AuditResult

CID = "[WEB1-1-1]"
EVIDENCE_TEXT = "据测算，企业知识库 Agent 平台市场规模达 180 亿元，增速约 30%。"


def _ev(cid: str = CID, content: str = EVIDENCE_TEXT) -> Evidence:
    return Evidence(
        citation_id=cid,
        source="web",
        title="测试来源",
        url="https://example.com/a",
        content=content,
        sub_question="市场规模",
        round_no=1,
        score=1.0,
    )


# --------------------------------------------------------------- 分节 / 分组


def test_extract_claims_sectioned_tags_section() -> None:
    report = (
        "# 报告\n\n## 核心摘要\n\n市场规模达 180 亿元 "
        f"{CID}。\n\n## 竞品格局\n\n共 5 家主要玩家 {CID}。\n"
    )
    claims = extract_claims_sectioned(report, [CID])
    assert [c[1] for c in claims] == [CID, CID]
    assert [c[2] for c in claims] == ["核心摘要", "竞品格局"], claims


def test_extract_claims_sectioned_ignores_reference_section() -> None:
    report = (
        "# 报告\n\n## 摘要\n\n市场规模达 180 亿元 "
        f"{CID}。\n\n## 参考资料\n\n- `{CID}` [标题](https://example.com)\n"
    )
    claims = extract_claims_sectioned(report, [CID])
    assert len(claims) == 1, "参考资料章节不应被当成正文引用"


def test_group_claims_by_citation_merges_same_cid() -> None:
    grouped = group_claims_by_citation([("句一", CID), ("句二", CID), ("句三", "[WEB1-2-1]")])
    assert grouped[CID] == ["句一", "句二"]
    assert grouped["[WEB1-2-1]"] == ["句三"]


# --------------------------------------------------------------- 规则版审计


def test_rule_auditor_accepts_supported_claim() -> None:
    report = f"# 报告\n\n## 摘要\n\n市场规模达 180 亿元 {CID}。\n"
    items = RuleCitationAuditor().audit(report, [_ev()])
    assert len(items) == 1
    assert items[0].verdict == "supported", items[0].reason
    assert items[0].sentence.startswith("市场规模达 180 亿元")
    assert items[0].section == "摘要"


def test_rule_auditor_flags_number_mismatch_as_unsupported() -> None:
    """数值不一致是硬否决——这是引用审计的头号风险场景。"""
    report = f"# 报告\n\n## 摘要\n\n市场规模达 999 亿元 {CID}。\n"
    items = RuleCitationAuditor().audit(report, [_ev()])
    assert items[0].verdict == "unsupported"
    assert "数值" in items[0].reason


def test_rule_auditor_no_claims_returns_empty() -> None:
    assert RuleCitationAuditor().audit("# 报告\n\n## 摘要\n\n没有引用的句子。\n", [_ev()]) == []


def test_multi_source_sentence_uses_union_of_cited_evidence() -> None:
    """一句话同时引用两个来源 → 按两者**并集**判定。

    否则"并列双方口径"的句子（争议与分歧小节里极常见）会因单条来源缺数被误判 unsupported
    并遭降级——这是端到端实跑时踩到的假阳性。
    """
    ev_a = _ev("[WEB1-1-1]", "市场规模达 180 亿元，增速约 30%。")
    ev_b = _ev("[WEB1-1-2]", "另一口径：市场规模约 62 亿元，增速约 38%。")
    report = (
        "# 报告\n\n## 争议与分歧\n\n"
        "一方口径 180 亿元 [WEB1-1-1]，另一方口径 62 亿元 [WEB1-1-2]。\n"
    )
    items = RuleCitationAuditor().audit(report, [ev_a, ev_b])
    assert {i.citation_id for i in items} == {"[WEB1-1-1]", "[WEB1-1-2]"}
    assert all(i.verdict != "unsupported" for i in items), [
        (i.citation_id, i.verdict, i.reason) for i in items
    ]


def test_single_source_sentence_still_flagged_for_foreign_number() -> None:
    """只引用一个来源却出现该来源没有的数值 → 仍必须判 unsupported（并集不能放宽单源）。"""
    ev_a = _ev("[WEB1-1-1]", "市场规模达 180 亿元。")
    report = "# 报告\n\n## 摘要\n\n市场规模达 180 亿元、增速 99% [WEB1-1-1]。\n"
    items = RuleCitationAuditor().audit(report, [ev_a])
    assert items[0].verdict == "unsupported", items[0].reason


def test_verify_claim_ignores_citation_marker_tokens() -> None:
    """引用编号本身不是 claim 内容。

    踩过的坑：碎片句「… [WEB1-1-1]」里的 "WEB1" 被当成实词 → 覆盖率 0% → 误判 unsupported
    → 对一份**正确**的报告触发假降级。修法是判定前剥掉编号。
    """
    from attest.quality.audit import verify_claim

    verdict, reason = verify_claim("… [WEB1-1-1]", "市场规模达 180 亿元。")
    assert verdict != "unsupported", f"含编号的占位句被误判：{reason}"


def test_rule_auditor_does_not_degrade_truncated_fragment() -> None:
    report = f"# 报告\n\n## 摘要\n\n… {CID}\n"
    items = RuleCitationAuditor().audit(report, [_ev()])
    assert items
    assert all(i.verdict != "unsupported" for i in items), [(i.verdict, i.reason) for i in items]


# --------------------------------------------------------------- 降级动作


def test_degrade_report_removes_citation_and_marks() -> None:
    report = f"# 报告\n\n## 摘要\n\n市场规模达 999 亿元 {CID}。\n"
    items = RuleCitationAuditor().audit(report, [_ev()])
    degraded, info = degrade_report(report, items)
    assert info["degraded"] == 1
    assert CID in info["removed_citations"]
    assert CID not in degraded, "unsupported 的引用编号必须被移除"
    assert "（未证实）" in degraded
    assert degraded.count("（未证实）") == 1


def test_degrade_report_leaves_supported_sentences_intact() -> None:
    report = f"# 报告\n\n## 摘要\n\n市场规模达 180 亿元 {CID}。\n"
    items = RuleCitationAuditor().audit(report, [_ev()])
    degraded, info = degrade_report(report, items)
    assert degraded == report, "supported 句子不应被改动"
    assert info["degraded"] == 0


def test_degrade_report_handles_mixed_sentences() -> None:
    report = (
        "# 报告\n\n## 摘要\n\n市场规模达 180 亿元 "
        f"{CID}。\n增速约 42% {CID}。\n"
    )
    items = RuleCitationAuditor().audit(report, [_ev()])
    verdicts = sorted(i.verdict for i in items)
    assert verdicts == ["supported", "unsupported"], verdicts
    degraded, info = degrade_report(report, items)
    assert info["degraded"] == 1
    # 被支持的那句保留编号，未证实的那句编号被去掉
    assert f"市场规模达 180 亿元 {CID}。" in degraded
    assert "增速约 42%。（未证实）" in degraded


def test_section_failure_ratios() -> None:
    items = [
        AuditItem(citation_id=CID, verdict="unsupported", sentence="a", section="摘要"),
        AuditItem(citation_id=CID, verdict="supported", sentence="b", section="摘要"),
        AuditItem(citation_id=CID, verdict="supported", sentence="c", section="竞品"),
    ]
    ratios = section_failure_ratios(items)
    assert ratios["摘要"] == 0.5
    assert ratios["竞品"] == 0.0


# --------------------------------------------------------------- LLM 版两段式


@dataclass
class _StubResp:
    parsed: Any


class _StubGateway:
    """桩网关：记录调用序列，按 task 返回预置结果；raise_on 里的 task 直接抛错。"""

    def __init__(self, fast: AuditResult, strong: AuditResult | None = None, raise_on: set[str] | None = None):
        self.fast = fast
        self.strong = strong or fast
        self.raise_on = raise_on or set()
        self.calls: list[str] = []

    def chat(
        self, messages, *, task, response_model=None, temperature=0.0, fuse_level=0, **kwargs
    ):  # noqa: ANN001
        self.calls.append(task)
        if task in self.raise_on:
            raise RuntimeError("stub failure")
        return _StubResp(self.fast if task == "audit_fast" else self.strong)


def _audit_llm(report: str, fast: AuditResult, strong: AuditResult | None = None, raise_on=None):
    gateway = _StubGateway(fast, strong, raise_on)
    items = LLMCitationAuditor(settings=None, gateway=gateway).audit(  # type: ignore[arg-type]
        report, [_ev()], objective="测试"
    )
    return items, gateway


def test_llm_auditor_escalates_flagged_claims_to_strong_model() -> None:
    report = f"# 报告\n\n## 摘要\n\n市场规模达 180 亿元 {CID}。\n增速约 42% {CID}。\n"
    sentences = [c[0] for c in extract_claims_sectioned(report, [CID])]
    fast = AuditResult(
        items=[
            AuditItem(citation_id=CID, sentence=sentences[0], verdict="supported", reason="ok"),
            AuditItem(citation_id=CID, sentence=sentences[1], verdict="unsupported", reason="数值缺失"),
        ]
    )
    strong = AuditResult(
        items=[AuditItem(citation_id=CID, sentence=sentences[1], verdict="supported", reason="复核通过")]
    )
    items, gateway = _audit_llm(report, fast, strong)
    assert gateway.calls == ["audit_fast", "audit_strong"], gateway.calls
    assert all(i.verdict == "supported" for i in items), [i.verdict for i in items]
    assert all(i.section == "摘要" for i in items)


def test_llm_auditor_skips_strong_when_all_supported() -> None:
    report = f"# 报告\n\n## 摘要\n\n市场规模达 180 亿元 {CID}。\n"
    sentences = [c[0] for c in extract_claims_sectioned(report, [CID])]
    fast = AuditResult(
        items=[AuditItem(citation_id=CID, sentence=sentences[0], verdict="supported", reason="ok")]
    )
    items, gateway = _audit_llm(report, fast)
    assert gateway.calls == ["audit_fast"], "全部 supported 时不应升级到强模型"
    assert items[0].verdict == "supported"


def test_llm_auditor_falls_back_to_rule_on_failure() -> None:
    report = f"# 报告\n\n## 摘要\n\n市场规模达 999 亿元 {CID}。\n"
    fast = AuditResult(items=[])
    items, _ = _audit_llm(report, fast, raise_on={"audit_fast"})
    assert items[0].verdict == "unsupported", "审计调用失败必须退回规则版，不能放行"
    assert items[0].section == "摘要"


def test_llm_auditor_fills_missing_sentence_via_rule() -> None:
    """模型漏判某句时，规则版必须补上，保证"应判定集合"不落空。"""
    report = f"# 报告\n\n## 摘要\n\n市场规模达 180 亿元 {CID}。\n增速约 42% {CID}。\n"
    sentences = [c[0] for c in extract_claims_sectioned(report, [CID])]
    fast = AuditResult(
        items=[AuditItem(citation_id=CID, sentence=sentences[0], verdict="supported", reason="ok")]
    )
    items, _ = _audit_llm(report, fast)
    got = {i.sentence: i.verdict for i in items}
    assert len(items) == 2
    assert "漏判" in [i for i in items if i.sentence == sentences[1]][0].reason


# --------------------------------------------------------------- 节点层


class _FakeTrace:
    def emit(self, *args, **kwargs):  # noqa: ANN002, ANN003
        pass


@dataclass
class _FakeSettings:
    """节点层用例只需这几个熔断/重写相关的配置项。"""

    audit_max_rewrites: int = 1
    audit_rewrite_ratio: float = 0.5
    fuse_ctx_top_k: int = 6


class _FakeCtx:
    def __init__(self, auditor, *, settings=None, gateway=None, cost=0.0, tokens=0):
        self.auditor = auditor
        self.trace = _FakeTrace()
        self.settings = settings or _FakeSettings()
        self.gateway = gateway
        self._cost = cost
        self._tokens = tokens

    def budget_snapshot(self, state):  # noqa: ANN001
        from attest.budget.account import BudgetConfig, snapshot

        return snapshot(self._cost, self._tokens, BudgetConfig())


def test_auditor_node_degrades_and_refinalizes() -> None:
    report = (
        "# 报告\n\n## 核心摘要\n\n市场规模达 999 亿元 "
        f"{CID}。\n\n## 参考资料\n\n- `{CID}` [测试来源](https://example.com/a) — 子问题：市场规模\n"
    )
    state = {"report": report, "evidence": [_ev()], "plan": {"objective": "测试"}}
    out = auditor_node.run(state, _FakeCtx(RuleCitationAuditor()))  # type: ignore[arg-type]

    assert out["audit_summary"]["unsupported"] == 1
    assert out["audit_summary"]["degraded_sentences"] == 1
    assert "（未证实）" in out["report"]
    assert CID not in out["report"], "被降级的编号不应再出现在正文或参考资料里"


def test_auditor_node_skips_when_disabled() -> None:
    state = {"report": "# 报告\n\n正文", "evidence": [_ev()], "plan": {}}
    out = auditor_node.run(state, _FakeCtx(None))  # type: ignore[arg-type]
    assert out == {}, "审计关闭时应返回空增量，而不是假装审计过"


# --------------------------- T7.9c 幻觉编号必须被质证（2026-09-13 真实运行暴露的漏洞）

def test_extract_claims_sectioned_keeps_unknown_citation() -> None:
    """引用了**不存在编号**的句子曾被静默过滤掉——幻觉因此漏过质检。

    实测：第 2 轮只补检了子问题 6–9，模型却在正文里编出 `[WEB2-2-2]`。
    正文留着它、参考资料里查无此条、审计也不吭声——只有 `CitationIndex` 报了个
    "未解析 1 处"，没人知道是哪句话的问题。这类句子恰恰最该被质证。
    """
    report = "# 报告\n\n## 竞争格局\n\n三类玩家重叠明显 [WEB2-2-2]。\n"
    claims = extract_claims_sectioned(report, [CID])  # 已知编号里没有 WEB2-2-2
    assert [c[1] for c in claims] == ["[WEB2-2-2]"], "未知编号必须送审，不能被过滤"


def test_verify_claim_unknown_citation_is_hard_unsupported() -> None:
    """证据为空（编号不存在）时理由要说清是"幻觉编号"，而非笼统的"覆盖率低"。"""
    verdict, reason = verify_claim("三类玩家重叠明显 [WEB2-2-2]", "")
    assert verdict == "unsupported"
    assert "不存在" in reason, f"理由要指出编号不存在：{reason}"


def test_auditor_node_degrades_hallucinated_citation() -> None:
    """端到端：幻觉编号的句子应被**去掉编号 + 标注「（未证实）」**，正常句子不受影响。"""
    report = (
        "# 报告\n\n## 竞争格局\n\n三类玩家重叠明显 [WEB2-2-2]。\n\n"
        "## 核心摘要\n\n市场规模达 180 亿元 [WEB1-1-1]。\n"
    )
    state = {"report": report, "evidence": [_ev()], "plan": {"objective": "测试"}}
    out = auditor_node.run(state, _FakeCtx(RuleCitationAuditor()))  # type: ignore[arg-type]

    assert out["audit_summary"]["unsupported"] == 1
    assert "[WEB2-2-2]" not in out["report"], "幻觉编号应从正文移除"
    assert "（未证实）" in out["report"], "该句应被标注"
    assert "[WEB1-1-1]" in out["report"], "被支持的编号不受影响"
