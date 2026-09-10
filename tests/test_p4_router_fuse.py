"""T4.2b / T4.3 / T4.4 · 熔断分级、模型路由器、章节重写的单测。

三块各自独立验证：
  1. **路由器（T4.4）**：策略表取值、L1 只降级轻任务、主链路不动、未映射回退；
  2. **熔断动作（T4.3）**：L2 审计只跑初筛、上下文按相关性截断、报告头降级横幅；
  3. **章节重写（T4.2b）**：失败率判定、只替换目标节、重写后复检、失败不中断链路。

全部离线：重写者（gateway）用桩注入——沿用 P3 的"离线兜底 + 注入点"范式，
使"重写→复检"这条时序无需真实模型即可断言。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from attest.agents import auditor as auditor_node
from attest.agents.analyst import truncate_topk
from attest.agents.auditor import _with_fuse_banner
from attest.config import Settings
from attest.llm.gateway import LLMGateway
from attest.llm.prompts import build_direct_messages
from attest.llm.router import CORE_TASKS, LIGHT_TASKS, ModelRouter
from attest.quality.audit import (
    citations_in_section,
    extract_claims_sectioned,
    replace_section,
    rewrite_targets,
    section_body,
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


def _settings() -> Settings:
    """离线配置：不读 .env、不联网，模型映射用默认值。"""
    return Settings(_env_file=None, llm_mode="mock", search_mode="mock")


# =============================================================== T4.4 路由器


def test_router_base_table_reads_settings() -> None:
    s = _settings()
    r = ModelRouter(s)
    assert r.base_model("planner") == s.model_planner
    assert r.base_model("analyst") == s.model_analyst
    assert r.base_model("audit_strong") == s.model_audit_strong
    # 重写复用撰稿模型：它是主链路的质量动作，不另开一个"更便宜"的模型
    assert r.base_model("analyst_rewrite") == s.model_analyst


def test_router_downgrades_light_tasks_only_at_l1() -> None:
    s = _settings()
    r = ModelRouter(s)
    for task in LIGHT_TASKS:
        assert r.route(task, fuse_level=0) == r.base_model(task)
        assert r.route(task, fuse_level=1) == s.model_fuse_light
        assert r.route(task, fuse_level=2) == s.model_fuse_light


def test_router_never_downgrades_core_tasks() -> None:
    """熔断的目的是"降质保交付"，不是把正文写坏——主链路任何级别都不切小模型。"""
    s = _settings()
    r = ModelRouter(s)
    for task in CORE_TASKS:
        assert r.route(task, fuse_level=2) == r.base_model(task), task


def test_router_unmapped_task_falls_back_to_direct() -> None:
    s = _settings()
    assert ModelRouter(s).route("完全不存在的任务") == s.model_direct


def test_router_tier_labels() -> None:
    r = ModelRouter(_settings())
    assert r.tier("planner") == "strong"
    assert r.tier("analyst_rewrite") == "strong"
    assert r.tier("judge") == "fast"
    # 降级后轻任务一律记 fast（trace 复盘用）
    assert r.tier("direct", fuse_level=1) == "fast"
    assert r.tier("planner", fuse_level=2) == "strong"


def test_gateway_delegates_routing_and_honors_fuse() -> None:
    """网关不再自带映射表——它必须委托路由器，并把 fuse_level 传下去。"""
    s = _settings()
    gw = LLMGateway(settings=s)
    assert gw.model_for("direct", fuse_level=0) == s.model_direct
    assert gw.model_for("direct", fuse_level=1) == s.model_fuse_light
    # 真实走一次调用：mock provider 会把 model 原样回填进响应，可断言
    resp = gw.chat(build_direct_messages("你好"), task="direct", fuse_level=1)
    assert resp.model == s.model_fuse_light
    resp0 = gw.chat(build_direct_messages("你好"), task="direct", fuse_level=0)
    assert resp0.model == s.model_direct


# =============================================================== T4.3 熔断动作


class _StubGateway:
    """返回预置 AuditResult 的桩网关。"""

    def __init__(self, fast: AuditResult, strong: AuditResult | None = None):
        self.fast = fast
        self.strong = strong or fast
        self.calls: list[str] = []

    def chat(self, messages, *, task, response_model=None, temperature=0.0, fuse_level=0, **kw):  # noqa: ANN001
        self.calls.append(task)
        return SimpleNamespace(parsed=self.fast if task == "audit_fast" else self.strong)


def _flagged_audit() -> tuple[str, AuditResult]:
    report = f"# 报告\n\n## 摘要\n\n市场规模达 180 亿元 {CID}。\n增速约 42% {CID}。\n"
    sentences = [c[0] for c in extract_claims_sectioned(report, [CID])]
    fast = AuditResult(
        items=[
            AuditItem(citation_id=CID, sentence=sentences[0], verdict="supported", reason="ok"),
            AuditItem(citation_id=CID, sentence=sentences[1], verdict="unsupported", reason="数值缺失"),
        ]
    )
    return report, fast


def test_l2_audit_runs_screen_only() -> None:
    """熔断 L2：「审计只跑初筛」——跳过强模型复核这一趟最贵的调用。"""
    report, fast = _flagged_audit()
    gw = _StubGateway(fast)
    LLMCitationAuditor(settings=None, gateway=gw).audit(  # type: ignore[arg-type]
        report, [_ev()], objective="测试", fuse_level=2
    )
    assert gw.calls == ["audit_fast"], gw.calls


def test_l1_audit_still_escalates() -> None:
    """L1 只降级轻任务的**模型档位**，复核流程照常——两者是不同维度，不能混为一谈。"""
    report, fast = _flagged_audit()
    gw = _StubGateway(fast)
    LLMCitationAuditor(settings=None, gateway=gw).audit(  # type: ignore[arg-type]
        report, [_ev()], objective="测试", fuse_level=1
    )
    assert gw.calls == ["audit_fast", "audit_strong"], gw.calls


def test_rule_auditor_accepts_fuse_level_kwarg() -> None:
    """规则版对 fuse_level 无感，但签名必须与 Protocol 一致（调用方无需分支）。"""
    report = f"# 报告\n\n## 摘要\n\n市场规模达 180 亿元 {CID}。\n"
    items = RuleCitationAuditor().audit(report, [_ev()], fuse_level=2)
    assert items[0].verdict == "supported"


def test_truncate_topk_keeps_most_relevant() -> None:
    evs = [_ev(f"[WEB1-1-{i}]", f"证据{i}内容") for i in range(1, 5)]
    judgments = [
        SimpleNamespace(citation_id="[WEB1-1-1]", relevance=1),
        SimpleNamespace(citation_id="[WEB1-1-2]", relevance=5),
        SimpleNamespace(citation_id="[WEB1-1-3]", relevance=2),
        SimpleNamespace(citation_id="[WEB1-1-4]", relevance=4),
    ]
    kept = truncate_topk(evs, judgments, 2)
    # 相关性最高的两条：2(rel5) 与 4(rel4)，内部保持原相对顺序
    assert [e.citation_id for e in kept] == ["[WEB1-1-2]", "[WEB1-1-4]"]


def test_truncate_topk_noop_when_below_k_or_no_judgments() -> None:
    evs = [_ev(f"[WEB1-1-{i}]") for i in range(1, 4)]
    assert truncate_topk(evs, [], 5) == evs
    # 无判别信息时退化为保留前 k 条（不是丢弃全部）
    assert len(truncate_topk(evs, [], 2)) == 2


def test_fuse_banner_inserted_after_h1_and_idempotent() -> None:
    report = "# 报告\n\n## 摘要\n\n正文\n"
    b2 = _with_fuse_banner(report, 2)
    assert b2.startswith("# 报告\n\n> ⚠️")
    assert "正文" in b2
    assert _with_fuse_banner(b2, 2).count("降级提示") == 1, "重复调用不应叠加横幅"


def test_fuse_banner_absent_below_l2() -> None:
    for level in (0, 1):
        assert "降级提示" not in _with_fuse_banner("# 报告\n\n正文", level)


def test_fuse_banner_without_h1_goes_to_top() -> None:
    out = _with_fuse_banner("正文没有标题", 2)
    assert out.startswith("> ⚠️") and "正文没有标题" in out


# =============================================================== T4.2b 章节重写


def test_rewrite_targets_only_over_threshold_and_skips_untitled() -> None:
    items = [
        # 摘要：1/2 = 0.5，恰好等于阈值 → 不重写（判据是严格大于）
        AuditItem(citation_id=CID, verdict="unsupported", sentence="a", section="摘要"),
        AuditItem(citation_id=CID, verdict="supported", sentence="b", section="摘要"),
        # 竞品：1/1 = 1.0 → 重写
        AuditItem(citation_id=CID, verdict="unsupported", sentence="c", section="竞品"),
        # 无标题段：有失败但也无锚点 → 不参与重写
        AuditItem(citation_id=CID, verdict="unsupported", sentence="d", section=""),
    ]
    assert rewrite_targets(items, 0.5) == {"竞品": 1.0}
    assert rewrite_targets(items, 0.1) == {"摘要": 0.5, "竞品": 1.0}


def test_section_body_and_citations_lookup() -> None:
    report = (
        "# 报告\n\n## 市场规模\n\n达 180 亿元 "
        f"{CID}，另有 [WEB1-1-2] 口径。\n\n## 竞品\n\n共 5 家。\n"
    )
    assert section_body(report, "市场规模").startswith("达 180 亿元")
    assert citations_in_section(report, "市场规模") == [CID, "[WEB1-1-2]"]
    assert section_body(report, "不存在的章节") == ""


def test_replace_section_only_touches_target() -> None:
    report = "# 报告\n\n## 市场规模\n\n旧正文\n\n## 竞品\n\n竞品正文\n\n## 参考资料\n\n- x\n"
    out = replace_section(report, "市场规模", "新正文")
    assert "新正文" in out and "旧正文" not in out
    assert "竞品正文" in out and "# 报告" in out and "## 参考资料" in out


def test_replace_section_missing_title_returns_original() -> None:
    report = "# 报告\n\n## 市场规模\n\n正文\n"
    assert replace_section(report, "不存在", "x") == report


@dataclass
class _RewriteStubGateway:
    """桩网关：只服务章节重写，返回预置正文。"""

    body: str
    fail: bool = False

    def __post_init__(self) -> None:
        self.calls: list[str] = []

    def chat(self, messages, *, task, response_model=None, temperature=0.0, fuse_level=0, **kw):  # noqa: ANN001
        self.calls.append(task)
        if self.fail:
            raise RuntimeError("stub rewrite failure")
        return SimpleNamespace(text=self.body, cost_cny=0.0123, total_tokens=456)


@dataclass
class _FakeSettings:
    audit_max_rewrites: int = 1
    audit_rewrite_ratio: float = 0.5
    fuse_ctx_top_k: int = 6


class _FakeTrace:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, name: str, **kw: Any) -> None:  # noqa: ANN003
        self.events.append((name, kw))


class _Ctx:
    def __init__(self, auditor, gateway=None, *, settings=None, cost=0.0, tokens=0):
        self.auditor = auditor
        self.gateway = gateway
        self.trace = _FakeTrace()
        self.settings = settings or _FakeSettings()
        self._cost = cost
        self._tokens = tokens

    def budget_snapshot(self, state):  # noqa: ANN001
        from attest.budget.account import BudgetConfig, snapshot

        return snapshot(self._cost, self._tokens, BudgetConfig())


def _bad_section_report() -> str:
    return (
        "# 报告\n\n## 市场规模\n\n"
        f"- 本市场规模达 9999 亿元 {CID}。\n"
        f"- 另一口径为 8888 亿元 {CID}。\n\n"
        "## 竞品\n\n- 共 5 家主要玩家（无引用）。\n"
    )


def test_node_rewrites_failing_section_then_rechecks() -> None:
    """整节 100% 不支撑 → 触发重写 → 复检通过 → 不再有「未证实」。"""
    gw = _RewriteStubGateway(body=f"- 据测算，市场规模达 180 亿元，增速约 30%。{CID}")
    ctx = _Ctx(RuleCitationAuditor(), gw)
    out = auditor_node.run(
        {"report": _bad_section_report(), "evidence": [_ev()], "plan": {"objective": "测试"}}, ctx
    )

    assert gw.calls == ["analyst_rewrite"], gw.calls
    assert out["audit_summary"]["rewrites"] == 1
    hist = out["audit_summary"]["rewrite_history"]
    assert hist[0]["section"] == "市场规模"
    assert hist[0]["ratio_before"] == 1.0 and hist[0]["ratio_after"] == 0.0
    assert "9999" not in out["report"], "重写应把证据支持不了的数值清掉"
    assert "（未证实）" not in out["report"]
    assert "竞品" in out["report"], "非目标章节必须原样保留"
    # 重写调用必须记账（绕过网关=漏账）
    assert out["cost_incurred"] == 0.0123
    assert out["tokens_incurred"] == 456


def test_node_does_not_rewrite_when_ratio_at_threshold() -> None:
    """失败率恰好等于阈值（0.5）不触发重写——判据是**严格大于**，边界不留模糊。"""
    report = (
        "# 报告\n\n## 摘要\n\n"
        f"- 市场规模达 180 亿元 {CID}。\n"
        f"- 增速约 42% {CID}。\n"
    )
    gw = _RewriteStubGateway(body="从不被调用")
    ctx = _Ctx(RuleCitationAuditor(), gw)
    out = auditor_node.run(
        {"report": report, "evidence": [_ev()], "plan": {"objective": "测试"}}, ctx
    )
    assert gw.calls == [], "失败率 = 阈值时不应重写"
    assert out["audit_summary"]["rewrites"] == 0
    assert "（未证实）" in out["report"], "未重写则落到逐句标注兜底"


def test_node_rewrite_failure_does_not_break_chain() -> None:
    """重写调用抛错 → 不中断链路，退回逐句标注（报告仍必须产出）。"""
    gw = _RewriteStubGateway(body="", fail=True)
    ctx = _Ctx(RuleCitationAuditor(), gw)
    out = auditor_node.run(
        {"report": _bad_section_report(), "evidence": [_ev()], "plan": {"objective": "测试"}}, ctx
    )
    assert out["audit_summary"]["rewrites"] == 0
    assert "（未证实）" in out["report"]
    assert out["report"].count("（未证实）") == 2


def test_node_respects_max_rewrites_budget() -> None:
    """重写次数受 `audit_max_rewrites` 硬约束——默认 1 次，模型改不好也不再烧钱。"""
    gw = _RewriteStubGateway(body=f"- 仍是 9999 亿元 {CID}")  # 重写后依旧不支撑
    ctx = _Ctx(RuleCitationAuditor(), gw, settings=_FakeSettings(audit_max_rewrites=1))
    out = auditor_node.run(
        {"report": _bad_section_report(), "evidence": [_ev()], "plan": {"objective": "测试"}}, ctx
    )
    assert gw.calls == ["analyst_rewrite"], f"超预算仍重写：{gw.calls}"
    assert out["audit_summary"]["rewrites"] == 1
    assert "（未证实）" in out["report"], "重写失败后必须有逐句标注兜底"


def test_node_marks_report_head_at_l2() -> None:
    """熔断 L2：报告头必须自证"这是降级状态下产出的"，否则读者无从判断覆盖度。"""
    report = f"# 报告\n\n## 摘要\n\n市场规模达 180 亿元 {CID}。\n"
    ctx = _Ctx(RuleCitationAuditor(), None, cost=1.9)  # 1.9 / 2.0 = 95% ≥ hard 90%
    out = auditor_node.run({"report": report, "evidence": [_ev()], "plan": {}}, ctx)
    assert out["audit_summary"]["fuse_level"] == 2
    assert "降级提示" in out["report"]
    assert out["report"].index("降级提示") < out["report"].index("摘要")
