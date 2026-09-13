"""T7.10 · 证据充足性闸门（拒编）回归测试。

**这一组要证明的**（而不是"跑过就算"）：
  1. 检索层对"主题无法判别"的查询，会把它返回的每一条都标 0 分（不再假装命中）；
  2. 证据充足性判据按**判别相关度**判定——判别器说全都不相关 ⇒ 判为不足；
  3. 即使判别器给了高分，若可用证据**全是检索兜底填充**（score=0）也判为不足；
  4. `analyst` 在证据不足时**不调用任何模型**就产出拒编页（把"拒编"交给 LLM 去说
     本身就是造假风险，所以必须确定性生成）；
  5. 拒编页如实交代检索与判别情况，并给出可执行的下一步；
  6. 审计与沉淀在拒编时跳过——对"没有产出的结论"做审计/沉淀没有意义。

真实缺陷背景（2026-09-13 实测，`data/trace/trace.jsonl` 可查）：
    问「2026 年大学生就业情况」时，判别器把全部 66 条证据判为 relevance ≤ 2（**判对了**），
    但 `analyst` 旧实现"筛选后为空 → 退回全量"，于是拿知识库/数据库/新能源汽车的资料
    硬凑出一份"大学生就业报告"——数字是真的、主题是假的。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from attest.agents import analyst as analyst_mod
from attest.agents import auditor as auditor_mod
from attest.agents.analyst import assess_evidence, build_insufficient_report
from attest.agents.base import NodeContext
from attest.budget.account import BudgetConfig
from attest.config import Settings
from attest.llm.gateway import LLMGateway
from attest.retrieval.mock_search import FixtureSearchClient
from attest.retrieval.ports import Evidence, SearchResult
from attest.schemas import Judgment
from attest.trace.events import TraceWriter

REPO = Path(__file__).resolve().parents[1]

OBJECTIVE = "2026年大学生就业市场调研"
SUBQS = ["大学生就业市场的规模", "就业市场的竞争格局"]


# --------------------------------------------------------------------- 工具


def _ev(cid: str, subq: str, *, score: float = 1.0) -> Evidence:
    return Evidence(
        citation_id=cid,
        source="web",
        title=f"标题{cid}",
        url=f"https://example.com/{cid.strip('[]')}",
        content="正文内容",
        sub_question=subq,
        round_no=1,
        score=score,
    )


def _jd(cid: str, rel: int, subq: str = SUBQS[0]) -> Judgment:
    return Judgment(citation_id=cid, sub_question=subq, relevance=rel, confidence=4)


def _state(evidence: list[Evidence], judgments: list[Judgment]) -> dict[str, Any]:
    return {
        "evidence": evidence,
        "judgments": judgments,
        "plan": {"objective": OBJECTIVE, "sub_questions": SUBQS, "outlines": ["摘要", "正文"]},
    }


def _settings(tmp_path: Path) -> Settings:
    s = Settings(
        llm_mode="mock",
        search_mode="mock",
        trace_dir=tmp_path / "trace",
        report_dir=tmp_path / "reports",
        fixture_dir=REPO / "data" / "fixtures",
        checkpoint_db=tmp_path / "cp.sqlite",
        profile_db=tmp_path / "profile.sqlite",
        local_docs_dir=tmp_path / "nodocs",
    )
    s.ensure_dirs()
    return s


# --------------------------------------------------------- 1. 检索层：未命中主题标 0


def test_fixture_search_marks_unrouted_results_as_zero() -> None:
    """query 判不出主题时，返回的每一条都必须 score=0（不得假装命中）。

    否则下游会把跨主题捞回来的资料当证据——这正是"张冠李戴报告"的源头。
    """
    client = FixtureSearchClient(fixture_dir=REPO / "data" / "fixtures" / "web")
    off, note, matched = client._route("2026年大学生就业市场的规模与薪酬")
    assert matched is False
    results = client.search("2026年大学生就业市场的规模与薪酬")
    assert results, "兜底仍会返回条目（保证链路不断），但……"
    assert all(r.score == 0.0 for r in results), "未命中主题的结果必须全部标 0 分"


def test_fixture_search_keeps_scores_when_topic_matched() -> None:
    """命中了主题时，真实命中条目必须保留原分（否则会把好证据也废掉）。"""
    client = FixtureSearchClient(fixture_dir=REPO / "data" / "fixtures" / "web")
    results = client.search("企业知识库 Agent 平台的市场规模")
    assert any(r.score > 0 for r in results), "命中主题时必须保留真实分数"


# --------------------------------------------------------- 2~3. 充足性判据


def test_assess_refuses_when_no_judgment_reaches_threshold() -> None:
    """判别器把全部证据判为低相关 ⇒ 证据不足（这是真实缺陷的修复点）。"""
    ev = [_ev("[WEB1-1-1]", SUBQS[0], score=0.9), _ev("[WEB1-1-2]", SUBQS[0], score=0.8)]
    jd = [_jd("[WEB1-1-1]", 2), _jd("[WEB1-1-2]", 1)]
    selected, info = assess_evidence(_state(ev, jd))
    assert info["sufficient"] is False
    assert selected == []
    assert info["reasons"], "必须给出拒编原因"


def test_assess_passes_when_some_evidence_is_relevant_and_grounded() -> None:
    ev = [_ev("[WEB1-1-1]", SUBQS[0], score=0.9), _ev("[WEB1-1-2]", SUBQS[1], score=0.7)]
    jd = [_jd("[WEB1-1-1]", 5, SUBQS[0]), _jd("[WEB1-1-2]", 4, SUBQS[1])]
    selected, info = assess_evidence(_state(ev, jd))
    assert info["sufficient"] is True
    assert len(selected) == 2
    assert info["n_grounded"] == 2


def test_assess_refuses_when_selected_are_all_fallback_padding() -> None:
    """判别器给了高分，但可用证据**全是兜底填充（score=0）** ⇒ 仍判不足。

    这是"第二道闸"：没有再信一遍打分模型，而是要求证据在检索里真实命中过。
    """
    ev = [_ev("[WEB1-1-1]", SUBQS[0], score=0.0), _ev("[WEB1-1-2]", SUBQS[0], score=0.0)]
    jd = [_jd("[WEB1-1-1]", 5), _jd("[WEB1-1-2]", 4)]
    selected, info = assess_evidence(_state(ev, jd))
    assert info["sufficient"] is False
    assert selected, "证据本身是通过相关度筛选的……"
    assert info["n_grounded"] == 0, "……但没有一条真实命中"
    assert any("兜底" in r for r in info["reasons"])


def test_assess_without_judgments_falls_back_to_grounded_score() -> None:
    """无判别结果时退化为"检索真实命中"，且不会把恒正的本地融合分当成相关性。"""
    ev = [_ev("[WEB1-1-1]", SUBQS[0], score=0.0)]
    _sel, info = assess_evidence(_state(ev, []))
    assert info["sufficient"] is False


# --------------------------------------------------------- 4~5. analyst 拒编页


class _ExplodingGateway:
    """任何 LLM 调用都会炸——用来证明拒编路径**不碰模型**。"""

    def chat(self, *a: Any, **k: Any) -> Any:  # pragma: no cover - 触发即失败
        raise AssertionError("证据不足时不得调用模型")

    def embed(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise AssertionError("证据不足时不得调用模型")


def _ctx(tmp_path: Path, *, gateway: Any = None) -> NodeContext:
    settings = _settings(tmp_path)
    trace = TraceWriter(path=tmp_path / "trace" / "trace.jsonl", run_id="t710")
    return NodeContext(
        settings=settings,
        gateway=gateway or LLMGateway(settings=settings, trace=trace),
        search=FixtureSearchClient(settings.fixture_dir / "web"),
        trace=trace,
        budget=BudgetConfig(),
    )


def test_analyst_refuses_without_calling_any_model(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, gateway=_ExplodingGateway())
    ev = [_ev("[WEB1-1-1]", SUBQS[0], score=0.9)]
    jd = [_jd("[WEB1-1-1]", 2)]
    out = analyst_mod.run(_state(ev, jd), ctx)

    assert "证据不足" in out["report"]
    assert out["reference_list"] == ""
    assert out["citation_check"]["insufficient"] is True
    assert out["evidence_sufficiency"]["sufficient"] is False
    assert out["model_tier"] == "refused"
    assert "[WEB" not in out["report"], "拒编页不得出现任何引用编号"
    events = [e for e in ctx.trace.events if e["event"] == "evidence_insufficient"]
    assert events, "必须留下 evidence_insufficient 事件供审计"


def test_insufficient_report_is_actionable(tmp_path: Path) -> None:
    _sel, info = assess_evidence(_state([], []))
    text = build_insufficient_report(info, search_mode="mock")
    for sq in SUBQS:
        assert sq in text, "拒编页要列出子问题及其证据情况"
    assert "TAVILY_API_KEY" in text, "离线模式下要给出可执行的下一步（接真实检索）"
    assert "不是故障" in text, "要让用户明白这是主动拒编，不是坏了"


def test_auditor_skips_on_insufficient_report(tmp_path: Path) -> None:
    """拒编页没有引用可审 —— 审计必须跳过，而不是把 citation_check 覆盖成"未通过"。"""
    ctx = _ctx(tmp_path)
    state = _state([_ev("[WEB1-1-1]", SUBQS[0])], [])
    state["report"] = "（拒编页正文，不含引用编号）"
    state["evidence_sufficiency"] = {"sufficient": False, "n_evidence": 1}
    out = auditor_mod.run(state, ctx)
    assert out == {}, "拒编时审计不得产出任何增量"
    assert ctx.trace.of("audit_skipped"), "必须留痕说明为什么跳过"


# --------------------------------------------- 6. 端到端：全零分检索必然拒编


def test_graph_refuses_when_retrieval_only_returns_unrouted(tmp_path: Path) -> None:
    """真实图端到端：检索全部返回 score=0（未命中主题）⇒ 报告必须是拒编页。

    这条钉住"第二道闸"在图里的实际效果——不依赖判别器是否严格。
    """
    from attest.graph.build import build_context, build_graph, initial_state

    class _ZeroSearch:
        def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
            return [
                SearchResult(
                    source="web",
                    title="跨主题的无关资料",
                    url="https://example.com/irrelevant",
                    content="企业知识库与 Agent 平台市场规模约 180 亿元，复合增速约 42%。",
                    score=0.0,  # 检索层已判定"非真实命中"
                )
            ]

    settings = _settings(tmp_path)
    ctx = build_context(settings, search=_ZeroSearch(), run_id="t710-e2e")
    app = build_graph(ctx)
    final = app.invoke(
        initial_state("调研2026年大学生就业市场的规模与薪酬水平，给出结构化分析报告", thread_id="t710-e2e")
    )
    assert final["route"] == "research"
    suff = final.get("evidence_sufficiency") or {}
    assert suff.get("sufficient") is False, f"应当拒编，实际 {suff}"
    assert "证据不足" in final["report"]
    assert "[WEB" not in final["report"]
