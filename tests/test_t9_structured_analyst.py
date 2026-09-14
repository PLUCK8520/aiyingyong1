"""T9.1 · analyst 结构化输出协议：编号由"模型的排版动作"改为"代码的确定性拼装"。

**为什么要有这一组测试**：T8.6 已经证明（2026-09-13/14，glm-4-flash 免费档四次复现）
弱模型最不服从事项是"在结论句末挂引用编号"——整篇零编号 → 审计无对象可审 →
凭空内容安全混过（"财政补贴/税收优惠"事件）。T9.1 的应对：
  1. 模型只负责**按章声明**依据编号（`<<CITE>>` 行，从证据块原样抄录）；
  2. 代码把声明**确定性注入**正文（章末依据行），解析不出来就降级回纯文本路径；
  3. 审计把"依据行"当元信息（不送审），但本章无编号句**关联本章声明编号**逐句受审。

本组钉住：解析容错、拼装规则、依据行的审计语义、analyst 节点的结构化优先/降级分支、
以及 mock 链路必须走"代码注入"主路径（否则离线 CI 对结构化零覆盖）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from attest.agents import analyst as analyst_mod
from attest.agents.base import NodeContext
from attest.budget.account import BudgetConfig
from attest.config import Settings
from attest.llm.gateway import LLMResponse
from attest.llm.prompts import (
    build_analyst_structured_messages,
    parse_structured_report,
    render_structured_report,
)
from attest.quality.audit import extract_claims_sectioned
from attest.quality.citation_auditor import RuleCitationAuditor
from attest.retrieval.citations import (
    CitationIndex,
    SECTION_BASIS_MARK,
    extract_ids,
    parse_basis_line,
    render_basis_line,
)
from attest.retrieval.ports import Evidence, SearchResult
from attest.trace.events import TraceWriter

REPO = Path(__file__).resolve().parents[1]

SUBQS = ["城市轨道交通建设的主要成本构成", "降低造价的常见做法"]
CID1 = "[LOC1-1-1]"
CID2 = "[LOC1-1-2]"
GHOST = "[WEB9-9-9]"
DOC1 = "土建工程约占总投资的 50% 至 60%，其中盾构掘进与车站土建是主要支出项。"
DOC2 = "高架线的造价约为地下线的 40% 至 60%，是降低造价的主要替代方案。"


def _ev(cid: str, content: str, subq: str = SUBQS[0], *, score: float = 0.7) -> Evidence:
    return Evidence(
        citation_id=cid,
        source="local",
        title=f"证据-{cid}",
        url=f"local://kb/{cid.strip('[]')}",
        content=content,
        sub_question=subq,
        round_no=1,
        score=score,
    )


def _index() -> CitationIndex:
    return CitationIndex.from_evidence([_ev(CID1, DOC1), _ev(CID2, DOC2, SUBQS[1])])


# --------------------------------------------------------------------- 1. 解析层


def test_parse_wellformed_output() -> None:
    text = (
        "<<CHAPTER>>\n标题：核心摘要\n<<CITE>>\n[LOC1-1-1] [LOC1-1-2]\n<<TEXT>>\n"
        "土建占比最高。\n<<END>>\n"
        "<<CHAPTER>>\n标题：一、成本构成\n<<CITE>>\n[LOC1-1-1]\n<<TEXT>>\n"
        "盾构是主要支出。\n<<END>>\n"
    )
    chapters = parse_structured_report(text)
    assert chapters is not None and len(chapters) == 2
    assert chapters[0].title == "核心摘要"
    assert chapters[0].cite_ids == [CID1, CID2]
    assert "土建占比最高" in chapters[0].body
    assert chapters[1].title == "一、成本构成"


def test_parse_tolerates_missing_end_and_messy_cite() -> None:
    """漏写 <<END>> 按下一 CHAPTER 切块；CITE 丢方括号/夹逗号也要收敛。"""
    text = (
        "<<CHAPTER>>\n标题:核心摘要\n<<CITE>>\nLOC1-1-1， [LOC1-1-2]、垃圾词\n<<TEXT>>\n摘要正文\n"
        "<<CHAPTER>>\n标题：一、成本构成\n<<CITE>>\n\n<<TEXT>>\n（证据不足）本章无直接证据。\n<<END>>\n"
    )
    chapters = parse_structured_report(text)
    assert chapters is not None and len(chapters) == 2
    assert chapters[0].title == "核心摘要"
    assert chapters[0].cite_ids == [CID1, CID2], "丢方括号/逗号分隔都要收敛，垃圾词丢弃"
    assert chapters[1].cite_ids == []
    assert "证据不足" in chapters[1].body


def test_parse_returns_none_without_chapter_block() -> None:
    assert parse_structured_report("# 普通报告\n\n没有任何标记。\n") is None
    assert parse_structured_report("") is None
    assert parse_structured_report(None) is None  # type: ignore[arg-type]


def test_parse_drops_invalid_chapters() -> None:
    """缺 TEXT / 空标题 / 空正文的章无效；全部无效时返回 None。"""
    text = (
        "<<CHAPTER>>\n标题：只有标题没有 TEXT 标记\n<<CITE>>\n[LOC1-1-1]\n<<END>>\n"
        "<<CHAPTER>>\n标题：\n<<CITE>>\n[LOC1-1-1]\n<<TEXT>>\n空标题章\n<<END>>\n"
        "<<CHAPTER>>\n标题：空正文章\n<<CITE>>\n[LOC1-1-1]\n<<TEXT>>\n\n<<END>>\n"
    )
    assert parse_structured_report(text) is None


# --------------------------------------------------------------------- 2. 拼装层


def test_render_injects_basis_line_only_for_missing_ids() -> None:
    chapters = parse_structured_report(
        "<<CHAPTER>>\n标题：一、成本构成\n<<CITE>>\n[LOC1-1-1] [LOC1-1-2]\n<<TEXT>>\n"
        "土建约占一半，盾构是主要支出。\n<<END>>\n"
    )
    assert chapters is not None
    report, dropped = render_structured_report("调研城市轨交成本", chapters, _index())
    assert dropped == []
    assert SECTION_BASIS_MARK in report, "正文无编号 → 章末必须注入依据行"
    assert f"{CID1} {CID2}" in report
    assert extract_ids(report) == [CID1, CID2]


def test_render_keeps_inline_citations_without_basis_line() -> None:
    """正文自带句末编号是最高质量形态——不画蛇添足。"""
    chapters = parse_structured_report(
        "<<CHAPTER>>\n标题：一、成本构成\n<<CITE>>\n[LOC1-1-1]\n<<TEXT>>\n"
        f"土建约占 50% 至 60% {CID1}。\n<<END>>\n"
    )
    assert chapters is not None
    report, _ = render_structured_report("调研", chapters, _index())
    assert SECTION_BASIS_MARK not in report
    assert CID1 in report


def test_render_drops_unregistered_ids_and_reports_them() -> None:
    """臆造编号必须被拼装层拦下——这是最后一道确定性关卡。"""
    chapters = parse_structured_report(
        "<<CHAPTER>>\n标题：一、成本构成\n<<CITE>>\n[LOC1-1-1] [WEB9-9-9]\n<<TEXT>>\n"
        "土建约占一半。\n<<END>>\n"
    )
    assert chapters is not None
    report, dropped = render_structured_report("调研", chapters, _index())
    assert dropped == [GHOST]
    assert GHOST not in report, "未注册编号不得进入正文"
    assert CID1 in report


def test_render_leaves_empty_cite_chapter_alone() -> None:
    chapters = parse_structured_report(
        "<<CHAPTER>>\n标题：二、降低造价\n<<CITE>>\n\n<<TEXT>>\n（证据不足）缺直接证据。\n<<END>>\n"
    )
    assert chapters is not None
    report, dropped = render_structured_report("调研", chapters, _index())
    assert dropped == []
    assert SECTION_BASIS_MARK not in report
    assert "（证据不足）" in report


def test_render_converges_bare_citation_lines() -> None:
    """编号裸露行（2026-09-14 真实档实测形态）：剥出编号走依据行，裸行删除。"""
    chapters = parse_structured_report(
        "<<CHAPTER>>\n标题：一、竞争格局\n<<CITE>>\n[LOC1-1-1]\n<<TEXT>>\n"
        "三类参与者共同推动市场发展。\n\n[LOC1-1-2]\n<<END>>\n"
    )
    assert chapters is not None
    report, dropped = render_structured_report("调研", chapters, _index())
    assert dropped == []
    # 裸露行被删除，编号收敛进依据行
    body = report.split("## 一、竞争格局", 1)[1]
    assert "\n[LOC1-1-2]\n" not in body
    assert SECTION_BASIS_MARK in body
    assert CID2 in body, "裸行里的编号不能丢——并入本章声明"
    assert extract_ids(report) == [CID1, CID2] or set(extract_ids(report)) == {CID1, CID2}


# --------------------------------------------------------------------- 3. 依据行的审计语义


def test_basis_line_roundtrip() -> None:
    line = render_basis_line([CID1, CID2])
    assert parse_basis_line(line) == [CID1, CID2]
    assert parse_basis_line("正文里普通一句提到依据二字。") is None
    assert parse_basis_line(f"（{SECTION_BASIS_MARK}）") == [], "空依据行也是依据行"


def test_basis_line_is_not_a_claim_but_section_sentences_are_audited() -> None:
    """依据行本身不送审；本章无编号句关联本章声明编号逐句受审。"""
    report = (
        "# 调研\n\n## 一、成本构成\n\n"
        f"土建工程约占总投资的 50% 至 60%。\n\n{render_basis_line([CID1])}\n"
    )
    claims = extract_claims_sectioned(report, [CID1])
    assert claims, "无编号句必须关联章节声明编号受审"
    assert all(cid == CID1 for _, cid, _ in claims)
    assert not any(SECTION_BASIS_MARK in sent for sent, _, _ in claims), "依据行不是 claim"


def test_rule_auditor_catches_fabrication_via_section_basis() -> None:
    """凭空数值在本章声明证据里找不到 → unsupported（章节级声明同样能兜住质证）。"""
    report = (
        "# 调研\n\n## 一、成本构成\n\n"
        "土建工程约占总投资的 97%。\n\n"
        f"{render_basis_line([CID1])}\n\n"
        "## 二、降低造价\n\n"
        "高架线造价约为地下线的 40% 至 60%。\n\n"
        f"{render_basis_line([CID2])}\n"
    )
    items = RuleCitationAuditor().audit(report, [_ev(CID1, DOC1), _ev(CID2, DOC2, SUBQS[1])])
    by_section: dict[str, list[Any]] = {}
    for it in items:
        by_section.setdefault(it.section, []).append(it)
    assert by_section["一、成本构成"][0].verdict == "unsupported", "97% 在证据里不存在"
    assert by_section["二、降低造价"][0].verdict == "supported"


# --------------------------------------------------------------------- 4. analyst 节点集成


class _Router:
    def tier(self, task: str, *, fuse_level: int = 0) -> str:
        return "strong"


class _StubGateway:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls: list[str] = []
        self.router = _Router()

    def chat(self, messages: Any, *, task: str, **kw: Any) -> LLMResponse:
        self.calls.append(task)
        text = self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]
        return LLMResponse(
            text=text,
            model="stub",
            task=task,
            provider="stub",
            input_tokens=100,
            output_tokens=200,
            cost_cny=0.001,
            latency_ms=5.0,
            token_estimate=False,
        )

    def embed(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise AssertionError("本组测试不应调用 embedding")


class _NoSearch:
    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        return []


def _ctx(tmp_path: Path, gateway: _StubGateway, *, structured: bool = True) -> NodeContext:
    settings = Settings(
        llm_mode="mock",
        search_mode="mock",
        trace_dir=tmp_path / "trace",
        report_dir=tmp_path / "reports",
        fixture_dir=REPO / "data" / "fixtures",
        checkpoint_db=tmp_path / "cp.sqlite",
        profile_db=tmp_path / "profile.sqlite",
        local_docs_dir=tmp_path / "nodocs",
        analyst_structured=structured,
    )
    settings.ensure_dirs()
    trace = TraceWriter(path=tmp_path / "trace" / "trace.jsonl", run_id="t91")
    return NodeContext(
        settings=settings,
        gateway=gateway,
        search=_NoSearch(),
        trace=trace,
        budget=BudgetConfig(),
        auditor=RuleCitationAuditor(),
    )


def _state() -> dict[str, Any]:
    return {
        "evidence": [_ev(CID1, DOC1), _ev(CID2, DOC2, SUBQS[1])],
        "judgments": [],
        "plan": {
            "objective": "调研城市轨道交通建设成本",
            "outlines": ["一、成本构成", "二、降低造价"],
            "sub_questions": SUBQS,
        },
    }


_STRUCTURED_OK = (
    "<<CHAPTER>>\n标题：核心摘要\n<<CITE>>\n[LOC1-1-1]\n<<TEXT>>\n"
    "土建是成本大头。\n<<END>>\n"
    "<<CHAPTER>>\n标题：一、成本构成\n<<CITE>>\n[LOC1-1-1]\n<<TEXT>>\n"
    "土建工程约占总投资的 50% 至 60%。\n<<END>>\n"
    "<<CHAPTER>>\n标题：二、降低造价\n<<CITE>>\n[LOC1-1-2]\n<<TEXT>>\n"
    "高架线是主要替代方案。\n<<END>>\n"
)


def test_analyst_structured_path_injects_citations_without_retry(tmp_path: Path) -> None:
    gw = _StubGateway([_STRUCTURED_OK])
    ctx = _ctx(tmp_path, gw)
    out = analyst_mod.run(_state(), ctx)

    assert gw.calls == ["analyst"], "结构化成功即带编号——不得再触发 T8.6 零引用重试"
    assert SECTION_BASIS_MARK in out["report"]
    assert out["citation_check"]["pass"] is True
    assert out["citation_check"]["referenced"] == 2
    assert out["citation_check"]["no_citations"] is False
    assert ctx.trace.of("analyst_structured_ok")
    assert not ctx.trace.of("citation_retry"), "结构化路径不该再烧一次纠偏调用"


def test_analyst_falls_back_to_plaintext_when_unparseable(tmp_path: Path) -> None:
    """模型不服从格式 → 降级纯文本路径，T8.6 纠偏逻辑不变。"""
    gw = _StubGateway(["# 报告\n\n土建约占一半。\n", f"# 报告\n\n土建约占 50% 至 60% {CID1}。\n"])
    ctx = _ctx(tmp_path, gw)
    out = analyst_mod.run(_state(), ctx)

    assert ctx.trace.of("analyst_structured_fallback")
    assert gw.calls == ["analyst", "analyst"], "降级后零引用仍按 T8.6 重试一次"
    assert CID1 in out["report"]
    assert out["citation_check"]["no_citations"] is False


def test_analyst_structured_disabled_restores_legacy_path(tmp_path: Path) -> None:
    gw = _StubGateway([f"# 报告\n\n土建约占 50% 至 60% {CID1}。\n"])
    ctx = _ctx(tmp_path, gw, structured=False)
    out = analyst_mod.run(_state(), ctx)

    assert not ctx.trace.of("analyst_structured_ok")
    assert not ctx.trace.of("analyst_structured_fallback")
    assert CID1 in out["report"]


# --------------------------------------------------------------------- 5. mock 链路走"代码注入"主路径


def test_mock_provider_emits_parseable_structured_output() -> None:
    """离线 mock 必须覆盖结构化主路径，否则 CI 对它零覆盖。

    且 mock 刻意模拟真实弱模型形态：TEXT 里**不挂编号**——走的就是"代码注入依据行"。
    """
    from attest.llm.providers import MockProvider

    evs = [_ev(CID1, DOC1), _ev(CID2, DOC2, SUBQS[1])]
    msgs = build_analyst_structured_messages("调研城市轨交成本", ["一、成本构成", "二、降低造价"], evs)
    result = MockProvider().chat(msgs, model="mock", task="analyst")

    chapters = parse_structured_report(result.text)
    assert chapters is not None, "mock 的结构化输出必须可解析"
    assert all(not extract_ids(c.body) for c in chapters), "mock 正文不应自带编号（模拟弱模型）"
    report, dropped = render_structured_report("调研城市轨交成本", chapters, _index())
    assert dropped == []
    assert SECTION_BASIS_MARK in report
    check_ids = set(extract_ids(report))
    assert {CID1, CID2} <= check_ids
