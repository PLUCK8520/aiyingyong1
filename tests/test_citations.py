"""T2.2 · 引用编号器与索引（纯单测，不联网）。"""

from __future__ import annotations

from attest.quality.citation_check import finalize
from attest.retrieval.citations import (
    CitationIndex,
    Evidence,
    assign_citation_ids,
    extract_ids,
    make_citation_id,
    parse_evidence_block,
    render_evidence_block,
)
from attest.retrieval.ports import SearchResult


def _results(*urls: str) -> list[SearchResult]:
    return [
        SearchResult(source="web", title=f"标题{i}", url=u, content=f"正文{i}", score=0.5)
        for i, u in enumerate(urls, start=1)
    ]


def test_id_format_matches_spec() -> None:
    assert make_citation_id("web", 1, 2, 3) == "[WEB1-2-3]"
    assert make_citation_id("local", 2, 1, 10) == "[LOC2-1-10]"


def test_assign_ids_are_positional_and_stable() -> None:
    ev = assign_citation_ids(_results("https://a", "https://b"), round_no=1, subq_no=1, sub_question="Q1")
    assert [e.citation_id for e in ev] == ["[WEB1-1-1]", "[WEB1-1-2]"]
    again = assign_citation_ids(_results("https://a", "https://b"), round_no=1, subq_no=1, sub_question="Q1")
    assert [e.citation_id for e in again] == [e.citation_id for e in ev], "编号必须可复现"


def test_extract_ids_keeps_first_order_and_dedups() -> None:
    text = "结论A [WEB1-1-1]；结论B [WEB1-1-2]；又见 [WEB1-1-1]。"
    assert extract_ids(text) == ["[WEB1-1-1]", "[WEB1-1-2]"]


def test_index_flags_unresolved_and_unused() -> None:
    ev = assign_citation_ids(_results("https://a", "https://b"), round_no=1, subq_no=1, sub_question="Q1")
    index = CitationIndex.from_evidence(ev)
    text = "用了 [WEB1-1-1]，还编了一个 [WEB1-1-9]。"
    assert index.unresolved_in(text) == ["[WEB1-1-9]"]
    assert index.unused(text) == ["[WEB1-1-2]"]


def test_evidence_block_round_trip() -> None:
    ev = assign_citation_ids(
        _results("https://a", "https://b"), round_no=1, subq_no=2, sub_question="市场规模"
    )
    # 内容里塞一个竖线，验证渲染时被转义、解析不串行
    ev = [ev[0], Evidence(**{**ev[1].__dict__, "content": "含 | 竖线 | 的正文"})]
    block = render_evidence_block(ev)
    parsed = parse_evidence_block(block)
    assert [p["citation_id"] for p in parsed] == ["[WEB1-2-1]", "[WEB1-2-2]"]
    assert parsed[1]["sub_question"] == "市场规模"
    assert "／" in parsed[1]["content"], "竖线应被替换为全角，避免字段错位"


def test_finalize_appends_reference_list_and_strips_model_written_one() -> None:
    ev = assign_citation_ids(_results("https://a"), round_no=1, subq_no=1, sub_question="Q1")
    index = CitationIndex.from_evidence(ev)
    raw = "# 报告\n\n## 正文\n\n- 一句带引用的话 [WEB1-1-1]\n\n## 参考资料\n\n- 模型自己写的，要被去掉\n"
    report, refs, check = finalize(raw, index)
    assert check["pass"] is True
    assert check["referenced"] == 1
    assert "模型自己写的" not in report
    assert report.count("## 参考资料") == 1
    assert "https://a" in refs


def test_finalize_marks_failure_when_citation_has_no_source() -> None:
    index = CitationIndex.from_evidence([])
    report, _, check = finalize("# 报告\n\n- 凭空引用 [WEB1-1-1]\n", index)
    assert check["pass"] is False
    assert check["unresolved"] == ["[WEB1-1-1]"]
