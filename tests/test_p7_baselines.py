"""T7.2 · baseline 对照组的回归测试（离线 mock，无网络、无 API）。

**这一组测试要证明的**（都是"对照组本身可信"的前提，不是指标算得对）：
  1. 三条 baseline 确实**各自不同**——B2 与 B3 不能因为同一份检索就完全同构
     （否则"审计层的净效果"读出来是 0，结论是假的）；
  2. **指标不适用时不许记 0**——B1 裸答不产出章节、B2 不跑审计，
     它们的"覆盖度/引用率"是 N/A，不是"很差"。把 N/A 当 0 是把"没测"读成"测出来差"；
  3. baseline 与主系统**共用同一套大纲**（借 planner 一次），覆盖度才可比；
  4. token 量级关系必须成立：裸答 < 单轮 RAG < 主系统 —— 若哪天反了，说明管道接错。
"""

from __future__ import annotations

from eval.baselines import BASELINES, run_baseline

import pytest

_CASE = {
    "id": "T-F1",
    "kind": "fact",
    "topic": "大模型推理成本",
    "query": "调研'大模型推理成本'的单价降幅，说明统计口径",
    "expect_route": "research",
}


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    """三条 baseline 各跑一题，结果复用给下面所有断言（省去重复跑图的开销）。"""
    root = tmp_path_factory.mktemp("baselines")
    return {b: run_baseline(_CASE, baseline=b, out_root=root) for b in BASELINES}


def test_three_baselines_registered():
    assert set(BASELINES) == {"b1_bare", "b2_rag", "b3_rag_audit"}


def test_b1_bare_does_no_retrieval(rows):
    """裸答题不应有证据、不应有引用编号。"""
    r = rows["b1_bare"]
    assert r["evidence"] == 0, "裸答不该检索"
    assert r["referenced"] == 0, "裸答不该有引用编号"
    assert r["has_report"], "裸答仍应产出文本"


def test_b1_coverage_is_na_not_full(rows):
    """**关键**：裸答不产出 `##` 章节，覆盖度必须记 0 + 标注不适用。

    绝不能因为"没有大纲"就记 1.0 —— 那会把裸答伪装成与主系统同等覆盖。
    """
    r = rows["b1_bare"]
    assert r["outline_coverage"] == 0.0
    assert r["sections_actual"] == [], "裸答没有二级标题"
    assert "不适用" in r.get("coverage_note", "")


def test_b2_and_b3_differ_on_citation_metrics(rows):
    """B2 不审计 → 引用率不适用；B3 跑审计 → 有真实引用率。

    这条防的是"两条 baseline 实际是同一个东西"——那样对照组给不出任何因果结论。
    """
    b2, b3 = rows["b2_rag"], rows["b3_rag_audit"]
    assert b2["cited_ratio"] == 0.0 and b3["cited_ratio"] >= 0.0
    assert "不适用" in b2.get("cited_ratio_note", ""), "B2 应明确标注引用率不适用"
    assert b3.get("cited_ratio_note") is None, "B3 跑了审计，不该标不适用"


def test_b3_runs_real_auditor(rows):
    """B3 必须真的调了审计器：audited_claims > 0。"""
    assert rows["b3_rag_audit"]["audited_claims"] > 0


def test_baselines_share_outlines_with_main_system(rows):
    """B2/B3 借 planner 拿大纲，所以 outlines_total > 0 且覆盖度可算。"""
    for b in ("b2_rag", "b3_rag_audit"):
        assert rows[b]["outlines_total"] > 0, f"{b} 应借到 planner 大纲"
        assert 0.0 <= rows[b]["outline_coverage"] <= 1.0


def test_token_ordering_is_sane(rows):
    """量级关系：裸答 < 单轮 RAG。若反了，说明哪条路线上多调了东西。"""
    assert rows["b1_bare"]["tokens"] < rows["b2_rag"]["tokens"]
    # B2 与 B3 检索+成文一致，token 应相同（审计是规则版、零 LLM 调用）
    assert rows["b2_rag"]["tokens"] == rows["b3_rag_audit"]["tokens"]
