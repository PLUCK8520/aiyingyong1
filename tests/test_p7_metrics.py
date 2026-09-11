"""T7.2 · 五指标与大纲覆盖度的回归测试（纯函数，无 IO、无 API）。

**这一组测试要证明的**：
  1. 大纲覆盖度**能抓出漏章节**——不是"有报告就算全覆盖"；
  2. 覆盖度对空格/标点差异**鲁棒**（模型加个后缀不该判成未覆盖）；
  3. 固定章节（核心摘要/争议与分歧/参考资料）**不算**大纲覆盖；
  4. `outlines` 为空时返回 1.0（无大纲即无要求），**不返回 0**——
     否则 direct 题或无大纲的 research 题会被误判成"覆盖不足"；
  5. **完成率的严格口径**：有报告但 unresolved>0 **不算完成**
     （产出报告 ≠ 完成；编号必须有来源是项目硬红线）；
  6. 汇总只统计 research 题，直答题不污染均值。
"""

from __future__ import annotations

from eval.metrics import (
    extract_sections,
    normalize,
    outline_coverage,
    summarize,
)


# ---------------------------------------------------------------- 章节抽取

def test_extract_sections_only_h2():
    md = "# 一级标题\n\n## 市场规模\n\n正文\n\n## 竞争格局\n\n正文\n\n### 三级不算"
    assert extract_sections(md) == ["市场规模", "竞争格局"]


def test_extract_sections_empty():
    assert extract_sections("") == []
    assert extract_sections("没有标题的正文") == []


def test_normalize_strips_space_and_punct():
    assert normalize(" 市场规模 ") == "市场规模"
    assert normalize("收费模式与落地成本：") == "收费模式与落地成本"
    assert normalize("市场规模（2025）") == "市场规模2025"


# ---------------------------------------------------------------- 覆盖度

def test_coverage_detects_missing_section():
    """**核心用例**：漏了一章必须能测出来，不能"有报告就满分"。"""
    report = "## 市场规模\n\n正文\n\n## 竞争格局\n\n正文\n"
    cov, missing = outline_coverage(report, ["市场规模", "竞争格局", "收费模式"])
    assert cov == 2 / 3
    assert missing == ["收费模式"]


def test_coverage_full():
    report = "## 市场规模\n\nA\n\n## 竞争格局\n\nB\n"
    cov, missing = outline_coverage(report, ["市场规模", "竞争格局"])
    assert cov == 1.0
    assert missing == []


def test_coverage_robust_to_space_and_suffix():
    """模型把大纲项写成「市场规模（2025）」不该判成未覆盖。"""
    report = "## 市场规模（2025）\n\n正文\n"
    cov, missing = outline_coverage(report, ["市场规模"])
    assert cov == 1.0, "带后缀的章节名应判为覆盖，否则会系统性低估覆盖度"
    assert missing == []


def test_coverage_ignores_fixed_sections():
    """固定章节不该被算作覆盖了大纲项。"""
    report = "## 核心摘要\n\nA\n\n## 争议与分歧\n\nB\n\n## 参考资料\n\nC\n"
    cov, missing = outline_coverage(report, ["市场规模", "竞争格局"])
    assert cov == 0.0, "固定章节不能冒充大纲章节"
    assert set(missing) == {"市场规模", "竞争格局"}


def test_coverage_empty_outlines_is_full_not_zero():
    """无大纲 → 1.0 而非 0.0。

    这条很容易写反：把"没有大纲要求"当成"覆盖度为 0"，
    会让所有无大纲的题（如直答、简化路径）被误判为覆盖不足。
    """
    cov, missing = outline_coverage("## 任意章节\n\n正文", [])
    assert cov == 1.0
    assert missing == []


def test_coverage_none_report():
    """报告为空但有大纲 → 覆盖度必须是 0（不是异常，也不能算满分）。"""
    cov, missing = outline_coverage("", ["市场规模"])
    assert cov == 0.0
    assert missing == ["市场规模"]


# ---------------------------------------------------------------- 汇总

def _row(**kw):
    base = {
        "route": "research",
        "tokens": 1000,
        "cost_cny": 0.01,
        "cited_ratio": 0.5,
        "unresolved": 0,
        "has_report": True,
        "outline_coverage": 1.0,
        "duration_s": 1.0,
    }
    base.update(kw)
    return base


def test_summarize_basic():
    rows = [_row(tokens=1000), _row(tokens=3000)]
    m = summarize(rows)
    assert m["research_cases"] == 2
    assert m["tokens_per_report"]["mean"] == 2000
    assert m["tokens_per_report"]["total"] == 4000
    assert m["completion_rate"] == 1.0


def test_completion_rate_excludes_unresolved():
    """有报告但 unresolved>0 → **不算完成**（严格口径）。"""
    rows = [
        _row(unresolved=0, has_report=True),
        _row(unresolved=2, has_report=True),  # 有报告，但编号查不到来源
        _row(unresolved=0, has_report=False),  # 没报告
        _row(unresolved=0, has_report=True),
    ]
    m = summarize(rows)
    assert m["completion_rate"] == 0.5, "4 题里只有 2 题真正完成"


def test_summarize_excludes_direct_cases():
    """直答题不进 research 指标分母（否则会把均值拉低成噪声）。"""
    rows = [_row(tokens=10000), _row(route="direct", tokens=200, has_report=False, outline_coverage=0.0)]
    m = summarize(rows)
    assert m["research_cases"] == 1
    assert m["direct_cases"] == 1
    assert m["tokens_per_report"]["mean"] == 10000, "直答的 200 token 不该被算进来"


def test_summarize_empty_is_safe():
    """零 research 题不能崩（除零），也不该报完成率 1.0。"""
    m = summarize([])
    assert m["research_cases"] == 0
    assert m["completion_rate"] == 0.0
    assert m["tokens_per_report"]["mean"] == 0.0
