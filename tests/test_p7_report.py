"""T7.3 · 评测报告生成的回归测试（纯函数 + 只读文件）。

**要证明的**（都是"报告不会骗人"的前提）：
  1. 口径声明**必然出现在报告里**——不允许生成一份只给数字不给 caveat 的报告；
  2. 对比时**格式按指标性质定**，不按数值量级猜——延迟是 0.05 但绝不能被渲染成百分数
     （早先版本真出过 `4.97%` 这种读不通的输出）；
  3. `_pick` 支持**任意位置子串**（人用时间戳尾段指代一次 run），多命中时报错而不是乱猜；
  4. baseline 对照表里，**不适用项必须印成 N/A**，不能把 0 印上去。
"""

from __future__ import annotations

import json

import pytest

from eval import report as R


def _mk_run(run_id: str, *, tag: str = "", tokens: float = 1000.0, cr: float = 0.4,
            cov: float = 1.0, comp: float = 1.0, lat: float = 0.05) -> dict:
    return {
        "run_id": run_id,
        "tag": tag,
        "dataset": "datasets.yaml",
        "caveat": "离线 mock 口径声明 ABC",
        "metrics": {
            "research_cases": 2,
            "direct_cases": 1,
            "tokens_per_report": {"mean": tokens, "median": tokens, "max": tokens + 1, "min": tokens - 1, "total": int(tokens) * 2},
            "cost_cny_total": 0.01,
            "cited_ratio": {"mean": cr, "median": cr, "max": cr, "min": cr},
            "outline_coverage": {"mean": cov, "median": cov, "max": cov, "min": cov},
            "completion_rate": comp,
            "latency_s": {"mean": lat, "median": lat, "max": lat, "min": lat},
            "unresolved_zero_rate": 1.0,
        },
        "rows": [],
        "expectation_health": {},
        "conflict_intent_notes": {},
        "exec_errors": {},
    }


def test_single_report_always_carries_caveat():
    """口径声明必须在报告里——报告只给数字不给口径 = 骗人。"""
    md = R.render_single(_mk_run("20260101-000000", tag="t1"))
    assert "口径声明" in md
    assert "离线 mock 口径声明 ABC" in md


def test_single_report_has_five_metrics():
    md = R.render_single(_mk_run("20260101-000000", tag="t1"))
    for marker in ("① 每份报告 token 成本", "② 引用有效率", "③ 大纲覆盖度", "④ **完成率**", "⑤ 延迟"):
        assert marker in md, f"报告缺指标：{marker}"


def test_compare_renders_latency_as_seconds_not_percent():
    """**关键回归**：延迟 0.05s 必须印成秒，不能因为 <1 就被当成比率印百分数。"""
    a, b = _mk_run("A", lat=0.05), _mk_run("B", lat=0.08)
    md = R.render_compare([a, b], ["A", "B"])
    line = next(l for l in md.splitlines() if "⑤ 延迟" in l)
    assert "0.050s" in line and "0.080s" in line, f"延迟行渲染异常：{line}"
    assert "%" not in line, f"延迟行不该出现百分号：{line}"


def test_compare_renders_ratio_as_percent():
    a, b = _mk_run("A", cr=0.40), _mk_run("B", cr=0.55)
    md = R.render_compare([a, b], ["A", "B"])
    line = next(l for l in md.splitlines() if "② 引用有效率" in l)
    assert "40.00%" in line and "55.00%" in line, f"比率行渲染异常：{line}"


def test_compare_renders_token_with_thousands_separator():
    a, b = _mk_run("A", tokens=15959), _mk_run("B", tokens=16594)
    md = R.render_compare([a, b], ["A", "B"])
    line = next(l for l in md.splitlines() if "① token 均值" in l)
    assert "15,959" in line and "16,594" in line, f"token 行缺少千分位：{line}"


def test_pick_matches_any_substring():
    runs = [_mk_run("20260911-213727"), _mk_run("20260911-220811", tag="p7-final")]
    assert R._pick(runs, "213727")["run_id"] == "20260911-213727"
    assert R._pick(runs, "p7-final")["run_id"] == "20260911-220811"


def test_pick_exact_beats_substring():
    """精确 run_id/tag 命中优先于子串命中。"""
    runs = [_mk_run("abc"), _mk_run("abcdef")]
    assert R._pick(runs, "abc")["run_id"] == "abc"


def test_pick_ambiguous_raises():
    """多命中时必须报错，不能静默挑一个——挑错了会拿错版本的报告去对比。"""
    runs = [_mk_run("20260911-aaa"), _mk_run("20260911-bbb")]
    with pytest.raises(SystemExit, match="匹配到多次"):
        R._pick(runs, "20260911")


def test_pick_missing_raises():
    runs = [_mk_run("A")]
    with pytest.raises(SystemExit, match="找不到"):
        R._pick(runs, "ZZZ")


def test_baseline_table_prints_na_for_not_applicable():
    """baseline 表里不适用项必须印 N/A，不能印 0（否则读成"做得很差"）。"""
    run = _mk_run("B")
    bl = {
        "baselines": {"b1_bare": "裸答"},
        "metrics": {
            "b1_bare": {
                "tokens_per_report": {"mean": 127},
                "cited_ratio": {"mean": 0.0},
                "outline_coverage": {"mean": 0.0},
                "latency_s": {"mean": 0.0},
                "cited_ratio_applicable": False,
                "coverage_applicable": False,
            }
        },
        "honesty_note": "不可对外宣称",
    }
    md = R.render_single(run, with_baselines=True, baselines=[bl])
    line = next(l for l in md.splitlines() if "b1_bare" in l)
    assert line.count("N/A") == 2, f"B1 的引用率与覆盖度都该印 N/A：{line}"
    assert "0.0%" not in line
