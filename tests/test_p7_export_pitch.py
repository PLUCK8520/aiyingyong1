"""T7.7 · 报告导出（md）与一分钟展示稿的回归测试。

钉住的是**交付契约**：
  - md 的结构顺序（元信息 → 正文 → 参考资料 → 审计摘要）；
  - 导出失败/无正文时不抛错（导出是交付动作，不能反过来弄挂已完成的运行）；
  - 展示稿只讲真实数据——没给的字段就不出现（不编数字）。
"""

from __future__ import annotations

from pathlib import Path

from attest.reporting import export_report, render_markdown, render_pitch

RESULT = {
    "query": "调研'企业知识库 Agent 平台'市场",
    "route": "research",
    "report": "## 核心摘要\n\n市场规模 180 亿元 [LOC1-1-1]。\n\n## 争议与分歧\n\n增速口径不一。",
    "reference_list": "- `[LOC1-1-1]` [资料A](local://docs/a#0) — 子问题：市场规模",
    "citation_check": {"cited_ratio": 0.4545, "unresolved": 0},
    "audit_summary": {"total": 12, "supported": 10, "partial": 2, "unsupported": 0},
    "cost_incurred": 0.0123,
    "tokens_incurred": 16594,
}


# ---------------------------------------------------------------- render_markdown


def test_render_markdown_structure_order() -> None:
    md = render_markdown(RESULT, exported_at="2026-09-12T00:00:00+00:00")
    # 结构顺序即契约：标题 → 元信息 → 正文 → 参考资料 → 审计摘要
    i_title = md.index("# 调研报告")
    i_meta = md.index("导出时间：2026-09-12T00:00:00+00:00")
    i_body = md.index("## 核心摘要")
    i_ref = md.index("## 参考资料")
    i_audit = md.index("## 引用审计摘要")
    assert i_title < i_meta < i_body < i_ref < i_audit
    # 正文与参考列表原样保留（不重排、不改写）
    assert "市场规模 180 亿元 [LOC1-1-1]。" in md
    assert "[资料A](local://docs/a#0)" in md


def test_render_markdown_audit_summary_numbers() -> None:
    md = render_markdown(RESULT)
    assert "supported（证据充分支持）：10" in md
    assert "partial（部分支持）：2" in md
    assert "未解析引用编号：0" in md
    assert "引用有效率：45.5%" in md  # 0.4545 → 45.5%（四舍五入到一位）


def test_render_markdown_without_reference_list() -> None:
    md = render_markdown({**RESULT, "reference_list": ""})
    assert "## 参考资料" not in md
    assert "## 引用审计摘要" in md


def test_render_markdown_without_audit() -> None:
    md = render_markdown({**RESULT, "audit_summary": {}, "citation_check": {}})
    assert "## 引用审计摘要" not in md
    assert "## 核心摘要" in md


# ---------------------------------------------------------------- export_report


def test_export_report_writes_file(tmp_path: Path) -> None:
    path = export_report(RESULT, tmp_path, thread_id="t1")
    assert path is not None and path.exists()
    assert path.name == "t1.md"
    assert "市场规模 180 亿元" in path.read_text(encoding="utf-8")


def test_export_report_direct_route_returns_none(tmp_path: Path) -> None:
    """direct 路由没有报告正文——如实跳过，不产出一个空壳 md。"""
    path = export_report(
        {**RESULT, "route": "direct", "report": ""}, tmp_path, thread_id="t2"
    )
    assert path is None
    assert not (tmp_path / "t2.md").exists()


def test_export_report_failure_does_not_raise(tmp_path: Path) -> None:
    """目录不可写时只记日志返回 None——报告已在内存，导出失败不算运行失败。"""
    blocked = tmp_path / "blocked"
    blocked.write_text("i am a file, not a dir", encoding="utf-8")
    path = export_report(RESULT, blocked / "sub", thread_id="t3")
    assert path is None


# ---------------------------------------------------------------- render_pitch


def test_pitch_contains_real_numbers() -> None:
    pitch = render_pitch(RESULT, elapsed_s=0.05)
    assert "调研'企业知识库 Agent 平台'市场" in pitch
    assert "16,594 token" in pitch
    assert "¥0.0123" in pitch
    assert "0.05s" in pitch
    assert "supported 10" in pitch
    assert "引用有效率 45.5%" in pitch
    assert "未解析编号 0" in pitch


def test_pitch_omits_metrics_not_given() -> None:
    """没给评测汇总就不讲对照数字——展示稿不许编。"""
    pitch = render_pitch(RESULT)
    assert "正式评测" not in pitch
    assert "耗时" not in pitch  # elapsed_s 没给就不写


def test_pitch_with_metrics() -> None:
    metrics = {
        "tokens_mean": "16,594",
        "cited_ratio_mean": "42.6%",
        "coverage_mean": "100.0%",
        "completion_rate": "100.0%",
    }
    pitch = render_pitch(RESULT, metrics=metrics)
    assert "token 均值 16,594" in pitch
    assert "引用率 42.6%" in pitch
    assert "完成率 100.0%" in pitch


def test_pitch_declares_offline_boundary() -> None:
    """展示稿必须自带边界声明（零造假铁律同样适用于展示材料）。"""
    pitch = render_pitch(RESULT)
    assert "离线 mock" in pitch
    assert "不代表真实模型质量/成本水平" in pitch
