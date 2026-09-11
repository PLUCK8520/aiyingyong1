"""T7.2 · 五指标实现（纯函数，可单测）。

指标定义（**每个都标了数据来源，不凭感觉**）：

① tokens_per_report  每份报告 token 成本
   来源：`final["tokens_incurred"]` / `final["cost_incurred"]`（图内记账，P0 起有）
② cited_ratio        引用有效率
   来源：`final["citation_check"]["cited_ratio"]` + `unresolved`（P4 审计产出）
③ outline_coverage   大纲覆盖度  ← 本批次新增
   定义：报告实际 `##` 章节 ∩ planner outlines / len(outlines)
   来源：报告正文的 markdown 标题 + `final["plan"]["outlines"]`
④ completion_rate    完成率  ← 本批次新增
   定义：跑完（有报告）**且** unresolved==0 的 research 题占比
   注：这是"完成"的严格口径——产出报告不等于完成，编号必须有来源（项目硬红线）
⑤ latency_s          延迟
   来源：`time.perf_counter()` 包裹 graph.invoke（wall-clock，非节点耗时之和）

⚠️ **口径声明（必须随数字一起给）**：全链路离线 mock，正文由脚本生成。
本模块算出的数字用于**回归与自比**，不代表真实模型下的质量或成本。
"""

from __future__ import annotations

import re
import statistics
from typing import Any

#: 报告里的固定章节（不由 planner 大纲产生），算覆盖度时排除
FIXED_SECTIONS = frozenset({"核心摘要", "争议与分歧", "参考资料"})

#: 匹配 `## 章节名`（二级标题），忽略 `#` 一级标题（那是 objective）
_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def extract_sections(report_md: str) -> list[str]:
    """从报告 markdown 里抽出二级标题（实际章节）。"""
    return [m.strip() for m in _H2_RE.findall(report_md or "")]


def normalize(title: str) -> str:
    """章节名归一化：去空白/标点，便于稳健匹配。

    真实场景里模型可能把「收费模式与落地成本」写成「收费模式与落地成本 」或
    带标点；这里做最小归一化，避免因空格差异误判未覆盖。
    """
    s = (title or "").strip()
    s = re.sub(r"[\s\u3000]+", "", s)
    s = re.sub(r"[：:，,。.、；;（）()\[\]【】]", "", s)
    return s


def outline_coverage(report_md: str, outlines: list[str]) -> tuple[float, list[str]]:
    """算出大纲覆盖度。

    Returns:
        (覆盖率, 未覆盖的章节名列表)。outlines 为空时返回 (1.0, [])——
        无大纲即无要求，不该算作"覆盖不足"（否则会把 direct 题误判成 0）。
    """
    wanted = [o for o in (outlines or []) if str(o).strip()]
    if not wanted:
        return 1.0, []

    actual = {normalize(s) for s in extract_sections(report_md)}
    missing: list[str] = []
    hit = 0
    for o in wanted:
        no = normalize(str(o))
        # 命中判定：完全相等 或 实际章节包含大纲项（模型可能加后缀，如「市场规模（2025）」）
        if no in actual or any(no and no in a for a in actual):
            hit += 1
        else:
            missing.append(str(o))
    return hit / len(wanted), missing


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """把逐题结果汇总成五指标。

    Args:
        rows: 每题的结果 dict（由 runner 产出，含上面列出的字段）。

    Returns:
        指标字典。`research` 相关的指标只统计 route=research 的题——
        直答题没有报告、没有大纲，混进去会把均值拉低成噪声。
    """
    research = [r for r in rows if r.get("route") == "research"]
    tokens = [int(r.get("tokens", 0) or 0) for r in research]
    ratios = [float(r.get("cited_ratio", 0.0) or 0.0) for r in research]
    coverages = [float(r.get("outline_coverage", 0.0) or 0.0) for r in research]
    latencies = [float(r.get("duration_s", 0.0) or 0.0) for r in research]

    # ④ 完成率：严格口径 —— 有报告 且 unresolved==0
    completed = sum(
        1 for r in research if r.get("has_report") and int(r.get("unresolved", 0) or 0) == 0
    )

    def _stat(xs: list[float | int]) -> dict[str, float]:
        if not xs:
            return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0}
        return {
            "mean": round(statistics.fmean(xs), 4),
            "median": round(statistics.median(xs), 4),
            "max": round(float(max(xs)), 4),
            "min": round(float(min(xs)), 4),
        }

    n = len(research)
    return {
        "research_cases": n,
        "direct_cases": len(rows) - n,
        "tokens_per_report": {**_stat(tokens), "total": sum(tokens)},
        "cost_cny_total": round(sum(float(r.get("cost_cny", 0.0) or 0.0) for r in research), 6),
        "cited_ratio": _stat(ratios),
        "unresolved_zero_rate": round(
            sum(1 for r in research if int(r.get("unresolved", 0) or 0) == 0) / n, 4
        )
        if n
        else 0.0,
        "outline_coverage": _stat(coverages),
        "completion_rate": round(completed / n, 4) if n else 0.0,
        "latency_s": _stat(latencies),
    }


def caveat() -> str:
    """口径声明 —— 打印与写入 JSON 时必须带上。"""
    return (
        "离线 mock 模式：正文由脚本按固定模板生成，证据来自合成 fixture，"
        "不是真实模型输出、不是真实检索结果。以上数字用于【回归与自比】，"
        "**不能**用于对外声称质量或成本水平。"
    )
