"""T7.7 · 一分钟展示稿：从一次真实运行结果生成 60 秒可讲的 pitch。

**定位**：不是营销文案，是"答辩/演示时照着念 60 秒"的稿子——
每一句都必须能被本次运行的真实数据支撑（零造假铁律同样适用于展示稿）。
所以它不是模板填空，而是**从 state 里取数**：取不到的字段如实写"未测"，
绝不编一个看起来漂亮的数字。
"""

from __future__ import annotations

from typing import Any


def render_pitch(
    result: dict[str, Any],
    *,
    metrics: dict[str, Any] | None = None,
    elapsed_s: float | None = None,
) -> str:
    """生成展示稿。

    Args:
        result: 一次运行的结果（session.result 同构 dict）。
        metrics: 可选的评测汇总（eval.metrics.summarize 的返回），给了就讲对照数字。
        elapsed_s: 本次运行墙钟耗时；不给则不写（不编）。
    """
    query = str(result.get("query") or "")
    audit = dict(result.get("audit_summary") or {})
    check = dict(result.get("citation_check") or {})
    tokens = int(result.get("tokens_incurred") or 0)
    cost = float(result.get("cost_incurred") or 0.0)
    conflicts = result.get("conflicts") or []
    n_conflicts = len(conflicts)

    cited = float(check.get("cited_ratio", 0.0)) * 100
    # citation_check["unresolved"] 是编号列表（契约），不是计数——取长度
    _u = check.get("unresolved", 0)
    unresolved = len(_u) if isinstance(_u, (list, tuple)) else int(_u or 0)

    lines: list[str] = [
        "# 一分钟展示稿 · Attest（质证）",
        "",
        "## 一句话",
        f"给定题目「{query}」，系统自动完成**检索 → 交叉验证 → 成文 → 引用审计**全流程，"
        "产出一份**每句话都能回查来源**的行业调研报告。",
        "",
        "## 三个可以现场验证的点",
        "",
        "1. **引用可回查**：正文里的 `[LOC1-1-2]` 这类编号，点击即达原始资料——"
        f"本次报告引用有效率 {cited:.1f}%，未解析编号 {unresolved} 个。",
        f"2. **审计兜底**：{audit.get('total', 0)} 句断言逐句过审——"
        f"supported {audit.get('supported', 0)} / partial {audit.get('partial', 0)} / "
        f"unsupported {audit.get('unsupported', 0)}；不支持的句子会被**降级标注**而不是悄悄放行。",
        f"3. **矛盾摆上台面**：检索到 {n_conflicts} 处信源互斥，报告里单开「争议与分歧」一节呈现，"
        "而不是取平均糊弄过去。",
        "",
        "## 成本与速度",
        "",
        f"- 本次运行：{tokens:,} token，约 ¥{cost:.4f}"
        # 耗时按实际量级给精度：亚秒级给两位小数（0.05s 不该被印成 0.1s）
        + (f"，耗时 {elapsed_s:.4g}s。" if elapsed_s is not None else "。"),
    ]

    if metrics:
        lines += [
            f"- 正式评测（15 题离线基准）：token 均值 {metrics.get('tokens_mean', '未测')}，"
            f"引用率 {metrics.get('cited_ratio_mean', '未测')}，"
            f"大纲覆盖度 {metrics.get('coverage_mean', '未测')}，"
            f"完成率 {metrics.get('completion_rate', '未测')}。",
        ]

    lines += [
        "",
        "## 如实声明的边界",
        "",
        "- 当前为离线 mock 演示：证据来自合成 fixture，数字用于**回归与自比**，"
        "不代表真实模型质量/成本水平；",
        "- 接真实模型（Qwen3/DashScope）只需配 `.env`，链路不变。",
        "",
    ]
    return "\n".join(lines)
