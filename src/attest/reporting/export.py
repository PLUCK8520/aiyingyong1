"""T7.7 · 报告导出为 Markdown。

**为什么单独一个模块而不是在 session.py 里拼字符串**：
  - 导出格式是**交付契约**——字段顺序、元信息头、参考列表的位置，验收与复用都依赖它，
    值得单测钉住，不该埋在请求处理流程里；
  - 离线 CLI / 评测 / FastAPI 三处都可能要导出，放这里共用。

**关于 PDF**（T7.7 原始表述是"导出 md/pdf"）：
  中文 PDF 需要内嵌 CJK 字体（reportlab/weasyprint，MB 级依赖 + 系统字体依赖），
  与本项目"离线默认、依赖最小"的约束冲突；前端浏览器「打印为 PDF」已覆盖该场景。
  故 PDF 导出**降级为浏览器打印**，代码只交付 md——这是工程取舍，已在任务清单 T7.7 留档。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..logging import get_logger

log = get_logger(__name__)


def _unresolved_count(check: dict[str, Any]) -> int:
    """citation_check["unresolved"] 是**编号列表**（citation_check.py 契约），
    但历史 payload 里也有过计数 int——两种都接，按真实类型取数。"""
    u = check.get("unresolved", 0)
    return len(u) if isinstance(u, (list, tuple)) else int(u or 0)


def render_markdown(result: dict[str, Any], *, exported_at: str | None = None) -> str:
    """把一次运行的结果渲染成独立 Markdown 文档。

    结构（顺序即契约，测试钉住）：
      1. 元信息头（非 frontmatter——给**人**看的导出，YAML 头对读者是噪音）
      2. 报告正文（analyst 产出，原样保留）
      3. 参考资料（reference_list，原样保留）
      4. 审计摘要（让读者能自己判断"这份报告的可信度"）
    """
    query = str(result.get("query") or "")
    route = str(result.get("route") or "research")
    report = str(result.get("report") or "")
    reference_list = str(result.get("reference_list") or "")
    audit = dict(result.get("audit_summary") or {})
    check = dict(result.get("citation_check") or {})
    exported = exported_at or datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines: list[str] = [
        f"# 调研报告：{query}",
        "",
        "> 由 Attest（质证）多 Agent 调研系统生成",
        f"> 导出时间：{exported} · 路由：{route} · "
        f"token：{int(result.get('tokens_incurred') or 0):,} · "
        f"成本：¥{float(result.get('cost_incurred') or 0.0):.4f}",
        "",
        "---",
        "",
        report.strip(),
        "",
    ]

    if reference_list.strip():
        lines += ["---", "", "## 参考资料", "", reference_list.strip(), ""]

    if audit:
        total = audit.get("total", 0)
        supported = audit.get("supported", 0)
        partial = audit.get("partial", 0)
        unsupported = audit.get("unsupported", 0)
        lines += [
            "---",
            "",
            "## 引用审计摘要",
            "",
            f"- 判定句子总数：{total}",
            f"- supported（证据充分支持）：{supported}",
            f"- partial（部分支持）：{partial}",
            f"- unsupported（证据不支持，已降级标注）：{unsupported}",
        ]
        if check:
            lines += [
                f"- 未解析引用编号：{_unresolved_count(check)}",
                f"- 引用有效率：{float(check.get('cited_ratio', 0.0)) * 100:.1f}%",
            ]
        lines.append("")

    return "\n".join(lines)


def export_report(
    result: dict[str, Any],
    report_dir: Path,
    *,
    thread_id: str,
) -> Path | None:
    """落盘到 `{report_dir}/{thread_id}.md`。返回路径；无正文可导（direct 路由）时返回 None。

    导出失败**不该让已完成的运行变成失败**——报告已经在内存里了，
    所以这里只记 warning 并返回 None，不抛。
    """
    if not str(result.get("report") or "").strip():
        log.info(f"[export] thread={thread_id} 无报告正文（direct 路由？），跳过 md 导出")
        return None
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"{thread_id}.md"
        path.write_text(render_markdown(result), encoding="utf-8")
        log.info(f"[export] 报告已导出：{path}")
        return path
    except Exception as exc:  # noqa: BLE001 - 见 docstring
        log.warning(f"[export] 导出失败（不影响本次报告）：{type(exc).__name__}: {exc}")
        return None
