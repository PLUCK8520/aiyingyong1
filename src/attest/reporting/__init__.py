"""⑦ 交付层 · 报告导出与展示稿（T7.7）。

只依赖 stdlib——导出是交付动作，不该为了它引入渲染/排版重依赖。
"""

from .export import export_report, render_markdown
from .pitch import render_pitch

__all__ = ["export_report", "render_markdown", "render_pitch"]
