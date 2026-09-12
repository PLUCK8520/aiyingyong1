"""T7.7 · 一键生成演示资产：跑一次真实（离线 mock）调研，落盘 md 报告 + 一分钟展示稿。

用法：
    .venv/Scripts/python scripts/make_pitch.py

产出（覆盖写，可重复跑）：
    docs/demo/report.md   —— 完整调研报告（md 导出的真实样例）
    docs/demo/pitch.md    —— 一分钟展示稿（数字全部来自本次运行 + 最近一次正式评测）

⚠️ 本脚本就是「闭环命中可现场演示」的载体（T7.7 验收第二条）：
   连跑两遍，第二遍的 pitch 会多出"历史结论复用"一段——那是 T7.4 研究闭环在干活。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from attest.config import Settings  # noqa: E402
from attest.graph.build import build_context, build_graph, initial_state  # noqa: E402
from attest.reporting import export_report, render_pitch  # noqa: E402

QUERY = "调研'企业知识库 Agent 平台'市场，按市场规模/竞品/收费模式三部分输出，附溯源链接"
DEMO_DIR = REPO / "docs" / "demo"


def main() -> int:
    settings = Settings(
        llm_mode="mock",
        search_mode="mock",
        log_level="WARNING",
        trace_dir=REPO / "data" / "trace",
        report_dir=DEMO_DIR,
        research_memory_enabled=True,  # 闭环开：演示"第二次调研复用第一次的结论"
        research_memory_dir=REPO / "data" / "index" / "research_memory",
    )
    settings.ensure_dirs()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    ctx = build_context(settings, run_id="demo")
    had_memory = ctx.memory.count() if ctx.memory else 0

    app = build_graph(ctx, interrupt_before=[])
    t0 = time.perf_counter()
    final = app.invoke(initial_state(QUERY, thread_id="demo"))
    elapsed = time.perf_counter() - t0

    result = {
        "query": QUERY,
        "route": final.get("route"),
        "report": final.get("report") or "",
        "reference_list": final.get("reference_list") or "",
        "citation_check": final.get("citation_check") or {},
        "audit_summary": final.get("audit_summary") or {},
        "cost_incurred": float(final.get("cost_incurred") or 0.0),
        "tokens_incurred": int(final.get("tokens_incurred") or 0),
        "conflicts": final.get("conflicts") or [],
    }

    md_path = export_report(result, DEMO_DIR, thread_id="report")

    # 闭环演示证据：本次检索命中了几条历史结论
    mem_hits = final.get("memory_hits") or []
    reused = sum(1 for h in mem_hits if h.get("in_context"))

    pitch = render_pitch(result, elapsed_s=elapsed)
    if had_memory or reused:
        pitch += (
            "\n## 研究闭环（第二遍跑才有这一段）\n\n"
            f"- 开跑前库里已有 **{had_memory}** 条历史结论（上一遍调研沉淀的）；\n"
            f"- 本次检索命中 **{len(mem_hits)}** 条，其中 **{reused}** 条进入正文证据——"
            "它们同样被编号、同样过了引用审计；\n"
            f"- 本次跑完又沉淀，库里现有 **{ctx.memory.count() if ctx.memory else 0}** 条。\n"
        )
    pitch_path = DEMO_DIR / "pitch.md"
    pitch_path.write_text(pitch, encoding="utf-8")

    print(f"报告：{md_path}")
    print(f"展示稿：{pitch_path}")
    print(f"闭环：跑前库存 {had_memory} 条 → 本次命中 {len(mem_hits)} 条（进正文 {reused} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
