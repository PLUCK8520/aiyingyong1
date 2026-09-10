"""T1.5 · CLI 入口。

用法：
    .venv/Scripts/python.exe scripts/chat.py "调研'企业知识库 Agent 平台'市场，按市场规模/竞品/收费模式三部分输出，附溯源链接"
    .venv/Scripts/python.exe scripts/chat.py --llm dashscope "..."     # 真实模型 + 离线证据
    .venv/Scripts/python.exe scripts/chat.py --search tavily "..."     # 真实检索（烧 credits）

默认全离线（mock LLM + fixture 检索）：**没有任何 API key 也能完整跑通并出报告**。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from attest.config import load_settings  # noqa: E402
from attest.graph.build import build_context, build_graph, initial_state  # noqa: E402
from attest.logging import setup_logging  # noqa: E402
from attest.trace.events import TraceWriter  # noqa: E402

DEFAULT_QUERY = "调研'企业知识库 Agent 平台'市场，按市场规模/竞品/收费模式三部分输出，附溯源链接"

NODE_LABEL = {
    "intent_router": "意图分流",
    "direct_responder": "快速回答",
    "planner": "任务规划",
    "scout_web": "并行检索",
    "scout_local": "本地检索",
    "evidence_judge": "证据判别",
    "reflect": "反思补检",
    "analyst": "撰写报告",
    "auditor": "引用审计",
}


def _brief(node: str, increment: dict[str, Any]) -> str:
    if not isinstance(increment, dict):
        return ""
    if node == "intent_router":
        return f"route={increment.get('route')}（{increment.get('route_reason', '')}）"
    if node == "planner":
        plan = increment.get("plan") or {}
        return f"子问题 {len(plan.get('sub_questions', []))} 个 / 大纲 {len(plan.get('outlines', []))} 章"
    if node in ("scout_web", "scout_local"):
        return f"命中 {len(increment.get('evidence', []))} 条证据"
    if node == "evidence_judge":
        return f"判别 {len(increment.get('judgments', []))} 条 / 缺口 {len(increment.get('gaps', []))} 项"
    if node == "analyst":
        check = increment.get("citation_check") or {}
        return f"引用 {check.get('referenced', 0)} 处 / 未解析 {len(check.get('unresolved', []))} 处"
    if node == "auditor":
        s = increment.get("audit_summary") or {}
        return (
            f"判定 {s.get('total', 0)} 句 | supported {s.get('supported', 0)} / "
            f"partial {s.get('partial', 0)} / unsupported {s.get('unsupported', 0)} | "
            f"降级 {s.get('degraded_sentences', 0)} 句"
        )
    if node == "direct_responder":
        return f"{len(increment.get('direct_answer', ''))} 字"
    return ""


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attest（质证）· 逐句质证的多 Agent 调研 CLI")
    p.add_argument("query", nargs="*", help=f"调研问题，留空用默认示例：{DEFAULT_QUERY}")
    p.add_argument("--llm", choices=["mock", "dashscope", "ollama"], help="覆盖 ATTEST_LLM_MODE")
    p.add_argument("--search", choices=["mock", "tavily"], help="覆盖 ATTEST_SEARCH_MODE")
    p.add_argument("--no-save", action="store_true", help="不把报告写入 data/reports/")
    p.add_argument("--quiet", action="store_true", help="不打印节点日志")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.llm:
        os.environ["ATTEST_LLM_MODE"] = args.llm
    if args.search:
        os.environ["ATTEST_SEARCH_MODE"] = args.search

    query = " ".join(args.query).strip() or DEFAULT_QUERY

    try:
        settings = load_settings()
    except Exception as exc:  # noqa: BLE001 - 配置错误要给人话
        print(f"\n[配置错误] {exc}\n", file=sys.stderr)
        return 2

    setup_logging("WARNING" if args.quiet else settings.log_level)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    settings.ensure_dirs()
    trace = TraceWriter(path=settings.trace_dir / "trace.jsonl", run_id=run_id)
    ctx = build_context(settings, trace=trace, run_id=run_id)
    app = build_graph(ctx)

    print("=" * 78)
    print(f"Attest（质证）· run_id={run_id}")
    print(f"问题：{query}")
    print(f"模式：LLM={settings.llm_mode} | 检索={settings.search_mode}")
    print("=" * 78)

    trace.emit("run_start", query=query, llm_mode=settings.llm_mode, search_mode=settings.search_mode)
    started = time.perf_counter()

    final: dict[str, Any] = {}
    try:
        for chunk in app.stream(initial_state(query, thread_id=run_id), stream_mode=["updates", "values"]):
            mode, payload = chunk  # type: ignore[misc]
            if mode == "updates":
                for node, increment in (payload or {}).items():
                    if node.startswith("__"):
                        continue
                    label = NODE_LABEL.get(node, node)
                    print(f"  ✓ {label:<6} {_brief(node, increment)}")
            else:
                final = payload  # type: ignore[assignment]
    except KeyboardInterrupt:
        print("\n[中断] 已停止。", file=sys.stderr)
        trace.emit("run_aborted")
        trace.close()
        return 130
    except Exception as exc:  # noqa: BLE001
        trace.emit("run_error", error=f"{type(exc).__name__}: {exc}")
        trace.close()
        print(f"\n[运行失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - started
    summary = trace.summary()
    trace.emit("run_end", duration_ms=round(elapsed * 1000, 1), **{k: summary[k] for k in ("total_tokens", "cost_cny")})
    trace.close()

    print("=" * 78)
    if final.get("route") == "direct":
        print(final.get("direct_answer", "（无输出）"))
    else:
        print(final.get("report", "（无报告产出）"))

    print("-" * 78)
    check = final.get("citation_check") or {}
    print(
        f"trace: 节点 {len(summary['nodes'])} 个 / start-end 成对 {summary['node_pairs_ok']} | "
        f"LLM 调用 {summary['llm_calls']} 次 | token {summary['total_tokens']} | "
        f"成本 ¥{summary['cost_cny']:.4f} | 耗时 {elapsed:.2f}s"
    )
    if check:
        print(
            f"引用: 有来源 {check.get('referenced', 0)} 处 / 未解析 {len(check.get('unresolved', []))} 处 / "
            f"引用段落占比 {check.get('cited_ratio', 0):.0%} | 校验 {'通过' if check.get('pass') else '未通过'}"
        )
    audit = final.get("audit_summary") or {}
    if audit:
        print(
            f"审计: 判定 {audit.get('total', 0)} 句 | 支持 {audit.get('supported', 0)} / "
            f"部分 {audit.get('partial', 0)} / 未证实 {audit.get('unsupported', 0)} | "
            f"降级 {audit.get('degraded_sentences', 0)} 句（审计器：{audit.get('auditor')}）"
        )

    if settings.llm_mode == "mock":
        print("⚠️  本次为【离线 mock 模式】：正文由脚本生成，不是模型输出；证据来自合成 fixture。")
        print("    想看真实输出：在 .env 填 DASHSCOPE_API_KEY，然后加 --llm dashscope")

    if not args.no_save:
        out = settings.report_dir / f"report-{run_id}.md"
        footer = [
            "",
            "---",
            "",
            f"<!-- 运行元数据：run_id={run_id} | LLM={settings.llm_mode} | 检索={settings.search_mode} "
            f"| token={summary['total_tokens']} | 成本=¥{summary['cost_cny']:.4f} | 耗时={elapsed:.2f}s -->",
        ]
        if settings.llm_mode == "mock":
            footer.append(
                "> ⚠️ **离线 mock 模式**：正文由脚本按固定模板生成，**不是语言模型输出**；"
                "证据来自合成 fixture，**不是真实检索结果**。仅用于验证链路。"
            )
        out.write_text((final.get("report") or final.get("direct_answer") or "") + "\n".join(footer), encoding="utf-8")
        print(f"报告已保存：{out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
