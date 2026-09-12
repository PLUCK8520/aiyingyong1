"""T1.5 / T5.1 / T5.2 / T5.4 · CLI 入口。

用法：
    .venv/Scripts/python.exe scripts/chat.py "调研'企业知识库 Agent 平台'市场，按市场规模/竞品/收费模式三部分输出"
    .venv/Scripts/python.exe scripts/chat.py --llm siliconflow "..."   # 真实模型（硅基流动）+ 离线证据
    .venv/Scripts/python.exe scripts/chat.py --llm dashscope "..."     # 真实模型（阿里百炼）+ 离线证据
    .venv/Scripts/python.exe scripts/chat.py --search tavily "..."     # 真实检索（烧 credits）

P5 新增（记忆与协同）：
    --thread NAME          指定会话线程（默认按时间戳新建）；同 thread 可断点续跑
    --resume               用最近一次挂起的 thread 恢复（配合 --thread）
    --confirm              打开人工确认大纲（编译期静态断点）；默认关（等价于"跳过"）
    --remember "键=值"     写入一条用户画像（跨会话生效）
    --profile              打印当前画像
    --forget KEY           删除一条画像

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
from attest.agents.human_confirm import confirm_payload  # noqa: E402
from attest.graph.build import build_context, build_graph, initial_state  # noqa: E402
from attest.logging import setup_logging  # noqa: E402
from attest.memory.profile import parse_remember_directive  # noqa: E402
from attest.memory.checkpointer import make_checkpointer  # noqa: E402
from attest.trace.events import TraceWriter  # noqa: E402

DEFAULT_QUERY = "调研'企业知识库 Agent 平台'市场，按市场规模/竞品/收费模式三部分输出，附溯源链接"

NODE_LABEL = {
    "memory_loader": "记忆注入",
    "intent_router": "意图分流",
    "direct_responder": "快速回答",
    "planner": "任务规划",
    "human_confirm": "大纲确认",
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
    if node == "memory_loader":
        return f"注入画像 {len(increment.get('profile') or {})} 条"
    if node == "intent_router":
        return f"route={increment.get('route')}（{increment.get('route_reason', '')}）"
    if node == "planner":
        plan = increment.get("plan") or {}
        return f"子问题 {len(plan.get('sub_questions', []))} 个 / 大纲 {len(plan.get('outlines', []))} 章"
    if node == "human_confirm":
        ap = increment.get("plan_approval") or {}
        return f"决策={ap.get('action', ap.get('decision', '-'))}"
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


def _interactive_confirm(payload: dict[str, Any]) -> dict[str, Any]:
    """T5.4 交互式大纲确认（CLI）。返回要 resume 进去的值。

    ⚠️ 这里是**终端交互**：不允许在非交互环境（管道/CI）里静默挂起——
    调用方（`main`）应保证只在真 TTY 下走到这里。
    """
    print()
    print("─" * 78)
    print("📋 大纲确认（人工介入点）")
    print(f"  目标：{payload.get('objective', '')}")
    print("  大纲：")
    for i, o in enumerate(payload.get("outlines") or [], 1):
        print(f"    {i}. {o}")
    subs = payload.get("sub_questions") or []
    if subs:
        print(f"  子问题（{len(subs)} 个）：")
        for i, s in enumerate(subs, 1):
            print(f"    {i}. {s}")
    print("─" * 78)
    print("  [Enter] 确认继续   [e] 编辑大纲   [s] 跳过   [q] 退出")
    try:
        choice = input("请选择> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n[确认] 未输入，按『确认』继续。")
        return {"action": "approve"}

    if choice in ("q", "quit", "退出"):
        return {"action": "abort"}
    if choice in ("e", "edit", "编辑"):
        outlines = list(payload.get("outlines") or [])
        print("  输入新大纲（每行一章，空行结束；直接回车保留原大纲）：")
        new_lines: list[str] = []
        while True:
            try:
                line = input("    > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            new_lines.append(line)
        if new_lines:
            return {"action": "edit", "plan": {"outlines": new_lines}}
        return {"action": "approve"}
    if choice in ("s", "skip", "跳过"):
        return {"action": "skip"}
    return {"action": "approve"}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attest（质证）· 逐句质证的多 Agent 调研 CLI")
    p.add_argument("query", nargs="*", help=f"调研问题，留空用默认示例：{DEFAULT_QUERY}")
    p.add_argument("--llm", choices=["mock", "dashscope", "siliconflow", "ollama"], help="覆盖 ATTEST_LLM_MODE")
    p.add_argument("--search", choices=["mock", "tavily"], help="覆盖 ATTEST_SEARCH_MODE")
    p.add_argument("--no-save", action="store_true", help="不把报告写入 data/reports/")
    p.add_argument("--quiet", action="store_true", help="不打印节点日志")
    # ---------- P5 ----------
    p.add_argument("--thread", help="会话线程 ID（默认按时间戳新建；同 thread 可断点续跑）")
    p.add_argument("--resume", action="store_true", help="恢复指定 thread 的挂起状态（需配合 --thread）")
    p.add_argument("--confirm", action="store_true", help="打开人工确认大纲（interrupt）")
    p.add_argument("--remember", help="写入一条用户画像，格式 '键=值' 或 '内容'")
    p.add_argument("--profile", action="store_true", help="打印当前用户画像后退出")
    p.add_argument("--forget", help="删除一条用户画像（按 key）")
    p.add_argument("--no-checkpoint", action="store_true", help="本次不接检查点（无状态单跑）")
    return p.parse_args(argv)


def _handle_profile_ops(args: argparse.Namespace, settings: Any) -> int | None:
    """画像的增/删/查在**进图之前**处理（这些是元操作，不是调研任务）。"""
    from attest.memory.profile import ProfileStore

    if args.profile:
        store = ProfileStore(settings.profile_db, enabled=settings.profile_enabled)
        items = store.all()
        if not items:
            print("（画像为空）")
        else:
            print(f"用户画像（{len(items)} 条）：")
            for it in items:
                print(f"  - {it.key}：{it.value}")
        return 0
    if args.forget:
        store = ProfileStore(settings.profile_db, enabled=settings.profile_enabled)
        ok = store.delete(args.forget)
        print(f"{'已删除' if ok else '未找到'}：{args.forget}")
        return 0
    if args.remember:
        store = ProfileStore(settings.profile_db, enabled=settings.profile_enabled)
        parsed = parse_remember_directive(args.remember) or ("偏好", args.remember)
        key, value = parsed
        store.set(key, value)
        print(f"已记住：{key} = {value}")
        return 0
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.llm:
        os.environ["ATTEST_LLM_MODE"] = args.llm
    if args.search:
        os.environ["ATTEST_SEARCH_MODE"] = args.search
    if args.confirm:
        os.environ["ATTEST_HUMAN_CONFIRM"] = "1"

    try:
        settings = load_settings()
    except Exception as exc:  # noqa: BLE001 - 配置错误要给人话
        print(f"\n[配置错误] {exc}\n", file=sys.stderr)
        return 2

    setup_logging("WARNING" if args.quiet else settings.log_level)
    settings.ensure_dirs()

    # 画像元操作（与调研链路无关，先处理完就退出）
    meta = _handle_profile_ops(args, settings)
    if meta is not None:
        return meta

    query = " ".join(args.query).strip() or DEFAULT_QUERY
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    thread_id = args.thread or run_id

    trace = TraceWriter(path=settings.trace_dir / "trace.jsonl", run_id=run_id)
    ctx = build_context(settings, trace=trace, run_id=run_id)

    print("=" * 78)
    print(f"Attest（质证）· run_id={run_id} · thread={thread_id}")
    print(f"问题：{query}")
    print(f"模式：LLM={settings.llm_mode} | 检索={settings.search_mode} | "
          f"检查点={'off' if args.no_checkpoint else 'on'} | 人工确认={'on' if settings.human_confirm_enabled else 'off'}")
    print("=" * 78)

    started = time.perf_counter()

    def _run(app: Any, cfg: dict[str, Any], input_value: Any, interactive: bool,
             *, resume_times: int = 0) -> dict[str, Any] | None:
        """跑图；遇**静态断点**则（若可交互）确认后放行。返回最终 state。

        T5.4 语义（编译期静态断点，`interrupt_before=["human_confirm"]`）：
          - 首次 `stream(input_state, cfg)` 会在 `human_confirm` **之前**停住；
          - 用 `app.get_state(cfg).next` 判断是否停在断点上；
          - **续跑方式（关键，2026-09-11 实测）**：
              静态断点下没有 `interrupt()` 调用点，`Command(resume=...)` 的 payload
              **无处投递**（resume 值由 `interrupt()` 从 scratchpad 消费）。
              正确做法是两步：
                1. `app.update_state(cfg, {"plan_approval": decision}, as_node="planner")`
                   把用户决策落进 state；
                2. `stream(None, cfg)`（或 `update_state` 后直接推进）继续执行。
              实测 `Command(resume=...)` 在静态断点下会**静默沿用旧 plan**（编辑不生效）。
        """
        from langgraph.types import Command  # noqa: F401 - 保留导入以兼容 resume 分支

        final: dict[str, Any] = {}
        budget = resume_times or 8  # 防死循环：正常最多 1 次确认
        while True:
            for chunk in app.stream(input_value, cfg, stream_mode=["updates", "values"]):
                mode, payload = chunk  # type: ignore[misc]
                if mode == "updates":
                    for node, increment in (payload or {}).items():
                        if node.startswith("__"):
                            continue
                        label = NODE_LABEL.get(node, node)
                        print(f"  ✓ {label:<6} {_brief(node, increment)}")
                else:
                    final = payload  # type: ignore[assignment]

            snap = app.get_state(cfg)
            if not snap.next:
                return final
            if budget <= 0:
                print("[确认] 断点未收敛，放弃续跑。", file=sys.stderr)
                return final

            pending = _pending_confirm_payload(snap)
            if interactive and sys.stdin.isatty():
                decision = _interactive_confirm(pending)
            else:
                decision = {"action": "approve"}  # 非交互环境：默认确认（不静默挂死）
                print("  ℹ 非交互环境，自动确认大纲继续")
            if decision.get("action") == "abort":
                print("[确认] 用户选择中止。", file=sys.stderr)
                return None
            budget -= 1
            # 决策落进 state（`as_node="planner"` 保证 plan 通道的写入来源正确），
            # 然后以 None 输入推进过断点。
            app.update_state(cfg, {"plan_approval": decision}, as_node="planner")
            input_value = None

    def _pending_confirm_payload(snap: Any) -> dict[str, Any]:
        """从待执行快照里取出要展示给人工的大纲。

        断点停在 `human_confirm` **之前**，所以 `plan` 已经在 state 里了——
        不需要依赖 interrupt payload（那是 09-10 旧方案的产物）。
        """
        values = snap.values or {}
        plan = values.get("plan") or {}
        payload = confirm_payload(plan)
        for task in (snap.tasks or ()):
            for ints in (getattr(task, "interrupts", ()) or ()):
                if getattr(ints, "value", None):
                    return ints.value
        return payload

    try:
        trace.emit("run_start", query=query, llm_mode=settings.llm_mode,
                   search_mode=settings.search_mode, thread_id=thread_id)

        if args.no_checkpoint:
            app = build_graph(ctx)
            cfg: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
            final = _run(app, cfg, initial_state(query, thread_id=thread_id), interactive=True)
        else:
            with make_checkpointer(settings.checkpoint_db) as ckpt:
                app = build_graph(ctx, checkpointer=ckpt)
                cfg = {"configurable": {"thread_id": thread_id}}
                if args.resume:
                    snap = app.get_state(cfg)
                    if not (snap.values or {}):
                        print(f"[恢复失败] thread={thread_id} 无检查点记录。", file=sys.stderr)
                        return 3
                    if not snap.next:
                        print(f"↻ thread={thread_id} 无待执行任务，本图为已结束状态。", file=sys.stderr)
                        return 3
                    print(f"↻ 从检查点恢复 thread={thread_id}（停在断点：{', '.join(snap.next)}）")
                    # 断点续跑：先以 None 推进（`_run` 会在断点处询问并 update_state 后继续）。
                    final = _run(app, cfg, None, interactive=True)
                else:
                    final = _run(app, cfg, initial_state(query, thread_id=thread_id), interactive=True)
        if final is None:
            trace.emit("run_aborted")
            trace.close()
            return 130
    except KeyboardInterrupt:
        print("\n[中断] 已停止。可用 --thread 同名 + --resume 恢复。", file=sys.stderr)
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
    prof = final.get("profile") or {}
    if prof:
        print(f"记忆: 本轮注入画像 {len(prof)} 条（{', '.join(sorted(prof))}）")

    if settings.llm_mode == "mock":
        print("⚠️  本次为【离线 mock 模式】：正文由脚本生成，不是模型输出；证据来自合成 fixture。")
        print("    想看真实输出：在 .env 填 SILICONFLOW_API_KEY，然后加 --llm siliconflow")

    if not args.no_save:
        out = settings.report_dir / f"report-{run_id}.md"
        footer = [
            "",
            "---",
            "",
            f"<!-- 运行元数据：run_id={run_id} | thread={thread_id} | LLM={settings.llm_mode} "
            f"| 检索={settings.search_mode} | token={summary['total_tokens']} "
            f"| 成本=¥{summary['cost_cny']:.4f} | 耗时={elapsed:.2f}s -->",
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
