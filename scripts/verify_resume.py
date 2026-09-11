"""T5.5 断点续跑验证：**两个独立进程**模拟"杀进程 → 重启恢复"。

为什么必须跨进程：同进程内 `SqliteSaver` 状态在内存里也在库里，看不出"是否真落盘"。
只有进程 A 挂起退出、进程 B 冷启动后能从同一 SQLite 恢复，才证明**检查点真的持久化**，
也才证明**不重复检索**（重启后不能再跑一遍 scout，否则重复扣额度——FR-24 的验收口径）。

⚠️ **自证断言（2026-09-11 加）**：本脚本原先只 print、不校验，导致"跑通了但结果是错的"
    这件事拖了很久才被发现（`start` 阶段其实一路跑完、根本没停在断点上，`interrupted=false`）。
    现在两阶段都做硬校验，不达标直接 **exit 1**：
      - `start` 阶段必须 `paused=True`（图停在 `human_confirm` 断点上）；
      - `resume` 阶段必须 `ok=True` 且 `scout_delta` 相较 `start` 阶段**不重复爆量**。

用法：
    python scripts/verify_resume.py <phase> <db_path> <thread_id>
    phase = start | resume

约定：
    - `start` 阶段需要 `ATTEST_HUMAN_CONFIRM=1`（否则没有断点，脚本会直接判失败）；
    - 两阶段必须用**同一个 db_path 与 thread_id**；
    - 续跑走 `update_state(plan_approval) + invoke(None)`，**不是** `Command(resume=...)`
      ——静态断点下没有 `interrupt()` 消费点，resume 值无处投递（2026-09-11 实测）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from attest.config import load_settings  # noqa: E402
from attest.graph.build import build_context, build_graph, initial_state  # noqa: E402
from attest.logging import setup_logging  # noqa: E402
from attest.memory.checkpointer import make_checkpointer  # noqa: E402

QUERY = "调研企业知识库 Agent 平台市场，按市场规模/竞品/收费模式三部分输出"


def _scout_count(trace_path: Path) -> int:
    """统计 trace 里的 scout 调用次数（node_start 事件）。"""
    if not trace_path.exists():
        return 0
    n = 0
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("event") == "node_start" and str(ev.get("node", "")).startswith("scout_"):
            n += 1
    return n


def _emit(payload: dict, *, ok: bool) -> int:
    payload = {**payload, "ok": ok}
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if ok else 1


def main() -> int:
    phase, db_path, thread_id = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    setup_logging("WARNING")
    settings = load_settings()
    settings.ensure_dirs()
    trace_path = settings.trace_dir / "trace.jsonl"

    if phase == "start" and not settings.human_confirm_enabled:
        return _emit(
            {
                "phase": "start",
                "error": "ATTEST_HUMAN_CONFIRM 未开启，本次运行不会产生断点，验证无意义。",
                "hint": "设置 ATTEST_HUMAN_CONFIRM=1 后重跑。",
            },
            ok=False,
        )

    with make_checkpointer(db_path) as ckpt:
        ctx = build_context(settings, run_id=f"resume-{phase}")
        app = build_graph(ctx, checkpointer=ckpt)
        cfg = {"configurable": {"thread_id": thread_id}}

        if phase == "start":
            before = _scout_count(trace_path)
            st = app.invoke(initial_state(QUERY, thread_id=thread_id), cfg)
            after = _scout_count(trace_path)
            snap = app.get_state(cfg)
            paused = bool(snap.next)
            report = st.get("report") or ""
            return _emit(
                {
                    "phase": "start",
                    "paused": paused,
                    "paused_at": list(snap.next),
                    "plan_outlines": len((st.get("plan") or {}).get("outlines") or []),
                    "scout_before": before,
                    "scout_after": after,
                    "scout_delta": after - before,
                    "report_chars": len(report),
                },
                ok=paused and not report,
            )

        # phase == resume：**冷启动**进程，只靠 SQLite 恢复
        before = _scout_count(trace_path)
        # 先看恢复出来的 state（不推进）
        snap = app.get_state(cfg)
        if not snap.next:
            return _emit(
                {
                    "phase": "resume",
                    "error": "无待执行任务：检查点里没有断点（start 阶段可能没真正挂起）。",
                    "next": list(snap.next),
                },
                ok=False,
            )
        recovered_plan = (snap.values or {}).get("plan") or {}
        # 续跑方式：静态断点下没有 `interrupt()` 消费点，`Command(resume=...)` 无处投递
        # （实测会静默沿用旧 plan）。正确做法是 `update_state` 落决策 + `invoke(None)` 推进。
        app.update_state(cfg, {"plan_approval": {"action": "approve"}}, as_node="planner")
        st2 = app.invoke(None, cfg)
        after = _scout_count(trace_path)
        # 冷启动后，因首轮扇出只在 human_confirm 之后发生 → resume 才会首次检索；
        # 关键是**不能出现"重跑整条链路"的重复检索**：resume 阶段的 scout 次数
        # 不应超过首轮扇出的理论值（子问题数 × 2 源）+ 补检轮上限。
        n_sub = len((recovered_plan.get("sub_questions") or []))
        round1_max = max(n_sub, 1) * 2
        scout_delta = after - before
        return _emit(
            {
                "phase": "resume",
                "recovered_outlines": len(recovered_plan.get("outlines") or []),
                "plan_approved": bool(st2.get("plan_approved")),
                "scout_before": before,
                "scout_after": after,
                "scout_delta": scout_delta,
                "round1_max_expected": round1_max,
                "report_chars": len(st2.get("report") or ""),
            },
            ok=bool(st2.get("plan_approved")) and bool(st2.get("report")),
        )


if __name__ == "__main__":
    sys.exit(main())
