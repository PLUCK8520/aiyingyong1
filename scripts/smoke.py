"""T0.6 / T3.11 · 冒烟脚本：一条命令验证「网关能调通 + trace 完整 + 节点成对 + P3 双源与矛盾」。

用法：
    .venv/Scripts/python.exe scripts/smoke.py

退出码 0 = 冒烟通过。任何一项不过都返回非 0，便于以后挂到 CI 或 pre-commit。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from attest.config import load_settings  # noqa: E402
from attest.graph.build import build_context, build_graph, initial_state  # noqa: E402
from attest.graph.state import reducer_fields_ok  # noqa: E402
from attest.llm.prompts import build_direct_messages  # noqa: E402
from attest.logging import setup_logging  # noqa: E402
from attest.trace.events import TraceWriter  # noqa: E402

QUERY = "调研'企业知识库 Agent 平台'市场，按市场规模/竞品/收费模式三部分输出"

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def main() -> int:
    settings = load_settings()
    setup_logging("WARNING")
    settings.ensure_dirs()

    print("=" * 72)
    print(f"冒烟：LLM={settings.llm_mode} | 检索={settings.search_mode}")
    print("=" * 72)

    # 1) state reducer 自检（漏了 reducer = 扇出静默丢数据）
    missing = reducer_fields_ok()
    check("state 并行字段全部声明 reducer", not missing, f"缺失：{missing}")

    trace = TraceWriter(path=settings.trace_dir / "smoke.jsonl", run_id="smoke")
    ctx = build_context(settings, trace=trace, run_id="smoke")

    # 2) 单次 LLM 调用 + 记账
    resp = ctx.gateway.chat(build_direct_messages("用一句话说明什么是向量检索。"), task="direct")
    check("LLM 网关调用成功", bool(resp.text), f"model={resp.model} tokens={resp.total_tokens}")
    check("成本已记账", resp.cost_cny >= 0.0, f"cost=¥{resp.cost_cny:.6f}")
    llm_events = trace.of("llm_call")
    check("trace 写入 llm_call 事件", len(llm_events) == 1, f"{len(llm_events)} 条")

    # 3) 全图跑两遍：分别覆盖 research 与 direct 两条分支
    #    （单次运行只走一条分支，所以不能一次断言全部节点都出现）
    app = build_graph(ctx)

    research = app.invoke(initial_state(QUERY, thread_id="smoke-research"))
    started_nodes = {e.get("node") for e in trace.of("node_start")}
    for node in ("intent_router", "planner", "scout_web", "scout_local", "evidence_judge", "reflect", "analyst"):
        check(f"[research] 节点 {node} 出现", node in started_nodes)
    check("[research] 产出非空报告", bool(research.get("report")), f"{len(research.get('report') or '')} 字")
    check(
        "[research] 引用完整性校验通过",
        bool((research.get("citation_check") or {}).get("pass")),
        str(research.get("citation_check")),
    )
    check(
        "[research] 3 路子问题全部扇出",
        len(research.get("evidence") or []) >= 3,
        f"evidence={len(research.get('evidence') or [])}",
    )

    # 3b) P3 切片：双源检索 + 反思 + 矛盾检测
    evidence = research.get("evidence") or []
    local_ev = [e for e in evidence if (e.url or "").startswith("local://")]
    web_ev = [e for e in evidence if not (e.url or "").startswith("local://")]
    check("[P3] 双源证据到齐（web + local）", bool(web_ev) and bool(local_ev),
          f"web={len(web_ev)} local={len(local_ev)}")
    check("[P3] reflect_count 已入 state", "reflect_count" in research,
          f"reflect_count={research.get('reflect_count')}")
    check("[P3] reflect_targets 字段存在", isinstance(research.get("reflect_targets"), list),
          f"{len(research.get('reflect_targets') or [])} 个补检任务")
    conflicts = research.get("conflicts")
    check("[P3] conflicts 字段存在且为列表", isinstance(conflicts, list), f"{len(conflicts or [])} 条")
    if conflicts:
        check("[P3] 报告渲染「争议与分歧」章节", "争议与分歧" in (research.get("report") or ""))
        check("[P3] 矛盾项带双方引用编号",
              all(getattr(c, "source_a", None) and getattr(c, "source_b", None) for c in conflicts))

    direct = app.invoke(initial_state("你好，用一句话说明什么是向量检索。", thread_id="smoke-direct"))
    check("[direct] 路由为 direct", direct.get("route") == "direct", str(direct.get("route")))
    check("[direct] 产出直接回答", bool(direct.get("direct_answer")), str(direct.get("direct_answer"))[:40])
    check(
        "[direct] 未触发检索",
        not direct.get("evidence"),
        f"evidence={len(direct.get('evidence') or [])}",
    )

    summary = trace.summary()
    check("node_start / node_end 全部成对", summary["node_pairs_ok"], str(summary["unmatched_nodes"]))

    trace.close()
    print("-" * 72)
    print(
        f"trace 摘要：节点 {len(summary['nodes'])} 个 | LLM 调用 {summary['llm_calls']} 次 | "
        f"token {summary['total_tokens']} | 成本 ¥{summary['cost_cny']:.6f}"
    )
    print(f"报告：{len(research.get('report') or '')} 字，引用校验 "
          f"{'通过' if (research.get('citation_check') or {}).get('pass') else '未通过'}")
    print("=" * 72)
    print("冒烟结果：", "PASS" if not failures else f"FAIL（{len(failures)} 项）：{failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
