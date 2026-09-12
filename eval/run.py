"""T7.2 · 正式评测 Runner：`python -m eval.run`。

与 `eval/mini.py` 的关系：
  - `mini.py` 是**最小关卡**（T3.9 遗留，6 题、2 指标、断言式）——留着，跑得快；
  - `run.py` 是**正式评测**（T7.1 的 15 题、五指标、可出报告、可版本对比）。
  两者共用图与 fixture，不重复实现检索/生成逻辑。

五指标：每份报告 token 成本 / 引用有效率 / 大纲覆盖度 / 完成率 / 延迟。
口径声明见 `eval.metrics.caveat()`，会写进 result.json 与 stdout。

用法：
    .venv/Scripts/python.exe -m eval.run                 # 全量 15 题
    .venv/Scripts/python.exe -m eval.run --kind dispute  # 只跑争议型
    .venv/Scripts/python.exe -m eval.run --limit 3       # 只跑前 3 题
    .venv/Scripts/python.exe -m eval.run --tag v0.9-pre  # 打标签便于版本对比
退出码：0 成功（指标无论高低都算成功，指标不是断言）；1 有题目执行异常。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from attest.config import Settings  # noqa: E402
from attest.graph.build import build_context, build_graph, initial_state  # noqa: E402
from attest.logging import setup_logging  # noqa: E402
from attest.retrieval.local_search import clear_local_cache  # noqa: E402

from eval.metrics import caveat, extract_sections, outline_coverage, summarize  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DATASET = EVAL_DIR / "datasets.yaml"
REPORT_DIR = EVAL_DIR / "reports"


def run_case(case: dict, *, out_root: Path) -> dict:
    """跑单题，返回含五指标原始字段的结果 dict。"""
    cid = case["id"]
    settings = Settings(
        llm_mode="mock",
        search_mode="mock",
        log_level="WARNING",
        trace_dir=out_root / "trace" / cid,
        report_dir=out_root / "reports" / cid,
        # T7.4：评测默认关闭研究闭环——15 题共享一个进程时，后跑的题会检索到
        # 先跑题的沉淀（跨题污染），结果随题目顺序漂移，不再可复现。
        # 闭环收益应作为独立对照实验测（开/关两组各跑一遍），不能混进校准过的基准。
        research_memory_enabled=False,
    )
    settings.ensure_dirs()
    ctx = build_context(settings, run_id=cid)
    app = build_graph(ctx)

    t0 = time.perf_counter()
    final = app.invoke(initial_state(case["query"], thread_id=cid))
    elapsed = time.perf_counter() - t0
    summary = ctx.trace.summary()
    ctx.trace.close()

    route = final.get("route")
    report_md = str(final.get("report") or "")
    check = final.get("citation_check") or {}
    outlines = list((final.get("plan") or {}).get("outlines") or [])
    cov, missing = outline_coverage(report_md, outlines)

    return {
        "id": cid,
        "kind": case.get("kind", "?"),
        "topic": case.get("topic", ""),
        "query": case["query"],
        "route": route,
        # ---- ① 成本
        "tokens": int(final.get("tokens_incurred", 0) or 0),
        "cost_cny": float(final.get("cost_incurred", 0.0) or 0.0),
        "llm_calls": summary["llm_calls"],
        # ---- ② 引用
        "referenced": int(check.get("referenced", 0) or 0),
        "unresolved": len(check.get("unresolved") or []),
        "cited_ratio": float(check.get("cited_ratio", 0.0) or 0.0),
        "citation_pass": bool(check.get("pass")),
        # ---- ③ 大纲覆盖度
        "outline_coverage": round(cov, 4),
        "outlines_total": len(outlines),
        "outlines_missing": missing,
        "sections_actual": extract_sections(report_md),
        # ---- ④ 完成率所需
        "has_report": bool(report_md.strip()),
        # ---- ⑤ 延迟
        "duration_s": round(elapsed, 3),
        # ---- 辅助（便于人工核对，不进指标）
        "evidence": len(final.get("evidence") or []),
        "conflicts": len(final.get("conflicts") or []),
        "gaps": len(final.get("gaps") or []),
        "rounds": int(final.get("round", 1) or 1),
        # ---- 与期望对照（不参与指标，只用于体检）
        "expect": {
            k: case[k]
            for k in ("expect_route", "expect_conflict", "expect_min_rounds")
            if k in case
        },
    }


def check_expectations(row: dict) -> tuple[list[str], list[str]]:
    """把 expect_* 与实际比对 → (硬闸失败, 软记录偏差)。

    **硬闸**（`expect_route` / `expect_min_rounds`）：这些是离线档能可靠验证的，
    不达标就是真问题，计入体检失败。

    **软记录**（`expect_conflict`）：离线矛盾启发式有已知保真度上限（详见 datasets.yaml
    头部说明与 `providers._t_conflict` 的局限注释）——planner 的通用子问题 + 按扇出贴标签，
    使"该报的没报 / 不该报的报了"都会出现，且**不能用调参修好**。所以它只如实记录，
    不判通过/失败；真实路径由 dashscope 的 LLM 判定，本项对那条路径才有验收意义。
    """
    exp = row.get("expect") or {}
    fails: list[str] = []
    softs: list[str] = []
    if "expect_route" in exp and row["route"] != exp["expect_route"]:
        fails.append(f"route={row['route']} 期望 {exp['expect_route']}")
    if "expect_min_rounds" in exp and int(row.get("rounds", 1) or 1) < int(exp["expect_min_rounds"]):
        fails.append(f"rounds={row['rounds']} 期望 ≥{exp['expect_min_rounds']}")
    if "expect_conflict" in exp:
        want = bool(exp["expect_conflict"])
        got = bool(int(row.get("conflicts", 0) or 0))
        if got != want:
            softs.append(f"conflicts={row['conflicts']} 设计期望{'有' if want else '无'}（离线启发式局限）")
    return fails, softs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Attest（质证）正式评测 Runner")
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--kind", help="只跑某一类（fact/compare/dispute/local）")
    ap.add_argument("--case", help="只跑某个 id")
    ap.add_argument("--limit", type=int, help="只跑前 N 题（冒烟用）")
    ap.add_argument("--tag", default="", help="给本次结果打标签（便于版本对比）")
    args = ap.parse_args(argv)

    setup_logging("WARNING")
    data = yaml.safe_load(Path(args.dataset).read_text(encoding="utf-8"))
    cases = data.get("cases") or []
    if args.kind:
        cases = [c for c in cases if c.get("kind") == args.kind]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("没有可跑的评测用例。")
        return 1

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{args.tag}" if args.tag else ""
    out_root = REPORT_DIR / f"run-{run_id}{suffix}"
    out_root.mkdir(parents=True, exist_ok=True)
    clear_local_cache()

    print("=" * 96)
    print(f"Attest 正式评测 · run={run_id}{suffix} · {len(cases)} 题")
    print("=" * 96)
    print(f"{'id':<5}{'kind':<9}{'route':<10}{'token':>7}{'引用率':>8}{'覆盖度':>8}{'耗时':>7}  体检")
    print("-" * 96)

    rows: list[dict] = []
    exec_errors: dict[str, str] = {}
    health: dict[str, list[str]] = {}
    soft_notes: dict[str, list[str]] = {}

    for case in cases:
        try:
            row = run_case(case, out_root=out_root)
        except Exception as exc:  # noqa: BLE001 - 评测要记录失败题而非整体崩掉
            exec_errors[case["id"]] = f"{type(exc).__name__}: {exc}"
            print(f"{case['id']:<5}!!! 执行异常：{type(exc).__name__}: {exc}")
            continue

        rows.append(row)
        fails, softs = check_expectations(row)
        if fails:
            health[case["id"]] = fails
        if softs:
            soft_notes[case["id"]] = softs
        mark = ("⚠ " + "；".join(fails)) if fails else ("· " + "；".join(softs) if softs else "OK")
        print(
            f"{row['id']:<5}{row['kind']:<9}{row['route']:<10}"
            f"{row['tokens']:>7,}{row['cited_ratio']:>7.0%}{row['outline_coverage']:>8.0%}"
            f"{row['duration_s']:>6.2f}s  {mark}"
        )

    m = summarize(rows)

    print("\n" + "=" * 96)
    print("五指标")
    print("=" * 96)
    print(f"① 每份报告 token 成本：均值 {m['tokens_per_report']['mean']:,.0f}"
          f" / 中位 {m['tokens_per_report']['median']:,.0f}"
          f" / 最大 {m['tokens_per_report']['max']:,.0f}"
          f"（合计 {m['tokens_per_report']['total']:,} token，¥{m['cost_cny_total']:.4f}）")
    print(f"② 引用有效率：cited_ratio 均值 {m['cited_ratio']['mean']:.1%}"
          f" / 未解析编号为 0 的报告占比 {m['unresolved_zero_rate']:.0%}")
    print(f"③ 大纲覆盖度：均值 {m['outline_coverage']['mean']:.1%}"
          f" / 最低 {m['outline_coverage']['min']:.0%}")
    print(f"④ 完成率（有报告且 unresolved=0）：{m['completion_rate']:.1%}")
    print(f"⑤ 延迟：均值 {m['latency_s']['mean']:.2f}s / 最大 {m['latency_s']['max']:.2f}s")
    print(f"   （research {m['research_cases']} 题 / direct {m['direct_cases']} 题）")

    if health:
        print("\n体检未达预期（硬闸，需修）：")
        for cid, msgs in health.items():
            print(f"  {cid}: " + "；".join(msgs))
    if soft_notes:
        print("\n设计意图偏差（软记录，离线启发式局限，不判失败）：")
        for cid, msgs in soft_notes.items():
            print(f"  {cid}: " + "；".join(msgs))

    payload = {
        "run_id": run_id,
        "tag": args.tag,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": Path(args.dataset).name,
        "trigger": {k: v for k, v in vars(args).items() if k != "dataset" and v},
        "metrics": m,
        "rows": rows,
        "expectation_health": health,
        "conflict_intent_notes": soft_notes,
        "exec_errors": exec_errors,
        "caveat": caveat(),
    }
    (out_root / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n结果已写入：{out_root / 'result.json'}")
    print(f"⚠️ {caveat()}")
    return 1 if exec_errors else 0


if __name__ == "__main__":
    sys.exit(main())
