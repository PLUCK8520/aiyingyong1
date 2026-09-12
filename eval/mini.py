"""T3.9 · 最小评测尺子：`python -m eval.mini`。

只出两项指标（刻意不贪多）：
  ① **每份报告的 token 成本** —— 没有这个数字，任何"优化了"都不可证伪（D6）
  ② **引用有效率** —— cited_ratio 均值 + `unresolved == 0` 的比例（后者的语义是
     "报告里不许出现查不到来源的编号"，是项目的硬红线）

用法：
    .venv/Scripts/python.exe -m eval.mini
    .venv/Scripts/python.exe -m eval.mini --case M4      # 只跑某一题
退出码：全部断言通过 0 / 有失败 1。
"""

from __future__ import annotations

import argparse
import json
import statistics
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

EVAL_DIR = Path(__file__).resolve().parent
DATASET = EVAL_DIR / "datasets.mini.yaml"
REPORT_DIR = EVAL_DIR / "reports"


def run_case(case: dict, *, out_root: Path, quiet: bool) -> dict:
    cid = case["id"]
    settings = Settings(
        llm_mode="mock",
        search_mode="mock",
        log_level="WARNING" if quiet else "INFO",
        trace_dir=out_root / "trace" / cid,
        report_dir=out_root / "reports" / cid,
        # 同 eval/run.py：跨题污染会让 M1~M6 冒烟随顺序漂移，默认关
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

    evidence = list(final.get("evidence") or [])
    check = final.get("citation_check") or {}
    return {
        "id": cid,
        "query": case["query"],
        "route": final.get("route"),
        "evidence": len(evidence),
        "loc_evidence": sum(1 for e in evidence if e.source == "local"),
        "conflicts": len(final.get("conflicts") or []),
        "gaps": len(final.get("gaps") or []),
        "rounds": int(final.get("round", 1) or 1),
        "reflect_count": int(final.get("reflect_count", 0) or 0),
        "tokens": int(final.get("tokens_incurred", 0) or 0),
        "cost_cny": float(final.get("cost_incurred", 0.0) or 0.0),
        "llm_calls": summary["llm_calls"],
        "referenced": int(check.get("referenced", 0) or 0),
        "unresolved": len(check.get("unresolved") or []),
        "cited_ratio": float(check.get("cited_ratio", 0.0) or 0.0),
        "citation_pass": bool(check.get("pass")),
        "duration_s": round(elapsed, 2),
        "expect": {
            k: case[k]
            for k in ("expect_route", "expect_conflict", "expect_min_rounds")
            if k in case
        },
    }


def check_case(row: dict) -> list[str]:
    exp = row["expect"]
    fails: list[str] = []
    if "expect_route" in exp and row["route"] != exp["expect_route"]:
        fails.append(f"route={row['route']} 期望 {exp['expect_route']}")
    if "expect_conflict" in exp:
        want = bool(exp["expect_conflict"])
        if bool(row["conflicts"]) != want:
            fails.append(f"conflicts={row['conflicts']} 期望{'有' if want else '无'}矛盾")
    if "expect_min_rounds" in exp and row["rounds"] < int(exp["expect_min_rounds"]):
        fails.append(f"rounds={row['rounds']} 期望 ≥{exp['expect_min_rounds']}")
    if row["route"] == "research" and row["unresolved"]:
        fails.append(f"unresolved={row['unresolved']}（硬红线：编号必须有来源）")
    return fails


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Attest（质证）最小评测尺子")
    ap.add_argument("--case", help="只跑指定 id")
    ap.add_argument("--quiet", action="store_true", help="不打印节点日志")
    args = ap.parse_args(argv)

    setup_logging("WARNING" if args.quiet else "INFO")
    data = yaml.safe_load(DATASET.read_text(encoding="utf-8"))
    cases = data.get("cases") or []
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
    if not cases:
        print("没有可跑的评测用例。")
        return 1

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = REPORT_DIR / f"mini-{run_id}"
    out_root.mkdir(parents=True, exist_ok=True)
    clear_local_cache()

    rows: list[dict] = []
    failures: dict[str, list[str]] = {}
    for case in cases:
        row = run_case(case, out_root=out_root, quiet=args.quiet)
        rows.append(row)
        fails = check_case(row)
        if fails:
            failures[case["id"]] = fails
        print(
            f"  {'✓' if not fails else '✗'} {row['id']:<4} route={row['route']:<9} "
            f"证据 {row['evidence']:>2}（LOC {row['loc_evidence']:>2}） 矛盾 {row['conflicts']} "
            f"轮次 {row['rounds']} token {row['tokens']:>6} 引用率 {row['cited_ratio']:.0%}"
        )

    research = [r for r in rows if r["route"] == "research"]
    tokens = [r["tokens"] for r in research] or [0]
    ratios = [r["cited_ratio"] for r in research] or [0.0]
    unresolved_ok = sum(1 for r in research if r["unresolved"] == 0)
    pass_ok = sum(1 for r in research if r["citation_pass"])

    print("\n" + "=" * 78)
    print(f"两项核心指标（research 题 {len(research)} 道）")
    print("=" * 78)
    print(
        f"① 每份报告 token 成本：均值 {statistics.fmean(tokens):,.0f} / "
        f"中位 {statistics.median(tokens):,.0f} / 最大 {max(tokens):,}  "
        f"（合计 {sum(tokens):,} token，¥{sum(r['cost_cny'] for r in research):.4f}）"
    )
    print(
        f"② 引用有效率：cited_ratio 均值 {statistics.fmean(ratios):.1%} / "
        f"未解析编号为 0 的报告 {unresolved_ok}/{len(research)} / "
        f"完整性校验通过 {pass_ok}/{len(research)}"
    )

    payload = {
        "run_id": run_id,
        "metrics": {
            "tokens_per_report_mean": round(statistics.fmean(tokens), 1),
            "tokens_per_report_median": statistics.median(tokens),
            "tokens_per_report_max": max(tokens),
            "cost_cny_total": round(sum(r["cost_cny"] for r in research), 6),
            "cited_ratio_mean": round(statistics.fmean(ratios), 4),
            "unresolved_zero_rate": round(unresolved_ok / len(research), 4) if research else 0.0,
            "citation_pass_rate": round(pass_ok / len(research), 4) if research else 0.0,
        },
        "rows": rows,
        "failures": failures,
        "note": "离线 mock（正文由脚本生成，非模型输出）；指标用于回归与自比，不与外部系统比。",
    }
    (out_root / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if failures:
        print("-" * 78)
        print(f"断言失败 {len(failures)} 题：")
        for cid, msgs in failures.items():
            print(f"  {cid}: " + "；".join(msgs))
    print(f"\n结果已写入：{out_root / 'result.json'}")
    print("⚠️ 离线 mock 模式：正文由脚本生成，token 成本受 mock 提示词长度驱动，不代表真实模型成本。")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
