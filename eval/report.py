"""T7.3 · 评测报告生成与版本对比：`python -m eval.report`。

把 `eval/reports/` 下的 result.json 汇总成**人可读的 markdown 报告**，供：
  - 求职展示：一页说清"这套系统现在什么水平、比上次好在哪"；
  - 回归对比：两次 run 的指标并排，涨跌一目了然。

三种用法：
    python -m eval.report                      # 用最新一次 run 生成报告
    python -m eval.report --run <run-dir 名>   # 指定某次
    python -m eval.report --compare A B        # 两版并排对比（可给 tag 或 run-id 前缀）
    python -m eval.report --with-baselines     # 附上 baseline 对照表

**设计纪律**：
  - 报告里**每个数字都必须能追到某个 result.json**，不做二次估算；
  - 与 baseline 并排时**标注"不可比项"**（如 B1 的覆盖度是 N/A）——不并排假数据；
  - 口径声明（caveat）**必然出现在报告开头**，不允许只放在末尾或被省略。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

REPORT_DIR = Path(__file__).resolve().parent / "reports"


# ------------------------------------------------------------------ 加载

def _load_runs() -> list[dict]:
    """读所有 run-*/result.json，按 run_id 升序（即时间升序）。"""
    out: list[dict] = []
    for p in sorted(REPORT_DIR.glob("run-*/result.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _load_baselines() -> list[dict]:
    out: list[dict] = []
    for p in sorted(REPORT_DIR.glob("baselines-*/result.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _pick(runs: list[dict], key: str) -> dict:
    """按 run-id / tag / 子串找一次 run。

    支持**任意位置子串**（不只是前缀）——人习惯用时间戳尾段指代一次 run
    （`213727` 指 `20260911-213727`），只匹配前缀会让人困惑。
    多个命中时报错并要求说得更具体，避免"猜错了一次 run 却静默出报告"。
    """
    if not runs:
        raise SystemExit("eval/reports/ 下没有可用的 run-*/result.json，先跑 `python -m eval.run`。")
    for r in runs:  # 精确 run_id / tag 优先
        if r.get("run_id") == key or r.get("tag") == key:
            return r
    hits = [
        r for r in runs
        if key in str(r.get("run_id", "")) or key in str(r.get("tag", ""))
    ]
    if len(hits) == 1:
        return hits[0]
    names = [r.get("tag") or r.get("run_id") for r in runs]
    if not hits:
        raise SystemExit(f"找不到 run：{key}。可用：{names}")
    raise SystemExit(f"“{key}” 匹配到多次：{[r.get('tag') or r.get('run_id') for r in hits]}，请说得更具体。")


# ------------------------------------------------------------------ 渲染

def _fmt_metrics_table(m: dict) -> list[str]:
    """五指标表。每个数字都注明来源字段，便于回查。"""
    tp, cr, oc, lat = (
        m["tokens_per_report"],
        m["cited_ratio"],
        m["outline_coverage"],
        m["latency_s"],
    )
    return [
        "| 指标 | 均值 | 中位 | 最大 | 最小 |",
        "|---|---|---|---|---|",
        f"| ① 每份报告 token 成本 | {tp['mean']:,.0f} | {tp['median']:,.0f} | {tp['max']:,.0f} | {tp['min']:,.0f} |",
        f"| ② 引用有效率 | {cr['mean']:.1%} | {cr['median']:.1%} | {cr['max']:.1%} | {cr['min']:.1%} |",
        f"| ③ 大纲覆盖度 | {oc['mean']:.1%} | {oc['median']:.1%} | {oc['max']:.1%} | {oc['min']:.1%} |",
        f"| ⑤ 延迟（秒） | {lat['mean']:.2f} | {lat['median']:.2f} | {lat['max']:.2f} | {lat['min']:.2f} |",
        "",
        f"- ④ **完成率**（有报告 且 unresolved=0）：**{m['completion_rate']:.1%}**"
        f"（research {m['research_cases']} 题 / direct {m['direct_cases']} 题）",
        f"- ① 合计 token：{tp['total']:,}，合计成本 ¥{m['cost_cny_total']:.4f}",
        f"- ② 未解析编号为 0 的报告占比：{m['unresolved_zero_rate']:.0%}",
    ]


def _delta(cur: float, prev: float | None) -> str:
    """给出涨跌标记。**方向好≠数字大**——token 降是好事、覆盖度升是好事，各自单独标。"""
    if prev is None:
        return "—（无对比）"
    diff = cur - prev
    if abs(diff) < 1e-9:
        return "持平"
    arrow = "↑" if diff > 0 else "↓"
    return f"{arrow} {diff:+.4g}"


def render_single(run: dict, *, with_baselines: bool = False, baselines: list[dict] | None = None) -> str:
    m = run["metrics"]
    lines: list[str] = []
    lines.append(f"# Attest 评测报告 · {run.get('tag') or run['run_id']}")
    lines.append("")
    lines.append(f"> 生成于 {datetime.now().isoformat(timespec='seconds')}｜"
                 f"run_id `{run['run_id']}`｜数据集 `{run.get('dataset', '?')}`")
    lines.append("")
    lines.append("> ⚠️ **口径声明（先读这段再读数字）**")
    lines.append(f"> {run.get('caveat', '')}")
    lines.append("")

    lines.append("## 五指标")
    lines.append("")
    lines.extend(_fmt_metrics_table(m))
    lines.append("")

    # 体检
    health = run.get("expectation_health") or {}
    softs = run.get("conflict_intent_notes") or {}
    lines.append("## 体检")
    lines.append("")
    if health:
        lines.append("**硬闸未达预期（需修）**：")
        for cid, msgs in health.items():
            lines.append(f"- `{cid}`：{'；'.join(msgs)}")
    else:
        lines.append("- 硬闸（`expect_route` / `expect_min_rounds`）**全部通过**。")
    lines.append("")
    if softs:
        lines.append("**设计意图偏差（软记录；离线启发式局限，不判失败）**：")
        for cid, msgs in softs.items():
            lines.append(f"- `{cid}`：{'；'.join(msgs)}")
        lines.append("")
        lines.append("> 离线档 planner 产出与主题无关的通用子问题，`sub_question` 按扇出分支贴标签，"
                     "导致对比/本地型题目仍会报出该主题的真实矛盾。这是**离线启发式的保真度上限**，"
                     "真实路径由 dashscope 的 LLM 判定，不依赖该启发式。")
        lines.append("")

    errs = run.get("exec_errors") or {}
    if errs:
        lines.append("## 执行异常")
        lines.append("")
        for cid, msg in errs.items():
            lines.append(f"- `{cid}`：{msg}")
        lines.append("")

    # 逐题明细
    lines.append("## 逐题明细")
    lines.append("")
    lines.append("| id | 类型 | route | token | 引用率 | 覆盖度 | 轮次 | 证据 | 矛盾 | 耗时 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in run.get("rows", []):
        lines.append(
            f"| {r['id']} | {r['kind']} | {r['route']} | {r['tokens']:,} | "
            f"{r['cited_ratio']:.0%} | {r['outline_coverage']:.0%} | {r['rounds']} | "
            f"{r['evidence']} | {r['conflicts']} | {r['duration_s']:.2f}s |"
        )
    lines.append("")

    if with_baselines and baselines:
        lines.append("## baseline 对照")
        lines.append("")
        lines.append("| 路线 | token | 引用率 | 覆盖度 | 延迟 | 说明 |")
        lines.append("|---|---|---|---|---|---|")
        b = baselines[-1]  # 最近一次
        for bid, desc in (b.get("baselines") or {}).items():
            bm = (b.get("metrics") or {}).get(bid) or {}
            tp = bm.get("tokens_per_report", {}).get("mean", 0)
            cr = bm.get("cited_ratio", {}).get("mean", 0)
            oc = bm.get("outline_coverage", {}).get("mean", 0)
            lat = bm.get("latency_s", {}).get("mean", 0)
            cr_s = f"{cr:.1%}" if bm.get("cited_ratio_applicable") else "N/A"
            oc_s = f"{oc:.1%}" if bm.get("coverage_applicable") else "N/A"
            lines.append(f"| `{bid}` | {tp:,.0f} | {cr_s} | {oc_s} | {lat:.2f}s | {desc} |")
        lines.append("")
        lines.append(f"- **主系统（本报告）**：token {m['tokens_per_report']['mean']:,.0f} / "
                     f"引用率 {m['cited_ratio']['mean']:.1%} / 覆盖度 {m['outline_coverage']['mean']:.1%}")
        lines.append("")
        lines.append(f"> {b.get('honesty_note', '')}")
        lines.append("")
    return "\n".join(lines)


def render_compare(runs: list[dict], keys: list[str]) -> str:
    a, b = _pick(runs, keys[0]), _pick(runs, keys[1])
    ma, mb = a["metrics"], b["metrics"]
    lines = [
        f"# Attest 版本对比 · {a.get('tag') or a['run_id']} → {b.get('tag') or b['run_id']}",
        "",
        f"> 生成于 {datetime.now().isoformat(timespec='seconds')}",
        "",
        "> ⚠️ 两次 run 必须**同数据集、同模式**才可比。若数据集版本不同，下表无意义。",
        f"> 数据集：`{a.get('dataset', '?')}` → `{b.get('dataset', '?')}`",
        "",
        "| 指标 | 前 | 后 | 变化 | 读法 |",
        "|---|---|---|---|---|",
    ]

    def row(name: str, va: float, vb: float, better: str, kind: str) -> str:
        """涨跌行。**格式由指标性质决定，不按数值量级猜**——
        猜量级会踩坑：延迟是 0.05（<1）但绝不该渲染成百分数（早先版本就这样出过 `4.97%`）。
        kind: "int" 大数（token，带千分位）/ "pct" 比率 / "sec" 秒。
        """
        diff = vb - va
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
        if kind == "int":
            va_s, vb_s, d_s = f"{va:,.0f}", f"{vb:,.0f}", f"{diff:+,.0f}"
        elif kind == "pct":
            va_s, vb_s, d_s = f"{va:.2%}", f"{vb:.2%}", f"{diff:+.2%}"
        else:  # sec
            va_s, vb_s, d_s = f"{va:.3f}s", f"{vb:.3f}s", f"{diff:+.3f}"
        note = {"lower": "越低越好", "higher": "越高越好"}[better]
        return f"| {name} | {va_s} | {vb_s} | {arrow} {d_s} | {note} |"

    lines.append(row("① token 均值", ma["tokens_per_report"]["mean"], mb["tokens_per_report"]["mean"], "lower", "int"))
    lines.append(row("② 引用有效率", ma["cited_ratio"]["mean"], mb["cited_ratio"]["mean"], "higher", "pct"))
    lines.append(row("③ 大纲覆盖度", ma["outline_coverage"]["mean"], mb["outline_coverage"]["mean"], "higher", "pct"))
    lines.append(row("④ 完成率", ma["completion_rate"], mb["completion_rate"], "higher", "pct"))
    lines.append(row("⑤ 延迟", ma["latency_s"]["mean"], mb["latency_s"]["mean"], "lower", "sec"))
    lines.append("")
    lines.append("## 口径声明")
    lines.append("")
    lines.append(f"> {b.get('caveat', a.get('caveat', ''))}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Attest 评测报告生成与版本对比")
    ap.add_argument("--run", help="指定 run-id / tag / 前缀（默认取最新）")
    ap.add_argument("--compare", nargs=2, metavar=("前", "后"), help="两版并排对比")
    ap.add_argument("--with-baselines", action="store_true", help="附 baseline 对照表")
    ap.add_argument("--out", help="输出路径（默认 eval/reports/report-<ts>.md）")
    args = ap.parse_args(argv)

    runs = _load_runs()
    baselines = _load_baselines() if args.with_baselines else []

    if args.compare:
        md = render_compare(runs, args.compare)
        default_name = f"compare-{args.compare[0]}-vs-{args.compare[1]}.md"
    else:
        run = _pick(runs, args.run) if args.run else runs[-1]
        md = render_single(run, with_baselines=args.with_baselines, baselines=baselines)
        default_name = f"report-{run.get('tag') or run['run_id']}.md"

    out = Path(args.out) if args.out else REPORT_DIR / default_name
    out.write_text(md, encoding="utf-8")
    print(f"已写入：{out}")
    print(f"（{len(md.splitlines())} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
