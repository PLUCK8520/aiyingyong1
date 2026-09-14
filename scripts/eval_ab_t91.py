# -*- coding: utf-8 -*-
"""T9.1 A/B 对照评测：结构化输出协议（新）vs 旧协议（P8 行为），真实档统计样本。

为什么需要它：T9.1 的单次真实档验证（thread `t91-real-1`）证明 glm-4-flash
**能服从** <<CHAPTER>> 格式，但单次 ≠ 统计显著。本脚本用真实模型各跑 6 轮，
量化对比"新协议 vs 旧协议"在引用编号生成上的差异，产出 Markdown 报告。

用法：
    .venv\\Scripts\\python.exe scripts\\eval_ab_t91.py            # 全量：2 档 × 6 轮
    .venv\\Scripts\\python.exe scripts\\eval_ab_t91.py --smoke    # 冒烟：2 档 × 1 轮（先验脚本本身）

约束（零造假铁律）：
  - 指标只从 chat.py 的真实 stdout / 落盘报告里解析，**不编造、不估算**；
  - 任何一轮失败（超时 / 网关耗尽 / 429 / 解析不到）都如实记 failed，不丢弃数据；
  - 报告里标注"实测"与"样本局限"。

环境前提：
  - LLM 走 .env 的 openai_compat（智谱 glm-4-flash 免费档，¥0）；
  - 检索走 mock（离线 fixture 只有 4 个主题，问题从这 4 个里轮换——问别的会被
    拒编闸门拦下，那是正确行为，不是本脚本要测的）。
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENV = REPO / ".venv" / "Scripts" / "python.exe"
CHAT = REPO / "scripts" / "chat.py"
OUT_DIR = REPO / "docs" / "eval-reports"

#: 离线 fixture 覆盖的 4 个主题（与 retrieval/mock_search.py 的语料一致）。
TOPICS = [
    "调研企业知识库 Agent 平台的市场规模与竞争格局",
    "调研大模型推理成本的构成与优化趋势",
    "调研国产数据库替换的驱动因素与落地情况",
    "调研新能源汽车出海的竞争格局与政策环境",
]

#: (档位名, ATTEST_ANALYST_STRUCTURED 值)
MODES = [("new", "1"), ("legacy", "0")]

_RE_REF = re.compile(r"引用[:：]\s*有来源\s*(\d+)\s*处\s*/\s*未解析\s*(\d+)\s*处[^|]*\|?\s*校验\s*(通过|未通过)")
_RE_AUDIT = re.compile(r"审计[:：]\s*判定\s*(\d+)\s*句\s*\|\s*支持\s*(\d+)\s*/\s*部分\s*(\d+)\s*/\s*未证实\s*(\d+)\s*\|\s*降级\s*(\d+)\s*句")
_RE_TIME = re.compile(r"耗时\s*([\d.]+)s")
_RE_CALLS = re.compile(r"LLM 调用\s*(\d+)\s*次")
_RE_SAVED = re.compile(r"报告已保存[:：]\s*(.+)")
_RE_SUMMARY = re.compile(r"报告[:：]\s*(\d+)\s*字")


def _int(m: re.Match | None, g: int) -> int | None:
    return int(m.group(g)) if m else None


def run_one(mode: str, struct_val: str, round_i: int, topic: str, timeout: int) -> dict:
    thread = f"ab-t91-{mode}-{round_i}"
    env = dict(os.environ)
    env["ATTEST_ANALYST_STRUCTURED"] = struct_val
    t0 = datetime.datetime.now()
    try:
        r = subprocess.run(
            [str(VENV), str(CHAT), "--thread", thread, topic],
            cwd=str(REPO), env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"mode": mode, "round": round_i, "topic": topic, "failed": True, "reason": "超时"}
    except Exception as e:  # noqa: BLE001
        return {"mode": mode, "round": round_i, "topic": topic, "failed": True, "reason": f"{type(e).__name__}: {e}"}
    dur = (datetime.datetime.now() - t0).total_seconds()
    out = (r.stdout or "") + "\n" + (r.stderr or "")

    saved = _RE_SAVED.search(out)
    report_path = saved.group(1).strip() if saved else ""
    report_text = ""
    if report_path and Path(report_path).exists():
        report_text = Path(report_path).read_text(encoding="utf-8", errors="replace")

    ref = _RE_REF.search(out)
    audit = _RE_AUDIT.search(out)
    return {
        "mode": mode, "round": round_i, "topic": topic,
        "failed": r.returncode != 0,
        "rc": r.returncode,
        "dur_s": round(dur, 1),
        "calls": _int(_RE_CALLS.search(out), 1),
        "referenced": _int(ref, 1),
        "unresolved": _int(ref, 2),
        "check_pass": (ref.group(3) == "通过") if ref else None,
        "audit_total": _int(audit, 1),
        "supported": _int(audit, 2),
        "partial": _int(audit, 3),
        "unsupported": _int(audit, 4),
        "degraded": _int(audit, 5),
        "has_unchecked_banner": "未通过逐句回查" in report_text,
        "basis_line_count": report_text.count("本节论述依据："),
        "report_chars": len(report_text),
        "report_path": report_path,
    }


def _avg(rows: list[dict], key: str) -> float | None:
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="每档只跑 1 轮，用于先验证脚本本身")
    args = ap.parse_args()

    rounds = 1 if args.smoke else 6
    timeout = 900
    rows: list[dict] = []
    for mode, val in MODES:
        for i in range(rounds):
            topic = TOPICS[i % len(TOPICS)]
            row = run_one(mode, val, i, topic, timeout)
            rows.append(row)
            status = "FAIL " if row["failed"] else "ok   "
            print(f"[{status}] {mode}/round{i}  ref={row.get('referenced')}  "
                  f"audit={row.get('audit_total')}  {row.get('dur_s')}s  {topic[:16]}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    out_path = OUT_DIR / f"ab-t91-{stamp}.md"
    _write_report(rows, rounds, out_path)
    print(f"报告已写：{out_path}")
    return 0


def _write_report(rows: list[dict], rounds: int, out_path: Path) -> None:
    new = [r for r in rows if r["mode"] == "new"]
    legacy = [r for r in rows if r["mode"] == "legacy"]
    new_ok = [r for r in new if not r["failed"]]
    legacy_ok = [r for r in legacy if not r["failed"]]

    def md_rows(rs: list[dict]) -> str:
        lines = []
        for r in rs:
            if r["failed"]:
                lines.append(f"| {r['round']} | ❌ {r.get('reason')} | — | — | — | — | — |")
                continue
            lines.append(
                f"| {r['round']} | {r['referenced']} | {r['unresolved']} | "
                f"{'✅' if r['check_pass'] else '❌'} | {r['audit_total']} | "
                f"{r['supported']}/{r['partial']}/{r['unsupported']} | {r['degraded']} |"
            )
        return "\n".join(lines)

    def avg_ok(rs: list[dict]) -> str:
        if not rs:
            return "（无成功轮）"
        return (
            f"引用 {_avg(rs, 'referenced')} 处 / 未解析 {_avg(rs, 'unresolved')} / "
            f"审计 {_avg(rs, 'audit_total')} 句 / 未证实 {_avg(rs, 'unsupported')} / "
            f"降级 {_avg(rs, 'degraded')} 句 / 依据行 {_avg(rs, 'basis_line_count')} 行"
        )

    n_new_banner = sum(1 for r in new_ok if r["has_unchecked_banner"])
    n_legacy_banner = sum(1 for r in legacy_ok if r["has_unchecked_banner"])

    body = f"""# T9.1 A/B 对照评测报告（{datetime.date.today()}）

> 脚本：`scripts/eval_ab_t91.py` · 档位：`ATTEST_ANALYST_STRUCTURED=1`（新协议）vs `=0`（旧协议，P8 行为）
> 模型：智谱 glm-4-flash 免费档（¥0）· 检索：mock（离线 fixture 4 主题轮换）
> 每档 **{rounds} 轮**。**零造假**：数字全部来自真实运行，失败轮如实列出。

## 结论速览

| 指标 | 新协议（structured=1） | 旧协议（structured=0） |
|---|---|---|
| 成功轮数 | {len(new_ok)}/{len(new)} | {len(legacy_ok)}/{len(legacy)} |
| 均值 | {avg_ok(new_ok)} | {avg_ok(legacy_ok)} |
| 出现「未通过逐句回查」横幅的轮数 | {n_new_banner} | {n_legacy_banner} |

## 新协议（structured=1）逐轮

| 轮 | 引用 | 未解析 | 校验 | 审计句数 | 支持/部分/未证实 | 降级 |
|---|---|---|---|---|---|---|
{md_rows(new)}

## 旧协议（structured=0）逐轮

| 轮 | 引用 | 未解析 | 校验 | 审计句数 | 支持/部分/未证实 | 降级 |
|---|---|---|---|---|---|---|
{md_rows(legacy)}

## 样本与局限（如实声明）

- 样本仅 **2 × {rounds} 轮**，主题限 fixture 覆盖的 4 个，检索为 mock（非真实 Tavily）——
  结论是"离线语料 + 免费档"下的对照，**不能外推到真实检索或更强模型**。
- 免费档有随机性；同一档内轮间波动属正常。
- 失败轮（超时/网关耗尽）**不代表协议优劣**，是环境约束，单独列出不参与均值。

## 原始行数据（JSON，供复核）

```json
{_json(rows)}
```
"""
    out_path.write_text(body, encoding="utf-8")


def _json(rows: list[dict]) -> str:
    import json
    return json.dumps(rows, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
