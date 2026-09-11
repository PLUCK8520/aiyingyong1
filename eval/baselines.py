"""T7.2 · baseline 对照组：`python -m eval.baselines`。

**为什么要有 baseline**：只报"我们的 token 均值是 16,594"，读者无法判断这个数
是高还是低。对照组给出**同一批题、同一批语料**下三条更弱的路线作参照，数字才有意义。

三条 baseline（**刻意做弱**，不是"竞品"）：
  B1 裸 LLM 直答    —— 不检索、不引用、不审计。代表"直接问模型"的下限。
  B2 单轮 RAG       —— 检索一轮 → 一次综合成文。代表"最朴素的 RAG"，无 planner /
                       无 Reflect 补检 / 无引用审计 / 无矛盾检测。
  B3 RAG 无审计     —— 与 B2 同检索，但**明确跳过引用审计**。
                       用来把"审计层"的贡献单独隔出来（B2 与 B3 的差 = 审计层的净效果）。

**实现纪律**：
  - 只复用现有 `build_context` / 网关 / 检索接口，**不碰主图、不改 src/**——
    baseline 是"外挂的对照组"，不是系统的第二种实现路径。改主图来"优化"baseline 对比
    就作弊了。
  - 输出与主系统**同构**的 row（沿用 `eval.run.run_case` 的字段名），便于并排。
    没有的字段如实置 0 / False，不估算、不编造。

⚠️ **离线 mock 下的诚实声明**：三条 baseline 用的都是同一个 mock LLM 与同一批 fixture，
所以正文质量差异**不是真实模型能力的差异**——它们证明的是"管道接通了、指标口径对得齐"，
**不能**用来对外声称"我们的系统比裸 GPT 好 X%"。
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

from attest.graph.build import build_context  # noqa: E402
from attest.llm.prompts import build_analyst_messages, build_planner_messages  # noqa: E402
from attest.logging import setup_logging  # noqa: E402
from attest.quality.citation_check import finalize  # noqa: E402
from attest.retrieval.citations import CitationIndex  # noqa: E402
from attest.retrieval.local_search import clear_local_cache  # noqa: E402
from attest.schemas import Plan  # noqa: E402

from eval.metrics import caveat, outline_coverage, summarize  # noqa: E402
from eval.run import DATASET  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parent / "reports"

#: 三条 baseline 的标识与说明（写进结果 JSON，报告页脚也用它）
BASELINES: dict[str, str] = {
    "b1_bare": "裸 LLM 直答（不检索、不引用、不审计、无章节结构）",
    "b2_rag": "单轮 RAG（检索一轮 → 一次成文；不审计）",
    "b3_rag_audit": "RAG + 引用审计（检索一轮 → 成文 → 跑真审计器；无 planner 扇出/无补检/无矛盾检测）",
}


def _empty_row(cid: str, case: dict, baseline: str, *, query: str) -> dict:
    """与主系统同构的空 row。没有的字段如实置 0/False，不估算。"""
    return {
        "id": cid,
        "baseline": baseline,
        "kind": case.get("kind", "?"),
        "topic": case.get("topic", ""),
        "query": query,
        "route": "research",
        "tokens": 0,
        "cost_cny": 0.0,
        "llm_calls": 0,
        "referenced": 0,
        "unresolved": 0,
        "cited_ratio": 0.0,
        "citation_pass": False,
        "outline_coverage": 0.0,
        "outlines_total": 0,
        "outlines_missing": [],
        "sections_actual": [],
        "has_report": False,
        "duration_s": 0.0,
        "evidence": 0,
        "conflicts": 0,
        "gaps": 0,
        "rounds": 1,
    }


def _plan_for(ctx, query: str) -> Plan:
    """借 planner 的**一次**调用拿大纲，让 baseline 与主系统用同一套大纲。

    这一步是刻意的：若 baseline 用自己编的大纲，"大纲覆盖度"就没法比。
    借大纲只调一次 planner，不跑扇出检索——baseline 的检索保持"单轮朴素"。
    """
    resp = ctx.gateway.chat(
        build_planner_messages(query),
        task="planner",
        response_model=Plan,
        temperature=0.0,
    )
    return resp.parsed  # type: ignore[return-value]


def _synthesize(ctx, objective: str, outlines: list[str], evidence: list, *, audit: bool):
    """一次成文。`audit=False` 时**不做**引用审计——这是 B3 与 B2 的唯一差别。

    返回 (报告正文, 累计 usage)。usage 从 trace 汇总取，与主系统同口径。
    """
    resp = ctx.gateway.chat(
        build_analyst_messages(objective, outlines, evidence),
        task="analyst",
        temperature=0.2,
    )
    text = resp.text or ""
    if audit:
        # B2/B3 都**不**审计（留 audit 参数只为让"将来要开审计"时有个明确开关）。
        # 这里刻意不调用 auditor：baseline 的意义就是"没有审计层"。
        pass
    return text


def run_baseline(case: dict, *, baseline: str, out_root: Path) -> dict:
    """跑一条 baseline 的一道题。

    三条路线的差别（刻意只差一层，便于归因）：
      B1 裸答      ：1 次 direct 调用，无检索、无大纲、无章节结构
      B2 单轮 RAG  ：借大纲 → 检索一轮 → 1 次成文（不审计）
      B3 RAG+审计  ：同 B2，但成文后**跑真引用审计器**并 finalize
                     （B3 − B2 = 审计层的净效果；这是"对照组"能给出因果的唯一边界）
    """
    cid = case["id"]
    query = case["query"]
    settings = Settings_for(out_root)(cid, baseline)
    ctx = build_context(settings, run_id=f"{baseline}-{cid}")
    row = _empty_row(cid, case, baseline, query=query)

    try:
        t0 = time.perf_counter()

        if baseline == "b1_bare":
            # 裸直答：不检索、不用大纲。这一条连 planner 都不该调。
            resp = ctx.gateway.chat(
                [{"role": "user", "content": query}], task="direct", temperature=0.2
            )
            text = resp.text or ""
            row["has_report"] = bool(text.strip())
            row["sections_actual"] = _h2(text)
            # ⚠️ 覆盖度如实置 0：裸答**不产出任何 `##` 章节结构**，
            # 与主系统的 100% 不可比，不能因为"无大纲"就记满分（那会把 baseline 伪装成同等水平）。
            row["outline_coverage"] = 0.0
            row["outlines_total"] = 0
            row["outlines_missing"] = []
            row["coverage_note"] = "裸答不产出章节结构，覆盖度不适用（记 0，不参与均值）"
            # 裸答没有引用编号，"引用有效率"无意义 → 如实置 0，并标注不适用
            row["cited_ratio_note"] = "裸答无引用编号，引用有效率不适用"
        else:
            plan = _plan_for(ctx, query)
            objective = plan.objective or query
            outlines = list(plan.outlines or [])
            row["outlines_total"] = len(outlines)

            # 单轮检索：直接对 objective 检索，不做 planner 扇出、不做补检
            results = ctx.search.search(objective, max_results=8)
            evidence = _to_evidence(results)
            row["evidence"] = len(evidence)
            index = _index_of(evidence)

            resp = ctx.gateway.chat(
                build_analyst_messages(objective, outlines, evidence),
                task="analyst",
                temperature=0.2,
            )
            draft = resp.text or ""
            row["has_report"] = bool(draft.strip())

            if baseline == "b3_rag_audit":
                # **关键**：跑真审计器（与主系统同一个 `ctx.auditor`），
                # 再走 `finalize` 补参考资料 —— 这样 cited_ratio / unresolved 才与主系统同口径。
                items = ctx.auditor.audit(draft, evidence, objective=objective) if ctx.auditor else []
                row["audited_claims"] = len(items)
                row["unsupported"] = sum(1 for it in items if it.verdict == "unsupported")
                final, _refs, check = finalize(draft, index)
                row["referenced"] = int(check["referenced"])
                row["unresolved"] = len(check["unresolved"])
                row["cited_ratio"] = float(check["cited_ratio"])
                row["citation_pass"] = bool(check["pass"])
                text_for_coverage = final
            else:
                # B2 不审计：编号出现情况照实记，但**无从判断**是否有来源。
                row["referenced"] = len(_ids(draft))
                row["unresolved"] = 0
                row["cited_ratio"] = 0.0
                row["citation_pass"] = False
                row["cited_ratio_note"] = "未跑审计 → 引用有效率不适用（主系统的口径需要审计层）"
                text_for_coverage = draft

            cov, missing = outline_coverage(text_for_coverage, outlines)
            row["outline_coverage"] = round(cov, 4)
            row["outlines_missing"] = missing
            row["sections_actual"] = _h2(text_for_coverage)

        row["duration_s"] = round(time.perf_counter() - t0, 3)
        s = ctx.trace.summary()
        row["llm_calls"] = s["llm_calls"]
        row["tokens"] = int(s.get("input_tokens", 0) + s.get("output_tokens", 0))
        row["cost_cny"] = float(s.get("cost_cny", 0.0) or 0.0)
    finally:
        ctx.trace.close()
    return row


# ------------------------------------------------------------- 小工具（避免 import 主图私有函数）

def _to_evidence(results):
    """把 SearchResult 转成 Evidence（复用主系统同一套编号语义）。

    这里**不**调 `assign_citation_ids`——那会引入扇出分支的编号空间，
    而 baseline 没有扇出。用最简单的全局递增编号即可，够渲染证据块。
    """
    from attest.retrieval.citations import assign_citation_ids

    return assign_citation_ids(results, round_no=1, subq_no=1, sub_question="单轮检索")


def _h2(text: str) -> list[str]:
    import re

    return [m.strip() for m in re.findall(r"^##\s+(.+?)\s*$", text or "", re.MULTILINE)]


def _ids(text: str) -> list[str]:
    from attest.retrieval.citations import extract_ids

    return extract_ids(text or "")


def _index_of(evidence) -> CitationIndex:
    """从证据列表建引用索引 —— 与主系统 `finalize` 用的是同一个类。"""
    return CitationIndex.from_evidence(evidence)


def Settings_for(out_root: Path):
    """把 Settings 的构造包一层，避免在模块顶层 import（保持与 run.py 一致的延迟导入）。"""

    def _make(cid: str, baseline: str):
        from attest.config import Settings

        s = Settings(
            llm_mode="mock",
            search_mode="mock",
            log_level="WARNING",
            trace_dir=out_root / "trace" / baseline / cid,
            report_dir=out_root / "reports" / baseline / cid,
        )
        s.ensure_dirs()
        return s

    return _make


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Attest baseline 对照组")
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--kind", help="只跑某一类（fact/compare/dispute/local）")
    ap.add_argument("--limit", type=int, help="只跑前 N 题")
    ap.add_argument("--baseline", help="只跑某条 baseline（b1_bare/b2_rag/b3_rag_audit）")
    args = ap.parse_args(argv)

    setup_logging("WARNING")
    data = yaml.safe_load(Path(args.dataset).read_text(encoding="utf-8"))
    cases = [c for c in (data.get("cases") or []) if c.get("expect_route") == "research"]
    if args.kind:
        cases = [c for c in cases if c.get("kind") == args.kind]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("没有可跑的 baseline 用例（只跑 research 题）。")
        return 1

    which = [args.baseline] if args.baseline else list(BASELINES)
    for b in which:
        if b not in BASELINES:
            print(f"未知 baseline：{b}（可选 {list(BASELINES)}）")
            return 1

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = REPORT_DIR / f"baselines-{run_id}"
    out_root.mkdir(parents=True, exist_ok=True)
    clear_local_cache()

    print("=" * 96)
    print(f"Attest baseline 对照组 · run={run_id} · {len(cases)} 题 × {len(which)} 条")
    print("=" * 96)

    all_rows: dict[str, list[dict]] = {}
    for b in which:
        rows: list[dict] = []
        print(f"\n[{b}] {BASELINES[b]}")
        print(f"{'id':<5}{'kind':<9}{'token':>7}{'引用率':>8}{'覆盖度':>8}{'耗时':>7}")
        print("-" * 60)
        for case in cases:
            try:
                row = run_baseline(case, baseline=b, out_root=out_root)
            except Exception as exc:  # noqa: BLE001
                print(f"{case['id']:<5}!!! {type(exc).__name__}: {exc}")
                continue
            rows.append(row)
            ratio = "  N/A" if row["cited_ratio"] == 0.0 else f"{row['cited_ratio']:>5.0%}"
            print(
                f"{row['id']:<5}{row['kind']:<9}{row['tokens']:>7,}"
                f"{ratio:>7}{row['outline_coverage']:>8.0%}"
                f"{row['duration_s']:>6.2f}s"
            )
        all_rows[b] = rows

    print("\n" + "=" * 96)
    print("对照（research 题）")
    print("=" * 96)
    # 主系统基线（同批题、同口径；由 eval.run 产出，这里是**引用**不是重算）
    print(f"{'路线':<18}{'token':>9}{'引用率':>8}{'覆盖度':>8}{'完成率':>8}{'延迟':>8}")
    print("-" * 96)
    summary: dict[str, dict] = {}
    for b, rows in all_rows.items():
        m = summarize(rows)
        # ⚠️ 指标"是否适用"要按**每个指标**单独判，不能按整条 baseline 一刀切：
        #   · B1 裸答：不产出 `##` 章节 → 覆盖度 N/A；无编号 → 引用率 N/A
        #   · B2 不审计：有章节 → 覆盖度**适用**；但没跑审计器 → **引用率 N/A**
        #     （把 0.0% 当"引用率差"是把"没测"读成"测出来很差"，是错的）
        #   · B3/B1 之外都有审计 → 引用率适用
        cov_applicable = b != "b1_bare"
        cr_applicable = b in ("b3_rag_audit",)
        m["coverage_applicable"] = cov_applicable
        m["cited_ratio_applicable"] = cr_applicable
        summary[b] = m
        cov = f"{m['outline_coverage']['mean']:>7.1%}" if cov_applicable else f"{'N/A':>8}"
        cr = f"{m['cited_ratio']['mean']:>6.1%}" if cr_applicable else f"{'N/A':>7}"
        print(
            f"{b:<18}{m['tokens_per_report']['mean']:>9,.0f}{cr}{cov}"
            f"{m['completion_rate']:>8.1%}{m['latency_s']['mean']:>7.2f}s"
        )
    print(
        "（主系统同批题：token 均值 16,594 / 引用率 42.6% / 覆盖度 100% / 完成率 100%，"
        "见 eval/reports/run-*-p7-final/result.json）"
    )

    payload = {
        "run_id": run_id,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "baselines": BASELINES,
        "cases": len(cases),
        "metrics": summary,
        "rows": all_rows,
        "caveat": caveat(),
        "honesty_note": (
            "三条 baseline 与主系统共用同一个 mock LLM 与同一批 fixture，"
            "所以**质量差异不反映真实模型能力**。本文件证明的是"
            "「对照管道接通、指标口径对得齐」，不代表任何对外可宣称的优劣。"
        ),
    }
    (out_root / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结果已写入：{out_root / 'result.json'}")
    print(f"⚠️ {caveat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
