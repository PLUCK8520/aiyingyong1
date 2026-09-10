"""T0.6 / T3.11 / T4.5 · 冒烟脚本：一条命令验证
「网关能调通 + trace 完整 + 节点成对 + P3 双源与矛盾 + P4 引用审计与降级」。

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

from attest.agents import auditor as auditor_node  # noqa: E402
from attest.agents.analyst import truncate_topk  # noqa: E402
from attest.agents.auditor import _with_fuse_banner  # noqa: E402
from attest.config import load_settings  # noqa: E402
from attest.graph.build import build_context, build_graph, initial_state  # noqa: E402
from attest.graph.state import reducer_fields_ok  # noqa: E402
from attest.llm.prompts import build_direct_messages  # noqa: E402
from attest.llm.router import ModelRouter  # noqa: E402
from attest.logging import setup_logging  # noqa: E402
from attest.quality.audit import degrade_report, replace_section, rewrite_targets  # noqa: E402
from attest.quality.citation_auditor import RuleCitationAuditor  # noqa: E402
from attest.schemas import AuditItem  # noqa: E402
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
    for node in ("intent_router", "planner", "scout_web", "scout_local", "evidence_judge", "reflect", "analyst", "auditor"):
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

    # 3c) P4 切片：引用审计（T4.1）+ 降级动作（T4.2）
    audit_summary = research.get("audit_summary") or {}
    check("[P4] 审计结果已入 state", bool(audit_summary), f"summary={audit_summary.get('auditor')}")
    check("[P4] 审计三档字段齐全",
          all(k in audit_summary for k in ("supported", "partial", "unsupported")),
          str({k: audit_summary.get(k) for k in ("supported", "partial", "unsupported")}))
    check("[P4] 审计逐句判定非空", int(audit_summary.get("total", 0)) > 0,
          f"total={audit_summary.get('total')}")

    # 故意注入假引用（数值 9999 亿元不在证据里）→ auditor 必须识别并降级
    if evidence:
        cid = evidence[0].citation_id
        fake = f"# 报告\n\n## 核心摘要\n\n本市场规模达 9999 亿元 {cid}。"
        fake_items = RuleCitationAuditor().audit(fake, evidence)
        check("[P4] 假引用被识别为 unsupported",
              any(i.verdict == "unsupported" for i in fake_items),
              str([(i.citation_id, i.verdict) for i in fake_items]))
        degraded, info = degrade_report(fake, fake_items)
        check("[P4] unsupported 触发降级（去编号 + 「未证实」）",
              "（未证实）" in degraded and cid not in degraded,
              f"degraded={info.get('degraded')} removed={info.get('removed_citations')}")

    # 3d) P4 切片（T4.4）：模型路由器策略表 + 熔断降级
    router = ModelRouter(settings)
    l1_direct = router.route("direct", fuse_level=1)
    check("[P4] 路由器：轻任务在 L1 切到便宜档",
          l1_direct == settings.model_fuse_light,
          f"direct L0={router.route('direct', fuse_level=0)} → L1={l1_direct}")
    check("[P4] 路由器：主链路不因熔断降级",
          router.route("analyst", fuse_level=0) == router.route("analyst", fuse_level=2),
          f"analyst={router.route('analyst', fuse_level=2)}")
    check("[P4] 路由器：可导出策略表 + 档位标签",
          router.tier("planner") == "strong" and router.tier("judge") == "fast"
          and len(router.table()) >= 8,
          f"planner={router.tier('planner')} judge={router.tier('judge')} entries={len(router.table())}")

    # 3e) P4 切片（T4.3）：熔断 L2 的上下文截断 + 报告头降级提示
    sample_ev = evidence[:4]
    truncated = truncate_topk(list(sample_ev), [], 2)
    check("[P4] 熔断 L2：上下文截断（证据按相关性取 top-k）",
          len(truncated) == min(2, len(sample_ev)),
          f"{len(sample_ev)} 条 → top-2 = {len(truncated)} 条")
    banner = _with_fuse_banner("# 报告\n\n正文", 2)
    check("[P4] 熔断 L2：报告头加降级提示",
          "降级提示" in banner and "正文" in banner and banner.count("降级提示") == 1)
    check("[P4] 熔断 L0/L1：不加降级提示",
          "降级提示" not in _with_fuse_banner("# 报告\n\n正文", 1))

    # 3f) P4 切片（T4.2b）：章节重写的判据与定位（纯函数）
    demo_items = [
        AuditItem(citation_id="[WEB1-1-1]", verdict="unsupported", sentence="s1", section="市场规模"),
        AuditItem(citation_id="[WEB1-1-1]", verdict="supported", sentence="s2", section="竞品"),
    ]
    targets = rewrite_targets(demo_items, 0.5)
    check("[P4] T4.2b：失败率超阈值的章节被标为待重写",
          targets == {"市场规模": 1.0}, str(targets))
    demo_report = "# 报告\n\n## 市场规模\n\n旧正文\n\n## 竞品\n\n竞品正文\n"
    replaced = replace_section(demo_report, "市场规模", "新正文")
    check("[P4] T4.2b：章节替换只动目标节",
          "新正文" in replaced and "旧正文" not in replaced and "竞品正文" in replaced,
          replaced.replace("\n", "⏎"))

    # 3g) P4 切片（T4.2b）：重写端到端——注入两处假数值，整节失败率 100% → 触发重写 → 复检收口
    if evidence:
        cid0 = evidence[0].citation_id
        bad_report = (
            "# 报告\n\n## 市场规模\n\n"
            f"- 本市场规模达 9999 亿元 {cid0}。\n"
            f"- 另一口径为 8888 亿元 {cid0}。\n"
        )
        out = auditor_node.run(
            {"report": bad_report, "evidence": evidence, "plan": {"objective": "冒烟 T4.2b"}},
            ctx,
        )
        rw = int((out.get("audit_summary") or {}).get("rewrites", 0))
        check("[P4] T4.2b：失败率超阈值触发章节重写",
              rw >= 1, str((out.get("audit_summary") or {}).get("rewrite_history")))
        check("[P4] T4.2b：重写后复检，该节不再有「未证实」",
              "（未证实）" not in (out.get("report") or ""),
              f"degraded={(out.get('audit_summary') or {}).get('degraded_sentences')}")

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
