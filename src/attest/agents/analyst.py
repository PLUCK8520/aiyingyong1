"""T2.5 · Analyst：按大纲分章撰写，内联引用编号；T2.6 后处理在同节点内完成。

一个刻意的设计：**低相关证据先被筛掉再交给撰稿**。
判别节点给出的 relevance ≤ 2 的证据不进正文——这让"证据判别"这一步是**有后果的**，
而不是流水线上的装饰。

⚠️ **T7.10：筛完为空时「拒编」，不再退回全量**（这是本项目最严重的一处质量缺陷修复）。
旧行为是"筛选后为空 → 退回全量并留痕"——本意是"链路别断"，实际后果是把判别器
「这些证据都不相关」的结论**丢掉**，再拿不相关素材硬凑一份看起来完整的报告。
真实运行实测（2026-09-13，glm-4-flash）：问「2026 年大学生就业情况」，
离线语料里根本没有就业类资料，判别器把全部 66 条判为 relevance ≤ 2（**判对了**），
但 analyst 退回全量，于是报告里全是"企业知识库/数据库市场"的规模与增速——
**数字是真的，主题是假的**，恰好踩中项目的「零造假」铁律。
现在的口径：`relevance ≥ MIN_RELEVANCE` 的一条都没有 ⇒ 证据不足 ⇒ **不出结论**，
产出一页如实说明"检索与判别情况 + 为什么不出结论 + 怎么办"的**拒编页**
（确定性生成，**不经过 LLM**——把"拒编"这件事交给模型去说，本身就是一次造假风险）。
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from ..budget.account import FUSE_HARD
from ..llm.prompts import build_analyst_messages
from ..logging import get_logger
from ..quality.citation_check import finalize
from ..retrieval.citations import CitationIndex, extract_ids, normalize_citation_id
from ..retrieval.ports import Evidence
from .base import NodeContext, accumulate, budget_after, report_budget
from .memory_loader import profile_block

log = get_logger(__name__)
NODE = "analyst"
TAG = "analyst"

MIN_RELEVANCE = 3

#: 拒编页里对"未覆盖子问题"的展示上限（问题很多时不至于刷屏）
_MAX_LISTED_SUBQ = 12

#: T8.6 零引用纠偏：追加在 **user 消息末尾** 的硬约束。
#:
#: 为什么是"挪到末尾"而不是"再强调一遍"：真实运行实测（2026-09-13，glm-4-flash，
#: thread `kb-e2e-2`）—— 同一条铁律写在 `ANALYST_SYSTEM` 里时被整条忽略，输出 905 字
#: 正文**零引用编号**，同时第五节还凭空生成了证据里根本没有的"财政补贴 / 税收优惠 /
#: 社会资本"。受控对照实验（同一份证据、同一温度，三个变体）：
#:     V0 基线（原提示词）      → 0 个编号，且出现凭空内容
#:     V1 末尾追加强制自检句    → **出现 `[LOC1-5-1]`**，无凭空内容
#:     V2 V1 + 显式编号清单     → 0 个编号（加了约束反而退化）
#: 所以：**尾部位置有效，但小模型的服从性不稳定**——只能拿它当"一次有界的纠偏尝试"，
#: 绝不能当成"提示词调好了"。真正的兜底必须是确定性的（见 `auditor` 的未回查横幅）。
CITATION_RETRY_TAIL = (
    "\n\n**输出前自检（硬要求）**：正文里每一句涉及事实 / 数据的结论，句末必须紧跟其证据编号，"
    "形如 `[LOC1-1-1]`。没有编号支撑的句子一律不要写；确需说明缺失就写「（证据不足）」。"
)


def _with_retry_tail(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """把纠偏约束追加到最后一条 user 消息末尾（不新增消息、不改 system，便于对照）。"""
    out = [dict(m) for m in messages]
    if not out:
        return out
    out[-1] = {"role": out[-1].get("role", "user"), "content": out[-1]["content"] + CITATION_RETRY_TAIL}
    return out


def assess_evidence(state: dict[str, Any]) -> tuple[list[Evidence], dict[str, Any]]:
    """成文前的**证据充足性闸门**：返回 (可用证据, 评估详情)。

    返回的 `selected` 就是可以送进正文的证据；**为空即拒编**。判据分两档：

      - 有判别结果：`relevance ≥ MIN_RELEVANCE`。这是主路径，也是真实运行里
        "判别器说不相关、撰稿却照写"那个缺陷的修复点（见模块文档串）。
      - 无判别结果（judge 跳过 / 失败）：退化为 `score > 0`——没有判别信息时，
        至少要求证据在检索里**真实命中过**（零分＝兜底/跨主题填充，不算命中）。

    为什么不用 `score` 当主判据：本地检索是 RRF 融合分，**恒为正**（1/(k+rank)），
    拿它当"相关性"会把任何检索都判成"有证据"。所以判定权交给判别器。

    ⚠️ **已知边界（如实记录，不掩饰）**：判据的有效性取决于判别器。
      - 真实模型（如 glm-4-flash）实测判定可靠：问「大学生就业」时它把全部 66 条判为
        relevance ≤ 2——闸门据此拒编，正确。
      - 但 `llm_mode=mock` 的**离线判别器是关键词启发式**（token 命中即给 3 分以上），
        「市场规模」这类通用词会让跨主题资料蒙混过关。此时若本地笔记/历史结论
        恰好含这些通用词，闸门可能放行。这是离线替身的近似性，不是闸门逻辑的问题——
        真实档与 Tavily 真实检索都不受影响。
    """
    evidence: list[Evidence] = list(state.get("evidence") or [])
    judgments = list(state.get("judgments") or [])
    plan = state.get("plan") or {}
    subqs = [str(s) for s in (plan.get("sub_questions") or []) if str(s).strip()]
    objective = str(plan.get("objective") or "")

    # 同一编号可能被多轮判别，取最高相关度（判别是"最有利证据"语义，不是取平均）。
    # ⚠️ **两侧都要规范化编号**：模型回填 `citation_id` 时常常丢掉方括号（写成 `LOC1-1-1`
    # 而非 `[LOC1-1-1]`），而这里是精确匹配——不收敛的话，判别器给的高分会被挂到一个
    # 不存在的键上，`selected` 恒为空，**一切查询都退化成拒编页**。
    # 详见 `retrieval.citations.normalize_citation_id`（含 2026-09-13 的实测数字）。
    relevance: dict[str, int] = {}
    for j in judgments:
        cid = normalize_citation_id(getattr(j, "citation_id", "") or "")
        if cid:
            relevance[cid] = max(relevance.get(cid, 0), int(getattr(j, "relevance", 0) or 0))

    if judgments:
        selected = [e for e in evidence if relevance.get(e.citation_id, 0) >= MIN_RELEVANCE]
        basis = f"证据判别相关度 ≥ {MIN_RELEVANCE}"
    else:
        selected = [e for e in evidence if float(getattr(e, "score", 0.0) or 0.0) > 0.0]
        basis = "检索真实命中（score > 0；本轮无判别结果）"

    # 第二道：可用证据里必须**至少有一条是真实命中**（score>0）。
    # score=0 的语义是"兜底补齐 / 检索未命中主题的填充"（见 `retrieval/mock_search.py`）——
    # 全靠填充物撑起的报告是没有落地证据的，不该出。
    grounded = [e for e in selected if float(getattr(e, "score", 0.0) or 0.0) > 0.0]

    covered = {e.sub_question for e in selected}
    uncovered = [s for s in subqs if s not in covered]

    reasons: list[str] = []
    if not evidence:
        reasons.append("本轮未检索到任何证据")
    elif not selected:
        reasons.append(
            f"检索到 {len(evidence)} 条原始条目，但没有一条通过可用性判据（{basis}）"
        )
    elif not grounded:
        reasons.append(
            f"可用证据 {len(selected)} 条**全部**是检索兜底/未命中主题的填充，没有一条真实命中"
        )

    info: dict[str, Any] = {
        "sufficient": bool(selected) and bool(grounded),
        "objective": objective,
        "n_evidence": len(evidence),
        "n_selected": len(selected),
        "n_grounded": len(grounded),
        "n_judged": len(judgments),
        "basis": basis,
        "sub_questions": subqs,
        "covered_sub_questions": sorted(covered),
        "uncovered_sub_questions": uncovered,
        "reasons": reasons,
    }
    return selected, info


def build_insufficient_report(info: dict[str, Any], *, search_mode: str = "") -> str:
    """证据不足时的**拒编页**（确定性文本，不调模型）。

    刻意写成"如实交代"而不是"一句抱歉"：用户需要知道系统**查了什么、为什么判它不够、
    下一步怎么才能拿到真报告**——否则只会以为系统坏了。
    """
    objective = info.get("objective") or "本次调研"
    subqs: list[str] = list(info.get("sub_questions") or [])
    covered = set(info.get("covered_sub_questions") or [])
    n_ev = int(info.get("n_evidence") or 0)
    n_sel = int(info.get("n_selected") or 0)

    lines: list[str] = [
        f"# {objective} —— 未能完成（证据不足）",
        "",
        "## 结论",
        "",
        "本轮检索**没有获得能支撑结论的相关证据**。按本系统的「零造假」原则，"
        "没有证据支撑的数字、判断与趋势一律不写——因此本次**不输出调研结论**，"
        "而不是用不相关的资料拼出一份看起来完整、实际张冠李戴的报告。",
        "",
        "## 检索与判别情况（如实记录）",
        "",
        f"- 检索到的原始条目：**{n_ev} 条**",
        f"- 通过可用性判据的条目：**{n_sel} 条**（判据：{info.get('basis') or '—'}）",
        f"- 其中**真实命中**（检索计分 > 0）：**{int(info.get('n_grounded') or 0)} 条**",
        f"- 子问题覆盖率：**{len(covered)}/{len(subqs)}**",
    ]

    if subqs:
        lines += ["", "各子问题的证据情况："]
        for i, sq in enumerate(subqs[:_MAX_LISTED_SUBQ], start=1):
            mark = "✅ 有可用证据" if sq in covered else "❌ 无可用证据"
            lines.append(f"{i}. {sq} —— {mark}")
        if len(subqs) > _MAX_LISTED_SUBQ:
            lines.append(f"（其余 {len(subqs) - _MAX_LISTED_SUBQ} 个子问题略）")

    lines += ["", "## 可能的原因", ""]
    if search_mode == "mock":
        lines += [
            "- 当前检索处于**离线回放模式**（`ATTEST_SEARCH_MODE=mock`）：内置样例语料"
            "只覆盖「企业知识库 Agent 平台」「大模型推理成本」「国产数据库替换」"
            "「新能源汽车出海」四个主题，**超出覆盖范围的问题匹配不到任何相关记录**。",
            "- 这不是检索失败，而是离线语料本来就没有这类资料。",
        ]
    else:
        lines += [
            "- 检索服务可能未返回与问题相关的资料，或问题措辞与可检索资料差异过大。",
            "- 该主题可能缺少公开信源（例如企业内部数据、非公开调研）。",
        ]
    lines += [
        "",
        "## 建议的下一步",
        "",
    ]
    if search_mode == "mock":
        lines += [
            "1. **接入真实检索（根本解法）**：配置 `TAVILY_API_KEY` 并把 `ATTEST_SEARCH_MODE` "
            "设为 `tavily` 后重跑——真实检索能覆盖任意主题。",
            "2. 或改用离线语料已覆盖的主题提问（见上方四个主题）。",
            "3. 或把问题拆得更具体、更接近公开行业报告的表述。",
        ]
    else:
        lines += [
            "1. 调整问题的措辞与范围，使其更贴近公开可查的资料表述。",
            "2. 补充本地资料库（把相关文档放进 `data/fixtures/local/`）后重跑。",
            "3. 若问题本身依赖非公开数据，公开信源检索无能为力——这是问题的固有边界。",
        ]
    lines += [
        "",
        "> 说明：本页**不是故障**，而是系统在证据不足时主动拒编。"
        "它保证你看到的每一份报告都有可回查的来源支撑。",
    ]
    return "\n".join(lines)


def truncate_topk(
    evidence: list[Evidence], judgments: list[Any], k: int
) -> list[Evidence]:
    """T4.3 / 熔断 L2 · 上下文截断：按相关性降序取前 k 条（内部保持原有相对顺序）。

    为什么是"按相关性"而不是"按顺序砍尾巴"：证据经 reducer 累加，顺序≈检索先后，
    与重要度无关；reflect 补检来的证据排在后面，砍尾会把补检成果整段丢掉。
    无判别信息（judgments 为空）时相关性一律记 0，退化为保留前 k 条——**已留痕**。
    """
    if k <= 0 or len(evidence) <= k:
        return list(evidence)
    # 同 `assess_evidence`：编号两侧都要规范化，否则模型丢掉方括号后相关性全成 0，
    # 截断会退化成"砍尾巴"——恰好丢掉最该留的那几条。
    relevance = {
        normalize_citation_id(j.citation_id): int(getattr(j, "relevance", 0) or 0)
        for j in judgments
    }
    ranked = sorted(range(len(evidence)), key=lambda i: (-relevance.get(evidence[i].citation_id, 0), i))
    return [evidence[i] for i in sorted(ranked[:k])]


def run(state: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    plan = state.get("plan") or {}
    objective = plan.get("objective", "调研报告")
    outlines: list[str] = plan.get("outlines") or []
    evidence = list(state.get("evidence") or [])
    conflicts = list(state.get("conflicts") or [])
    fuse = ctx.budget_snapshot(state).fuse_level

    selected, suff = assess_evidence(state)

    # ---- T7.10 证据充足性闸门：没有可用证据 ⇒ 拒编（确定性产物，不经过 LLM）----
    if not suff["sufficient"]:
        report = build_insufficient_report(suff, search_mode=str(ctx.settings.search_mode))
        ctx.trace.emit(
            "evidence_insufficient",
            node=NODE,
            n_evidence=suff["n_evidence"],
            n_selected=suff["n_selected"],
            n_judged=suff["n_judged"],
            reasons=suff["reasons"],
            uncovered=suff["uncovered_sub_questions"],
        )
        log.warning(
            f"[analyst] 证据不足，拒编 | 原始 {suff['n_evidence']} 条 / 可用 {suff['n_selected']} 条 | "
            f"原因：{'；'.join(suff['reasons']) or '未覆盖任何子问题'}"
        )
        log.node(TAG, NODE, "完成", chars=len(report), refused=True,
                 n_evidence=suff["n_evidence"], n_selected=0)
        return {
            "report": report,
            "reference_list": "",
            # 拒编页不含任何引用编号 → 引用校验必然通过；`insufficient` 供 auditor 跳过审计
            "citation_check": {
                "pass": True,
                "referenced": 0,
                "unresolved": [],
                "unused": [],
                "evidence_total": suff["n_evidence"],
                "unique_sources": 0,
                "duplicate_sources": {},
                "cited_paragraphs": 0,
                "paragraphs": 0,
                "cited_ratio": 0.0,
                "no_citations": False,
                "insufficient": True,
            },
            "evidence_sufficiency": suff,
            "model_tier": "refused",
        }

    # T4.3 / L2：预算硬熔断 → 上下文截断（证据按相关性取 top-k）。**不改证据本身**，
    # 只影响"这一次写正文时送进上下文多少"；引用索引仍基于全量证据（回查能力不受影响）。
    trunc_info: dict[str, Any] = {}
    if fuse >= FUSE_HARD and len(selected) > ctx.settings.fuse_ctx_top_k:
        before = len(selected)
        selected = truncate_topk(selected, list(state.get("judgments") or []), ctx.settings.fuse_ctx_top_k)
        trunc_info = {"truncated": True, "before": before, "after": len(selected)}
        ctx.trace.emit(
            "context_truncated",
            node=NODE,
            fuse_level=fuse,
            top_k=ctx.settings.fuse_ctx_top_k,
            **trunc_info,
        )

    ctx.trace.emit(
        "evidence_selected",
        node=NODE,
        selected=len(selected),
        dropped=suff["n_evidence"] - suff["n_selected"],
        conflicts=len(conflicts),
        fuse_level=fuse,
        **trunc_info,
    )
    log.node(
        TAG,
        NODE,
        "开始",
        evidence=len(selected),
        outlines=len(outlines),
        conflicts=len(conflicts),
        fuse_level=fuse,
        **trunc_info,
    )

    messages = build_analyst_messages(
        objective, outlines, selected, conflicts, profile_block=profile_block(state)
    )
    resp = ctx.gateway.chat(messages, task="analyst", temperature=0.3, fuse_level=fuse)

    # 索引基于**全量证据**建：筛选只影响"写什么"，不影响"能不能回查"
    index = CitationIndex.from_evidence(evidence)

    # ---- T8.6 零引用纠偏：一次、有界、可留痕 ----
    # 触发条件必须同时满足两条：① 有可用证据（否则"没有引用"是正常的）；
    # ② 首轮正文一个编号都没有。第二次仍为零引用就**不再重试**——继续烧调用换不来服从度，
    # 该做的是把"这份报告没通过回查"如实标出来（auditor 的未回查横幅 + citation_check.no_citations）。
    retried = False
    if selected and not extract_ids(resp.text):
        retried = True
        log.warning(
            f"[analyst] 首轮正文 0 个引用编号（证据 {len(selected)} 条），"
            f"追加末尾硬约束重试一次"
        )
        ctx.trace.emit("citation_retry", node=NODE, reason="首轮正文无任何引用编号", evidence=len(selected))
        try:
            retry = ctx.gateway.chat(
                _with_retry_tail(messages), task="analyst", temperature=0.3, fuse_level=fuse
            )
        except Exception as exc:  # noqa: BLE001 - 纠偏是补救动作，失败就用首轮结果 + 兜底标注
            log.warning(f"[analyst] 纠偏重试失败：{type(exc).__name__}: {exc}")
            retry = None
        if retry is not None:
            # **两轮都算进成本**：只记一轮会让成本面板漏掉这次重试的真实开销。
            resp = replace(
                resp,
                text=retry.text,
                input_tokens=resp.input_tokens + retry.input_tokens,
                output_tokens=resp.output_tokens + retry.output_tokens,
                cost_cny=resp.cost_cny + retry.cost_cny,
                latency_ms=resp.latency_ms + retry.latency_ms,
                attempts=resp.attempts + retry.attempts,
            )
            ctx.trace.emit(
                "citation_retry_done",
                node=NODE,
                citations=len(extract_ids(retry.text)),
                recovered=bool(extract_ids(retry.text)),
            )

    final_report, refs, check = finalize(resp.text, index)

    if not check["pass"]:
        ctx.trace.emit("citation_check_failed", node=NODE, **check)
        log.warning(
            f"[analyst] 引用完整性校验未通过：unresolved={check['unresolved']} "
            f"referenced={check['referenced']}"
        )
    if check.get("no_citations"):
        # 这里**不能**只记一条日志：零引用会让审计「无对象可审」，
        # 摘要里的 unsupported=0 会被读成"全部通过"。必须显式留痕，
        # 并由 `auditor` 在报告头加上"未通过回查"横幅。
        ctx.trace.emit(
            "citation_missing",
            node=NODE,
            evidence=len(selected),
            retried=retried,
        )
        log.warning(
            f"[analyst] 正文零引用（已重试={retried}）：逐句回查对本次报告不适用，"
            f"reader 必须以「未通过回查」对待"
        )

    cost, toks = budget_after(state, resp)
    report_budget(ctx, NODE, cost, toks)
    log.node(
        TAG,
        NODE,
        "完成",
        chars=len(final_report),
        referenced=check["referenced"],
        unresolved=len(check["unresolved"]),
        cited_ratio=check["cited_ratio"],
    )

    return {
        "report": final_report,
        "reference_list": refs,
        "citation_check": check,
        # T7.10：把充足性评估写进 state，供审计/沉淀/前端判断"这份报告建立在多少证据上"
        "evidence_sufficiency": suff,
        # T4.4：记录正文是用哪个档位写的，供 trace / 成本面板复盘"降级后报告由谁产出"
        "model_tier": ctx.gateway.router.tier("analyst", fuse_level=fuse),
        **accumulate(resp),
    }
