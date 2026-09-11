"""T2.4 / T3.7 · 证据判别（升级）：逐条评相关度/可信度 + 缺口 + **矛盾检测**。

两个工程决策，都写在代码里以便答辩时能解释：

1. **只判本轮新证据**。`evidence` 经 reducer 跨轮累加，若每轮都对全量重新判别，
   `judgments` 会随轮次出现重复编号，且 token 成本按轮次线性放大。所以按
   `round_no == state.round` 取新证据送判；**缺口覆盖度则按全量证据算**（那才是事实）。
2. **矛盾检测独立成一次受限调用**，不复用判别调用。若让模型在判别时顺带做 O(n²) 两两比对，
   成本不可控（正是《功能设计》§6.3 要避免的）。所以先由 `quality/conflict.py` 做
   聚类 + 配额截断，再把**有限的对**送进提示词。
"""

from __future__ import annotations

import re
from typing import Any

from ..llm.prompts import build_conflict_messages, build_judge_messages
from ..logging import get_logger
from ..quality.conflict import pairs_as_ids, select_pairs
from ..schemas import Conflict, ConflictResult, JudgeResult
from .base import NodeContext, report_budget

log = get_logger(__name__)
NODE = "evidence_judge"
TAG = "judge"

#: 低于该相关度的证据不进正文（与 analyst 的 MIN_RELEVANCE 同源语义）
MIN_RELEVANCE = 3

_GAP_SUBJ_RE = re.compile(r"[「『\"']([^」』\"']+)[」』\"']")


def merge_gaps(
    llm_gaps: list[str], sub_questions: list[str], covered: set[str]
) -> list[str]:
    """LLM 报的缺口 ∪ 按覆盖度算出的缺口，并剔除**已有证据覆盖**的项。

    剔除这一步是防"补检死循环"的关键：补检第二轮后该子问题已有证据，
    但模型仍可能凭残缺上下文重复报同一缺口——不拦就会把轮数上限全烧光。
    """
    out: list[str] = []
    seen: set[str] = set()
    for gap in llm_gaps:
        s = (gap or "").strip()
        if not s:
            continue
        m = _GAP_SUBJ_RE.search(s)
        if m and m.group(1).strip() in covered:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    for sq in sub_questions:
        if sq in covered:
            continue
        g = f"缺少「{sq}」的直接证据"
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def _embed_fn(ctx: NodeContext):
    def embed(texts: list[str]) -> list[list[float]]:
        vectors, _, _ = ctx.gateway.embed(list(texts), task="conflict_cluster")
        return vectors

    return embed


def _detect_conflicts(
    ctx: NodeContext, objective: str, evidence: list[Any], *, fuse_level: int = 0
) -> tuple[list[Conflict], Any]:
    """聚类 → 配额配对 → 一次受限 LLM 调用。返回 (conflicts, 该次调用的响应或 None)。"""
    if len(evidence) < 2:
        return [], None
    pairs = select_pairs(
        evidence,
        embed_fn=_embed_fn(ctx),
        threshold=ctx.settings.conflict_cluster_threshold,
        per_cluster=ctx.settings.conflict_pairs_per_cluster,
        global_cap=ctx.settings.conflict_pairs_global,
    )
    if not pairs:
        return [], None
    resp = ctx.gateway.chat(
        build_conflict_messages(objective, evidence, pairs_as_ids(pairs)),
        task="conflict",
        response_model=ConflictResult,
        temperature=0.0,
        fuse_level=fuse_level,  # T4.3：轻任务，熔断 L1 起可切便宜档
    )
    result: ConflictResult = resp.parsed  # type: ignore[assignment]
    return list(result.conflicts), resp


def _new_gaps_only(state: dict[str, Any], gaps: list[str]) -> list[str]:
    """`gaps` 的 reducer 是 `operator.add`，而缺口是**每轮重算的全量**——

    若每轮都把全量缺口交回去，state 里的 gaps 会 1 → 2 → 3 地长（同一缺口重复计入），
    报告与评测都会看到重复条目。所以只回传"新增的"那些，保持累加语义的同时不重复。
    """
    existing = set(state.get("gaps") or [])
    return [g for g in gaps if g not in existing]


def _conflict_key(c: Conflict) -> tuple[str, str, str]:
    """去重键 = **双方原话**（规范化后），**不含** `topic`，也**不含**引用编号。

    为什么两个字段都要从键里拿掉（T7.2 实测，逐条踩出来的）：

    1. **去掉 `topic`（子问题/扇出分支名）**——分支不是争议的身份。词法检索会把同一条证据
       塞进多个分支，于是"同一处分歧"被每个分支各报一次。
    2. **去掉引用编号**——这更隐蔽：每个扇出分支有**自己的编号空间**，所以**同一份文档**
       被 3 个分支检索到时会拿到 3 个不同编号（`WEB1-1-2` / `WEB1-2-2` / `WEB1-3-3`）。
       按编号去重会把它们当成 3 条不同证据，`conflicts` 计数从 1 虚增到 3（C3/C4/L2/L3 实测）。

    一处分歧的真实身份是"**什么口径对什么口径不一致**"，即双方原话的组合——
    与它在哪个分支被发现、被编了哪个号都无关。所以键只保留规范化后的双方原话。
    """
    a, b = sorted([_norm_claim(c.claim_a), _norm_claim(c.claim_b)])
    return ("|", a, b)


def _norm_claim(text: str) -> str:
    """把原话规范化后用于比较：去空白、去 markdown 标记干扰。

    同一份文档在不同分支被截断到不同长度时（`_snippet(limit=60)` 的截断点一致，
    但前缀可能带不同的 markdown 残留），规范化能提高命中去重的概率。
    """
    return re.sub(r"[\s*`_]+", "", text or "")


def _conflict_sides(c: Conflict) -> tuple[tuple[str, str], tuple[str, str]]:
    """返回两侧的 (规范化原话, 引用编号)。**锚点比较用规范化原话**，编号只作展示。"""
    return ((_norm_claim(c.claim_a), c.source_a), (_norm_claim(c.claim_b), c.source_b))


def _dedupe_conflicts(found: list[Conflict], existing: list[Conflict]) -> list[Conflict]:
    """跨轮 + 同轮去重。两条判据：

    1. **完全相同**（双方原话都相同）→ 直接丢。见 `_conflict_key`：键不含分支名、
       也不含引用编号，因为同一份文档在不同扇出分支会拿到不同编号。
    2. **同一处分歧被另一个来源重复报**：若某侧的原话已经出现过，说明同一个争议换了
       对面来源又来报一次，合并掉（保留先出现的）。

    判据 2 刻意收紧到"原话相同"——只有**同一句话**才算重复；谈**另一个**指标
    （原话不同）不会被误并。**锚点用规范化原话而非引用编号**，理由同上。

    实测教训（P3 收尾）：180 亿 vs 62 亿的口径差，被 `WEB1-1-1` 和 `WEB1-1-2` 分别
    与同一句 `LOC1-1-1`（62 亿）配对，报告里"争议与分歧"于是把同一出处报了两遍。
    实测教训（T7.2）：`topic` 是分支名、编号按分支独立，同一处分歧被 3 个分支各报一次
    → 见 `_conflict_key`。
    """
    kept: list[Conflict] = list(existing)
    seen = {_conflict_key(c) for c in kept}
    anchors: set[str] = set()
    for c in kept:
        for claim, _src in _conflict_sides(c):
            anchors.add(claim)

    out: list[Conflict] = []
    for c in found:
        key = _conflict_key(c)
        if key in seen:
            continue
        sides = _conflict_sides(c)
        if any(claim in anchors for claim, _src in sides):
            continue
        seen.add(key)
        for claim, _src in sides:
            anchors.add(claim)
        out.append(c)
    return out


def run(state: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    plan = state.get("plan") or {}
    objective = plan.get("objective", "")
    sub_questions: list[str] = plan.get("sub_questions") or []
    all_evidence = list(state.get("evidence") or [])
    cur_round = int(state.get("round", 1) or 1)
    fuse = ctx.budget_snapshot(state).fuse_level

    if not all_evidence:
        log.node(TAG, NODE, "跳过", reason="本轮无证据")
        return {
            "judgments": [],
            "gaps": ["本轮未检索到任何证据，建议放宽查询词或补充信源"],
        }

    new_evidence = [e for e in all_evidence if int(getattr(e, "round_no", 1) or 1) == cur_round]
    if not new_evidence:
        # 补检这一轮没拿到任何新证据 → **不再重复判别**（否则 judgments 出现重复编号、白烧一次调用）。
        # 只按已有判别重算缺口，把收口交给 reflect 的轮数上限。
        keep_ids = {
            j.citation_id
            for j in (state.get("judgments") or [])
            if j.relevance >= MIN_RELEVANCE
        }
        covered = {e.sub_question for e in all_evidence if e.citation_id in keep_ids}
        gaps = merge_gaps([], sub_questions, covered)
        ctx.trace.emit("judge_skipped", node=NODE, round=cur_round, reason="本轮无新证据")
        log.node(TAG, NODE, "跳过", reason="本轮无新证据", gaps=len(gaps))
        return {"judgments": [], "gaps": gaps}

    log.node(
        TAG, NODE, "开始", evidence=len(all_evidence), new_evidence=len(new_evidence), round=cur_round
    )

    resp = ctx.gateway.chat(
        build_judge_messages(objective, sub_questions, new_evidence),
        task="judge",
        response_model=JudgeResult,
        temperature=0.0,
        fuse_level=fuse,
    )
    result: JudgeResult = resp.parsed  # type: ignore[assignment]

    cost = float(state.get("cost_incurred", 0.0) or 0.0) + resp.cost_cny
    toks = int(state.get("tokens_incurred", 0) or 0) + resp.total_tokens
    inc_cost, inc_toks = resp.cost_cny, resp.total_tokens

    all_judgments = list(state.get("judgments") or []) + list(result.judgments)
    keep_ids = {j.citation_id for j in all_judgments if j.relevance >= MIN_RELEVANCE}
    covered = {e.sub_question for e in all_evidence if e.citation_id in keep_ids}
    gaps = merge_gaps(list(result.gaps), sub_questions, covered)

    conflicts, cresp = _detect_conflicts(ctx, objective, all_evidence)
    conflicts = _dedupe_conflicts(conflicts, list(state.get("conflicts") or []))
    if cresp is not None:
        cost += cresp.cost_cny
        toks += cresp.total_tokens
        inc_cost += cresp.cost_cny
        inc_toks += cresp.total_tokens

    report_budget(ctx, NODE, cost, toks)
    low = [j for j in result.judgments if j.relevance <= 2]
    log.node(
        TAG,
        NODE,
        "完成",
        judgments=len(result.judgments),
        low_relevance=len(low),
        gaps=len(gaps),
        covered=len(covered),
        conflicts=len(conflicts),
    )

    return {
        "judgments": result.judgments,
        "gaps": _new_gaps_only(state, gaps),
        "conflicts": conflicts,
        "cost_incurred": inc_cost,
        "tokens_incurred": inc_toks,
    }
