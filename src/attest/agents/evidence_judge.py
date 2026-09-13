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
from difflib import SequenceMatcher
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

#: claim 是"口径原话"，不是证据正文。
#: 真实运行实测（2026-09-13，智谱 glm-4-flash 免费档）：模型把**整段证据原文**（数百字）
#: 串进了 claim，报告「争议与分歧」章节直接不可读。提示词里明明写了「≤60 字原话片段」——
#: 但**提示词是软约束，小模型会不服从**，所以必须在代码里兜住。
MAX_CLAIM_CHARS = 80

#: 近重复判定的字符相似度阈值（`difflib.SequenceMatcher.ratio`）。
#: 实测案例："市场规模和增长趋势" vs "市场规模与增长趋势" → 0.889（同一话题换了个连词）。
#: 取 0.88 是**贴着实测值下限**走：再低就会开始吞掉"同一指标不同措辞"的正常差异。
NEAR_DUP_RATIO = 0.88

#: 短于该长度的原话不做近重复判定。
#: 理由：`62亿` / `180亿` 这类短口径字符相似度天然很高，而它们**恰恰是真冲突本身**——
#: 误并等于把矛盾检测关掉（宁可漏并，不可误并）。
NEAR_DUP_MIN_CHARS = 6

_GAP_SUBJ_RE = re.compile(r"[「『\"']([^」』\"']+)[」』\"']")

#: claim 里内嵌的引用编号：展示层已由 `source_a` / `source_b` 负责，claim 里再带一遍是重复的，
#: 而且编号**按扇出分支独立**（同一文档在不同分支拿到不同编号），留在 claim 里会让去重键漂移。
_CITE_IN_CLAIM_RE = re.compile(r"\[(?:WEB|LOC|MEM)\d+(?:-\d+)*\]", re.I)

_NUM_IN_CLAIM_RE = re.compile(r"\d+(?:\.\d+)?")


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


def sanitize_claim(text: str) -> str:
    """把模型报的 claim 清成**一句可读、可回查**的口径原话。

    为什么必须在代码里做（不能只靠提示词，真实运行实测 2026-09-13）：
    - 提示词要求「原话片段 ≤60 字」，但免费档小模型把**整段证据原文**串了进来；
    - claim 会**原样进报告正文**（「争议与分歧」章节）——几百字的 claim 直接毁掉可读性；
    - claim 还会进**去重键**，噪音越大越判不出重复。
    """
    s = text or ""
    s = _CITE_IN_CLAIM_RE.sub("", s)  # 内嵌编号交给 source_a/source_b 展示，claim 里不留
    s = re.sub(r"[*`_#>]+", "", s)
    s = re.sub(r"\s+", " ", s)
    # 去掉编号后会留下"悬空空格"（`约 1800 亿元 ，含交付口径`）——中文标点前不该有空格
    s = re.sub(r"\s+([，。；、,;：:！!？?）)】」])", r"\1", s)
    s = s.strip()
    if len(s) > MAX_CLAIM_CHARS:
        s = s[:MAX_CLAIM_CHARS].rstrip("，。；、,; ") + "…"
    return s


def sanitize_conflict(c: Conflict) -> Conflict:
    """清洗一条 Conflict 的文本字段。构造后立刻调用，state 里存的就是干净的。

    无改动时**原样返回同一个对象**——保持对象身份（有测试断言"保留先出现的那条"），
    也避免每轮去重都无谓地复制一遍。
    """
    updates = {
        "claim_a": sanitize_claim(c.claim_a),
        "claim_b": sanitize_claim(c.claim_b),
        "summary": re.sub(r"\s+", " ", c.summary or "").strip(),
        "topic": (c.topic or "").strip(),
    }
    if all(getattr(c, k) == v for k, v in updates.items()):
        return c
    return c.model_copy(update=updates)


def _numbers_in(text: str) -> set[str]:
    return set(_NUM_IN_CLAIM_RE.findall(text or ""))


def _is_near_dup(a: str, b: str) -> bool:
    """两条原话是否只是**同一口径的不同措辞**，而不是两个互斥口径。

    为什么不能只看字符相似度（真实运行踩到的反例）：
    - `2024年市场规模为62亿元` vs `2024年市场规模为180亿元` 相似度 ≈0.91，**但它们正是
      真冲突**——只差数字。所以要先排除"数值不同"：两侧都含数值且集合不同 → 判为真冲突，不合并。
    - `市场规模和增长趋势` vs `市场规模与增长趋势`（无数值、只差一个连词）→ 同一话题被重复报，应合并。
    """
    if len(a) < NEAR_DUP_MIN_CHARS or len(b) < NEAR_DUP_MIN_CHARS:
        return False
    na, nb = _numbers_in(a), _numbers_in(b)
    if na and nb and na != nb:
        return False  # 数值不同 = 口径互斥的证据本身，不能被当成重复措辞并掉
    hi, lo = max(len(a), len(b)), min(len(a), len(b))
    if hi > lo * 2:
        return False  # 长度悬殊：一个是短语、一个是整句，不判近重复
    return SequenceMatcher(None, a, b).ratio() >= NEAR_DUP_RATIO


def _matches_anchor(claim: str, anchors: set[str]) -> bool:
    """claim 是否命中已有锚点：先精确（快），再近重复（慢，放后面）。"""
    if claim in anchors:
        return True
    return any(_is_near_dup(claim, a) for a in anchors)


def _dedupe_conflicts(found: list[Conflict], existing: list[Conflict]) -> list[Conflict]:
    """跨轮 + 同轮去重。三条判据：

    1. **完全相同**（双方原话都相同）→ 直接丢。见 `_conflict_key`：键不含分支名、
       也不含引用编号，因为同一份文档在不同扇出分支会拿到不同编号。
    2. **同一处分歧被另一个来源重复报**：若某侧的原话已经出现过，说明同一个争议换了
       对面来源又来报一次，合并掉（保留先出现的）。
    3. **近重复措辞**（2026-09-13 真实运行实测新增）：同一话题被两种措辞各报一次
       （"市场规模和增长趋势" / "市场规模与增长趋势"），精确匹配抓不到。
       判据见 `_is_near_dup`——**数值不同不算近重复**，因为那正是冲突本身。

    判据 2/3 刻意收紧到"原话相同或仅措辞差异"——谈**另一个**指标不会被误并。
    **锚点用规范化原话而非引用编号**，理由同上。

    实测教训（P3 收尾）：180 亿 vs 62 亿的口径差，被 `WEB1-1-1` 和 `WEB1-1-2` 分别
    与同一句 `LOC1-1-1`（62 亿）配对，报告里"争议与分歧"于是把同一出处报了两遍。
    实测教训（T7.2）：`topic` 是分支名、编号按分支独立，同一处分歧被 3 个分支各报一次
    → 见 `_conflict_key`。
    实测教训（T7.9c）：免费档模型对同一话题给出两种措辞，精确匹配漏判 → 判据 3。
    """
    # 防御：旧 checkpoint 里的 conflict 可能是清洗前落下的脏数据（claim 串了整段证据）。
    kept: list[Conflict] = [sanitize_conflict(c) for c in existing]
    seen = {_conflict_key(c) for c in kept}
    anchors: set[str] = set()
    for c in kept:
        for claim, _src in _conflict_sides(c):
            anchors.add(claim)

    out: list[Conflict] = []
    for c in found:
        c = sanitize_conflict(c)  # 幂等：调用方可能直接传未清洗的（本函数是公开入口）
        key = _conflict_key(c)
        if key in seen:
            continue
        sides = _conflict_sides(c)
        if any(_matches_anchor(claim, anchors) for claim, _src in sides):
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
