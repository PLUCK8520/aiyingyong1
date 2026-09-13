"""P3 切片回归测试：本地混合检索 / 矛盾检测成本闸门 / 审计判定 / Reflect 早停。

这一批用例的共同点：**都是纯函数或近纯函数**（无 LLM、无网络、无 Chroma），
所以能像断言算术一样断言它们——这正是 P3 把"检索/矛盾/审计"从节点里拆出来单独成模块的回报。

覆盖的坑（每条都对应 P3 开发中真实踩过的）：
  - T3.1 分块：段落优先与超长硬切，且 chunk_id 必须稳定可复现（换机器也一样）
  - T3.2 中文 BM25：必须分词，否则整句一个 token，检索近似失效
  - T3.2 RRF：只在单路出现的下标也要被召回（混合检索的收益来源）
  - T3.7 矛盾闸门：**零分/兜底证据不得参与比对**（否则凭空造冲突，实测教训）
  - T3.7 成本上限：每簇 ≤N 对、全局 ≤M 对
  - T3.10 审计：数值不一致是**硬否决**（"数字被改过却看起来对"是最危险的失分）
  - T3.6 Reflect：无缺口不补检；缺口能定位到原子问题序号；轮数上限生效
"""

from __future__ import annotations

from pathlib import Path

import pytest

from attest.agents.evidence_judge import (
    MAX_CLAIM_CHARS,
    _dedupe_conflicts,
    _is_near_dup,
    sanitize_claim,
    sanitize_conflict,
)
from attest.agents.reflect import targets_from_gaps
from attest.llm.providers import _primary_number
from attest.quality.audit import extract_claims, verify_claim
from attest.quality.conflict import Pair, group_by_sub_question, select_pairs
from attest.retrieval.hybrid import BM25Retriever, rrf_fuse, tokenize
from attest.retrieval.ports import Evidence
from attest.retrieval.text import chunk_text, load_chunks, strip_markdown
from attest.schemas import Conflict

# ----------------------------------------------------------------- T3.1 分块


def test_chunk_text_packs_paragraphs_without_cutting_sentences() -> None:
    """短段落应被贪心合并进同一 chunk，而不是一段一块。"""
    text = "第一段较短。\n\n第二段也短。\n\n第三段短。"
    chunks = chunk_text(text, size=200, overlap=20)
    assert len(chunks) == 1, f"短段落应合并为 1 块，实际 {len(chunks)}：{chunks}"
    assert "第一段" in chunks[0] and "第三段" in chunks[0]


def test_chunk_text_hard_splits_oversized_paragraph_with_overlap() -> None:
    """超长无空行段落必须硬切，且相邻块有 overlap（避免边界信息被截断）。"""
    para = "".join(f"{i:04d}" for i in range(50))  # 200 字符，无空行
    chunks = chunk_text(para, size=80, overlap=20)
    assert len(chunks) >= 3, f"应被切成 >=3 块，实际 {len(chunks)}"
    assert all(len(c) <= 80 for c in chunks), "任一块都不得超过 size"
    # 相邻块的尾部/头部应重叠：第 0 块尾 20 字符 == 第 1 块首 20 字符
    assert chunks[0][-20:] == chunks[1][:20], "相邻块没有按 overlap 重叠"


def test_load_chunks_id_is_stable_and_path_scoped(tmp_path: Path) -> None:
    """chunk_id 由「相对路径#序号」哈希而来 → 同内容同序号必须稳定。"""
    (tmp_path / "a.md").write_text("# 标题A\n\n甲段落内容。\n\n乙段落内容。", encoding="utf-8")
    first = load_chunks(tmp_path, size=200, overlap=20)
    again = load_chunks(tmp_path, size=200, overlap=20)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in again], "chunk_id 不稳定"
    assert all(c.source_path == "a.md" for c in first)
    assert first[0].title == "标题A"


def test_load_chunks_drops_h1_and_markdown_markers(tmp_path: Path) -> None:
    """文档 H1 已存进 title，正文里不该再出现 `# 标题`；`**`/反引号也应降级为纯文本。

    实测教训（P3 收尾）：摘要首句变成「# 内部定价与落地成本手册 …」，标题标记直接串进报告。
    """
    (tmp_path / "a.md").write_text(
        "# 标题A\n\n正文含 **重点** 与 `代码` 标记。", encoding="utf-8"
    )
    chunks = load_chunks(tmp_path, size=200, overlap=20)
    body = chunks[0].text
    assert "#" not in body, f"正文不该含标题标记：{body!r}"
    assert "**" not in body and "`" not in body, f"强调/代码标记未剥离：{body!r}"
    assert "重点" in body and "代码" in body, "剥离标记不得丢字"
    assert chunks[0].title == "标题A"


def test_strip_markdown_keeps_inner_text() -> None:
    assert strip_markdown("## 小标题\n\n**加粗** 与 `code`") == "小标题\n\n加粗 与 code"


# ----------------------------------------------------------- T3.2 中文 BM25 / RRF


def test_tokenize_segments_cjk_instead_of_whole_sentence() -> None:
    """中文必须切词：#整句一个 token 是 BM25 失效的根因。"""
    toks = tokenize("企业知识库 Agent 平台的市场规模")
    assert len(toks) > 3, f"中文没有被有效切分：{toks}"
    assert "企业" in toks or "知识库" in toks


def test_bm25_ranks_relevant_doc_first() -> None:
    docs = [
        "本报告讨论市场规模与增速，2025 年约 180 亿元。",
        "本报告讨论竞品格局与差异化优势。",
        "本报告讨论收费模式与席位定价。",
    ]
    r = BM25Retriever(["d0", "d1", "d2"], docs)
    hits = r.search("竞品 差异化 优势", top_k=3)
    assert hits, "相关查询应有命中"
    assert hits[0][0] == 1, f"应命中第 2 篇（竞品），实际 {hits}"
    # 零分文档不得返回（零分＝一个词都没命中，不是弱相关）
    assert all(score > 0 for _, score in hits)


def test_rrf_fuse_recalls_item_present_in_only_one_list() -> None:
    """RRF 的关键收益：某下标只在一路出现也能进融合结果（另一路是 0 分而非缺席）。"""
    fused = rrf_fuse([[0, 1], [2]], k=60)
    idxs = [i for i, _ in fused]
    assert set(idxs) == {0, 1, 2}, f"单路独有项被漏掉：{fused}"
    score = dict(fused)
    # 0 与 2 都是各自列表的 rank1 → 分数相同；1 是 rank2 → 更低
    assert score[0] == pytest.approx(score[2])
    assert score[0] > score[1]


def test_rrf_fuse_rewards_agreement_across_lists() -> None:
    """同一项在两路都排高位 → 融合分应高于只在单路排同位的项（RRF 的"共识加成"）。"""
    fused = rrf_fuse([[0, 1], [0, 2]], k=60)
    score = dict(fused)
    # 0 在两路都是 rank1 → 2/(k+1)；1、2 只在单路 rank2 → 1/(k+2)
    assert score[0] > score[1]
    assert score[1] == pytest.approx(score[2])


# --------------------------------------------------------- T3.7 矛盾检测闸门


def _ev(cid: str, subq: str, content: str, *, score: float = 1.0, rnd: int = 1) -> Evidence:
    return Evidence(
        citation_id=cid,
        source="local" if cid.startswith("[LOC") else "web",
        title=cid,
        url=f"local://x#{cid}",
        content=content,
        sub_question=subq,
        round_no=rnd,
        score=score,
    )


def test_group_by_sub_question_buckets_evidence() -> None:
    groups = group_by_sub_question([_ev("[WEB1-1-1]", "市场规模", "a"), _ev("[WEB1-2-1]", "竞品", "b")])
    assert set(groups) == {"市场规模", "竞品"}
    assert len(groups["市场规模"]) == 1


def test_select_pairs_excludes_zero_score_fallback_evidence() -> None:
    """零分/兜底证据不参与矛盾比对——否则兜底补齐会凭空造出冲突（实测教训）。"""
    real_a = _ev("[WEB1-1-1]", "市场规模", "2025 年市场规模约 180 亿元，增速 30%。", score=1.0)
    fallback = _ev("[LOC1-1-2]", "市场规模", "按此口径 2025 年约 62 亿元。", score=0.0)
    pairs = select_pairs(
        [real_a, fallback],
        embed_fn=lambda texts: [[1.0, 0.0] for _ in texts],
        threshold=0.25,
        per_cluster=6,
        global_cap=30,
    )
    assert pairs == [], f"零分证据不应产生任何待比对pair，实际 {pairs}"


def test_select_pairs_respects_global_cap() -> None:
    """全局上限是成本闸门，必须真的截断（不能因为"想多比几对"就突破）。"""
    items = [
        _ev(f"[WEB1-1-{i}]", "市场规模", f"第 {i} 条证据，2025 年规模说法。", score=1.0)
        for i in range(1, 9)
    ]
    # 全部同簇（embedding 恒等 + 高阈值），会生成 C(8,2)=28 对
    pairs = select_pairs(
        items,
        embed_fn=lambda texts: [[1.0, 0.0] for _ in texts],
        threshold=0.0,
        per_cluster=100,
        global_cap=5,
    )
    assert len(pairs) == 5, f"全局上限 5 未生效：{len(pairs)}"


def test_select_pairs_keeps_only_within_cluster_shortlist() -> None:
    """每簇上限同样生效：一个簇最多贡献 per_cluster 对。"""
    items = [
        _ev(f"[WEB1-1-{i}]", "市场规模", f"第 {i} 条。", score=1.0) for i in range(1, 6)
    ]
    pairs = select_pairs(
        items,
        embed_fn=lambda texts: [[1.0, 0.0] for _ in texts],
        threshold=0.0,
        per_cluster=2,
        global_cap=100,
    )
    assert len(pairs) == 2, f"每簇上限 2 未生效：{len(pairs)}"


# ------------------------------------------------------------- T3.10 审计判定


def test_verify_claim_supported_when_terms_and_numbers_match() -> None:
    verdict, reason = verify_claim(
        "2025 年国内市场规模约 180 亿元",
        "示例研究院测算，2025 年国内企业知识库市场规模约为 180 亿元人民币。",
    )
    assert verdict == "supported", reason


def test_verify_claim_numeric_mismatch_is_hard_reject() -> None:
    """数值不一致必须直接 unsupported——哪怕文字覆盖率很高（防"数字被改过"）。"""
    verdict, reason = verify_claim(
        "2025 年国内市场规模约 300 亿元",
        "示例研究院测算，2025 年国内企业知识库市场规模约为 180 亿元人民币。",
    )
    assert verdict == "unsupported", f"数值不符未硬否决：{verdict} / {reason}"
    assert "数值" in reason


def test_verify_claim_unsupported_on_low_overlap() -> None:
    verdict, _ = verify_claim("该厂商提供完全免费的私有化部署方案", "本段讨论的是席位定价与按量计费。")
    assert verdict in {"unsupported", "partial"}
    assert verdict != "supported"


def test_extract_claims_keeps_only_cited_sentences() -> None:
    report = "市场规模约 180 亿元 [WEB1-1-1]。这句话没有引用。\n价格是 300 元 [LOC1-3-2]。"
    known = {"[WEB1-1-1]", "[LOC1-3-2]"}
    claims = extract_claims(report, known)
    ids = [cid for _, cid in claims]
    assert ids == ["[WEB1-1-1]", "[LOC1-3-2]"], f"应只抽带引用的句子：{claims}"


# ------------------------------------------------------------- T3.6 Reflect


def test_targets_from_gaps_maps_back_to_original_subq_number() -> None:
    """缺口文本里的原子问题要能定位回原序号——否则补检证据编号会串。"""
    subs = ["市场规模", "竞品格局", "收费模式"]
    gaps = ["子问题「竞品格局」缺少 2025 年最新数据"]
    targets = targets_from_gaps(gaps, subs, "评估市场", next_round=2)
    assert len(targets) == 1
    assert targets[0]["sub_question"] == "竞品格局"
    assert targets[0]["subq_no"] == 2, f"应复用原子问题序号 2，实际 {targets[0]['subq_no']}"
    assert targets[0]["round"] == 2


def test_targets_from_gaps_dedupes_and_assigns_new_number_for_unknown() -> None:
    subs = ["市场规模"]
    gaps = ["「新增维度」资料不足", "「新增维度」再次缺失"]
    targets = targets_from_gaps(gaps, subs, "目标", next_round=2)
    assert len(targets) == 1, "同缺口应去重"
    # 未知子问题追加到末尾，序号 = 1(已有) + 1 = 2
    assert targets[0]["subq_no"] == 2


def test_pair_dataclass_is_frozen() -> None:
    """Pair 冻结：融合/排序过程中不能被就地改写（确定性依赖此约束）。"""
    a = _ev("[WEB1-1-1]", "市场规模", "a")
    b = _ev("[WEB1-1-2]", "市场规模", "b")
    p = Pair(a=a, b=b, topic="市场规模", similarity=0.5)
    with pytest.raises(Exception):
        p.topic = "改了"  # type: ignore[misc]


# ------------------------------------------------------ T3.7 矛盾去重（同处分歧）

_LOC_62 = "按此口径，2025 年国内市场规模约 62 亿元人民币，增速约 38%"


def _cf(topic: str, ca: str, sa: str, cb: str = _LOC_62, sb: str = "[LOC1-1-1]") -> Conflict:
    return Conflict(
        sub_question=topic, topic=topic, claim_a=ca, source_a=sa, claim_b=cb, source_b=sb
    )


def test_dedupe_conflicts_merges_same_dispute_from_another_source() -> None:
    """同一处分歧（对面都是同一句 62 亿）换了个来源又来报 → 只保留一条。

    实测教训（P3 收尾）：`WEB1-1-1` 与 `WEB1-1-2` 分别和同一句 `LOC1-1-1` 配对，
    报告"争议与分歧"把同一出处报了两遍。
    """
    c1 = _cf("市场规模", "2025 年规模约 180 亿元", "[WEB1-1-1]")
    c2 = _cf("市场规模", "不把定制交付计入盘子，与含交付口径差约 2.9 倍", "[WEB1-1-2]")
    kept = _dedupe_conflicts([c1, c2], [])
    assert len(kept) == 1, f"同一处分歧应合并为 1 条，实际 {len(kept)}"
    assert kept[0] is c1, "应保留先出现的那条"


def test_dedupe_conflicts_keeps_distinct_metrics_in_same_topic() -> None:
    """同一来源谈**不同指标**（原话不同）不得误并——去重判据必须收紧到"来源+原话都相同"。"""
    size = _cf("市场规模", "规模约 180 亿元", "[WEB1-1-1]")
    growth = _cf(
        "市场规模",
        "增速约 42%",
        "[WEB1-1-1]",
        cb="增速约 38%",
        sb="[LOC1-1-1]",
    )
    kept = _dedupe_conflicts([size, growth], [])
    assert len(kept) == 2, f"不同指标的分歧不应合并，实际 {len(kept)}"


def test_dedupe_conflicts_drops_exact_repeat_across_rounds() -> None:
    c = _cf("市场规模", "规模约 180 亿元", "[WEB1-1-1]")
    kept = _dedupe_conflicts([c], [c])
    assert kept == [], "跨轮完全相同应被去掉"


def test_dedupe_conflicts_ignores_sub_question_bucket() -> None:
    """**T7.2 实测缺陷**：同一处分歧被不同扇出分支各报一次 → 应合并为 1 条。

    `topic` / `sub_question` 是**扇出分支名**，不是争议的身份。词法检索把同一条证据
    塞进多个分支时，同一个"45% vs 78%"会在 3 个分支各报一次，`conflicts` 从 1 虚增到 3。
    """
    a = _cf("大模型推理成本的市场规模与增长情况", "降幅约 45%", "[WEB1-1-2]",
            cb="降幅约 78%", sb="[LOC1-1-1]")
    b = _cf("大模型推理成本的主要参与方与竞争格局", "降幅约 45%", "[WEB1-2-2]",
            cb="降幅约 78%", sb="[LOC1-2-1]")
    c = _cf("大模型推理成本的收费模式与落地成本", "降幅约 45%", "[WEB1-3-3]",
            cb="降幅约 78%", sb="[LOC1-3-2]")
    kept = _dedupe_conflicts([a, b, c], [])
    assert len(kept) == 1, f"同一处分歧跨分支应合并为 1 条，实际 {len(kept)}"


def test_dedupe_conflicts_ignores_citation_ids_per_branch() -> None:
    """**T7.2 实测缺陷（更隐蔽）**：每个扇出分支有**自己的编号空间**——

    同一份文档被 3 个分支检索到会拿到 3 个不同编号（`WEB1-1-2` / `WEB1-2-2` / `WEB1-3-3`）。
    若去重键含引用编号，就会把它们当成 3 条不同证据，计数再次虚增。
    所以键必须建立在**原话内容**上，而非编号上。这里两边编号全不同、原话相同 → 应合并。
    """
    x = _cf("甲分支", "降幅约 45%", "[WEB1-1-2]", cb="降幅约 78%", sb="[LOC1-1-1]")
    y = _cf("乙分支", "降幅约 45%", "[WEB2-7-9]", cb="降幅约 78%", sb="[LOC4-5-6]")
    kept = _dedupe_conflicts([x, y], [])
    assert len(kept) == 1, f"编号不同但原话相同应合并，实际 {len(kept)}"


def test_dedupe_conflicts_tolerates_whitespace_and_markup() -> None:
    """同一份文档在不同分支被 `_snippet` 截断/清洗后可能带不同空白或 markdown 残留，

    规范化后应仍能识别为同一处。这条防的是"去重键对格式过敏"。
    """
    x = _cf("甲", "降幅约 45%", "[W1]", cb="降幅约 78%", sb="[L1]")
    y = _cf("乙", "降幅约 **45%**", "[W2]", cb="降幅约 `78%` ", sb="[L2]")
    kept = _dedupe_conflicts([x, y], [])
    assert len(kept) == 1, f"仅格式差异应合并，实际 {len(kept)}"


# --------------------------------------------- T3.7 主数值抽取（T7.2 实测校准）

def test_primary_number_takes_first_not_largest() -> None:
    """**T7.2 实测缺陷**：口径对比类资料里，第二份资料几乎总会**引述第一份的数字**。

    若按量级最大取，两条证据会抽到**同一个数**，比值恒 1.0 → 矛盾检测静默失效。
    实测 F1（180亿/62亿）/ F3（78%/45%）/ D1（12%/8.5%）三例全因此漏报。
    """
    # 第二份资料引述了第一份的 180 亿，但**它自己的口径是 62 亿**
    second = "示例咨询仅统计**平台软件**口径，约 62 亿元；与含交付口径的 1800 亿元相差近三倍。"
    got = _primary_number(second)
    assert got is not None
    assert got[0] == 62e8, f"应抽本方口径 62 亿，实际 {got[0]}"


def test_primary_number_extracts_competing_calibers_from_pair() -> None:
    """两条证据必须各自抽出**本方**口径，比值才达阈值（这里 2.90×）。"""
    a = _primary_number("示例研究院测算：含交付口径约 1800 亿元，2023–2025 复合增速约 42%。")
    b = _primary_number("示例咨询仅统计平台软件口径，约 620 亿元；与含交付口径的 1800 亿元相差近三倍。")
    assert a and b
    lo, hi = sorted((a[0], b[0]))  # sorted 升序：先小后大
    assert hi / lo >= 1.5, f"应达冲突阈值，实际 {hi / lo:.2f}"


def test_primary_number_same_unit_family_required() -> None:
    """`万辆` 与 `亿` 不同族，不可比——闸门①的单元级验证。"""
    wan = _primary_number("出口量约 210 万辆，同比增速约 32%。")
    yi = _primary_number("市场规模约 1800 亿元。")
    assert wan and yi
    assert wan[1] == "万" and yi[1] == "亿"
    assert wan[1] != yi[1], "万辆 与 亿 不属同一单位族，不应参与比对"


def test_primary_number_none_when_no_number() -> None:
    assert _primary_number("本文没有给出任何可量化口径。") is None


# ------------------------------------- T7.9c 真实运行暴露的两个质量缺陷（claim 清洗 / 近重复）

def _c(
    a: str,
    b: str,
    *,
    sa: str = "[WEB1-1-1]",
    sb: str = "[LOC1-1-1]",
    topic: str = "市场规模",
) -> Conflict:
    return Conflict(
        sub_question="市场规模与增长情况",
        topic=topic,
        claim_a=a,
        source_a=sa,
        claim_b=b,
        source_b=sb,
    )


def test_sanitize_claim_strips_citation_and_markdown() -> None:
    """claim 里不该留引用编号（展示层有 source_*），markdown 标记也要去掉。"""
    dirty = "**约 1800 亿元** [WEB1-1-1]，含交付口径"
    assert sanitize_claim(dirty) == "约 1800 亿元，含交付口径"


def test_sanitize_claim_truncates_full_evidence_dump() -> None:
    """**T7.9c 真实运行缺陷**：glm-4-flash 把整段证据原文串进了 claim。

    提示词写了「≤60 字原话」，但小模型不服从——必须在代码里截断，
    否则报告「争议与分歧」章节会被几百字的 claim 撑爆。
    """
    dump = "示例研究院《2025 企业知识库市场追踪》指出，" + "根据多方访谈与问卷回收结果，" * 40
    out = sanitize_claim(dump)
    assert len(out) <= MAX_CLAIM_CHARS + 1, f"未截断：{len(out)} 字"
    assert out.endswith("…"), "截断后应有省略号，让人知道被截了"


def test_sanitize_claim_keeps_short_original_intact() -> None:
    """正常的短 claim 不能被改动（否则会破坏去重键的稳定性）。"""
    ok = "平台软件口径约 620 亿元"
    assert sanitize_claim(ok) == ok


def test_sanitize_conflict_cleans_both_sides_and_summary() -> None:
    c = sanitize_conflict(
        _c("**1800 亿元**\n[WEB1-1-1]", "620 亿元 [LOC1-1-1]")
    )
    assert "\n" not in c.claim_a and "[" not in c.claim_a
    assert c.claim_b == "620 亿元"


def test_is_near_dup_catches_connective_variants() -> None:
    """**T7.9c 实测案例**：同一话题两种措辞（和 / 与），精确匹配抓不到。"""
    assert _is_near_dup("市场规模和增长趋势", "市场规模与增长趋势")


def test_is_near_dup_never_merges_different_numbers() -> None:
    """反例（这条比上一条更重要）：只差数字的两句话**正是冲突本身**。

    `2024年市场规模为62亿元` vs `...180亿元` 字符相似度 ≈0.91，
    若只看相似度会把真冲突当重复并掉——等于把矛盾检测关掉。
    """
    assert not _is_near_dup("2024年市场规模为62亿元", "2024年市场规模为180亿元")


def test_is_near_dup_rejects_length_mismatch() -> None:
    """短语 vs 整句不判近重复（长度悬殊时短串会被包含，相似度虚高）。"""
    assert not _is_near_dup("市场规模", "市场规模与增长趋势" * 5)


def test_dedupe_merges_near_duplicate_phrasing() -> None:
    """跨轮/跨分支把同一话题用两种措辞各报一次 → 只留一条。"""
    existing = [_c("市场规模和增长趋势", "数据来源口径不同", sb="[LOC1-1-2]")]
    found = [_c("市场规模与增长趋势", "样本覆盖范围不同", sb="[LOC1-2-2]")]
    out = _dedupe_conflicts(found, existing)
    assert out == [], "近重复措辞应被合并掉（保留先出现的）"


def test_dedupe_keeps_conflicts_with_different_numbers() -> None:
    """数值口径不同的冲突必须留下——这是矛盾检测的核心产出，不能被去重吃掉。

    注意构造：两侧都要不同。若 claim_b 与原话相同，会先命中**既有判据 2**
    （任一侧原话重复即合并），那是 P3 收尾定下的取舍，与近重复无关。
    """
    existing = [_c("2024年市场规模为62亿元", "统计范围含交付与实施")]
    found = [_c("2024年市场规模为180亿元", "样本仅覆盖平台软件")]
    out = _dedupe_conflicts(found, existing)
    assert len(out) == 1, "数值互斥的口径差异是真冲突，不该被当重复"


def test_dedupe_still_merges_when_one_side_repeats_exactly() -> None:
    """钉住既有取舍（判据 2）：同一句原话与两个不同对手配对时只留先出现的。

    这条不是新逻辑，是防止我加近重复时把原有语义改掉。
    """
    existing = [_c("A 方口径 62 亿元", "B 方口径 180 亿元")]
    found = [_c("A 方口径 62 亿元", "C 方口径 1800 亿元")]
    assert _dedupe_conflicts(found, existing) == []


def test_dedupe_matches_dirty_claim_after_sanitize() -> None:
    """历史 checkpoint 里的脏 claim（带编号 + markdown）仍能与新报的同一条去重。"""
    dirty = "**市场规模和增长趋势** [WEB1-1-1]"
    existing = [_c(dirty, "数据来源口径不同", sb="[LOC1-1-2]")]
    found = [_c("市场规模与增长趋势", "样本覆盖范围不同", sb="[LOC1-2-2]")]
    assert _dedupe_conflicts(found, existing) == []
