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

from attest.agents.reflect import targets_from_gaps
from attest.quality.audit import extract_claims, verify_claim
from attest.quality.conflict import Pair, group_by_sub_question, select_pairs
from attest.retrieval.hybrid import BM25Retriever, rrf_fuse, tokenize
from attest.retrieval.ports import Evidence
from attest.retrieval.text import chunk_text, load_chunks

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
