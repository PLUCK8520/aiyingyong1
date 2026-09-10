"""T3.2 · 混合检索：jieba 分词 + BM25 稀疏检索 + RRF 融合。

⚠️ 中文必须先分词（任务清单 T3.2 明确）：不分词的中文 BM25 近似失效——
整句被当成一个 token，词频统计失去意义。所以 `tokenize()` 是这条链路的地基。

RRF（Reciprocal Rank Fusion）取 k=60（《功能设计》§6.2）：
    score(d) = Σ_{r ∈ 结果集} 1 / (k + rank_r(d))
只用**排名**不用原始分，所以 BM25 的无界分与余弦相似度可以直接融合，不需要归一化——
这是选 RRF 而不是加权求和的理由。
"""

from __future__ import annotations

import logging
from typing import Sequence

import jieba
from rank_bm25 import BM25Okapi

RRF_K = 60

#: 极小的中文停用词表：只为压掉噪声，不做语言学完备性追求（BM25 自身有 IDF 抑制常见词）
_STOPWORDS = frozenset(
    "的 了 和 与 及 或 是 在 对 为 以 于 中 有 也 就 都 而 之 其 上 下 不 这 那 我们 你们 他们 "
    "一个 一种 一些 什么 怎么 如何 哪些 可以 需要 进行 通过 以及 但是 因为 所以 如果 这个 那个".split()
)

_jieba_ready = False


def _ensure_jieba() -> None:
    global _jieba_ready
    if not _jieba_ready:
        jieba.setLogLevel(logging.WARNING)  # 关掉首次加载词典的冗余输出
        _jieba_ready = True


def _has_cjk(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


def tokenize(text: str) -> list[str]:
    """中文分词 + 轻量清洗。纯 ASCII 单字母（如 "a"）会被丢掉，避免噪声命中。"""
    _ensure_jieba()
    out: list[str] = []
    for raw in jieba.lcut(text or ""):
        tok = raw.strip()
        if not tok or tok in _STOPWORDS:
            continue
        if tok.isascii():
            if not tok.isalnum() or len(tok) == 1:
                continue
            out.append(tok.lower())
        else:
            out.append(tok)
    return out


class BM25Retriever:
    """稀疏检索。语料在构造时一次分词建索引（离线场景语料小，成本可接受）。"""

    def __init__(self, ids: Sequence[str], documents: Sequence[str]) -> None:
        self.ids = list(ids)
        self._tokenized = [tokenize(d) for d in documents]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def search(self, query: str, *, top_k: int = 20) -> list[tuple[int, float]]:
        """返回 [(语料下标, 分数)]，按分数降序，只保留分数 > 0 的（零分＝无任何词命中）。"""
        if self._bm25 is None:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in order[:top_k] if scores[i] > 0]


def rrf_fuse(rank_lists: Sequence[Sequence[int]], *, k: int = RRF_K) -> list[tuple[int, float]]:
    """多路排名融合 → [(下标, RRF 分)]，按分数降序。

    某个下标只在一路出现，也能被召回（另一路是 0 分而不是没参与）——这正是混合检索的收益：
    稀疏/稠密各有盲区，融合后互相兜底。
    """
    agg: dict[int, float] = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks, start=1):
            agg[idx] = agg.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
