"""T3.4 支撑 · 本地知识库混合检索客户端（BM25 + 向量 → RRF → rerank）。

对外只暴露 `SearchClient` 协议（`search(query, max_results)`），所以 `scout_local` 节点
既不知道底层是 Chroma 还是内存向量库，也不知道 BM25 的存在——与 `scout_web` 完全对称。
这是"依赖倒置"在这里的实际收益：换检索后端，节点代码零改动。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from ..logging import get_logger
from .hybrid import BM25Retriever, rrf_fuse, tokenize
from .ports import Hit, Reranker, SearchResult, VectorStore
from .text import load_chunks
from .stores import ChromaVectorStore, NumpyVectorStore

log = get_logger(__name__)

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


def local_url(source_path: str, chunk_id: str) -> str:
    """本地文档的可回查地址（`local://` 方案，避免假造 http 链接）。"""
    return f"local://{source_path}#{chunk_id}"


@dataclass
class LocalIndex:
    ids: list[str]
    documents: list[str]
    metadatas: list[dict]
    bm25: BM25Retriever
    store: VectorStore
    embedder: str
    dim: int
    _pos: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._pos = {cid: i for i, cid in enumerate(self.ids)}

    def position(self, chunk_id: str) -> int | None:
        return self._pos.get(chunk_id)


def build_local_index(
    root: Path,
    embed_fn: EmbedFn,
    *,
    store: VectorStore,
    embedder: str,
    dim: int,
    chunk_chars: int = 800,
    chunk_overlap: int = 120,
) -> LocalIndex:
    """装载 `root` 下文档 → 分块 → 向量入库（若库为空）→ 建 BM25。"""
    chunks = load_chunks(root, size=chunk_chars, overlap=chunk_overlap)
    ids = [c.chunk_id for c in chunks]
    docs = [c.text for c in chunks]
    metas = [
        {
            "chunk_id": c.chunk_id,
            "source_path": c.source_path,
            "title": c.title,
            "ordinal": c.ordinal,
            "embedder": embedder,
            "dim": dim,
        }
        for c in chunks
    ]

    if not chunks and store.count() > 0:
        # 文档目录不在，但持久索引还在（例：只部署了索引的机器）——从向量库恢复元数据，
        # 否则 BM25 会退化成空索引，混合检索名存实亡。
        ids, docs, metas = store.dump()
        log.info(f"[local] 文档目录缺失，从已有向量库恢复 {len(ids)} 个 chunk 元数据")

    if store.count() == 0:
        if not chunks:
            log.warning(f"[local] 文档目录为空，未建立任何本地索引：{root}")
        else:
            vectors = embed_fn(docs)
            store.upsert(ids, vectors, docs, metas)
            log.info(f"[local] 建立本地索引：{len(chunks)} 个 chunk，embedder={embedder} dim={dim}")

    return LocalIndex(
        ids=ids,
        documents=docs,
        metadatas=metas,
        bm25=BM25Retriever(ids, docs),
        store=store,
        embedder=embedder,
        dim=dim,
    )


@dataclass
class LocalSearchClient:
    index: LocalIndex
    embed_fn: EmbedFn
    reranker: Reranker | None = None
    candidate_n: int = 20

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        if not self.index.ids:
            return []

        # ① 稀疏：jieba 分词后的 BM25
        sparse = [i for i, _ in self.index.bm25.search(query, top_k=self.candidate_n)]
        # ② 稠密：向量召回（embedding_function=None，查询向量自传）
        qvec = self.embed_fn([query])[0]
        dense_hits: list[Hit] = self.index.store.query(qvec, top_k=self.candidate_n)
        dense = [p for h in dense_hits if (p := self.index.position(h.chunk_id)) is not None]
        hit_score = {h.chunk_id: h.score for h in dense_hits}

        # ③ RRF 融合（只用排名，无需归一化两种分数）
        fused = rrf_fuse([sparse, dense])[: self.candidate_n]
        if not fused:
            return []

        cand_idx = [i for i, _ in fused]
        cand_score = {i: s for i, s in fused}
        docs = [self.index.documents[i] for i in cand_idx]

        # ④ 精排（失败不致命：退回融合排序并留痕——降级不停机，D4）
        if self.reranker is not None and len(cand_idx) > 1:
            try:
                top = self.reranker.rerank(query, docs, top_n=max_results)
                picked = [(cand_idx[i], score) for i, score in top]
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    f"[local] rerank 失败，退回 RRF 融合排序 | {type(exc).__name__}: {exc}"
                )
                picked = [(i, cand_score[i]) for i in cand_idx[:max_results]]
        else:
            picked = [(i, cand_score[i]) for i in cand_idx[:max_results]]

        out: list[SearchResult] = []
        for idx, score in picked:
            meta = self.index.metadatas[idx]
            cid = self.index.ids[idx]
            out.append(
                SearchResult(
                    source="local",
                    title=str(meta.get("title") or "(本地文档)"),
                    url=local_url(str(meta.get("source_path", "")), cid),
                    content=self.index.documents[idx],
                    # 融合分与向量分都很小，展示时放大到可读区间；只用于排序观测，不参与判定
                    score=round(float(score), 6),
                )
            )
        log.debug(
            f"[local] query={query!r} 稀疏={len(sparse)} 稠密={len(dense)} 融合={len(fused)} "
            f"输出={len(out)}"
        )
        return out


# ------------------------------------------------------------------ 装配（带缓存）

_INDEX_CACHE: dict[tuple[str, str, str], LocalIndex] = {}


def make_local_client(
    *,
    docs_dir: Path,
    embed_fn: EmbedFn,
    embedder: str,
    dim: int,
    use_chroma: bool = False,
    chroma_dir: Path | None = None,
    reranker: Reranker | None = None,
    chunk_chars: int = 800,
    chunk_overlap: int = 120,
) -> LocalSearchClient:
    """装配本地检索客户端。索引按 (目录, embedder, 后端) 缓存，避免每轮重复分块与嵌入。"""
    kind = "chroma" if (use_chroma and chroma_dir) else "numpy"
    key = (str(docs_dir), embedder, kind)
    index = _INDEX_CACHE.get(key)
    if index is None:
        if kind == "chroma":
            assert chroma_dir is not None
            store: VectorStore = ChromaVectorStore(
                path=chroma_dir, collection="docs", embedder=embedder, dim=dim
            )
        else:
            store = NumpyVectorStore(embedder=embedder, dim=dim)
        index = build_local_index(
            docs_dir,
            embed_fn,
            store=store,
            embedder=embedder,
            dim=dim,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
        )
        _INDEX_CACHE[key] = index
    return LocalSearchClient(index=index, embed_fn=embed_fn, reranker=reranker)


def clear_local_cache() -> None:
    _INDEX_CACHE.clear()


__all__ = [
    "LocalIndex",
    "LocalSearchClient",
    "build_local_index",
    "clear_local_cache",
    "local_url",
    "make_local_client",
    "tokenize",
]
