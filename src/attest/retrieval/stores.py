"""T3.1 · 向量库双实现：内存余弦（离线默认） / Chroma 持久化（验收档）。

**Chroma 三个静默坑**（P-1 已实测①，见 docs/env-report.md 与任务清单 T3.1）：
  1. 必须显式 `embedding_function=None` 并自传向量——否则首次调用会**静默下载约 200MB**
     的 onnxruntime 模型，无进度提示，看起来像卡死；
  2. `PersistentClient` 不支持多进程并发（SQLite 文件锁）→ 本模块保持**单例 client**；
  3. 设 `ANONYMIZED_TELEMETRY=False` 关闭默认外发。

`embedder` + `dim` 写进 collection 元数据：**换 embedding 模型必须全量重建索引**，
不一致时直接抛中文错，而不是返回两套向量空间混算出来的脏结果（NFR-11）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 必须在 import chromadb 之前关掉埋点
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from ..logging import get_logger  # noqa: E402
from .ports import Hit  # noqa: E402

log = get_logger(__name__)


@dataclass
class NumpyVectorStore:
    """内存余弦检索。离线默认档：无 IO、无文件锁，测试与 mock 链路首选。"""

    embedder: str
    dim: int
    _ids: list[str] = field(default_factory=list, init=False)
    _docs: list[str] = field(default_factory=list, init=False)
    _metas: list[dict] = field(default_factory=list, init=False)
    _mat: list[list[float]] = field(default_factory=list, init=False)

    def upsert(self, ids, vectors, documents, metadatas) -> None:
        import numpy as np

        for i, v in enumerate(vectors):
            if len(v) != self.dim:
                raise ValueError(
                    f"向量维度不一致：期望 {self.dim}（embedder={self.embedder}），实得 {len(v)}。"
                    "换 embedding 模型必须全量重建索引。"
                )
        self._ids.extend(ids)
        self._docs.extend(documents)
        self._metas.extend(metadatas)
        mat = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._mat.extend((mat / norms).tolist())

    def query(self, vector, *, top_k: int = 10, where: dict | None = None) -> list[Hit]:
        import numpy as np

        if not self._mat:
            return []
        q = np.asarray(vector, dtype="float32")
        n = float(np.linalg.norm(q)) or 1.0
        q = q / n
        mat = np.asarray(self._mat, dtype="float32")
        sims = mat @ q
        order = np.argsort(-sims)
        out: list[Hit] = []
        for i in order:
            meta = self._metas[int(i)]
            if where and any(meta.get(k) != v for k, v in where.items()):
                continue
            out.append(
                Hit(
                    chunk_id=self._ids[int(i)],
                    score=float(sims[int(i)]),
                    document=self._docs[int(i)],
                    metadata=meta,
                )
            )
            if len(out) >= top_k:
                break
        return out

    def count(self) -> int:
        return len(self._ids)

    def dump(self) -> tuple[list[str], list[str], list[dict]]:
        """导出 (ids, documents, metadatas)——VectorStore 协议要求的恢复路径。

        T7.4 之前只有 Chroma 实现了它（`build_local_index` 的"文档目录缺失"恢复路径
        只发生在 chroma 档），NumpyVectorStore 漏了——T7.4 的冷启动恢复
        （`ResearchMemoryStore.search`）把这条缺口暴露了出来，此处补齐。
        """
        return list(self._ids), list(self._docs), [dict(m) for m in self._metas]


@dataclass
class ChromaVectorStore:
    """Chroma 持久化向量库。单例 client + 自传向量 + 关闭 telemetry。"""

    path: Path
    collection: str = "docs"
    embedder: str = "unknown"
    dim: int = 0
    #: 显式重建：embedder/dim 变化时本应报错；`ingest_local.py --rebuild` 传 True 表示"知道要重建"
    rebuild: bool = False
    _client: object = field(default=None, init=False, repr=False)
    _col: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._open(force_rebuild=self.rebuild)

    def _open(self, *, force_rebuild: bool = False) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.path.mkdir(parents=True, exist_ok=True)
        # 单例 client：PersistentClient 不支持多进程并发（SQLite 文件锁）
        self._client = chromadb.PersistentClient(
            path=str(self.path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        want = {"embedder": self.embedder, "dim": self.dim, "hnsw:space": "cosine"}
        col = None
        try:
            col = self._client.get_collection(name=self.collection, embedding_function=None)
        except Exception:  # noqa: BLE001 - 不存在就是 None，交由下面创建
            col = None
        if col is not None:
            md = col.metadata or {}
            if (md.get("embedder"), md.get("dim")) != (self.embedder, self.dim):
                if not force_rebuild:
                    raise ValueError(
                        f"Chroma collection {self.collection!r} 的向量空间与当前配置不符：\n"
                        f"  已有 embedder={md.get('embedder')} dim={md.get('dim')}\n"
                        f"  当前 embedder={self.embedder} dim={self.dim}\n"
                        "  换 embedding 模型必须全量重建索引："
                        "scripts/ingest_local.py --rebuild（或删除 data/index/chroma）"
                    )
                log.warning("[chroma] embedder/dim 变化，按 --rebuild 重建 collection")
                self._client.delete_collection(self.collection)
                col = None
        if col is None:
            # 关键：embedding_function=None，向量全部自传（否则静默下载 ~200MB 模型）
            col = self._client.create_collection(
                name=self.collection, embedding_function=None, metadata=want
            )
        self._col = col

    def reset(self) -> None:
        self._client.delete_collection(self.collection)
        self._col = None
        self._open()

    def upsert(self, ids, vectors, documents, metadatas) -> None:
        if not ids:
            return
        for v in vectors:
            if self.dim and len(v) != self.dim:
                raise ValueError(
                    f"向量维度不一致：期望 {self.dim}（embedder={self.embedder}），实得 {len(v)}。"
                )
        self._col.upsert(ids=list(ids), embeddings=list(vectors), documents=list(documents), metadatas=list(metadatas))

    def query(self, vector, *, top_k: int = 10, where: dict | None = None) -> list[Hit]:
        res = self._col.query(
            query_embeddings=[list(vector)],
            n_results=max(1, top_k),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return [
            Hit(
                chunk_id=cid,
                score=round(1.0 - float(dist), 6),  # hnsw:space=cosine → similarity = 1 - distance
                document=docs[i] if i < len(docs) else "",
                metadata=metas[i] if i < len(metas) else {},
            )
            for i, cid in enumerate(ids)
            for dist in [dists[i] if i < len(dists) else 1.0]
        ]

    def count(self) -> int:
        return int(self._col.count())

    def dump(self) -> tuple[list[str], list[str], list[dict]]:
        res = self._col.get(include=["documents", "metadatas"])
        ids = list(res.get("ids") or [])
        docs = [d or "" for d in (res.get("documents") or [])]
        metas = [m or {} for m in (res.get("metadatas") or [])]
        return ids, docs, metas
