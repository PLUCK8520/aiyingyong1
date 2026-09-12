"""④ 服务层 · Ports 定义（依赖倒置的落点）。

硬原则：**依赖只能向下**。能力层（agents）只依赖这里的 Protocol，不依赖任何具体实现；
生产 Adapter 与测试 Adapter 是**同等的**实现（Mock 是正式 Adapter，不是补丁）。

约定：所有外部调用（LLM / 搜索 / Rerank / 向量库）都必须经端口注入，禁止在节点里直接 import SDK。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

#: `memory` = 历史调研沉淀的研究结论（T7.4）。它与 `local` 走同一路检索、同一套编号，
#: 但来源语义不同（我们自己的结论 vs 用户的资料），评测要能分开统计，故单列一档。
#: ⚠️ 编号仍映射到 `LOC`（`citations.make_citation_id` 只认 web/非 web 两档），
#: 这是刻意的：正文里不应出现"MEM"这种让读者莫名其妙的前缀。
SourceKind = Literal["web", "local", "memory"]


@dataclass(frozen=True)
class SearchResult:
    """一次检索返回的原始结果（未编号）。"""

    source: SourceKind
    title: str
    url: str
    content: str
    score: float = 0.0


@dataclass(frozen=True)
class Evidence:
    """带引用编号的证据。编号规则见 `retrieval/citations.py`。"""

    citation_id: str
    source: SourceKind
    title: str
    url: str
    content: str
    sub_question: str
    round_no: int
    score: float = 0.0


@dataclass(frozen=True)
class Hit:
    """向量库召回结果（T3.1，《功能设计》§5 的 `Hit`）。"""

    chunk_id: str
    score: float
    document: str
    metadata: dict


@runtime_checkable
class SearchClient(Protocol):
    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, docs: list[str], *, top_n: int = 10) -> list[tuple[int, float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    """向量库（T3.1）。

    实现有两档，**接口一致**：
      - `NumpyVectorStore`：内存余弦检索，离线默认档（无 IO、无锁，测试友好）
      - `ChromaVectorStore`：Chroma 持久化，`ingest_local.py` 与验收档

    切换 embedding 模型必须全量重建索引——`embedder` / `dim` 写进 collection 元数据，
    不一致时**直接抛错**而不是返回脏结果（NFR-11）。
    """

    def upsert(self, ids: list[str], vectors: list[list[float]], documents: list[str], metadatas: list[dict]) -> None: ...

    def query(self, vector: list[float], *, top_k: int = 10, where: dict | None = None) -> list[Hit]: ...

    def count(self) -> int: ...

    def delete(self, ids: list[str]) -> None:
        """按 id 删除——T7.4 的 purge_report（按报告撤沉淀）需要它，
        否则撤除只能停留在"内存里看不见"，底层向量还在，重启即复活。"""
        ...

    def dump(self) -> tuple[list[str], list[str], list[dict]]:
        """导出 (ids, documents, metadatas)——用于"文档目录缺失但索引仍在"的恢复路径。"""
        ...


@runtime_checkable
class Store(Protocol):
    """键值/关系存储（检查点、画像、research_memory）。"""

    def put(self, namespace: str, key: str, value: dict) -> None: ...

    def get(self, namespace: str, key: str) -> dict | None: ...
