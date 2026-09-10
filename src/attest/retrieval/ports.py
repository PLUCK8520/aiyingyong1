"""④ 服务层 · Ports 定义（依赖倒置的落点）。

硬原则：**依赖只能向下**。能力层（agents）只依赖这里的 Protocol，不依赖任何具体实现；
生产 Adapter 与测试 Adapter 是**同等的**实现（Mock 是正式 Adapter，不是补丁）。

约定：所有外部调用（LLM / 搜索 / Rerank / 向量库）都必须经端口注入，禁止在节点里直接 import SDK。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

SourceKind = Literal["web", "local"]


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


@runtime_checkable
class SearchClient(Protocol):
    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, docs: list[str], *, top_n: int = 10) -> list[tuple[int, float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, ids: list[str], vectors: list[list[float]], documents: list[str], metadatas: list[dict]) -> None: ...

    def query(self, vector: list[float], *, top_k: int = 10) -> list[SearchResult]: ...


@runtime_checkable
class Store(Protocol):
    """键值/关系存储（检查点、画像、research_memory）。"""

    def put(self, namespace: str, key: str, value: dict) -> None: ...

    def get(self, namespace: str, key: str) -> dict | None: ...
