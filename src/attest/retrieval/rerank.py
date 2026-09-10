"""T3.3 · Rerank。

主选 **qwen3.7-text-rerank**（官方标注 recommended，单条 30,000 token / 请求 120,000）。
备用 **qwen3-rerank** 的 endpoint 与请求体结构**不同**，且单条上限仅 4,000 token、
**超限直接 HTTP 400 不截断**——所以 chunk 必须 ≤ 4000 token（T3.1 已按 800 字符控制）。
gte-rerank 已于 2026-05-30 下线，勿用。

离线档用 `LexicalReranker`（jieba 词重叠打分）：它**不是** rerank 模型，是一个确定性的
词典兜底，目的是让"精排"这一步在无 key 时也能跑通并有可解释行为。
真实档 `DashScopeReranker` 代码完整，但**本机无 key，未实跑**（详见交付说明的未验证清单）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import httpx

from ..logging import get_logger
from .hybrid import tokenize

log = get_logger(__name__)

#: qwen3.7-text-rerank（推荐档）；备用 qwen3-rerank 的 endpoint 见类文档
RERANK_PATH = "/api/v1/services/rerank/text-rerank/text-rerank"
RERANK_MODEL = "qwen3.7-text-rerank"
RERANK_MODEL_BACKUP = "qwen3-rerank"
RERANK_MAX_DOC_TOKENS = 30_000
RERANK_BACKUP_MAX_DOC_TOKENS = 4_000


@dataclass
class LexicalReranker:
    """离线兜底：查询词 → 文档词的覆盖度。确定性、可解释、零成本。"""

    name: str = "lexical"

    def rerank(self, query: str, docs: Sequence[str], *, top_n: int = 10) -> list[tuple[int, float]]:
        q = set(tokenize(query))
        if not q:
            return [(i, 0.0) for i in range(min(top_n, len(docs)))]
        scored: list[tuple[int, float]] = []
        for i, doc in enumerate(docs):
            d = set(tokenize(doc))
            cover = len(q & d) / len(q)
            # 轻微偏好较长文本（信息量更大），但权重远低于覆盖度，避免长文霸榜
            scored.append((i, round(cover + 0.01 * min(len(doc), 800) / 800, 6)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]


@dataclass
class DashScopeReranker:
    """真实 Rerank（未实跑：本机无 key）。

    注意 endpoint 与请求体与 qwen3-rerank **不同**：本类按主选模型
    `qwen3.7-text-rerank` 的 `/api/v1/services/rerank/text-rerank/text-rerank` 构造。
    若日后改用备用模型，需换 endpoint 为
    `{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-api/v1/reranks` 并自行保证
    单条 ≤ 4000 token（超限是 400，不是截断）。
    """

    api_key: str
    base_url: str = "https://dashscope.aliyuncs.com"
    model: str = RERANK_MODEL
    name: str = "dashscope-rerank"
    timeout: float = 30.0

    def rerank(self, query: str, docs: Sequence[str], *, top_n: int = 10) -> list[tuple[int, float]]:
        payload = {
            "model": self.model,
            "input": {"query": query, "documents": [{"text": d} for d in docs]},
            "parameters": {"return_documents": False, "top_n": top_n},
        }
        resp = httpx.post(
            f"{self.base_url.rstrip('/')}{RERANK_PATH}",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        results = ((resp.json().get("output") or {}).get("results")) or []
        return [(int(r["index"]), float(r["relevance_score"])) for r in results]


def truncate_for_rerank(docs: Sequence[str], *, backup: bool = False) -> list[str]:
    """按 rerank 单条上限截断（字符数≈token 数的保守估计：CJK 1 字 1 token）。"""
    limit = RERANK_BACKUP_MAX_DOC_TOKENS if backup else RERANK_MAX_DOC_TOKENS
    return [d if len(d) <= limit else d[:limit] for d in docs]
