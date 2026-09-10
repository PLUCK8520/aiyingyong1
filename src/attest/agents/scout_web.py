"""T2.3 · Web Scout：每个子问题生成 2 个查询词 → 检索 → 截断 → 编号。

这个节点是 Send 扇出的**目标节点**，所以它收到的是 Send payload（子问题包），不是全量 state。
它不知道也不关心检索来自 Tavily 还是离线 fixture——那是 Adapter 的事。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..logging import get_logger
from ..retrieval.citations import assign_citation_ids
from ..retrieval.ports import SearchResult
from .base import NodeContext

log = get_logger(__name__)
NODE = "scout_web"
TAG = "scout"

RESULTS_PER_QUERY = 3


def make_queries(sub_question: str, objective: str) -> list[str]:
    """确定性生成 2 个查询词：子问题本体 + 叠加调研目标的首个限定词。"""
    queries = [sub_question.strip()]
    head = (objective or "").replace("调研", "").strip()
    head = head.split("，")[0].split(",")[0][:20].strip()
    if head and head not in sub_question:
        queries.append(f"{head} {sub_question}")
    return queries[:2]


def _truncate(r: SearchResult, limit: int) -> SearchResult:
    if len(r.content) <= limit:
        return r
    return replace(r, content=r.content[:limit] + "…（正文已截断）")


def run(payload: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    sub_question = payload.get("sub_question", "")
    subq_no = int(payload.get("subq_no", 1) or 1)
    round_no = int(payload.get("round", 1) or 1)
    objective = payload.get("objective", "")

    queries = make_queries(sub_question, objective)
    log.node(TAG, NODE, "开始", subq_no=subq_no, sub_question=sub_question)

    merged: dict[str, SearchResult] = {}
    errors: list[dict[str, Any]] = []
    for q in queries:
        try:
            for r in ctx.search.search(q, max_results=RESULTS_PER_QUERY):
                merged.setdefault(r.url or r.title, r)
        except Exception as exc:  # noqa: BLE001 - 单路检索失败不该炸掉整轮
            errors.append({"node": NODE, "sub_question": sub_question, "error": f"{type(exc).__name__}: {exc}"})
            log.warning(f"[scout] 检索失败 | query={q!r} | {type(exc).__name__}: {exc}")

    limit = ctx.settings.context_truncate_chars
    truncated = [_truncate(r, limit) for r in merged.values()]
    evidence = assign_citation_ids(
        truncated, round_no=round_no, subq_no=subq_no, sub_question=sub_question
    )

    ctx.trace.emit(
        "retrieval",
        node=NODE,
        subq_no=subq_no,
        sub_question=sub_question,
        queries=queries,
        hits=len(evidence),
        ids=[e.citation_id for e in evidence],
    )
    log.node(TAG, NODE, "完成", subq_no=subq_no, queries=len(queries), hits=len(evidence))

    out: dict[str, Any] = {"evidence": evidence}
    if errors:
        out["errors"] = errors
    return out
