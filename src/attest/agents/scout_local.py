"""T3.4 · Local Scout：本地知识库混合检索（LOC 编号，与 WEB 统一编号体系）。

与 `scout_web` 完全对称：同样接收 Send payload、同样返回 `evidence` 增量。
它不知道底层是 Chroma 还是内存向量库、有没有 BM25/RRF——那是 Ports 后面的事。
**唯一差别**是来源标签：`LOC` vs `WEB`，编号规则共用（`citations.assign_citation_ids`）。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..logging import get_logger
from ..retrieval.citations import assign_citation_ids
from .base import NodeContext

log = get_logger(__name__)
NODE = "scout_local"
TAG = "scout_local"

RESULTS_PER_QUERY = 3


def run(payload: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    sub_question = payload.get("sub_question", "")
    subq_no = int(payload.get("subq_no", 1) or 1)
    round_no = int(payload.get("round", 1) or 1)

    if ctx.local is None:
        ctx.trace.emit(
            "retrieval",
            node=NODE,
            kind="local",
            subq_no=subq_no,
            sub_question=sub_question,
            queries=[],
            hits=0,
            skipped="本地知识库未启用",
        )
        return {}

    query = sub_question.strip()
    seen = set(payload.get("seen_keys") or [])
    log.node(TAG, NODE, "开始", subq_no=subq_no, sub_question=sub_question, skip_seen=len(seen))

    errors: list[dict[str, Any]] = []
    results = []
    try:
        results = ctx.local.search(query, max_results=RESULTS_PER_QUERY + len(seen))
    except Exception as exc:  # noqa: BLE001 - 本地检索失败不该炸掉整轮
        errors.append(
            {"node": NODE, "sub_question": sub_question, "error": f"{type(exc).__name__}: {exc}"}
        )
        log.warning(f"[scout_local] 本地检索失败 | query={query!r} | {type(exc).__name__}: {exc}")

    kept = [r for r in results if (r.url or r.title) not in seen]
    skipped = len(results) - len(kept)
    limit = ctx.settings.context_truncate_chars
    trimmed = [
        r if len(r.content) <= limit else replace(r, content=r.content[:limit] + "…（正文已截断）")
        for r in kept[:RESULTS_PER_QUERY]
    ]
    evidence = assign_citation_ids(
        trimmed, round_no=round_no, subq_no=subq_no, sub_question=sub_question
    )

    ctx.trace.emit(
        "retrieval",
        node=NODE,
        kind="local",
        subq_no=subq_no,
        sub_question=sub_question,
        queries=[query],
        hits=len(evidence),
        skipped_seen=skipped,
        ids=[e.citation_id for e in evidence],
    )
    log.node(TAG, NODE, "完成", subq_no=subq_no, hits=len(evidence), skipped_seen=skipped)

    out: dict[str, Any] = {"evidence": evidence}
    if errors:
        out["errors"] = errors
    return out
