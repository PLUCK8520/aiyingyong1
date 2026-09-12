"""T3.4 / T7.4 · Local Scout：本地知识库混合检索（LOC 编号，与 WEB 统一编号体系）。

与 `scout_web` 完全对称：同样接收 Send payload、同样返回 `evidence` 增量。
它不知道底层是 Chroma 还是内存向量库、有没有 BM25/RRF——那是 Ports 后面的事。
**唯一差别**是来源标签：`LOC` vs `WEB`，编号规则共用（`citations.assign_citation_ids`）。

**T7.4 起的第二路检索（研究闭环）**：
  除用户资料库（docs）外，同时检索 `research_memory`（历史调研已验证的结论）。
  历史结论被命中后**同样进入 evidence、同样编号、同样受引用审计**——
  它不是"提示"，是一个**来源**：说出来的每句话都必须能被回查。
  检索结果另写一份到 `memory_hits`（只作观测，不参与正文），便于评测"闭环有没有真被用上"。
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


def _retrieve(label: str, fn: Any, query: str, *, max_results: int) -> tuple[list, str | None]:
    """跑一路检索并把异常收进返回值——**不抛**：一路检索失败不该炸掉整轮（降级不停机，D4）。

    返回 `(结果列表, 错误描述|None)`。
    """
    try:
        return list(fn(query, max_results=max_results) or []), None
    except Exception as exc:  # noqa: BLE001 - 见 docstring
        err = f"{type(exc).__name__}: {exc}"
        log.warning(f"[scout_local] {label} 检索失败 | query={query!r} | {err}")
        return [], err


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

    # ---------- ① 用户资料库（docs）----------
    raw, err = _retrieve(
        "docs", ctx.local.search, query, max_results=RESULTS_PER_QUERY + len(seen)
    )
    if err:
        errors.append({"node": NODE, "sub_question": sub_question, "error": f"docs: {err}"})

    # ---------- ② 历史研究结论（research_memory，T7.4）----------
    # 关闭时（memory 为 None 或未启用）**如实不检索**，不假装跑过。
    mem = getattr(ctx, "memory", None)
    mem_results: list = []
    if mem is not None and getattr(ctx.settings, "research_memory_enabled", False):
        mem_results, mem_err = _retrieve(
            "research_memory",
            mem.search,
            query,
            max_results=int(getattr(ctx.settings, "research_memory_top_k", 3) or 3),
        )
        if mem_err:
            errors.append(
                {"node": NODE, "sub_question": sub_question, "error": f"research_memory: {mem_err}"}
            )

    # ---------- ③ 合并去重 → 编号 ----------
    # 去重键与 web/local 一致（url 优先，退标题）：`memory://` 与 `local://` 方案不同，
    # 天然不会互相顶掉。
    def _dedupe(results: list) -> tuple[list, int]:
        out, skip = [], 0
        for r in results:
            key = r.url or r.title
            if key in seen or key in seen_here:
                skip += 1
                continue
            seen_here.add(key)
            out.append(r)
        return out, skip

    seen_here: set[str] = set()
    docs_kept, skip_docs = _dedupe(raw)
    mem_kept, skip_mem = _dedupe(mem_results)
    skipped = skip_docs + skip_mem

    # 槽位分配（共 RESULTS_PER_QUERY 个）：**docs 优先，但给历史结论留 1 个固定位**——
    # 实测教训（2026-09-12 smoke 翻红）：memory 命中若排在 docs 前面会把 3 个槽位占满，
    # 用户资料反而一条不进；全给 docs 则闭环形同虚设。2+1 是两头都在场的最小方案。
    if mem_kept:
        docs_take = docs_kept[: RESULTS_PER_QUERY - 1]
        mem_take = mem_kept[: RESULTS_PER_QUERY - len(docs_take)]
    else:
        docs_take = docs_kept[:RESULTS_PER_QUERY]
        mem_take = []
    pool = docs_take + mem_take

    limit = ctx.settings.context_truncate_chars
    trimmed = [
        r if len(r.content) <= limit else replace(r, content=r.content[:limit] + "…（正文已截断）")
        for r in pool
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
        docs_hits=len(raw),
        memory_hits=len(mem_results),
        skipped_seen=skipped,
        ids=[e.citation_id for e in evidence],
    )
    log.node(
        TAG,
        NODE,
        "完成",
        subq_no=subq_no,
        hits=len(evidence),
        docs=len(raw),
        memory=len(mem_results),
        skipped_seen=skipped,
    )

    out: dict[str, Any] = {"evidence": evidence}
    # memory_hits 只作观测（谁被复用了），不参与正文——正文只认 evidence。
    if mem_results:
        out["memory_hits"] = [
            {
                "sub_question": sub_question,
                "report_id": r.title,
                "url": r.url,
                "score": r.score,
                "in_context": any(e.url == r.url for e in evidence),
            }
            for r in mem_results
        ]
    if errors:
        out["errors"] = errors
    return out

