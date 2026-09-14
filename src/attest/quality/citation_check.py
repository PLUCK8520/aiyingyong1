"""T2.6 · 报告后处理与引用完整性校验。

判据（T2.6 验收）：**出现编号必有来源**。所以校验分三问：
  1. 正文里的编号，索引里查得到吗？→ `unresolved`（必须为空）
  2. 索引里的来源，正文用上了吗？→ `unused`（多检索到的正常，但值得观测）
  3. 有引用的句子占比 → `cited_ratio`（P3 的"引用有效率"先埋这里）

参考资料章节由本模块**统一追加**，不让模型自己写——避免编号漂移与重复。
"""

from __future__ import annotations

from typing import Any

from ..retrieval.citations import CitationIndex, extract_ids, render_reference_list

REF_HEADING = "## 参考资料"
EMPTY_REF_TEXT = "（本轮未产生可引用来源）"


def check_report(report: str, index: CitationIndex) -> dict[str, Any]:
    referenced = index.referenced_in(report)
    unresolved = index.unresolved_in(report)
    unused = index.unused(report)

    # 引用有效率（粗口径）：含有编号的段落数 / 总段落数
    paragraphs = [p for p in (report or "").split("\n\n") if p.strip()]
    cited_paragraphs = [p for p in paragraphs if extract_ids(p)]

    # 同一 URL 拿到多个编号（跨子问题重复检索命中同一来源）——观测项，用于后续去重优化
    url_counts: dict[str, int] = {}
    for ev in index.by_id.values():
        key = ev.url or ev.title
        url_counts[key] = url_counts.get(key, 0) + 1
    duplicates = {u: c for u, c in url_counts.items() if c > 1}

    return {
        "pass": not unresolved and (bool(referenced) or len(index) == 0),
        "referenced": len(referenced),
        "unresolved": unresolved,
        "unused": unused,
        "evidence_total": len(index),
        # T8.6：正文**一个编号都没有**，而索引里明明有证据 —— 引用契约整体失效。
        # 必须单独标出来：这不是"引用有瑕疵"，而是"逐句回查对这份报告不适用"。
        # 实测（2026-09-13，glm-4-flash）正文 0 编号时，`unresolved` 与
        # `unsupported` 会**双双为 0**，审计摘要看起来一片干净——凭空生成的内容
        # 反而比"有编号但引错"更安全地混过去。`pass` 虽已为 False，
        # 但只有这个显式标志能让审计/前端做出正确处置（见 `auditor` 的未回查横幅）。
        "no_citations": bool(len(index)) and not referenced,
        "unique_sources": len(url_counts),
        "duplicate_sources": duplicates,
        "cited_paragraphs": len(cited_paragraphs),
        "paragraphs": len(paragraphs),
        "cited_ratio": round(len(cited_paragraphs) / len(paragraphs), 4) if paragraphs else 0.0,
    }


def finalize(report: str, index: CitationIndex) -> tuple[str, str, dict[str, Any]]:
    """返回 (含参考资料的最终报告, 参考资料文本, 校验结果)。"""
    check = check_report(report, index)
    refs = render_reference_list(index, report)

    body = (report or "").rstrip()
    if REF_HEADING in body:
        body = body.split(REF_HEADING, 1)[0].rstrip()

    tail = refs if refs else EMPTY_REF_TEXT
    final = f"{body}\n\n{REF_HEADING}\n\n{tail}\n"
    return final, refs, check
