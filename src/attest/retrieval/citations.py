"""T2.2 · 引用编号器：`[WEB{轮次}-{子问题}-{序号}]` 的分配与反查。

设计要点：
  - **纯函数**：不持有全局可变注册表。并行分支各自分配自己的编号（子问题序号天然隔离），
    汇总后由 `CitationIndex` 从 state 的 evidence 列表重建索引——对 resume/重放是确定的。
  - 编号必须**稳定**：同一份检索结果重复跑，编号一致（按位置分配 → 确定）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .ports import Evidence, SearchResult, SourceKind

ID_PATTERN = re.compile(r"\[((?:WEB|LOC)\d+-\d+-\d+)\]")


def make_citation_id(kind: SourceKind, round_no: int, subq_no: int, seq: int) -> str:
    tag = "WEB" if kind == "web" else "LOC"
    return f"[{tag}{round_no}-{subq_no}-{seq}]"


def assign_citation_ids(
    results: Iterable[SearchResult],
    *,
    round_no: int,
    subq_no: int,
    sub_question: str,
) -> list[Evidence]:
    """给一批检索结果按顺序编号。序号从 1 开始。"""
    out: list[Evidence] = []
    for seq, r in enumerate(results, start=1):
        out.append(
            Evidence(
                citation_id=make_citation_id(r.source, round_no, subq_no, seq),
                source=r.source,
                title=r.title,
                url=r.url,
                content=r.content,
                sub_question=sub_question,
                round_no=round_no,
                score=r.score,
            )
        )
    return out


def extract_ids(text: str) -> list[str]:
    """抽出文本里出现过的引用编号（含方括号），保持首次出现顺序、去重。"""
    seen: dict[str, None] = {}
    for m in ID_PATTERN.finditer(text or ""):
        seen.setdefault(m.group(1), None)
    return [f"[{i}]" for i in seen]


@dataclass(frozen=True)
class CitationIndex:
    """从 evidence 列表重建的引用索引（只读）。"""

    by_id: dict[str, Evidence]

    @classmethod
    def from_evidence(cls, evidence: Iterable[Evidence]) -> "CitationIndex":
        return cls({e.citation_id: e for e in evidence})

    def resolve(self, citation_id: str) -> Evidence | None:
        return self.by_id.get(citation_id)

    def referenced_in(self, text: str) -> list[str]:
        return [i for i in extract_ids(text) if i in self.by_id]

    def unresolved_in(self, text: str) -> list[str]:
        """文本里出现、但索引里查不到的编号——T2.6「出现编号必有来源」的判据。"""
        return [i for i in extract_ids(text) if i not in self.by_id]

    def unused(self, text: str) -> list[str]:
        used = set(extract_ids(text))
        return sorted(i for i in self.by_id if i not in used)

    def __len__(self) -> int:
        return len(self.by_id)

    def __contains__(self, citation_id: str) -> bool:
        return citation_id in self.by_id


def render_reference_list(index: CitationIndex, text: str) -> str:
    """按正文出现顺序渲染文末参考资料。

    **按 URL 去重**：同一来源可能被多个子问题各自检索到，于是拿到多个编号
    （编号格式天生含子问题序号）。参考列表只列一次，取正文中先出现的那个编号，
    避免"同一篇资料出现三四遍"的观感问题。重复情况由 `citation_check` 单独计数。
    """
    lines: list[str] = []
    seen_urls: set[str] = set()
    for kid in index.referenced_in(text):
        ev = index.resolve(kid)
        if ev is None:
            continue
        key = ev.url or ev.title
        if key in seen_urls:
            continue
        seen_urls.add(key)
        lines.append(f"- `{kid}` [{ev.title}]({ev.url}) — 子问题：{ev.sub_question}")
    return "\n".join(lines)


# ------------------------------------------------------------------ 证据序列化契约
# 提示词与离线 mock 共用同一份格式，避免"两边各写一套解析"的隐性耦合。
# 行格式：`[WEB1-1-1] | 标题 | 链接 | 子问题 | 正文`（字段内的 `|` 渲染前被替换成 `／`）

EVIDENCE_MARK_BEGIN = "<<EVIDENCE>>"
EVIDENCE_MARK_END = "<<EVIDENCE_END>>"
EVIDENCE_LINE_RE = re.compile(
    r"^(\[(?:WEB|LOC)\d+-\d+-\d+\]) \| ([^|]*) \| ([^|]*) \| ([^|]*) \| (.*)$", re.MULTILINE
)

_PIPE_SAFE = str.maketrans({"|": "／", "\n": " "})


def _clean(s: str) -> str:
    return (s or "").translate(_PIPE_SAFE).strip()


def render_evidence_block(evidence: Iterable[Evidence]) -> str:
    lines = [EVIDENCE_MARK_BEGIN]
    for e in evidence:
        lines.append(
            f"{e.citation_id} | {_clean(e.title)} | {_clean(e.url)} | "
            f"{_clean(e.sub_question)} | {_clean(e.content)}"
        )
    lines.append(EVIDENCE_MARK_END)
    return "\n".join(lines)


def parse_evidence_block(text: str) -> list[dict[str, str]]:
    """把证据块解析回结构化记录（离线 mock 与测试用）。"""
    if EVIDENCE_MARK_BEGIN in text:
        body = text.split(EVIDENCE_MARK_BEGIN, 1)[1].split(EVIDENCE_MARK_END, 1)[0]
    else:
        body = text
    return [
        {
            "citation_id": m.group(1),
            "title": m.group(2),
            "url": m.group(3),
            "sub_question": m.group(4),
            "content": m.group(5),
        }
        for m in EVIDENCE_LINE_RE.finditer(body)
    ]
