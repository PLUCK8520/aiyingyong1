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


#: 编号的规范内核（不含方括号）。大小写不敏感——模型偶尔写成小写。
CID_CORE_RE = re.compile(r"(?:WEB|LOC|MEM)\d+-\d+-\d+", re.IGNORECASE)

#: 规范化时要剥掉的包裹字符：半角/全角括号、中文角括号、书名号、引号、空白。
_CID_TRIM = " \t\r\n[]()（）【】<>《》「」\"'`"


def normalize_citation_id(raw: object) -> str:
    """把引用编号收敛成规范形态 `[KIND{轮次}-{子问题}-{序号}]`。

    **为什么必须有这一层**（2026-09-13 真实档实测，P0 缺陷的修复点）：
    提示词、证据块、`make_citation_id()` 产出的编号一律是 `[LOC1-1-1]`（**带方括号**），
    但模型回填 `citation_id` 时会**把方括号丢掉**，写成 `LOC1-1-1`。下游是**字符串精确匹配**
    （`analyst.assess_evidence` 里的 `relevance.get(e.citation_id)`），于是：

        判别器给出 relevance=5 → 挂在键 `LOC1-1-1` 上
        → evidence 的 `[LOC1-1-1]` 永远取不到分 → `n_selected=0` → 产出拒编页

    实测数字（thread `kb-e2e-1`，可从 checkpoint 复现）：
    `judgments` 56 条、`evidence` 72 条，**编号交集 0**；去掉方括号后交集 **56/56**。
    最恶劣的地方在于**失败被伪装成正常行为**——界面上是一份措辞得体的"证据不足，故拒编"，
    完全符合零造假铁律的叙事，而实际上是全盘误判：判别器明明把知识库里的文档判成了 5 分。

    这个函数用在两处，缺一不可：
      - **校验层**（`schemas.py` 的 `field_validator`）：挡住新产生的 LLM 输出；
      - **消费层**（`evidence_judge` / `analyst`）：兜住 checkpoint 里**已经存下的**脏数据，
        否则用户"续跑"一个旧会话时缺陷依旧复现。

    非编号输入原样返回（不猜、不构造），保证幂等且不会把无关文本变成编号。
    """
    s = str(raw if raw is not None else "").strip().strip(_CID_TRIM).strip()
    if not s:
        return ""
    if CID_CORE_RE.fullmatch(s):
        return f"[{s.upper()}]"
    # 编号被整句包住（例："来源：LOC1-1-1"）→ 抽出第一个。宁可不改也不猜。
    m = CID_CORE_RE.search(s)
    return f"[{m.group(0).upper()}]" if m else s


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


# ------------------------------------------------------------------ 矛盾对序列化契约
# 与证据块同理：提示词与离线 mock 共用一份定义。
# 行格式：`[WEB1-1-1] || [WEB1-1-3] || 主题`
# 只传编号对 + 一份证据块，避免把正文重复送进上下文（矛盾检测是 O(对数) 的成本敏感步骤）。

PAIRS_MARK_BEGIN = "<<PAIRS>>"
PAIRS_MARK_END = "<<PAIRS_END>>"
PAIR_LINE_RE = re.compile(
    r"^(\[(?:WEB|LOC)\d+-\d+-\d+\]) \|\| (\[(?:WEB|LOC)\d+-\d+-\d+\]) \|\| (.*)$", re.MULTILINE
)


def render_pairs_block(pairs: Iterable[tuple[str, str, str]]) -> str:
    """pairs: (citation_id_a, citation_id_b, topic) 序列。"""
    lines = [PAIRS_MARK_BEGIN]
    for a, b, topic in pairs:
        lines.append(f"{a} || {b} || {_clean(topic)}")
    lines.append(PAIRS_MARK_END)
    return "\n".join(lines)


def parse_pairs_block(text: str) -> list[dict[str, str]]:
    if PAIRS_MARK_BEGIN in text:
        body = text.split(PAIRS_MARK_BEGIN, 1)[1].split(PAIRS_MARK_END, 1)[0]
    else:
        body = text
    return [
        {"id_a": m.group(1), "id_b": m.group(2), "topic": m.group(3)}
        for m in PAIR_LINE_RE.finditer(body)
    ]


# ------------------------------------------------------------------ 矛盾结果序列化契约
# 行格式：`- 主题: {topic} | A: {id_a} {claim_a} | B: {id_b} {claim_b} | 严重度: {severity}`

CONFLICTS_MARK_BEGIN = "<<CONFLICTS>>"
CONFLICTS_MARK_END = "<<CONFLICTS_END>>"
CONFLICT_LINE_RE = re.compile(
    r"^- 主题: (.*?) \| A: (\[(?:WEB|LOC)\d+-\d+-\d+\]) (.*?) \| "
    r"B: (\[(?:WEB|LOC)\d+-\d+-\d+\]) (.*?) \| 严重度: (\w+)$",
    re.MULTILINE,
)


def render_conflicts_block(conflicts: Iterable["Conflict"]) -> str:  # noqa: F821
    from ..schemas import Conflict as _Conflict  # 局部导入：避免根命名空间在模块导入期被拉起

    lines = [CONFLICTS_MARK_BEGIN]
    for c in conflicts:
        if not isinstance(c, _Conflict):
            continue
        lines.append(
            f"- 主题: {_clean(c.topic)} | A: {c.source_a} {_clean(c.claim_a)} | "
            f"B: {c.source_b} {_clean(c.claim_b)} | 严重度: {c.severity}"
        )
    lines.append(CONFLICTS_MARK_END)
    return "\n".join(lines)


def parse_conflicts_block(text: str) -> list[dict[str, str]]:
    if CONFLICTS_MARK_BEGIN in text:
        body = text.split(CONFLICTS_MARK_BEGIN, 1)[1].split(CONFLICTS_MARK_END, 1)[0]
    else:
        return []
    return [
        {
            "topic": m.group(1),
            "id_a": m.group(2),
            "claim_a": m.group(3),
            "id_b": m.group(4),
            "claim_b": m.group(5),
            "severity": m.group(6),
        }
        for m in CONFLICT_LINE_RE.finditer(body)
    ]
