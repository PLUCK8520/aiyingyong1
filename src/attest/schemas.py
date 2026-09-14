"""共享领域契约（Pydantic）。

放在包根、与 `config.py` 同级，理由：这些模型被 **④ llm 层**（结构化输出的目标类型）与
**③ agents 层**（节点产出）同时使用。若放到 agents/ 里，llm/ 就得反向 import，违反"依赖只能向下"。
（`架构设计.md` §11 只列了 config.py，本文件是同类共享内核，待下版文档补齐。）
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .retrieval.citations import normalize_citation_id

Route = Literal["direct", "research"]


class RouteDecision(BaseModel):
    """T1.2 意图分流输出。"""

    route: Route
    reason: str = Field("", description="判定理由，一句话")


class Plan(BaseModel):
    """T1.4 任务规划输出。"""

    objective: str
    sub_questions: list[str] = Field(min_length=1, max_length=8)
    outlines: list[str] = Field(min_length=1, max_length=8)
    requires_data: bool = False

    @field_validator("sub_questions", "outlines")
    @classmethod
    def _dedup_and_strip(cls, v: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for item in v:
            s = (item or "").strip()
            if s:
                seen.setdefault(s, None)
        if not seen:
            raise ValueError("不能为空")
        return list(seen)


class Judgment(BaseModel):
    """T2.4 单条证据的判别结果。"""

    citation_id: str
    sub_question: str
    relevance: int = Field(ge=1, le=5, description="与子问题的相关度 1-5")
    confidence: int = Field(ge=1, le=5, description="内容可信度 1-5")
    note: str = ""

    @field_validator("citation_id", mode="before")
    @classmethod
    def _normalize_citation_id(cls, v: object) -> str:
        """收敛模型输出的编号写法（模型常常丢方括号 → 见 `normalize_citation_id` 的说明）。

        用 `mode="before"`：模型可能回非字符串（数字/None），必须在进入类型校验前收敛。
        """
        return normalize_citation_id(v)


class JudgeResult(BaseModel):
    """T2.4 证据判别整体输出。"""

    judgments: list[Judgment] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list, description="证据不足的缺口，供 Reflect 补检")


Severity = Literal["low", "medium", "high"]


class Conflict(BaseModel):
    """T3.7 矛盾检测（字段对齐《功能设计》§6.3）。

    注意 `source_a` / `source_b` 存的是**引用编号**（如 `[WEB1-1-1]`），不是 URL——
    编号才可回查，URL 会随引用去重变化。`claim_*` 存双方各自的口径原话（截断）。
    """

    sub_question: str
    topic: str
    claim_a: str
    source_a: str
    claim_b: str
    source_b: str
    severity: Severity = "medium"
    summary: str = ""

    @field_validator("source_a", "source_b", mode="before")
    @classmethod
    def _normalize_sources(cls, v: object) -> str:
        """`source_a/b` 也是引用编号，同样会被模型丢掉方括号。

        矛盾清单里的编号要能和 `CitationIndex` 对上，否则报告「争议与分歧」章节
        会给出查不到的编号——比不写编号更糟（看起来可回查，实际查不到）。
        """
        return normalize_citation_id(v)


class ConflictResult(BaseModel):
    """T3.7 矛盾检测整体输出。"""

    conflicts: list[Conflict] = Field(default_factory=list)


AuditVerdict = Literal["supported", "partial", "unsupported"]


class AuditItem(BaseModel):
    """T4.1 引用审计的单条结果。"""

    citation_id: str
    verdict: AuditVerdict
    reason: str = ""
    #: 被判定的句子原文与所属章节——T4.2 降级要靠它定位到正文的哪一句。
    sentence: str = ""
    section: str = ""

    @field_validator("citation_id", mode="before")
    @classmethod
    def _normalize_citation_id(cls, v: object) -> str:
        """同 `Judgment`：审计结果要按编号回查证据，编号对不上就等于审计失效。"""
        return normalize_citation_id(v)


class AuditResult(BaseModel):
    """T4.1 引用审计整体输出。

    是 LLM 批量核验（FR-17：按 citation_id 分组，一组一次调用）的返回结构；
    规则版审计器直接产出 `list[AuditItem]`，落进 state 时同构。
    """

    items: list[AuditItem] = Field(default_factory=list)
