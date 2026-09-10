"""共享领域契约（Pydantic）。

放在包根、与 `config.py` 同级，理由：这些模型被 **④ llm 层**（结构化输出的目标类型）与
**③ agents 层**（节点产出）同时使用。若放到 agents/ 里，llm/ 就得反向 import，违反"依赖只能向下"。
（`架构设计.md` §11 只列了 config.py，本文件是同类共享内核，待下版文档补齐。）
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

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


class ConflictResult(BaseModel):
    """T3.7 矛盾检测整体输出。"""

    conflicts: list[Conflict] = Field(default_factory=list)


AuditVerdict = Literal["supported", "partial", "unsupported"]


class AuditItem(BaseModel):
    """T4.1 引用审计的单条结果。"""

    citation_id: str
    verdict: AuditVerdict
    reason: str = ""
