"""T1.1 · 图状态定义。

**并行写入的字段必须声明 reducer**（`Annotated[..., operator.add]`），否则 Send 扇出时
多路分支写同一字段会互相覆盖——且 **LangGraph 默认覆盖语义不报错**，极难定位。

预算按 ADR-02 走派生值：state 只保留两个**标量增量** `cost_incurred` / `tokens_incurred`，
`fuse_level` 由 `budget/account.py` 现算，不写入 state（避免"忘记更新"的不一致）。

字段清单以《功能设计.md》§3 为准；此处为实现。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from ..retrieval.ports import Evidence
from ..schemas import AuditItem, Conflict, Judgment


class GraphState(TypedDict, total=False):
    # ---------- 输入 ----------
    query: str
    thread_id: str

    # ---------- 编排 ----------
    route: str  # direct | research
    route_reason: str
    plan: dict[str, Any]  # 存 dict 而非 Pydantic 实例：便于检查点序列化（P5 复核）
    round: int
    reflect_count: int
    #: T5.4 人工确认：大纲是否已确认。**resume 重跑时的状态守卫**——`human_confirm`
    #: 节点首行检查它，避免 interval resume 时重复询问/重复副作用（《功能设计》§4.3 硬性约定）。
    plan_approved: bool
    #: T5.4 确认动作留痕：{action: approve|skip|edit}
    plan_approval: dict[str, Any]
    #: T5.2 用户画像（跨会话偏好）。memory_loader 是**唯一写入者**，覆盖语义。
    profile: dict[str, str]
    #: T3.6：reflect 产出的补检任务包（Send payload 列表）。**覆盖语义**——reflect 是唯一写入者。
    #: 放到 state 而不是让路由函数现算，是为了让"为什么补检这几路"在 trace/检查点里可见。
    reflect_targets: list[dict[str, Any]]

    # ---------- 并行写入：以下字段必须有累加 reducer ----------
    evidence: Annotated[list[Evidence], operator.add]
    judgments: Annotated[list[Judgment], operator.add]
    gaps: Annotated[list[str], operator.add]
    conflicts: Annotated[list[Conflict], operator.add]
    memory_hits: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[dict[str, Any]], operator.add]
    cost_incurred: Annotated[float, operator.add]
    tokens_incurred: Annotated[int, operator.add]

    # ---------- 产出 ----------
    direct_answer: str
    report: str
    reference_list: str
    citation_check: dict[str, Any]
    #: T4.1 引用审计结果。auditor 是**唯一写入者**（analyst → auditor 单线），故无需 reducer。
    audit_items: list[AuditItem]
    audit_summary: dict[str, Any]
    #: T4.2b 章节重写次数（auditor 写，覆盖语义）
    rewrite_count: int
    #: T4.4 模型路由器：正文（analyst）本次使用的档位 strong|fast。供 trace 与降级判断。
    model_tier: str


#: 供 build.py 初始化用：所有 reducer 字段必须给初值
INITIAL_STATE: dict[str, Any] = {
    "round": 1,
    "reflect_count": 0,
    "reflect_targets": [],
    "plan_approved": False,
    "plan_approval": {},
    "profile": {},
    "evidence": [],
    "judgments": [],
    "gaps": [],
    "conflicts": [],
    "memory_hits": [],
    "errors": [],
    "cost_incurred": 0.0,
    "tokens_incurred": 0,
    "audit_items": [],
    "audit_summary": {},
    "rewrite_count": 0,
    "model_tier": "strong",
}


#: 冒烟断言用：这些字段若漏了 reducer，扇出就会丢数据
REDUCER_FIELDS: tuple[str, ...] = (
    "evidence",
    "judgments",
    "gaps",
    "conflicts",
    "memory_hits",
    "errors",
    "cost_incurred",
    "tokens_incurred",
)


def reducer_fields_ok() -> list[str]:
    """返回**缺失 reducer** 的字段名列表（空列表 = 全部正确）。"""
    import typing

    hints = typing.get_type_hints(GraphState, include_extras=True)
    missing: list[str] = []
    for name in REDUCER_FIELDS:
        hint = hints.get(name)
        meta = getattr(hint, "__metadata__", ())
        if not any(m is operator.add for m in meta):
            missing.append(name)
    return missing
