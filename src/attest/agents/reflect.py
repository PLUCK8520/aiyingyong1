"""T3.6 · Reflect：判断是否补检 + 生成补检查询（≤2 轮 + 熔断只读）。

设计取舍（**与文档不同的一处，明确备案**）：补检查询**不走 LLM**，由缺口文本 + 原子问题
确定性生成。理由：
  1. 额度红线——单份报告已约 30 万 token，单模型仅够约 3 次；reflect 若再造一次调用，
     等于把"补一轮"的成本翻倍，而它要产出的只是一个检索 query；
  2. 可复现——确定性 query 让"第二轮检索到的证据编号"可预测，便于单测断言；
  3. 职责最小——reflect 的判定（轮数上限 / 熔断）本来就是纯算术。
LLM 改写 query 留作 P4 的优化项（模型路由器策略表落地后再评估收益）。

熔断只读：本节点**不决定**熔断等级，只读派生值 `fuse_level` 并执行（《功能设计》§2）。
"""

from __future__ import annotations

import re
from typing import Any

from ..logging import get_logger
from .base import NodeContext

log = get_logger(__name__)
NODE = "reflect"
TAG = "reflect"

_GAP_SUBJ_RE = re.compile(r"[「『\"']([^」』\"']+)[」』\"']")


def _has_evidence_in_round(state: dict[str, Any], rnd: int) -> bool:
    """本轮是否真的检索到了新证据（用来判断"补检有没有用"）。"""
    return any(
        int(getattr(e, "round_no", 1) or 1) == rnd for e in (state.get("evidence") or [])
    )


def targets_from_gaps(
    gaps: list[str], sub_questions: list[str], objective: str, next_round: int
) -> list[dict[str, Any]]:
    """缺口 → 补检 payload。**subq_no 必须是原子问题序号**：引用编号 = 轮次-子问题-序号，
    补检沿用原序号，报告里 `WEB1-2-1` 与 `WEB2-2-1` 才指向同一个子问题的两轮证据。"""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    subs = list(sub_questions)
    for gap in gaps:
        m = _GAP_SUBJ_RE.search(gap or "")
        sq = (m.group(1) if m else (gap or "")).strip()
        if not sq or sq in seen:
            continue
        seen.add(sq)
        if sq in subs:
            subq_no = subs.index(sq) + 1
        else:
            subs.append(sq)
            subq_no = len(subs)
        out.append(
            {
                "objective": objective,
                "sub_question": sq,
                "subq_no": subq_no,
                "round": next_round,
            }
        )
    return out


def run(state: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    plan = state.get("plan") or {}
    objective = plan.get("objective", "")
    sub_questions: list[str] = plan.get("sub_questions") or []
    gaps: list[str] = list(state.get("gaps") or [])
    rnd = int(state.get("round", 1) or 1)
    done = int(state.get("reflect_count", 0) or 0)
    fuse = ctx.budget_snapshot(state).fuse_level
    max_rounds = ctx.settings.max_reflect_rounds

    if not gaps:
        reason, targets = "无缺口，直接成文", []
    elif rnd > 1 and not _has_evidence_in_round(state, rnd):
        # 上一轮补检**一条新证据都没拿到** → 再补也是重复烧额度（检索词没变，来源也已去重）。
        # 这条早停来自 eval.mini 的实测：补检过度触发时，每问会白跑两轮。
        reason, targets = f"第 {rnd} 轮补检未获得任何新证据，提前停止", []
    elif done >= max_rounds:
        reason, targets = f"已达补检上限（{max_rounds} 轮），停止补检", []
    elif fuse >= 2:
        reason, targets = "预算熔断 L2：停补检（报告仍产出，D5/不变式）", []
    else:
        targets = targets_from_gaps(gaps, sub_questions, objective, rnd + 1)
        reason = f"{len(gaps)} 项缺口 → 生成 {len(targets)} 路补检"

    ctx.trace.emit(
        "reflect",
        node=NODE,
        gaps=len(gaps),
        do_reflect=bool(targets),
        reflect_count=done,
        round=rnd,
        fuse_level=fuse,
        reason=reason,
        targets=[t["sub_question"] for t in targets],
    )
    log.node(
        TAG,
        NODE,
        "完成",
        do_reflect=bool(targets),
        gaps=len(gaps),
        reflect_count=done,
        fuse_level=fuse,
        reason=reason,
    )

    if not targets:
        return {"reflect_targets": []}
    return {"reflect_targets": targets, "round": rnd + 1, "reflect_count": done + 1}
