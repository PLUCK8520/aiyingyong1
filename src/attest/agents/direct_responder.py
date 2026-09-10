"""T1.3 · 快速回答：direct 分支的秒回节点（不检索、不规划）。"""

from __future__ import annotations

from typing import Any

from ..llm.prompts import build_direct_messages
from ..logging import get_logger
from .base import NodeContext, accumulate, budget_after, report_budget

log = get_logger(__name__)
NODE = "direct_responder"
TAG = "direct"


def run(state: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    query = state.get("query", "")
    log.node(TAG, NODE, "开始", query=query)

    resp = ctx.gateway.chat(build_direct_messages(query), task="direct", temperature=0.3)

    cost, toks = budget_after(state, resp)
    report_budget(ctx, NODE, cost, toks)
    log.node(TAG, NODE, "完成", chars=len(resp.text))

    return {"direct_answer": resp.text.strip(), **accumulate(resp)}
