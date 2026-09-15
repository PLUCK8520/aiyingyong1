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
    fuse = ctx.budget_snapshot(state).fuse_level
    log.node(TAG, NODE, "开始", query=query, fuse_level=fuse)

    try:
        resp = ctx.gateway.chat(
            build_direct_messages(query), task="direct", temperature=0.3, fuse_level=fuse
        )
        answer = resp.text.strip()
    except Exception as exc:  # noqa: BLE001 - 回答失败也要给用户一个交代（不静默空白）
        # 这里没有"更好的降级答案"可给（direct 分支的全部价值就是那句回答），
        # 但**绝不能静默返回空白**——用户会以为系统坏了或自己没说清。
        # 如实说明失败原因，并指出可用出路（重跑 / 走调研路径）。
        ctx.trace.emit(
            "direct_failed",
            node=NODE,
            query=query,
            error=f"{type(exc).__name__}: {exc}",
        )
        log.error(f"[direct] 回答调用失败（已重试耗尽）：{type(exc).__name__}: {exc}")
        log.node(TAG, NODE, "降级完成", degraded=True)
        return {
            "direct_answer": (
                "（本次未能生成回答：模型调用失败，已按网关策略重试仍未成功。）\n\n"
                f"失败原因：{type(exc).__name__}: {exc}\n\n"
                "可以稍后重试；如果这是一次性故障，重跑通常即可恢复。"
                "若这个问题需要检索多源资料，也可以改用调研模式提问。"
            )
        }

    cost, toks = budget_after(state, resp)
    report_budget(ctx, NODE, cost, toks)
    log.node(TAG, NODE, "完成", chars=len(answer))

    return {"direct_answer": answer, **accumulate(resp)}
