"""T5.4 · 人工确认大纲（Human-in-the-loop / 编译期静态断点）。

**流程**（《功能设计》§4.3）：
    planner 产出 plan
      → 图在 `human_confirm` 之前**静态挂起**（`interrupt_before=["human_confirm"]`）
          ├─ 用户直接放行（不改）→ 继续
          ├─ 用户改大纲后 resume → 用新 plan 继续
          └─ 用户选择跳过 → 走配置项直接继续

⚠️⚠️ **本模块最容易踩的坑（2026-09-11 重写，实测推翻 09-10 的实现）**：

    09-10 的实现用**节点内 `interrupt()`**，在 PostgresSaver/SqliteSaver 下会**静默失效**：
      1. `interrupt()` 抛 `GraphInterrupt` 之前，已执行
         `conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])`
         （`langgraph/types.py:956`）——**即使 resume 是空列表，这条写仍进了
         `checkpoint_pending_writes`**；
      2. `is_resuming` 分支把所有通道（含 `input`）标记为
         `versions_seen[INTERRUPT] = 当前版本`（`pregel/_loop.py:946-951`）；
      3. 但 `apply_writes` 靠 `get_next_version` 让 `channel_versions` **单调递增**
         （`pregel/_algo.py:270-288`），`_triggers()` 仍判定"有更新" → 节点被重跑；
      4. 重跑时 `get_null_resume()`（`_algo.py:1320`）从残留的 pending write 里
         读出那个空值 → `v is not None` 成立 → **`interrupt()` 不再抛异常，直接返回 `{}`**。

    净效果：`human_confirm` 被静默放行，`invoke` 返回的 state 里**没有 `__interrupt__`**，
    `interrupted=false`，整条链路一轮跑完。**它不报错，只是不再挂起**——最难查的一类故障。

    改用**编译期静态断点**后（`build_graph(interrupt_before=["human_confirm"])`）：
      - 断点由 `should_interrupt()`（`_algo.py:156`）在**节点执行前**判定，
        不依赖 `interrupt()` 的 scratchpad 时序，行为确定；
      - 恢复用 `invoke(None, cfg)`（不放行）/ `invoke(Command(resume=...), cfg)`（放行）；
      - 官方 API 显式支持：`Pregel.invoke(..., interrupt_before=...)`（`pregel/main.py:3836`）。

⚠️ **本节点现在无副作用**（纯询问 + 改写 plan），故不再需要"守卫放在 interrupt 之前"那套约定。
    但**状态守卫仍然保留**——未来若有改动把副作用插进来，守卫是唯一的幂等防线。
"""

from __future__ import annotations

from typing import Any

from ..logging import get_logger

log = get_logger(__name__)
NODE = "human_confirm"
TAG = "confirm"


def confirm_payload(plan: dict[str, Any]) -> dict[str, Any]:
    """构造给人工看的确认载荷（CLI / 前端共用同一份结构）。"""
    return {
        "node": NODE,
        "objective": plan.get("objective", ""),
        "outlines": list(plan.get("outlines") or []),
        "sub_questions": list(plan.get("sub_questions") or []),
    }


def run(state: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """人工确认节点（在**静态断点之后**执行）。

    返回值语义：
      - 已放行 / 配置关闭 → 返回 `{plan_approved: True, ...}`；
      - `plan_approval` 里带 `action=edit` → 用新大纲覆盖 `plan`。
    """
    # ---------------- 守卫：幂等防线（防未来插入副作用后 resume 重跑重复触发）----------------
    if state.get("plan_approved"):
        log.node(TAG, NODE, "已确认，跳过（守卫命中）")
        return {}

    plan = state.get("plan") or {}
    approval = state.get("plan_approval") or {}
    action = str(approval.get("action") or approval.get("decision") or "approve")

    if action in ("abort", "quit"):
        ctx.trace.emit("human_confirm_aborted", node=NODE)
        log.node(TAG, NODE, "用户中止")
        return {"plan_approved": True, "plan_approval": {**approval, "aborted": True}}

    if action == "edit" and isinstance(approval.get("plan"), dict):
        merged = {**plan, **approval["plan"]}
        ctx.trace.emit(
            "human_confirm_edited", node=NODE, outlines=len(merged.get("outlines") or [])
        )
        log.node(TAG, NODE, "大纲已修改", outlines=len(merged.get("outlines") or []))
        return {"plan": merged, "plan_approved": True, "plan_approval": approval}

    ctx.trace.emit("human_confirm_approved", node=NODE, action=action)
    log.node(TAG, NODE, "已确认", action=action)
    return {"plan_approved": True, "plan_approval": approval}


def auto_approval(settings: Any) -> dict[str, Any]:
    """配置关闭人工确认时的默认决策（等价于用户选「跳过」）。"""
    return {"action": "skip", "decision": "skipped", "auto": True}
