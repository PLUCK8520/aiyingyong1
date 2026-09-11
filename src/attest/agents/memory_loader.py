"""T5.2 · 记忆加载节点：每轮调研前把用户画像注入 state。

**位置**：图入口（`START → memory_loader → intent_router`）。
**职责**：只读画像，不写。写入发生在 CLI 层（「记住 xx」指令）或未来前端设置页。

为什么单独成节点而不是在 CLI 里读：
  1. 画像要进 **state**（能被检查点存、能在 trace 里看见），CLI 里读就丢了这层可观测性；
  2. P6 的 FastAPI 会复用同一张图，届时**不用再写一遍**注入逻辑（同一入口，同一行为）。

⚠️ 画像读取**必须容错**：画像库损坏/被占用不应让整条调研链路挂掉——
   记忆是**增强**，不是**前提**。读不到就留痕 + 空画像继续。
"""

from __future__ import annotations

from typing import Any

from ..logging import get_logger

log = get_logger(__name__)
NODE = "memory_loader"
TAG = "memory"


def run(state: dict[str, Any], ctx: Any) -> dict[str, Any]:
    store = getattr(ctx, "profile", None)
    if store is None:
        ctx.trace.emit("memory_loaded", node=NODE, enabled=False, entries=0)
        return {"profile": {}}

    try:
        items = store.all()
    except Exception as exc:  # noqa: BLE001 - 记忆是增强，不能拖垮主链路
        log.warning(f"[{TAG}] 画像读取失败，按空画像继续：{type(exc).__name__}: {exc}")
        ctx.trace.emit("memory_loaded", node=NODE, enabled=True, entries=0, error=str(exc))
        return {"profile": {}}

    profile = {it.key: it.value for it in items}
    ctx.trace.emit(
        "memory_loaded",
        node=NODE,
        enabled=True,
        entries=len(profile),
        keys=sorted(profile.keys()),
    )
    log.node(TAG, NODE, f"注入画像 {len(profile)} 条")
    return {"profile": profile}


def profile_block(state: dict[str, Any]) -> str:
    """把 state 里的画像渲染成提示词片段。空则空串（不注入空段落）。"""
    profile = state.get("profile") or {}
    if not profile:
        return ""
    lines = [f"- {k}：{v}" for k, v in profile.items()]
    return "【用户偏好（跨会话记忆）】\n" + "\n".join(lines)
