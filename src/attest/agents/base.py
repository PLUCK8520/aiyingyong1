"""③ 能力层 · 节点公共设施（NodeContext + 追踪包装 + 预算上报助手）。

分层：节点是**纯函数** `(state, ctx) -> 增量`，不自己建客户端、不自己写日志格式、不自己算钱。
把"包 trace / 记预算 / 捕获异常"这些横切动作收在这里，节点只写业务逻辑。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ..budget.account import BudgetConfig, BudgetSnapshot, snapshot
from ..config import Settings
from ..llm.gateway import LLMGateway, LLMResponse
from ..logging import get_logger
from ..retrieval.ports import SearchClient
from ..trace.events import TraceWriter

if TYPE_CHECKING:  # 仅类型标注用：避免 agents → quality 的运行期硬依赖
    from ..quality.citation_auditor import CitationAuditor

from langgraph.errors import GraphInterrupt  # T5.4：interrupt 异常需显式放行（见 `bind`）

log = get_logger(__name__)

NodeFn = Callable[[dict[str, Any], "NodeContext"], dict[str, Any]]


@dataclass
class NodeContext:
    """节点运行所需的一切外部能力（依赖注入点）。

    节点只认这里的接口，不认具体实现——把 `search` 换成 mock 还是 tavily、
    把 `local` 换成 Chroma 还是内存向量库，节点代码一行都不用改。
    """

    settings: Settings
    gateway: LLMGateway
    search: SearchClient
    trace: TraceWriter
    budget: BudgetConfig
    #: 本地知识库检索（T3.4）。为 None 表示未启用——`scout_local` 会如实留痕并跳过，
    #: 而不是假装检索过。
    local: SearchClient | None = None
    #: 引用审计器（T4.1）。为 None 表示审计关闭——`auditor` 节点会如实留痕并跳过。
    auditor: "CitationAuditor | None" = None
    #: 用户画像存储（T5.2）。为 None 表示记忆关闭——`memory_loader` 会如实留痕并返回空画像。
    profile: "Any | None" = None

    def budget_snapshot(self, state: dict[str, Any]) -> BudgetSnapshot:
        return snapshot(
            float(state.get("cost_incurred", 0.0) or 0.0),
            int(state.get("tokens_incurred", 0) or 0),
            self.budget,
        )


def accumulate(resp: LLMResponse) -> dict[str, Any]:
    """把一次 LLM 调用的成本与 token 转成 state 增量（交给累加 reducer）。"""
    return {"cost_incurred": resp.cost_cny, "tokens_incurred": resp.total_tokens}


def report_budget(ctx: NodeContext, node: str, cost_incurred: float, tokens_incurred: int) -> None:
    """按**累计后**的值上报熔断等级（增量尚未并入 state，故由调用方传入累计值）。"""
    snap = snapshot(cost_incurred, tokens_incurred, ctx.budget)
    ctx.trace.emit(
        "fuse",
        node=node,
        fuse_level=snap.fuse_level,
        cost_incurred=snap.cost_incurred,
        tokens_incurred=snap.tokens_incurred,
        used_ratio=snap.used_ratio,
    )
    if snap.fuse_level > 0:
        log.warning(f"[budget] {snap.describe(ctx.budget)}")


def budget_after(state: dict[str, Any], resp: LLMResponse) -> tuple[float, int]:
    """(累计费用, 累计 token) = state 现值 + 本次调用。"""
    return (
        float(state.get("cost_incurred", 0.0) or 0.0) + resp.cost_cny,
        int(state.get("tokens_incurred", 0) or 0) + resp.total_tokens,
    )


def bind(fn: NodeFn, *, name: str, ctx: NodeContext) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """把 `(state, ctx)` 纯函数 + 依赖，包成 LangGraph 可用的 `(state) -> increment`。

    trace 的 `node_start` / `node_end` 在这里成对写入——这是冒烟清单第 3 条的落点。

    ⚠️ 关于 `config`（T5.4）：LangGraph 会把运行期 `config` 以**关键字参数**注入节点函数
    （`RunnableConfig`）。我们**不接收它**（签名里没有 `config` 形参），这是有意的：
      - 节点不需要读 config；
      - `interrupt()` 走的是**上下文变量**（NodeContext/运行时栈）而非 config 参数，
        所以不接收 config **也能正常 interrupt**（P5 实测）。
    若将来某节点确需 config（如读 `config["configurable"]`），再给那个节点单独加形参，
    不要在这里全局透传——那会让所有节点都被迫接受一个它们不用的参数。
    """

    def node(state: dict[str, Any]) -> dict[str, Any]:
        ctx.trace.emit("node_start", node=name)
        t0 = time.perf_counter()
        try:
            out = fn(state, ctx)
        except GraphInterrupt:
            # T5.4：`interrupt()` 是用**异常**实现挂起的（GraphInterrupt）。这是**正常控制流**，
            # 不是错误——若在此记 node_error，日志与 trace 里会出现刺眼的"失败"，
            # 掩盖真正的故障。故显式放行：记一条 node_interrupt 后原样抛出（LangGraph 需要它）。
            ctx.trace.emit("node_interrupt", node=name, duration_ms=round((time.perf_counter() - t0) * 1000, 1))
            log.node(name, name, "挂起等待人工确认（interrupt）")
            raise
        except Exception as exc:  # noqa: BLE001 - 要留下失败痕迹再抛出
            ctx.trace.emit("node_error", node=name, error=f"{type(exc).__name__}: {exc}")
            log.error(f"[{name}] 失败 | agent={name} | error={type(exc).__name__}: {exc}")
            raise
        duration = round((time.perf_counter() - t0) * 1000, 1)
        ctx.trace.emit(
            "node_end",
            node=name,
            duration_ms=duration,
            outputs=sorted(k for k in out if not k.startswith("_")),
        )
        return out

    node.__name__ = name
    return node
