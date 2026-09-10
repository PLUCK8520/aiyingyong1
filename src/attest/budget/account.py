"""横切 · 派生式预算与分级熔断（ADR-02）。

设计（为什么是派生值）：
  原设计让 `budget.spent` 作为可变复合对象被多路并行写，Send 扇出时 LangGraph 默认覆盖语义会丢账。
  改为：state 只保留**两个标量增量**（`cost_incurred` 元 / `tokens_incurred`），声明累加 reducer；
  阈值属静态配置 `budget_cfg`；`fuse_level` 随时由 `f(增量, 配置)` 算出——不存在"忘记更新"的不一致。
  代价：所有 LLM 调用必须经网关上报，绕过网关会漏账。

不变式：**报告必须产出**。熔断只降级（切小模型 / 停补检 / 截断上下文），绝不杀进程。
"""

from __future__ import annotations

from dataclasses import dataclass

FUSE_NORMAL = 0
FUSE_DEGRADE = 1  # ① 轻任务切小模型
FUSE_HARD = 2  # ② 停补检 + 截断上下文


@dataclass(frozen=True)
class BudgetConfig:
    """静态预算配置（不进 state，因为不会被并发写）。"""

    total_cny: float = 2.0
    total_tokens: int = 300_000
    warn_ratio: float = 0.70
    hard_ratio: float = 0.90

    def __post_init__(self) -> None:
        if not 0 < self.warn_ratio < self.hard_ratio <= 1:
            raise ValueError("熔断阈值必须满足 0 < warn < hard <= 1")
        if self.total_cny <= 0 or self.total_tokens <= 0:
            raise ValueError("预算总额必须为正")


@dataclass(frozen=True)
class BudgetSnapshot:
    cost_incurred: float
    tokens_incurred: int
    cny_ratio: float
    token_ratio: float
    fuse_level: int

    @property
    def used_ratio(self) -> float:
        """取两个口径里更紧张的那个——任一触线即降级。"""
        return max(self.cny_ratio, self.token_ratio)

    def describe(self, cfg: BudgetConfig) -> str:
        label = {
            FUSE_NORMAL: "正常",
            FUSE_DEGRADE: "降级(切小模型)",
            FUSE_HARD: "硬熔断(停补检+截断)",
        }[self.fuse_level]
        return (
            f"预算状态 {label}｜已花 {self.cost_incurred:.4f}/{cfg.total_cny:.2f} 元 "
            f"({self.cny_ratio:.1%})｜已用 {self.tokens_incurred}/{cfg.total_tokens} token "
            f"({self.token_ratio:.1%})"
        )


def fuse_level(cost_incurred: float, tokens_incurred: int, cfg: BudgetConfig) -> int:
    """派生熔断等级，不写 state。"""
    ratio = max(cost_incurred / cfg.total_cny, tokens_incurred / cfg.total_tokens)
    if ratio >= cfg.hard_ratio:
        return FUSE_HARD
    if ratio >= cfg.warn_ratio:
        return FUSE_DEGRADE
    return FUSE_NORMAL


def snapshot(cost_incurred: float, tokens_incurred: int, cfg: BudgetConfig) -> BudgetSnapshot:
    return BudgetSnapshot(
        cost_incurred=round(cost_incurred, 6),
        tokens_incurred=tokens_incurred,
        cny_ratio=round(cost_incurred / cfg.total_cny, 6),
        token_ratio=round(tokens_incurred / cfg.total_tokens, 6),
        fuse_level=fuse_level(cost_incurred, tokens_incurred, cfg),
    )
