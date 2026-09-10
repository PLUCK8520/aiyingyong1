"""T4.4 · 模型路由器：任务类型 → 模型（策略表 + 熔断降级）。

为什么独立成模块（《开发任务清单》第 40 行明确要求）：
  - **记账是网关的职责，"选哪个模型"是另一个关注点**。混在一个文件里，改路由会碰记账代码，
    反之亦然——两者失败模式不同，独立才有单一改动点；
  - 策略表要能被 trace 复盘、被评测尺子回填，独立模块才能"只改一处"。

⚠️ **诚实声明（证据等级 ④未核）**：本表此刻是**可配置的路由骨架**，不是"已调优的策略"。
`Settings.model_*` 的取值来自技术选型的经验判断；T3.9 的评测尺子目前只在**离线 mock** 上跑过，
"换模型后质量/成本怎么变"**尚无任何真实数据**。真实策略待接 dashscope 后用 `eval/mini.py`
的 token 成本 + 引用有效率两指标回填——这正是《功能设计》§11 残留风险 #3 的缓解路径：
**先立尺子（T3.9，已完成）再换零件（T4.4，本模块）**，避免"盲切"。

熔断降级规则（《功能设计》§6.5 / FR-21，职责边界：路由器只决定模型，不记账、不改 state）：

  | fuse_level | 触发 | 本模块的动作 |
  |---|---|---|
  | 0 正常 | r < warn | 按策略表原样路由 |
  | 1 降级 | r ≥ warn | **轻任务**切到 `model_fuse_light`；**主链路不动** |
  | 2 硬熔断 | r ≥ hard | 同上（另两条动作不在本模块：停补检在 reflect，截断在 analyst，审计只初筛在审计器） |

"主链路不动"是刻意的：熔断的目的是**降质保交付**，不是把报告写坏。正文（analyst）与
大纲（planner）是报告质量的承重墙，切小模型省下的钱抵不过报告变烂的代价。

⚠️ 另一条必须说清的实情：**在当前单价表下，L1 对多数轻任务是无操作**——intent / judge /
conflict / audit_fast 的默认模型本就是最便宜的 `qwen3.7-flash`（0.3/1.2 元每百万），
真正会被换掉的是 `direct`（qwen3.8-flash → qwen3.7-flash）。这不是 bug，是"模型集里
便宜档只有一档"的现实。要让 L1 有实质收益，二选一：
  ① 把 `model_fuse_light` 换成**本地模型名**（如 `qwen3:4b`）——但仅在
     `ATTEST_LLM_MODE=ollama` 下成立；dashscope 模式会因模型不存在直接报错，故不作默认；
  ② 在 `trace/pricing.py` 补一档更便宜的 API 模型，由评测尺子验证质量后再设为默认。
两条都留给"真实评测回填"这一步，不在本模块内拍脑袋决定。
"""

from __future__ import annotations

from ..budget.account import FUSE_DEGRADE
from ..config import Settings
from ..logging import get_logger

log = get_logger(__name__)

#: 档位标签——供 trace 复盘"这次调用用的是省钱档还是能力档"
STRONG = "strong"
FAST = "fast"

#: **轻任务**：熔断 L1 起切便宜档（设计 §6.5「轻任务（intent / 摘要 / 审计初筛）切小模型」）。
#: 判据是"判错代价低 + 调用频次高"：分流判错顶多多检索一次，审计初筛判错还有复核兜底。
LIGHT_TASKS: frozenset[str] = frozenset(
    {"intent", "direct", "judge", "conflict", "audit_fast"}
)

#: **主链路 / 高风险任务**：任何熔断级别都不降级。
#: `audit_strong` 不在此列但也不属轻任务——它在 L2 下直接被**跳过**（只跑初筛），
#: 由审计器负责，路由器不参与（职责边界）。
CORE_TASKS: frozenset[str] = frozenset({"planner", "analyst", "analyst_rewrite"})

#: 任务 → `Settings` 字段名（映射表是**唯一真值来源**，避免多处硬编码漂移）
_TASK_FIELD: dict[str, str] = {
    "intent": "model_intent",
    "direct": "model_direct",
    "planner": "model_planner",
    "judge": "model_judge",
    "conflict": "model_conflict",
    "analyst": "model_analyst",
    "analyst_rewrite": "model_analyst",
    "audit_fast": "model_audit_fast",
    "audit_strong": "model_audit_strong",
}

_TASK_TIER: dict[str, str] = {
    "intent": FAST,
    "direct": FAST,
    "planner": STRONG,
    "judge": FAST,
    "conflict": FAST,
    "analyst": STRONG,
    "analyst_rewrite": STRONG,
    "audit_fast": FAST,
    "audit_strong": STRONG,
}


class ModelRouter:
    """任务类型 + 熔断级别 → 模型名。**无状态、纯函数式**，可安全复用。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ---------------- 对外 ----------------

    def route(self, task: str, *, fuse_level: int = 0) -> str:
        """返回该任务在当前熔断级别下应使用的模型名。"""
        model = self.base_model(task)
        if fuse_level >= FUSE_DEGRADE and task in LIGHT_TASKS:
            cheap = self.settings.model_fuse_light
            if cheap and cheap != model:
                log.info(f"[router] 熔断 L{fuse_level}：轻任务 {task} {model} → {cheap}")
            return cheap or model
        return model

    def tier(self, task: str, *, fuse_level: int = 0) -> str:
        """当前档位标签（trace / 降级判断用）。熔断降级后轻任务一律记为 fast。"""
        if fuse_level >= FUSE_DEGRADE and task in LIGHT_TASKS:
            return FAST
        return _TASK_TIER.get(task, FAST)

    def base_model(self, task: str) -> str:
        """策略表原值（不含熔断覆盖）。未映射任务回退 `model_direct` 并留痕。"""
        field_name = _TASK_FIELD.get(task)
        if field_name is None:
            log.warning(f"[router] 任务 {task!r} 没有模型映射，回退 model_direct")
            return self.settings.model_direct
        return getattr(self.settings, field_name)

    def table(self) -> dict[str, str]:
        """导出策略表快照（trace 复盘 / 报告页脚可自证"用了哪个模型写的"）。"""
        return {task: self.base_model(task) for task in sorted(_TASK_FIELD)}
