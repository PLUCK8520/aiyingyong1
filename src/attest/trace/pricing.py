"""T0.5 · 二维单价表：`模型 × 上下文长度档位 → (输入价, 输出价)`。

为什么是二维：百炼是**阶梯定价 + 按上下文长度分档**（例：qwen3.8-max 在 0–32K 档约 12/36 元每百万，
128K–256K 档约 15/60）。用单一费率会让成本必然算错，FR-20 的"记账误差 < 5%"直接达不成。

**价格来源与证据等级**（按厂商分列，因为等级不同）：
  - 百炼系列：**③第三方**——社区文章汇总，不同来源同档已有出入（有文章写 qwen-flash 输入 0.15 元/百万，
    与本表 0.3 差一倍）。开工后必须在百炼控制台价格页核对回填（审查报告 H 节 #12）。
  - 硅基流动系列：**①一手官方**（2026-09-12 取自 siliconflow.cn/pricing，人民币 / 百万 token）。
    ⚠️ DeepSeek-V4-Flash 是**分时段定价**（2–8 点半价），本表取白天高价档——
    宁可高估不可低估：低估会让预算熔断形同虚设。
本文件是唯一价格来源，改价只改这里。
"""

from __future__ import annotations

from dataclasses import dataclass

MILLION = 1_000_000


@dataclass(frozen=True)
class Tier:
    """一个上下文长度档位。`max_context_tokens` 为该档覆盖上限（含），单位 token。"""

    max_context_tokens: int
    input_cny_per_million: float
    output_cny_per_million: float


@dataclass(frozen=True)
class ModelPrice:
    tiers: tuple[Tier, ...]  # 按 max_context_tokens 升序

    def tier_for(self, context_tokens: int) -> Tier:
        """选档：第一个能容纳 context_tokens 的档；超出最大档则用最大档（并在调用方留痕）。"""
        for tier in self.tiers:
            if context_tokens <= tier.max_context_tokens:
                return tier
        return self.tiers[-1]


# ---------------------------------------------------------------- 对话模型（③第三方，待核）
PRICING: dict[str, ModelPrice] = {
    "qwen3.8-max": ModelPrice(
        (
            Tier(32_768, 12.0, 36.0),
            Tier(131_072, 12.0, 36.0),
            Tier(262_144, 15.0, 60.0),
        )
    ),
    "qwen3.7-max": ModelPrice(
        (
            Tier(32_768, 1.65, 4.95),
            Tier(262_144, 1.65, 4.95),
        )
    ),
    "qwen3.8-flash": ModelPrice((Tier(262_144, 0.8, 2.7),)),
    "qwen3.7-flash": ModelPrice((Tier(262_144, 0.3, 1.2),)),
    # ---------------- 硅基流动（①一手官方 siliconflow.cn/pricing，2026-09-12）----------------
    "Qwen/Qwen3.5-35B-A3B": ModelPrice(
        (Tier(131_072, 0.40, 3.20), Tier(1_000_000, 1.60, 12.80))
    ),
    "Qwen/Qwen3.5-27B": ModelPrice((Tier(131_072, 0.60, 4.80), Tier(1_000_000, 1.80, 14.40))),
    "Qwen/Qwen3.5-122B-A10B": ModelPrice(
        (Tier(131_072, 0.80, 6.40), Tier(1_000_000, 2.00, 16.00))
    ),
    "Qwen/Qwen3.8-27B": ModelPrice((Tier(262_144, 3.00, 12.00),)),
    "inclusionAI/Ling-flash-2.0": ModelPrice((Tier(262_144, 1.00, 4.00),)),
    "inclusionAI/Ling-mini-2.0": ModelPrice((Tier(262_144, 0.50, 2.00),)),
    "stepfun-ai/Step-3.5-Flash": ModelPrice((Tier(262_144, 0.70, 2.10),)),
    #: 分时段定价，取白天高价档（宁高不低估，见模块文档串）
    "deepseek-ai/DeepSeek-V4-Flash": ModelPrice((Tier(1_000_000, 3.00, 9.00),)),
    "deepseek-ai/DeepSeek-V4-Pro": ModelPrice((Tier(1_000_000, 12.00, 24.00),)),
    "deepseek-ai/DeepSeek-V3.2": ModelPrice((Tier(1_000_000, 4.00, 6.00),)),
}

# 本地模型（Ollama）不计费，但要把 token 记账，便于对比"离线兜底的代价"
LOCAL_MODELS: frozenset[str] = frozenset({"qwen3:4b", "qwen3:8b", "llama3.1", "local"})

# 托管但免费的模型（①官方价格页标"免费"）——同样计 token、不计费。
# 单列一类而不是混进 LOCAL_MODELS：语义不同（一个是"跑在我机器上"，一个是"厂商送的额度"），
# 混在一起将来排查"为什么这次没收钱"时就分不清了。
FREE_HOSTED_MODELS: frozenset[str] = frozenset(
    {
        "Qwen/Qwen2.5-7B-Instruct",
        "Pro/Qwen/Qwen2.5-7B-Instruct",
        "BAAI/bge-m3",
        "BAAI/bge-large-zh-v1.5",
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-reranker-v2-m3",
        "tencent/Hunyuan-MT-7B",
    }
)

#: 免费向量模型（①官方标"免费"：bge 系列；以及本地 Ollama 拉的 bge-m3）。
#: **必须收在这里**：`compute_embed_cost` 对未知模型是直接抛错的（不许静默算成 0），
#: 所以本地模型不登记就会被自己的守卫拦下——这是守卫在正常工作，不是 bug。
FREE_EMBED_MODELS: frozenset[str] = frozenset(
    {"BAAI/bge-m3", "BAAI/bge-large-zh-v1.5", "BAAI/bge-large-en-v1.5", "bge-m3"}
)

# ---------------------------------------------------------------- 向量模型（③第三方，待核）
# 单位：元 / 千 token（注意与对话模型不同量纲）
EMBED_PRICING_CNY_PER_1K: dict[str, float] = {
    "text-embedding-v4": 0.0005,
}


class UnknownModelError(KeyError):
    """单价表里没有这个模型——不允许静默按 0 计费。"""


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    context_tokens: int | None = None,
) -> float:
    """返回本次调用的费用（元）。

    `context_tokens` 用于选档；不传时用 input_tokens 近似。
    """
    if model in LOCAL_MODELS or model in FREE_HOSTED_MODELS:
        return 0.0
    price = PRICING.get(model)
    if price is None:
        raise UnknownModelError(
            f"模型 {model!r} 不在单价表里。请在 src/attest/trace/pricing.py 的 PRICING 中补充，"
            f"否则成本会被静默算成 0（FR-20 直接失效）。"
        )
    tier = price.tier_for(context_tokens if context_tokens is not None else input_tokens)
    return (
        input_tokens / MILLION * tier.input_cny_per_million
        + output_tokens / MILLION * tier.output_cny_per_million
    )


def compute_embed_cost(model: str, tokens: int) -> float:
    if model in FREE_EMBED_MODELS:
        return 0.0
    if model not in EMBED_PRICING_CNY_PER_1K:
        raise UnknownModelError(f"向量模型 {model!r} 不在单价表里，请补充 EMBED_PRICING_CNY_PER_1K。")
    return tokens / 1000 * EMBED_PRICING_CNY_PER_1K[model]
