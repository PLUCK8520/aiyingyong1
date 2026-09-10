"""T0.5 · 二维单价表与成本计算（纯单测，不联网）。"""

from __future__ import annotations

import pytest

from attest.budget.account import FUSE_DEGRADE, FUSE_HARD, FUSE_NORMAL, BudgetConfig, snapshot
from attest.trace.pricing import (
    PRICING,
    UnknownModelError,
    compute_cost,
    compute_embed_cost,
)


def test_tier_selection_by_context_length() -> None:
    """同样是 qwen3.8-max，上下文越长单价越高——单一费率会算错。"""
    price = PRICING["qwen3.8-max"]
    short = price.tier_for(32_000)
    long = price.tier_for(200_000)
    assert short.input_cny_per_million == 12.0
    assert long.input_cny_per_million == 15.0
    assert long.output_cny_per_million == 60.0


def test_tier_falls_back_to_largest_when_over_max() -> None:
    price = PRICING["qwen3.8-max"]
    assert price.tier_for(10_000_000) is price.tiers[-1]


def test_compute_cost_uses_selected_tier() -> None:
    # 100 万输入 + 100 万输出，128K+ 档：15 + 60 = 75 元
    cost = compute_cost("qwen3.8-max", 1_000_000, 1_000_000, context_tokens=200_000)
    assert cost == pytest.approx(75.0)
    # 32K 档：12 + 36 = 48 元
    cost_short = compute_cost("qwen3.8-max", 1_000_000, 1_000_000, context_tokens=30_000)
    assert cost_short == pytest.approx(48.0)


def test_local_model_is_free_but_not_silent_error() -> None:
    assert compute_cost("qwen3:4b", 10_000, 10_000) == 0.0


def test_unknown_model_raises_instead_of_charging_zero() -> None:
    """静默按 0 计费会让 FR-20 的记账误差直接失效——必须抛错。"""
    with pytest.raises(UnknownModelError):
        compute_cost("qwen9-ultra", 1000, 1000)


def test_embed_cost_uses_per_1k_scale() -> None:
    assert compute_embed_cost("text-embedding-v4", 1000) == pytest.approx(0.0005)


def test_fuse_level_is_derived_from_the_tighter_ratio() -> None:
    cfg = BudgetConfig(total_cny=1.0, total_tokens=100_000)
    assert snapshot(0.1, 10_000, cfg).fuse_level == FUSE_NORMAL
    # 费用只用 50%，但 token 已用 75% → 取更紧的那个 → 降级
    assert snapshot(0.5, 75_000, cfg).fuse_level == FUSE_DEGRADE
    assert snapshot(0.5, 95_000, cfg).fuse_level == FUSE_HARD


def test_budget_config_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        BudgetConfig(total_cny=1.0, total_tokens=1000, warn_ratio=0.9, hard_ratio=0.8)
