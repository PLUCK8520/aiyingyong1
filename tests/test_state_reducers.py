"""T1.1 · 状态 reducer 与初值（这是 L2-1 的第一道防线，纯单测）。"""

from __future__ import annotations

from attest.graph.state import INITIAL_STATE, REDUCER_FIELDS, reducer_fields_ok


def test_every_parallel_field_declares_a_reducer() -> None:
    """漏一个 reducer = Send 扇出时该字段静默丢数据，且不报错。"""
    assert reducer_fields_ok() == []


def test_initial_state_provides_every_reducer_field() -> None:
    missing = [f for f in REDUCER_FIELDS if f not in INITIAL_STATE]
    assert missing == [], f"初值缺失：{missing}"
    # 列表字段给空列表、数值字段给 0，否则 operator.add 会因类型不符炸掉
    for f in REDUCER_FIELDS:
        assert INITIAL_STATE[f] in ([], 0, 0.0), f"{f} 的初值类型不对：{INITIAL_STATE[f]!r}"


def test_budget_fields_are_scalar_increments_not_objects() -> None:
    """ADR-02：预算只保留标量增量，派生值 fuse_level 不进 state。"""
    assert isinstance(INITIAL_STATE["cost_incurred"], float)
    assert isinstance(INITIAL_STATE["tokens_incurred"], int)
    assert "fuse_level" not in INITIAL_STATE
