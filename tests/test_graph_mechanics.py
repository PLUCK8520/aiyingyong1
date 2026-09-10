"""LangGraph 核心机制回归测试（T0.7 · P0 测试底座的第一批用例）。

这三条不是"跑通 API"的演示，而是把**方案 §3 L2 的亮点**变成可回归的断言：
    - L2-1  Send 动态扇出 + 状态 reducer 正确性（多路并行写入不互相覆盖）
    - L2-2  interrupt / Command(resume=) 与 checkpointer 的暂停-恢复闭环

对应 P-1 探针 scripts/langgraph_probe.py 的固化版本。
版本依据：langgraph 1.2.11 / langgraph-checkpoint-sqlite 3.1.1（2026-09-10 本机实测）。
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt


def test_send_import_path_is_types_module() -> None:
    """1.x 里 Send 归 langgraph.types；从 langgraph.graph 导入必须失败。

    这条断言的意义：把"导入路径"从人脑记忆变成回归防线——
    将来有人（包括我）顺手写成 `from langgraph.graph import Send` 时立刻红。
    """
    with pytest.raises(ImportError):
        from langgraph.graph import Send  # noqa: F401


# ---------------------------------------------------------------- L2-1 扇出


class FanState(TypedDict):
    subqs: list[str]
    # 并行节点写入的列表字段必须声明 reducer，否则多路写 state 互相覆盖
    hits: Annotated[list[str], operator.add]
    # 预算增量同为并行写入，必须累加（方案 §6 / ADR-02）
    cost_incurred: Annotated[float, operator.add]


def _plan(state: FanState) -> dict:
    return {"subqs": ["q1", "q2", "q3"]}


def _fanout(state: FanState) -> list[Send]:
    # Send 的 payload 是"目标节点的输入"，不是父 state
    return [Send("worker", {"q": q}) for q in state["subqs"]]


def _worker(payload: dict) -> dict:
    return {"hits": [payload["q"]], "cost_incurred": 0.1}


def _build_fanout_app():
    g = StateGraph(FanState)
    g.add_node("plan", _plan)
    g.add_node("worker", _worker)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", _fanout, ["worker"])
    g.add_edge("worker", END)
    return g.compile()


def test_send_fanout_merges_all_branches() -> None:
    """3 路 Send 并行 → hits 应为 3 条，而不是被覆盖成 1 条。"""
    out = _build_fanout_app().invoke({"subqs": [], "hits": [], "cost_incurred": 0.0})
    assert len(out["hits"]) == 3, f"并行写入被覆盖：{out['hits']}"
    assert sorted(out["hits"]) == ["q1", "q2", "q3"]


def test_send_fanout_accumulates_cost() -> None:
    """并行分支同时记账 → cost_incurred 必须累加，不能丢账。"""
    out = _build_fanout_app().invoke({"subqs": [], "hits": [], "cost_incurred": 0.0})
    assert out["cost_incurred"] == pytest.approx(0.3, rel=1e-9)


# ------------------------------------------------------- L2-2 interrupt/resume


class HitlState(TypedDict):
    approved: bool
    value: int
    calls: int


def _approve_node(state: HitlState) -> dict:
    # 约定：interrupt() 必须置于节点最前，之后才是副作用
    answer = interrupt({"ask": "接受这份大纲吗？", "outline": ["A", "B"]})
    return {"approved": bool(answer), "value": 1, "calls": state.get("calls", 0) + 1}


def _build_hitl_app(cp: SqliteSaver):
    g = StateGraph(HitlState)
    g.add_node("approve", _approve_node)
    g.add_edge(START, "approve")
    g.add_edge("approve", END)
    return g.compile(checkpointer=cp)


def test_interrupt_pauses_then_resume_completes() -> None:
    """中断 → 恢复闭环：暂停态带 __interrupt__，Command(resume=) 后跑完并携带结果。"""
    with SqliteSaver.from_conn_string(":memory:") as cp:
        app = _build_hitl_app(cp)
        cfg = {"configurable": {"thread_id": "t-hitl"}}

        paused = app.invoke({"approved": False, "value": 0, "calls": 0}, cfg)
        assert "__interrupt__" in paused, f"图未在节点处暂停：{list(paused)}"

        resumed = app.invoke(Command(resume=True), cfg)
        assert resumed["approved"] is True
        assert resumed["value"] == 1
