"""P-1 补充：LangGraph 核心机制探针（在真实版本上验证，替代文档里 ④未核 的断言）。

验证三件事：
  1. 关键符号的正确导入路径（langgraph 1.x 下 Send 在 langgraph.types，不在 langgraph.graph）；
  2. Send 动态扇出 + Annotated reducer 的归并语义（多路写入是否真的累加）；
  3. interrupt / Command(resume=) + SqliteSaver 的"暂停-恢复"闭环。

用法：
    .venv/Scripts/python.exe scripts/langgraph_probe.py
退出码：全通过 0 / 有失败 1。
"""

from __future__ import annotations

import operator
import sys
from typing import Annotated, TypedDict

import langgraph

failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"[{'OK' if cond else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")
    if not cond:
        failures.append(label)


print(f"langgraph: {getattr(langgraph, '__version__', '?')}")
print("-" * 72)

# ---- 1. 导入路径 ----
try:
    from langgraph.graph import END, START, StateGraph  # noqa: F401
    check("graph: StateGraph/START/END", True)
except Exception as exc:  # noqa: BLE001
    check("graph: StateGraph/START/END", False, f"{type(exc).__name__}: {exc}")

try:
    from langgraph.graph import Send  # noqa: F401
    check("graph: Send（旧路径，1.x 应为 False）", False, "仍可从 langgraph.graph 导入，与 1.2.11 实际不符")
except ImportError:
    check("graph: Send 不在 langgraph.graph（符合 1.2.11）", True)

try:
    from langgraph.types import Command, Send, interrupt  # noqa: F401
    check("types: Send/interrupt/Command", True)
except Exception as exc:  # noqa: BLE001
    check("types: Send/interrupt/Command", False, f"{type(exc).__name__}: {exc}")

try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: F401
    check("checkpoint: SqliteSaver + AsyncSqliteSaver", True)
except Exception as exc:  # noqa: BLE001
    check("checkpoint: SqliteSaver + AsyncSqliteSaver", False, f"{type(exc).__name__}: {exc}")

print("-" * 72)

# ---- 2. Send 扇出 + reducer ----
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command, Send, interrupt  # noqa: E402


class FanState(TypedDict):
    subqs: list[str]
    hits: Annotated[list[str], operator.add]
    cost: Annotated[float, operator.add]


def plan(state: FanState) -> dict:
    return {"subqs": ["a", "b", "c"]}


def fanout(state: FanState) -> list[Send]:
    return [Send("worker", {"q": q}) for q in state["subqs"]]


def worker(payload: dict) -> dict:
    return {"hits": [payload["q"]], "cost": 0.1}


g = StateGraph(FanState)
g.add_node("plan", plan)
g.add_node("worker", worker)
g.add_edge(START, "plan")
g.add_conditional_edges("plan", fanout, ["worker"])
g.add_edge("worker", END)
app = g.compile()
out = app.invoke({"subqs": [], "hits": [], "cost": 0.0})

check("Send 扇出：3 路写入归并为 3 条", len(out["hits"]) == 3, f"hits={out['hits']}")
check("Send 扇出：累加 reducer 生效", abs(out["cost"] - 0.3) < 1e-9, f"cost={out['cost']}")

print("-" * 72)

# ---- 3. interrupt / resume ----
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402


class HitlState(TypedDict):
    approved: bool
    value: int


def approve_node(state: HitlState) -> dict:
    answer = interrupt({"ask": "接受这份大纲吗？", "outline": ["A", "B"]})
    return {"approved": bool(answer), "value": 1}


g2 = StateGraph(HitlState)
g2.add_node("approve", approve_node)
g2.add_edge(START, "approve")
g2.add_edge("approve", END)

try:
    with SqliteSaver.from_conn_string(":memory:") as cp:
        app2 = g2.compile(checkpointer=cp)
        cfg = {"configurable": {"thread_id": "probe-1"}}
        r1 = app2.invoke({"approved": False, "value": 0}, cfg)
        paused = "__interrupt__" in r1
        check("interrupt：图在节点处暂停", paused, f"keys={list(r1)}")
        r2 = app2.invoke(Command(resume=True), cfg)
        check("Command(resume=)：恢复并跑完", r2.get("approved") is True and r2.get("value") == 1, f"{r2}")
except Exception as exc:  # noqa: BLE001
    check("interrupt/resume 闭环", False, f"{type(exc).__name__}: {exc}")

print("-" * 72)
print("RESULT:", "PASS" if not failures else f"FAIL ({len(failures)}): {failures}")
sys.exit(0 if not failures else 1)
