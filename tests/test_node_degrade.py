"""四个节点的**降级不变式**：模型调用失败不许杀进程（报告/回答必须产出）。

**为什么单独一组**：2026-09-15 实测 `glm-4.5-flash`（免费档推理模型）在大证据量下
连续超时 + 429，网关重试耗尽后抛 `RuntimeError` → **整轮 23 分钟白跑、零产出**。
顺着这条故障链做了一次系统性排查：8 处 `ctx.gateway.chat()` 调用里，
`analyst`（主调用）/ `planner` / `intent_router` / `direct_responder` **四处都没有保护**。

项目不变式（`docs/开发任务清单.md` ADR-02）：「**熔断只降级不杀进程：不变式是报告必须产出**」。

本组钉住四处降级各自的**语义**（不是"随便返回点东西"就算降级）：
  - analyst：产出「成文失败页」+ **真实证据清单**，且**不伪造任何结论**；
  - planner：降级为**最小可用计划**（单子问题），宁可少不可假；
  - intent  ：降级走 **research**（宁可多做，不可漏做）；
  - direct ：**如实说明失败**，绝不静默返回空白。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from attest.agents import analyst as analyst_mod
from attest.agents import direct_responder, intent_router, planner
from attest.agents.base import NodeContext
from attest.budget.account import BudgetConfig
from attest.config import Settings
from attest.retrieval.ports import SearchResult
from attest.trace.events import TraceWriter
from attest.retrieval.ports import Evidence

REPO = Path(__file__).resolve().parents[1]


class _Router:
    def tier(self, task: str, *, fuse_level: int = 0) -> str:
        return "strong"


class _FailingGateway:
    """所有 chat 调用都失败——模拟网关重试耗尽。"""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.router = _Router()

    def chat(self, messages: Any, *, task: str, **kw: Any) -> Any:
        self.calls.append(task)
        raise RuntimeError("zhipu-glm 调用失败（已重试 3 次）：可重试状态码 429")

    def embed(self, texts: Any, **kw: Any) -> Any:
        raise RuntimeError("embedding 失败")


class _NoSearch:
    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        return []


def _ctx(tmp_path: Path, gateway: Any = None) -> NodeContext:
    settings = Settings(
        llm_mode="mock",
        search_mode="mock",
        trace_dir=tmp_path / "trace",
        report_dir=tmp_path / "reports",
        fixture_dir=REPO / "data" / "fixtures",
        checkpoint_db=tmp_path / "cp.sqlite",
        profile_db=tmp_path / "profile.sqlite",
        local_docs_dir=tmp_path / "nodocs",
    )
    settings.ensure_dirs()
    trace = TraceWriter(path=tmp_path / "trace" / "trace.jsonl", run_id="nd")
    return NodeContext(
        settings=settings, gateway=gateway or _FailingGateway(), search=_NoSearch(),
        trace=trace, budget=BudgetConfig(),
    )


def _ev(cid: str = "[LOC1-1-1]", score: float = 0.7) -> Evidence:
    return Evidence(
        citation_id=cid,
        source="local",
        title="城市轨道交通成本观察",  # 真实标题不含编号形态（否则会被 extract_ids 误抽）
        url="local://kb/doc#1",
        content="土建工程约占总投资的 50% 至 60%。",
        sub_question="成本构成",
        round_no=1,
        score=score,
    )


# ------------------------------------------------------- analyst：成文失败页

def _analyst_state() -> dict[str, Any]:
    return {
        "evidence": [_ev()],
        "judgments": [],
        "plan": {
            "objective": "调研城市轨道交通建设成本",
            "outlines": ["一、成本构成"],
            "sub_questions": ["成本构成"],
        },
    }


def test_analyst_failure_yields_failure_page_with_evidence(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    out = analyst_mod.run(_analyst_state(), ctx)  # 不得抛异常

    report = out["report"]
    assert "成文失败" in report
    assert "不是「证据不足」" in report, "必须说清这是模型故障，不是证据问题"
    assert "429" in report, "失败原因要如实写出来"
    assert "城市轨道交通成本观察" in report, "已检索到的证据要交出来（它是真的）"
    assert "local://kb/doc#1" in report, "带链接，用户能直接查阅"
    assert out["model_tier"] == "failed"


def test_analyst_failure_page_has_no_fabricated_body(tmp_path: Path) -> None:
    """**绝不伪造正文**：产物里只能有失败说明 + 证据清单，不能有结论。"""
    ctx = _ctx(tmp_path)
    out = analyst_mod.run(_analyst_state(), ctx)
    check = out["citation_check"]
    # 不含引用编号（没有正文就没有引用契约）→ no_citations 如实为真
    assert check["no_citations"] is True, "失败页没有可回查的正文，必须如实标记"
    assert ctx.trace.of("analyst_failed"), "必须留痕"


# ------------------------------------------------------- planner：最小可用计划

def test_planner_failure_degrades_to_minimal_plan(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    out = planner.run({"query": "调研城市轨道交通建设成本"}, ctx)

    plan = out["plan"]
    assert plan["objective"] == "调研城市轨道交通建设成本"
    assert plan["sub_questions"] == ["调研城市轨道交通建设成本"], "用原问题当唯一子问题"
    assert plan["outlines"] == ["调研发现"]
    assert ctx.trace.of("planner_failed"), "必须留痕"


def test_planner_degraded_plan_is_schema_valid(tmp_path: Path) -> None:
    """降级 plan 必须过 Plan 的校验（否则下游从 state 读出来就炸）。"""
    from attest.schemas import Plan

    ctx = _ctx(tmp_path)
    out = planner.run({"query": "调研 X"}, ctx)
    Plan(**out["plan"])  # 不抛即合法


# ------------------------------------------------------- intent：降级走 research

def test_intent_failure_degrades_to_research(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    out = intent_router.run({"query": "帮我调研一下"}, ctx)
    assert out["route"] == "research", "宁可多做，不可漏做"
    assert out["route_reason"], "要说明为什么走这条路"
    assert ctx.trace.of("intent_failed")


# ------------------------------------------------------- direct：如实说明失败

def test_direct_failure_returns_explicit_message_not_blank(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    out = direct_responder.run({"query": "什么是向量检索"}, ctx)
    ans = out["direct_answer"]
    assert ans.strip(), "绝不能静默返回空白"
    assert "未能生成回答" in ans
    assert "429" in ans, "失败原因要写出来"
    assert ctx.trace.of("direct_failed")


# ------------------------------------------------------- 横切：四处都不抛异常

def test_all_four_nodes_survive_gateway_outage(tmp_path: Path) -> None:
    """网关全挂时，四个节点都必须**返回**而不是抛异常——这是不变式的底线。"""
    gw = _FailingGateway()
    ctx = _ctx(tmp_path, gw)
    intent_router.run({"query": "q"}, ctx)
    planner.run({"query": "q"}, ctx)
    direct_responder.run({"query": "q"}, ctx)
    analyst_mod.run(_analyst_state(), ctx)
    assert sorted(set(gw.calls)) == ["analyst", "direct", "intent", "planner"]
