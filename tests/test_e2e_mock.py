"""T2.6 端到端（离线）：两条分支各跑一遍，断言产出与 trace 完整性。

这是"改坏了立刻能发现"的第一块地板——全程 mock，不联网、不烧额度。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from attest.config import Settings
from attest.graph.build import build_context, build_graph, initial_state

REPO = Path(__file__).resolve().parents[1]
RESEARCH_QUERY = "调研'企业知识库 Agent 平台'市场，按市场规模/竞品/收费模式三部分输出，附溯源链接"
DIRECT_QUERY = "你好，用一句话说明什么是向量检索。"


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_mode="mock",
        search_mode="mock",
        trace_dir=tmp_path / "trace",
        report_dir=tmp_path / "reports",
        fixture_dir=REPO / "data" / "fixtures",
    )


def _run(settings: Settings, query: str, thread: str):
    ctx = build_context(settings, run_id=thread)
    app = build_graph(ctx)
    return app.invoke(initial_state(query, thread_id=thread)), ctx


def test_research_branch_produces_cited_report(settings: Settings) -> None:
    final, ctx = _run(settings, RESEARCH_QUERY, "e2e-research")

    assert final["route"] == "research"
    assert len(final["plan"]["sub_questions"]) >= 1
    # Send 扇出：3 路子问题各检索到证据并**归并**（不是互相覆盖）
    assert len(final["evidence"]) >= 3, f"evidence={len(final.get('evidence') or [])}"

    check = final["citation_check"]
    assert check["unresolved"] == [], f"报告里有查无来源的编号：{check['unresolved']}"
    assert check["pass"] is True
    assert check["referenced"] >= 1
    assert "[WEB" in final["report"]
    assert "## 参考资料" in final["report"]

    # 预算：网关记账 → 节点累加 → state 有值
    assert final["tokens_incurred"] > 0
    assert final["cost_incurred"] >= 0.0

    # trace：node_start / node_end 必须成对
    summary = ctx.trace.summary()
    assert summary["node_pairs_ok"] is True, summary["unmatched_nodes"]
    assert summary["llm_calls"] >= 4  # intent + planner + judge + analyst


def test_direct_branch_skips_retrieval(settings: Settings) -> None:
    final, ctx = _run(settings, DIRECT_QUERY, "e2e-direct")

    assert final["route"] == "direct"
    assert final["direct_answer"].strip()
    assert not final.get("evidence"), "direct 分支不应触发检索"
    assert not final.get("report")
    nodes = {e.get("node") for e in ctx.trace.of("node_start")}
    assert "direct_responder" in nodes
    assert "scout_web" not in nodes
    assert ctx.trace.summary()["node_pairs_ok"] is True


def test_budget_incurred_matches_trace_llm_cost(settings: Settings) -> None:
    """state 的累计成本必须等于 trace 里所有 llm_call 的合计——网关是唯一记账点。"""
    final, ctx = _run(settings, RESEARCH_QUERY, "e2e-cost")
    from_trace = round(sum(e["cost_cny"] for e in ctx.trace.of("llm_call")), 6)
    assert round(final["cost_incurred"], 6) == pytest.approx(from_trace, abs=1e-6)
    tokens_from_trace = sum(e["total_tokens"] for e in ctx.trace.of("llm_call"))
    assert final["tokens_incurred"] == tokens_from_trace
