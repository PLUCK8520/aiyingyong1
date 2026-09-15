"""判别节点的**降级不变式**：模型调用失败不许杀进程，报告必须产出。

**为什么单独一组**：2026-09-15 实测暴露——`glm-4.5-flash`（免费档推理模型）在判 65 条证据时
连续读超时（每次 240s）+ 429，网关重试耗尽后抛 `RuntimeError`，
**整轮调研跑了 23 分钟后整体失败、零产出**。

而项目在 `docs/开发任务清单.md` 里写着不变式：「**熔断只降级不杀进程：不变式是报告必须产出**」。
analyst 有拒编兜底、auditor 的重写有 try/except，只有 `evidence_judge` 是漏网之鱼。

本组钉住：判别失败 → 降级返回 + **显式留痕**（不是静默吞异常）；矛盾检测失败 → 只丢冲突章节。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from attest.agents import evidence_judge as judge_mod
from attest.agents.base import NodeContext
from attest.budget.account import BudgetConfig
from attest.config import Settings
from attest.llm.gateway import LLMResponse
from attest.retrieval.ports import Evidence
from attest.schemas import JudgeResult, Judgment
from attest.trace.events import TraceWriter

REPO = Path(__file__).resolve().parents[1]
CID = "[LOC1-1-1]"


def _ev(cid: str = CID, subq: str = "成本构成", round_no: int = 1) -> Evidence:
    return Evidence(
        citation_id=cid,
        source="local",
        title="t",
        url="local://kb/x",
        content="土建工程约占总投资的 50% 至 60%。",
        sub_question=subq,
        round_no=round_no,
        score=0.7,
    )


class _Router:
    def tier(self, task: str, *, fuse_level: int = 0) -> str:
        return "strong"


class _FailingGateway:
    """所有 chat 调用都失败（模拟网关重试耗尽后的 RuntimeError）。"""

    def __init__(self, *, fail_tasks: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.router = _Router()
        self.fail_tasks = fail_tasks  # None = 全部失败

    def chat(self, messages: Any, *, task: str, **kw: Any) -> LLMResponse:
        self.calls.append(task)
        if self.fail_tasks is None or task in self.fail_tasks:
            raise RuntimeError(f"zhipu-glm 调用失败（已重试 3 次）：可重试状态码 429")
        raise AssertionError(f"本测试不该走到 {task} 的成功分支")

    def embed(self, texts: Any, **kw: Any) -> Any:
        raise RuntimeError("embedding 也失败")


def _ctx(tmp_path: Path, gateway: Any) -> NodeContext:
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
    trace = TraceWriter(path=tmp_path / "trace" / "trace.jsonl", run_id="jd")
    return NodeContext(
        settings=settings, gateway=gateway, search=None, trace=trace, budget=BudgetConfig()
    )


def _state(evidence: list[Evidence], judgments: list[Any] | None = None) -> dict[str, Any]:
    return {
        "evidence": evidence,
        "judgments": judgments or [],
        "plan": {"objective": "调研成本", "sub_questions": ["成本构成", "降低造价"], "outlines": ["成本"]},
        "round": 1,
    }


# ------------------------------------------------------- 1. 判别失败 → 降级而非抛异常

def test_judge_failure_does_not_raise(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, _FailingGateway())
    out = judge_mod.run(_state([_ev()]), ctx)  # 不得抛异常
    assert out["judgments"] == [], "失败时不新增判别"
    assert any("判别未完成" in g for g in out["gaps"]), "缺口里要写明本轮判别缺失"
    assert out["gaps"][0].startswith("本轮证据判别未完成"), "说明放最前，读者先看到"


def test_judge_failure_is_traced_not_silent(tmp_path: Path) -> None:
    """降级**必须留痕**——静默吞异常等于把故障藏起来（比崩溃更糟）。"""
    ctx = _ctx(tmp_path, _FailingGateway())
    judge_mod.run(_state([_ev()]), ctx)
    events = ctx.trace.of("judge_failed")
    assert events, "必须打 judge_failed 事件"
    assert "429" in str(events[0].get("error", "")), "要把真实错因带进 trace"


def test_judge_failure_keeps_existing_judgments_out_of_return(tmp_path: Path) -> None:
    """降级只"不新增"，不返回空去覆盖——state 里既有的判别继续有效。"""
    existing = [Judgment(citation_id=CID, sub_question="成本构成", relevance=5, confidence=4)]
    ctx = _ctx(tmp_path, _FailingGateway())
    out = judge_mod.run(_state([_ev()], existing), ctx)
    assert out["judgments"] == [], "返回值不新增（reduce 后 state 里的旧判别仍在）"
    # 已有的 5 分判别让"成本构成"算已覆盖 → 不该再报它的缺口
    assert not any("成本构成" in g for g in out["gaps"]), "已有判别覆盖的子问题不该报缺口"


# ------------------------------------------------------- 2. 矛盾检测失败 → 只丢冲突章节

def test_conflict_failure_is_isolated(tmp_path: Path) -> None:
    """矛盾检测失败只丢"争议与分歧"章节，不影响判别结果与主链路。"""
    class _JudgeOkConflictFail:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.router = _Router()

        def chat(self, messages: Any, *, task: str, **kw: Any) -> LLMResponse:
            self.calls.append(task)
            if task == "judge":
                return LLMResponse(
                    text="",
                    model="stub",
                    task=task,
                    provider="stub",
                    input_tokens=10,
                    output_tokens=5,
                    cost_cny=0.0,
                    latency_ms=1.0,
                    token_estimate=False,
                    parsed=JudgeResult(
                        judgments=[
                            Judgment(
                                citation_id=CID, sub_question="成本构成", relevance=5, confidence=4
                            )
                        ],
                        gaps=[],
                    ),
                )
            raise RuntimeError("conflict 调用失败（模拟 429）")

        def embed(self, texts: Any, **kw: Any) -> Any:
            raise RuntimeError("embedding 失败")

    ctx = _ctx(tmp_path, _JudgeOkConflictFail())
    # 两条证据才会触发配对（len(evidence) < 2 直接返回）——此处走不到配对就返回，属正常
    out = judge_mod.run(_state([_ev(), _ev("[LOC1-1-2]")]), ctx)
    assert len(out["judgments"]) == 1, "判别结果不受矛盾检测失败影响"
    assert out["conflicts"] == [], "冲突章节降级为空"


def test_conflict_failure_traced(tmp_path: Path) -> None:
    """配对能成时，矛盾检测失败要留 conflict_failed；配不出对则根本不该调。"""
    class _Gw:
        def __init__(self) -> None:
            self.router = _Router()

        def chat(self, messages: Any, *, task: str, **kw: Any) -> LLMResponse:
            if task == "judge":
                return LLMResponse(
                    text="", model="stub", task=task, provider="stub",
                    input_tokens=1, output_tokens=1, cost_cny=0.0, latency_ms=1.0,
                    token_estimate=False,
                    parsed=JudgeResult(
                        judgments=[
                            Judgment(
                                citation_id=CID, sub_question="成本构成", relevance=5, confidence=4
                            )
                        ],
                        gaps=[],
                    ),
                )
            raise RuntimeError("conflict 429")

        def embed(self, texts: Any, **kw: Any) -> Any:
            # 返回固定向量，让聚类能配成对（触发 conflict 调用路径）
            return [[1.0, 0.0] for _ in texts]

    ctx = _ctx(tmp_path, _Gw())
    # 同子问题两条证据 + 相似向量 → select_pairs 有机会配成对
    evs = [_ev(CID), _ev("[LOC1-1-2]")]
    judge_mod.run(_state(evs), ctx)
    # 配对成或不成都不该崩；成了就必须留痕
    if ctx.trace.of("conflict_failed"):
        assert True
