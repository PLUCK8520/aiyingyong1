"""T8.6 · 引用契约的**确定性兜底**：零引用必须显示为「未通过回查」，而不是"看起来干净"。

**为什么单独一组测试**：真实运行暴露的漏洞不是"引用写错了"，而是"**没有引用**"：
  1. `analyst` 可能整篇不写引用编号（实测 glm-4-flash：905 字正文，0 个编号）；
  2. 此时 `auditor` **没有任何对象可审** → `total=0 / unsupported=0`，摘要一片干净；
  3. 于是**证据根本不支持的句子**（实测凭空生成过"财政补贴 / 税收优惠 / 社会资本"）
     反而比"引错编号"更容易混过去——引错会被 unresolved 抓住，不引则整张网失效。

所以本组测试钉住三条：
  - 校验层能**识别**"零引用"（`citation_check.no_citations`）；
  - 审计摘要能把"没有结论"与"审计通过"**区分开**（`audit_summary.inconclusive`）；
  - 报告正文里出现**读者看得见的**横幅，且该判定基于**降级前**的正文
    （否则审计降级会把两类故障混成一句横幅）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from attest.agents import analyst as analyst_mod
from attest.agents import auditor as auditor_mod
from attest.agents.base import NodeContext
from attest.budget.account import BudgetConfig
from attest.config import Settings
from attest.llm.gateway import LLMResponse
from attest.quality.citation_auditor import RuleCitationAuditor
from attest.quality.citation_check import check_report
from attest.retrieval.citations import CitationIndex
from attest.retrieval.ports import Evidence, SearchResult
from attest.trace.events import TraceWriter

REPO = Path(__file__).resolve().parents[1]

SUBQS = ["城市轨道交通建设的主要成本构成", "降低造价的常见做法"]
CID = "[LOC1-1-1]"
DOC = "土建工程约占总投资的 50% 至 60%，其中盾构掘进与车站土建是主要支出项。"


# --------------------------------------------------------------------- 工具


def _ev(cid: str = CID, content: str = DOC, subq: str = SUBQS[0], *, score: float = 0.7) -> Evidence:
    return Evidence(
        citation_id=cid,
        source="local",
        title="城市轨道交通建设成本观察",
        url="local://kb_upload_test.md#abc",
        content=content,
        sub_question=subq,
        round_no=1,
        score=score,
    )


class _Router:
    def tier(self, task: str, *, fuse_level: int = 0) -> str:
        return "strong"


class _StubGateway:
    """按顺序返回预置正文；用真实的 `LLMResponse`（重试路径会 `dataclasses.replace` 它）。

    `raise_on` 用来关掉章节重写：审计节点的重写分支会拿模型输出**整段替换**该章节
    （`parse_rewrite_output` 在没有 `<<SECTION_BODY>>` 标记时宽松退化为整段），
    会把本节正文连同引用一起换掉，干扰"横幅判定"这组断言。
    """

    def __init__(self, texts: list[str], *, raise_on: set[str] | None = None) -> None:
        self.texts = texts
        self.raise_on = raise_on or set()
        self.calls: list[str] = []
        self.router = _Router()

    def chat(self, messages: Any, *, task: str, **kw: Any) -> LLMResponse:
        self.calls.append(task)
        if task in self.raise_on:
            raise RuntimeError(f"stub: {task} 被禁用")
        text = self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]
        return LLMResponse(
            text=text,
            model="stub",
            task=task,
            provider="stub",
            input_tokens=100,
            output_tokens=200,
            cost_cny=0.001,
            latency_ms=5.0,
            token_estimate=False,
        )

    def embed(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise AssertionError("本组测试不应调用 embedding")


def _settings(tmp_path: Path) -> Settings:
    s = Settings(
        llm_mode="mock",
        search_mode="mock",
        trace_dir=tmp_path / "trace",
        report_dir=tmp_path / "reports",
        fixture_dir=REPO / "data" / "fixtures",
        checkpoint_db=tmp_path / "cp.sqlite",
        profile_db=tmp_path / "profile.sqlite",
        local_docs_dir=tmp_path / "nodocs",
    )
    s.ensure_dirs()
    return s


class _NoSearch:
    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        return []


def _ctx(tmp_path: Path, *, gateway: Any = None, with_auditor: bool = True) -> NodeContext:
    settings = _settings(tmp_path)
    trace = TraceWriter(path=tmp_path / "trace" / "trace.jsonl", run_id="t86")
    return NodeContext(
        settings=settings,
        gateway=gateway or _StubGateway(["# 报告\n\n（无）\n"], raise_on={"analyst_rewrite"}),
        search=_NoSearch(),
        trace=trace,
        budget=BudgetConfig(),
        auditor=RuleCitationAuditor() if with_auditor else None,
    )


def _state(evidence: list[Evidence], report: str = "") -> dict[str, Any]:
    return {
        "evidence": evidence,
        "judgments": [],
        "plan": {"objective": "调研城市轨道交通建设成本", "outlines": ["一、成本构成", "二、降低造价"], "sub_questions": SUBQS},
        "report": report,
        "evidence_sufficiency": {"sufficient": True, "n_evidence": len(evidence)},
    }


# ------------------------------------------- 1. 校验层：能否识别"零引用"


def test_check_flags_body_without_any_citation() -> None:
    idx = CitationIndex.from_evidence([_ev()])
    check = check_report("# 报告\n\n土建约占一半，盾构是主要支出。\n", idx)
    assert check["no_citations"] is True
    assert check["pass"] is False
    assert check["referenced"] == 0


def test_check_does_not_flag_when_cited_or_index_empty() -> None:
    idx = CitationIndex.from_evidence([_ev()])
    assert check_report(f"# 报告\n\n土建约占 50%~60% {CID}。\n", idx)["no_citations"] is False
    # 索引为空时"没有引用"是正常的（拒编页 / 无证据），不该报
    assert check_report("# 报告\n\n没有证据。\n", CitationIndex.from_evidence([]))["no_citations"] is False


# ------------------------------------------- 2. 纠偏重试：只在必要时、且只一次


def test_retry_tail_appends_to_last_message_without_mutating_input() -> None:
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    out = analyst_mod._with_retry_tail(msgs)
    assert msgs[-1]["content"] == "U", "不得就地修改调用方传入的消息"
    assert out[0] == msgs[0], "system 消息不该动"
    assert out[-1]["content"].startswith("U")
    assert "自检" in out[-1]["content"], "硬约束要追加在末尾"


def test_analyst_retries_once_and_recovers_citations(tmp_path: Path) -> None:
    """首轮零引用 → 追加末尾约束重试 → 拿到编号；**两轮 token 都要计入成本**。"""
    gw = _StubGateway(["# 报告\n\n土建约占一半。\n", f"# 报告\n\n土建约占 50%~60% {CID}。\n"])
    ctx = _ctx(tmp_path, gateway=gw)
    out = analyst_mod.run(_state([_ev()]), ctx)

    assert gw.calls == ["analyst", "analyst"], "应当正好重试一次"
    assert CID in out["report"]
    assert out["citation_check"]["referenced"] == 1
    assert out["citation_check"]["no_citations"] is False
    assert out["tokens_incurred"] == 2 * (100 + 200), "重试的开销必须计入，否则成本面板会漏账"


def test_analyst_gives_up_after_one_retry_and_flags_missing(tmp_path: Path) -> None:
    gw = _StubGateway(["# 报告\n\n土建约占一半。\n"])  # 两次都零引用
    ctx = _ctx(tmp_path, gateway=gw)
    out = analyst_mod.run(_state([_ev()]), ctx)

    assert len(gw.calls) == 2, "只重试一次，不得无界重试"
    assert out["citation_check"]["no_citations"] is True
    assert out["citation_check"]["pass"] is False
    assert ctx.trace.of("citation_missing"), "零引用必须留痕（否则只剩下日志）"


def test_analyst_does_not_retry_when_no_evidence(tmp_path: Path) -> None:
    """无可用证据时"没有引用"是正常的——不该白烧一次调用。"""
    gw = _StubGateway(["# 报告\n\n（证据不足）\n"])
    ctx = _ctx(tmp_path, gateway=gw)
    # 无判别结果 → 退化为 score>0；score=0 会被判为"非真实命中"→ 不足 → 拒编路径
    out = analyst_mod.run(_state([_ev(score=0.0)]), ctx)
    assert out["model_tier"] == "refused"
    assert gw.calls == [], "拒编路径不得调用模型"


# ------------------------------------------- 3. 审计节点：横幅 + 显式"无结论"


def test_auditor_flags_unchecked_report(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    out = auditor_mod.run(_state([_ev()], report="# 城市轨道交通成本\n\n土建约占一半。\n"), ctx)

    assert "未通过逐句回查" in out["report"], "读者必须能在正文里看到"
    assert out["audit_summary"]["inconclusive"] is True
    assert out["audit_summary"]["inconclusive_reason"], "要说明为什么没有结论"
    assert out["audit_summary"]["total"] == 0
    assert ctx.trace.of("audit_inconclusive"), "必须留痕，供评测统计"


def test_auditor_keeps_quiet_when_report_is_cited(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    out = auditor_mod.run(
        _state([_ev()], report=f"# 城市轨道交通成本\n\n土建工程约占总投资的 50% 至 60% {CID}。\n"), ctx
    )
    assert "未通过逐句回查" not in out["report"]
    assert out["audit_summary"]["inconclusive"] is False


def test_auditor_banner_judges_before_degrade(tmp_path: Path) -> None:
    """**判定必须基于降级前的正文**：审计把不支撑的编号降级成「（未证实）」后，
    正文里就没有编号了——若在降级后判定，会把"引了但没通过"误报成"根本没引"，
    两类故障显示成同一句横幅，排查时会被带偏。
    """
    ctx = _ctx(tmp_path)
    # 句子里的数字在证据里不存在 → 规则审计判 unsupported → 降级去掉编号。
    # 正文形状与其它审计测试一致（有 `## 章节`，否则拆句器拿不到 claim）。
    out = auditor_mod.run(
        _state([_ev()], report=f"# 城市轨道交通成本\n\n## 摘要\n\n土建占比高达 97% {CID}。\n"), ctx
    )
    assert out["audit_summary"]["total"] == 1, "这句话是被审计过的……"
    assert out["audit_summary"]["unsupported"] == 1
    assert out["audit_summary"]["inconclusive"] is False, "……所以不属于「零引用」那类故障"
    assert "未通过逐句回查" not in out["report"]
