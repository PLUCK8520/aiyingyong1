"""T7.4 · 研究闭环（research_memory）回归测试。

覆盖四块：
  1. `MemoryItem` / `distill_from_report`：只有 supported 能进候选（筛选侧）；
  2. `ResearchMemoryStore.add`：非 supported **一律拒收**（兜底侧，防自我投毒硬闸）、
     同 report 同 claim 去重、超长截断；
  3. `ResearchMemoryStore.search`：命中返回 `source="memory"` + `memory://` URL、
     空库返回空、冷启动从 store.dump() 恢复；
  4. 节点接线：`memory_writer`（审计后沉淀，可关断如实留痕）、
     `scout_local`（双路检索，memory 命中进 evidence 且编号仍是 LOC 系）。

原则与全项目一致：默认离线、不联网、不烧额度。Chroma 用 tmp_path 隔离。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from attest.agents import memory_writer as memory_writer_node
from attest.agents import scout_local as scout_local_node
from attest.agents.base import NodeContext
from attest.budget.account import BudgetConfig
from attest.config import Settings
from attest.llm.gateway import LLMGateway
from attest.memory.research_memory import (
    MAX_CLAIM_CHARS,
    MemoryItem,
    ResearchMemoryStore,
    distill_from_report,
)
from attest.retrieval.mock_search import FixtureSearchClient
from attest.retrieval.ports import Hit, SearchResult
from attest.retrieval.stores import NumpyVectorStore
from attest.schemas import AuditItem
from attest.trace.events import TraceWriter

REPO = Path(__file__).resolve().parents[1]
DIM = 8


# ---------------------------------------------------------------- 小桩


def _embed(texts: list[str]) -> list[list[float]]:
    """确定性伪向量：长度 8，按文本逐字节散列。只用于测试，不追求语义。"""
    out: list[list[float]] = []
    for t in texts:
        v = [0.0] * DIM
        for i, ch in enumerate(t.encode("utf-8")):
            v[i % DIM] += ((ch % 31) - 15) / 15.0
        out.append(v)
    return out


def _store() -> ResearchMemoryStore:
    return ResearchMemoryStore(
        store=NumpyVectorStore(embedder="test", dim=DIM),
        embed_fn=_embed,
        enabled=True,
    )


def _settings(tmp_path: Path, *, enabled: bool = True) -> Settings:
    return Settings(
        llm_mode="mock",
        search_mode="mock",
        trace_dir=tmp_path / "trace",
        report_dir=tmp_path / "reports",
        fixture_dir=REPO / "data" / "fixtures",
        checkpoint_db=tmp_path / "checkpoints.sqlite",
        profile_db=tmp_path / "profile.sqlite",
        research_memory_enabled=enabled,
        research_memory_dir=tmp_path / "rm",
    )


def _ctx(settings: Settings, tmp_path: Path, *, memory: Any = None) -> NodeContext:
    trace = TraceWriter(path=tmp_path / "trace" / "trace.jsonl", run_id="t74-test")
    gateway = LLMGateway(settings=settings, trace=trace)
    return NodeContext(
        settings=settings,
        gateway=gateway,
        search=FixtureSearchClient(settings.fixture_dir / "web"),
        trace=trace,
        budget=BudgetConfig(),
        memory=memory,
    )


class _LocalStub:
    """scout_local 的用户资料库桩：只回一条 docs 命中。"""

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self._results = results or [
            SearchResult(
                source="local",
                title="用户资料A",
                url="local://docs/a#0",
                content="用户资料正文：市场规模 180 亿元。",
                score=0.9,
            )
        ]

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        return list(self._results)


# ---------------------------------------------------------------- 1. 沉淀筛选（distill）


def test_distill_keeps_only_supported() -> None:
    items = [
        AuditItem(citation_id="[LOC1-1-1]", verdict="supported", sentence="市场规模 180 亿元。", section="核心摘要"),
        AuditItem(citation_id="[LOC1-1-2]", verdict="unsupported", sentence="增速 40%。", section="核心摘要"),
        AuditItem(citation_id="[LOC1-1-3]", verdict="partial", sentence="共 5 家玩家。", section="竞品格局"),
    ]
    out = distill_from_report(report_id="r1", audit_items=items)
    assert len(out) == 1
    assert out[0].claim == "市场规模 180 亿元。"
    assert out[0].evidence_ids == ("[LOC1-1-1]",)
    assert out[0].section == "核心摘要"


def test_distill_skips_empty_claim() -> None:
    items = [AuditItem(citation_id="[LOC1-1-1]", verdict="supported", sentence="  ", section="x")]
    assert distill_from_report(report_id="r1", audit_items=items) == []


# ---------------------------------------------------------------- 2. 入库硬闸（防自我投毒）


def test_add_rejects_non_supported_verdicts() -> None:
    mem = _store()
    written, rejected = mem.add(
        [
            MemoryItem(report_id="r1", claim="可信结论", audit_verdict="supported"),
            MemoryItem(report_id="r1", claim="不可信结论", audit_verdict="unsupported"),
            MemoryItem(report_id="r1", claim="半可信结论", audit_verdict="partial"),
        ]
    )
    assert (written, rejected) == (1, 2)
    assert mem.count() == 1
    assert mem.items()[0].claim == "可信结论"


def test_add_rejects_even_if_distill_bypassed() -> None:
    """兜底侧：即使调用方漏筛，直接把 unsupported 塞给 add 也必须被拒——双保险。"""
    mem = _store()
    written, rejected = mem.add(
        [MemoryItem(report_id="r1", claim="漏网之鱼", audit_verdict="unsupported")]
    )
    assert (written, rejected) == (0, 1)
    assert mem.count() == 0


def test_add_dedupes_same_report_same_claim() -> None:
    mem = _store()
    item = MemoryItem(report_id="r1", claim="市场规模 180 亿元", audit_verdict="supported")
    first = mem.add([item])
    second = mem.add([item])  # 重跑/rewrite 重复提交
    assert first == (1, 0)
    assert second == (0, 0)
    assert mem.count() == 1


def test_add_truncates_overlong_claim() -> None:
    mem = _store()
    long_claim = "长" * (MAX_CLAIM_CHARS + 50)
    written, _ = mem.add(
        [MemoryItem(report_id="r1", claim=long_claim, audit_verdict="supported")]
    )
    assert written == 1
    assert len(mem.items()[0].claim) == MAX_CLAIM_CHARS + 1  # +1 是省略号「…」


def test_add_disabled_store_is_noop() -> None:
    mem = _store()
    mem.enabled = False
    written, rejected = mem.add(
        [MemoryItem(report_id="r1", claim="x", audit_verdict="supported")]
    )
    assert (written, rejected) == (0, 0)


# ---------------------------------------------------------------- 3. 检索


def test_search_returns_memory_source_hits() -> None:
    mem = _store()
    mem.add([MemoryItem(report_id="r1", claim="市场规模 180 亿元", audit_verdict="supported")])
    hits = mem.search("市场规模是多少", max_results=3)
    assert len(hits) == 1
    assert hits[0].source == "memory"
    assert hits[0].url.startswith("memory://r1#")
    assert hits[0].content == "市场规模 180 亿元"


def test_search_empty_store_returns_empty() -> None:
    mem = _store()
    assert mem.search("随便什么问题") == []


def test_search_disabled_returns_empty() -> None:
    mem = _store()
    mem.enabled = False
    assert mem.search("随便什么问题") == []


def test_search_recovers_from_store_dump_after_cold_start() -> None:
    """冷启动：进程重启后 `_ids/_metas` 是空的，但向量库里还有数据——
    search 必须能从 store.dump() 把元数据捞回来，而不是静默返回空。"""
    shared = NumpyVectorStore(embedder="test", dim=DIM)
    mem1 = ResearchMemoryStore(store=shared, embed_fn=_embed)
    mem1.add([MemoryItem(report_id="r1", claim="历史结论 X", audit_verdict="supported")])

    # 模拟重启：同一块底层 store，新的门面实例（内存副本为空）
    mem2 = ResearchMemoryStore(store=shared, embed_fn=_embed)
    # count() 冷启动回退到底层 store 计数——宁可如实报"库里有 1 条"，也不报 0 误导观测
    assert mem2.count() == 1
    hits = mem2.search("历史结论", max_results=3)
    assert len(hits) == 1
    assert hits[0].content == "历史结论 X"


def test_purge_report_removes_in_memory_items() -> None:
    """purge 是真删：内存索引与底层向量库同步撤除，count（以 store 为准）同步减少。"""
    mem = _store()
    mem.add(
        [
            MemoryItem(report_id="r1", claim="r1 的结论", audit_verdict="supported"),
            MemoryItem(report_id="r2", claim="r2 的结论", audit_verdict="supported"),
        ]
    )
    removed = mem.purge_report("r1")
    assert removed == 1
    assert mem.count() == 1
    assert mem.items()[0].report_id == "r2"


def test_purge_report_works_after_cold_start() -> None:
    """冷启动后 purge：内存副本为空也能撤——会先从 store.dump() 恢复再删。"""
    shared = NumpyVectorStore(embedder="test", dim=DIM)
    mem1 = ResearchMemoryStore(store=shared, embed_fn=_embed)
    mem1.add([MemoryItem(report_id="r1", claim="r1 的结论", audit_verdict="supported")])

    mem2 = ResearchMemoryStore(store=shared, embed_fn=_embed)
    assert mem2.purge_report("r1") == 1
    assert mem2.count() == 0
    assert mem2.search("r1 的结论") == []


# ---------------------------------------------------------------- 4a. memory_writer 节点


def test_memory_writer_skips_when_disabled(tmp_path: Path) -> None:
    ctx = _ctx(_settings(tmp_path, enabled=False), tmp_path, memory=_store())
    out = memory_writer_node.run({"thread_id": "r1", "audit_items": []}, ctx)
    assert out == {}
    events = ctx.trace.of("memory_write")
    assert len(events) == 1 and "未启用" in events[0].get("skipped", "")


def test_memory_writer_skips_when_not_assembled(tmp_path: Path) -> None:
    ctx = _ctx(_settings(tmp_path), tmp_path, memory=None)
    out = memory_writer_node.run({"thread_id": "r1", "audit_items": []}, ctx)
    assert out == {}
    events = ctx.trace.of("memory_write")
    assert len(events) == 1 and "未装配" in events[0].get("skipped", "")


def test_memory_writer_writes_only_supported(tmp_path: Path) -> None:
    mem = _store()
    ctx = _ctx(_settings(tmp_path), tmp_path, memory=mem)
    state = {
        "thread_id": "r1",
        "audit_items": [
            AuditItem(citation_id="[LOC1-1-1]", verdict="supported", sentence="可信句子", section="核心摘要"),
            AuditItem(citation_id="[LOC1-1-2]", verdict="unsupported", sentence="不可信句子", section="核心摘要"),
        ],
    }
    out = memory_writer_node.run(state, ctx)
    assert out == {}
    assert mem.count() == 1
    events = ctx.trace.of("memory_write")
    assert events[0]["written"] == 1
    assert events[0]["rejected"] == 0  # unsupported 在 distill 侧就被筛掉了，不进 add


# ---------------------------------------------------------------- 4b. scout_local 双路检索


def test_scout_local_dual_retrieval_merges_docs_and_memory(tmp_path: Path) -> None:
    mem = _store()
    mem.add([MemoryItem(report_id="r0", claim="历史结论：市场规模约 180 亿元", audit_verdict="supported")])
    ctx = _ctx(_settings(tmp_path), tmp_path, memory=mem)
    ctx.local = _LocalStub()

    out = scout_local_node.run(
        {"sub_question": "市场规模", "subq_no": 1, "round": 1}, ctx
    )
    evidence = out["evidence"]
    urls = {e.url for e in evidence}
    assert any(u.startswith("memory://") for u in urls), "历史结论没进 evidence"
    assert any(u.startswith("local://") for u in urls), "docs 命中丢了"
    # 编号体系不破坏：仍是 LOC 系（memory 是 local 语义的一档）
    assert all(e.citation_id.startswith("[LOC1-1-") for e in evidence)
    # 观测字段：谁被复用了、有没有进正文
    assert out["memory_hits"] and out["memory_hits"][0]["in_context"] is True
    # trace 分源计数
    ret = ctx.trace.of("retrieval")[-1]
    assert ret["memory_hits"] == 1 and ret["docs_hits"] == 1


def test_scout_local_without_memory_behaves_as_before(tmp_path: Path) -> None:
    ctx = _ctx(_settings(tmp_path), tmp_path, memory=None)
    ctx.local = _LocalStub()
    out = scout_local_node.run(
        {"sub_question": "市场规模", "subq_no": 1, "round": 1}, ctx
    )
    assert len(out["evidence"]) == 1
    assert "memory_hits" not in out


def test_scout_local_slot_allocation_docs_first_memory_reserved(tmp_path: Path) -> None:
    """槽位分配：docs 优先 + 历史结论固定位（2026-09-12 smoke 翻红的回归钉）。

    3 条 docs + 3 条 memory 时，结果必须是 docs 2 + memory 1——
    不能 memory 占满（用户资料消失），也不能 docs 占满（闭环形同虚设）。
    """
    mem = _store()
    for i in range(3):
        mem.add([MemoryItem(report_id=f"r{i}", claim=f"历史结论 {i}", audit_verdict="supported")])
    docs = [
        SearchResult(source="local", title=f"资料{i}", url=f"local://docs/{i}#0",
                     content=f"用户资料 {i}", score=0.9 - i * 0.01)
        for i in range(3)
    ]
    ctx = _ctx(_settings(tmp_path), tmp_path, memory=mem)
    ctx.local = _LocalStub(docs)

    out = scout_local_node.run({"sub_question": "q", "subq_no": 1, "round": 1}, ctx)
    urls = [e.url for e in out["evidence"]]
    assert len(urls) == 3
    n_docs = sum(1 for u in urls if u.startswith("local://"))
    n_mem = sum(1 for u in urls if u.startswith("memory://"))
    assert (n_docs, n_mem) == (2, 1), f"槽位分配错误：docs={n_docs} memory={n_mem}"


def test_scout_local_memory_failure_does_not_break_run(tmp_path: Path) -> None:
    """一路检索炸了不能炸掉整轮（降级不停机，D4）：memory 检索抛异常 →
    docs 结果照常返回，错误进 errors 留痕。"""

    class _BoomMemory:
        def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
            raise RuntimeError("向量库连接断开")

    ctx = _ctx(_settings(tmp_path), tmp_path, memory=_BoomMemory())
    ctx.local = _LocalStub()
    out = scout_local_node.run(
        {"sub_question": "市场规模", "subq_no": 1, "round": 1}, ctx
    )
    assert len(out["evidence"]) == 1
    assert out["errors"] and "research_memory" in out["errors"][0]["error"]


# ---------------------------------------------------------------- 5. 装配（build_context）


def test_build_context_auto_assembles_memory(tmp_path: Path) -> None:
    from attest.graph.build import build_context

    settings = _settings(tmp_path, enabled=True)
    ctx = build_context(settings, run_id="t74-auto")
    assert isinstance(ctx.memory, ResearchMemoryStore)


def test_build_context_disabled_gives_none(tmp_path: Path) -> None:
    from attest.graph.build import build_context

    settings = _settings(tmp_path, enabled=False)
    ctx = build_context(settings, run_id="t74-off")
    assert ctx.memory is None


def test_build_context_explicit_override(tmp_path: Path) -> None:
    """测试/对照组需要隔离存储时，显式传 None 或自带实例必须生效。"""
    from attest.graph.build import build_context

    settings = _settings(tmp_path, enabled=True)
    ctx = build_context(settings, run_id="t74-override", memory=None)
    assert ctx.memory is None
    mem = _store()
    ctx2 = build_context(settings, run_id="t74-override2", memory=mem)
    assert ctx2.memory is mem
