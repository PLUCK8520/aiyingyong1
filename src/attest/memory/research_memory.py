"""T7.4 · 研究闭环：research_memory 沉淀与双 collection 检索。

**要解决的问题**：每次调研都把结论丢掉，下次同类问题从零开始。本模块把**已核验为
`supported` 的结论**沉淀进独立向量集合 `research_memory`，下次调研时可被 `scout_local`
检索命中并复用（计入 LOC 引用）。

**为什么要单开一个 collection**（而不是塞进 `docs`）：
  - **来源不同**：`docs` 是用户上传/采集的原始资料；`research_memory` 是我们**自己产出的结论**。
    两者混在一起，"这条结论是谁说的"就分不清了，而本项目整个立论就是"结论必须能回指来源"。
  - **生命周期不同**：原始资料长存；研究结论需要能**按 report_id 回溯清理**（万一某份报告
    后来被证明有误，要能把它的沉淀一起撤掉）。带 `report_id` 才能做到。

🔴 **防自我投毒（本模块最重要的约束）**：
  只允许 `audit_verdict == "supported"` 的结论入库。
  `unsupported`（证据不支持）与 `partial`（部分支持）**一律拒收**——
  否则本次调研的错误结论会在下次调研里变成"可信来源"，错误被复利放大。
  这条不是可配置的开关，是写死的硬闸（见 `ResearchMemoryStore.add`）。

⚠️ **与用户画像（T5.2）的区别**：画像是**用户偏好**（小、键值对）；
research_memory 是**研究结论**（大、向量库）。两者受众与生命周期都不同，故分模块分存储。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from ..logging import get_logger
from ..retrieval.ports import Hit, SearchResult, VectorStore

log = get_logger(__name__)

#: 允许入库的审计结论——**只有这一个**。改这里等于关掉防自我投毒，需评审。
ALLOWED_VERDICTS = frozenset({"supported"})

#: 单条沉淀的 claim 长度上限——超长的"结论"其实是正文片段，入库会污染检索
MAX_CLAIM_CHARS = 400


@dataclass(frozen=True)
class MemoryItem:
    """一条沉淀的研究结论（字段对齐《项目方案》§6 MemoryItem）。

    - `report_id`：来源报告，用于**回溯清理**（哪份报告错了就撤哪批沉淀）
    - `evidence_ids`：支撑该结论的证据编号——保留它才能"二次复用时可回查原始出处"
    - `audit_verdict`：**必须 supported 才允许入库**
    """

    report_id: str
    claim: str
    evidence_ids: tuple[str, ...] = ()
    audit_verdict: str = "supported"
    created_at: str = ""
    section: str = ""

    def to_metadata(self) -> dict[str, Any]:
        """转成向量库 metadata（Chroma 不支持嵌套 list，故 `evidence_ids` 用逗号串）。"""
        return {
            "report_id": self.report_id,
            "claim": self.claim,
            "evidence_ids": ",".join(self.evidence_ids),
            "audit_verdict": self.audit_verdict,
            "created_at": self.created_at or _now_iso(),
            "section": self.section,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ResearchMemoryStore:
    """research_memory 的读写门面。**唯一入口**——不允许绕过它直接写向量库。"""

    store: VectorStore
    embed_fn: Any  # Callable[[Sequence[str]], list[list[float]]]
    enabled: bool = True
    _ids: list[str] = field(default_factory=list)
    _metas: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------ 写

    def add(self, items: Iterable[MemoryItem]) -> tuple[int, int]:
        """写入沉淀。返回 (写入条数, 拒收条数)。

        **拒收是常态而不是异常**：一次调研里总有 unsupported/partial 的结论，
        它们被安静地挡在门外——这正是防自我投毒在工作。所以不抛错、不告警刷屏，
        只记 debug，并把计数返回给调用方留痕。
        """
        if not self.enabled:
            return 0, 0

        accepted: list[MemoryItem] = []
        rejected = 0
        for it in items:
            claim = (it.claim or "").strip()
            if it.audit_verdict not in ALLOWED_VERDICTS:
                rejected += 1
                log.debug(f"[memory] 拒收非 supported 结论（verdict={it.audit_verdict}）：{claim[:40]!r}")
                continue
            if not claim:
                rejected += 1
                continue
            if len(claim) > MAX_CLAIM_CHARS:
                claim = claim[:MAX_CLAIM_CHARS] + "…"
            accepted.append(
                MemoryItem(
                    report_id=it.report_id,
                    claim=claim,
                    evidence_ids=tuple(it.evidence_ids),
                    audit_verdict=it.audit_verdict,
                    created_at=it.created_at or _now_iso(),
                    section=it.section,
                )
            )

        if not accepted:
            return 0, rejected

        # 去重：同一 report 下同一 claim 只存一次（重跑/rewrite 会重复提交）
        existing = {(m.get("report_id"), m.get("claim")) for m in self._metas}
        fresh = [a for a in accepted if (a.report_id, a.claim) not in existing]
        dup = len(accepted) - len(fresh)
        if not fresh:
            log.info(f"[memory] 本次 {len(accepted)} 条结论全部已存在（去重 {dup} 条），无新增")
            return 0, rejected

        ids = [f"{a.report_id}::{i}" for i, a in enumerate(fresh)]
        metas = [a.to_metadata() for a in fresh]
        try:
            vectors = self.embed_fn([a.claim for a in fresh])
            self.store.upsert(ids, vectors, [a.claim for a in fresh], metas)
        except Exception as exc:  # noqa: BLE001 - 沉淀失败不该影响主流程（报告已经产出了）
            log.warning(f"[memory] 沉淀写入失败（不影响本次报告）：{type(exc).__name__}: {exc}")
            return 0, rejected

        self._ids.extend(ids)
        self._metas.extend(metas)
        log.info(f"[memory] 沉淀 {len(fresh)} 条 supported 结论；拒收 {rejected} 条；去重 {dup} 条")
        return len(fresh), rejected

    # ------------------------------------------------------------ 读

    def search(self, query: str, *, max_results: int = 3) -> list[SearchResult]:
        """检索历史研究结论。返回 `source="memory"` 的 SearchResult，与 docs 结果同构。"""
        if not self.enabled or self.store.count() == 0:
            return []
        try:
            qvec = self.embed_fn([query])[0]
            hits: list[Hit] = self.store.query(qvec, top_k=max_results)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[memory] 检索失败，跳过历史结论：{type(exc).__name__}: {exc}")
            return []

        # 向量库只回 id/score；正文与元数据从内存副本取（与 docs 检索同一套路）
        by_id = {i: m for i, m in zip(self._ids, self._metas)}
        # 冷启动（进程重启）：store 里可能有数据但内存副本为空 → 从 store 恢复
        if not by_id and hits:
            ids, docs, metas = self.store.dump()
            by_id = {i: {**m, "claim": d} for i, m, d in zip(ids, metas, docs)}

        out: list[SearchResult] = []
        for h in hits:
            meta = by_id.get(h.chunk_id)
            if not meta:
                continue
            out.append(
                SearchResult(
                    source="memory",
                    title=f"历史研究结论 · {meta.get('report_id', '?')}",
                    # 用 memory:// 方案，与 local:// 一样不假造 http 链接
                    url=f"memory://{meta.get('report_id', '?')}#{h.chunk_id}",
                    content=str(meta.get("claim", "")),
                    score=round(float(h.score), 6),
                )
            )
        return out

    # ------------------------------------------------------------ 维护

    def purge_report(self, report_id: str) -> int:
        """按 report_id 撤掉某份报告的全部沉淀（回溯清理）——**真删**。

        这是"带 report_id"的实际收益：某份报告后来被发现有误，能定向撤掉它的沉淀，
        而不是重建整个集合。底层向量库同步删除（`VectorStore.delete`），
        不存在"内存里看不见、重启又复活"的残留。
        """
        # 冷启动（内存副本为空但库里有数据）：先从 store 恢复，否则 purge 会漏
        if not self._ids and self.store.count():
            ids, docs, metas = self.store.dump()
            self._ids = list(ids)
            self._metas = [{**m, "claim": d} for m, d in zip(metas, docs)]
        kept_ids, kept_metas, removed_ids = [], [], []
        for i, m in zip(self._ids, self._metas):
            if m.get("report_id") == report_id:
                removed_ids.append(i)
            else:
                kept_ids.append(i)
                kept_metas.append(m)
        if removed_ids:
            try:
                self.store.delete(removed_ids)
            except Exception as exc:  # noqa: BLE001 - 底层删失败也要撤内存索引，并如实留痕
                log.warning(f"[memory] 底层向量删除失败（内存索引已撤，重启可能残留）：{type(exc).__name__}: {exc}")
            self._ids, self._metas = kept_ids, kept_metas
            log.info(f"[memory] 已撤销 report={report_id} 的 {len(removed_ids)} 条沉淀")
        return len(removed_ids)

    def count(self) -> int:
        """可检索的沉淀条数——以底层 store 为准（冷启动内存副本可能还没恢复）。"""
        return self.store.count()

    def items(self) -> list[MemoryItem]:
        """导出当前沉淀（供评测与人工核对）。"""
        return [
            MemoryItem(
                report_id=str(m.get("report_id", "")),
                claim=str(m.get("claim", "")),
                evidence_ids=tuple(filter(None, str(m.get("evidence_ids", "")).split(","))),
                audit_verdict=str(m.get("audit_verdict", "")),
                created_at=str(m.get("created_at", "")),
                section=str(m.get("section", "")),
            )
            for m in self._metas
        ]


# ---------------------------------------------------------------- 沉淀的筛选规则

def distill_from_report(
    *,
    report_id: str,
    audit_items: Sequence[Any],
    sentences: Sequence[tuple[str, str, str]] | None = None,
) -> list[MemoryItem]:
    """从审计结果里挑出**可沉淀**的结论。

    只有 `verdict == supported` 的句子才进候选——这一步是"防自我投毒"的**筛选**侧，
    `ResearchMemoryStore.add` 里还有**兜底**侧（双保险：即使调用方漏筛也会被拒）。

    Args:
        audit_items: `AuditItem` 列表（含 `verdict` / `sentence` / `citation_id` / `section`）。
        sentences: 可选的 (句子, 证据编号, 章节) 覆盖列表；不给则从 audit_items 自取。
    """
    out: list[MemoryItem] = []
    for it in audit_items:
        verdict = str(getattr(it, "verdict", "") or "")
        if verdict not in ALLOWED_VERDICTS:
            continue
        claim = str(getattr(it, "sentence", "") or "").strip()
        if not claim:
            continue
        cid = str(getattr(it, "citation_id", "") or "")
        out.append(
            MemoryItem(
                report_id=report_id,
                claim=claim,
                evidence_ids=(cid,) if cid else (),
                audit_verdict=verdict,
                section=str(getattr(it, "section", "") or ""),
            )
        )
    return out
