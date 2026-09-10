"""T3.7 · 矛盾检测：先聚类、再簇内两两比对，成本上限写进配置。

**为什么不能朴素两两比对**（《功能设计》§6.3）：O(n²) 且判定标准模糊。所以：

    同子问题证据 → embedding 聚类 → 仅簇内两两比对
                 → 每簇 ≤ N 对、全局 ≤ M 对，超出按相似度降序截断

"口径不同 ≠ 结论冲突"是本项目刻意强调的一点：判为 `conflict` 的前提是**同一指标出现
互斥数值**（或结论互斥），而不是"两份资料侧重不同"。因此提示词与离线启发式都以
"同一指标、数值差异 ≥ 阈值倍数"为最低门槛，宁缺毋滥——宁可漏报，不可把正常差异渲染成冲突。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from ..logging import get_logger
from ..retrieval.ports import Evidence

log = get_logger(__name__)

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


@dataclass(frozen=True)
class Pair:
    a: Evidence
    b: Evidence
    topic: str
    similarity: float


def group_by_sub_question(evidence: Iterable[Evidence]) -> dict[str, list[Evidence]]:
    groups: dict[str, list[Evidence]] = {}
    for ev in evidence:
        groups.setdefault(ev.sub_question or "(未标注子问题)", []).append(ev)
    return groups


def _cosine(u: Sequence[float], v: Sequence[float]) -> float:
    import numpy as np

    a = np.asarray(u, dtype="float32")
    b = np.asarray(v, dtype="float32")
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def cluster_group(
    items: Sequence[Evidence], *, embed_fn: EmbedFn, threshold: float
) -> list[list[Evidence]]:
    """贪心聚类：按余弦相似度把同一子问题下的证据分簇。

    贪心（而非 k-means）的理由：无需指定簇数、结果确定（同输入同输出），
    便于单测断言——聚类只是**成本闸门**，不是要产出一个漂亮的分类。
    """
    if len(items) <= 1:
        return [list(items)]
    try:
        vecs = embed_fn([e.content for e in items])
    except Exception as exc:  # noqa: BLE001 - 向量服务不可用时应退化为单簇（保守：全比）
        log.warning(f"[conflict] 聚类 embedding 失败，退化为单簇：{type(exc).__name__}: {exc}")
        return [list(items)]

    clusters: list[dict] = []
    for ev, vec in zip(items, vecs):
        best_idx, best_sim = -1, -1.0
        for ci, c in enumerate(clusters):
            sim = _cosine(vec, c["centroid"])
            if sim > best_sim:
                best_sim, best_idx = sim, ci
        if best_idx >= 0 and best_sim >= threshold:
            c = clusters[best_idx]
            c["members"].append((ev, vec))
            c["centroid"] = _mean_vec([v for _, v in c["members"]])
        else:
            clusters.append({"centroid": list(vec), "members": [(ev, vec)]})
    return [[ev for ev, _ in c["members"]] for c in clusters]


def _mean_vec(vecs: Sequence[Sequence[float]]) -> list[float]:
    n = len(vecs)
    dim = len(vecs[0])
    return [sum(v[i] for v in vecs) / n for i in range(dim)]


def select_pairs(
    evidence: Sequence[Evidence],
    *,
    embed_fn: EmbedFn,
    threshold: float,
    per_cluster: int,
    global_cap: int,
) -> list[Pair]:
    """产出**受限数量**的待比对证据对。这是矛盾检测的成本闸门。"""
    # 兜底/零分证据不参与矛盾判定：它不是"命中"，拿它两两比对照样会凭空造出冲突
    # （实测教训：fixture 的兜底补齐曾让"竞品"话题报出市场规模矛盾）。
    usable = [e for e in evidence if float(getattr(e, "score", 0.0) or 0.0) > 0.0]
    dropped = len(evidence) - len(usable)
    if dropped:
        log.info(f"[conflict] 跳过 {dropped} 条兜底/零分证据，仅对真实命中做矛盾比对")

    out: list[Pair] = []
    for subq, items in group_by_sub_question(usable).items():
        for cluster in cluster_group(items, embed_fn=embed_fn, threshold=threshold):
            if len(cluster) < 2:
                continue
            try:
                vecs = embed_fn([e.content for e in cluster])
            except Exception:  # noqa: BLE001
                vecs = [[] for _ in cluster]
            local: list[Pair] = []
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    sim = _cosine(vecs[i], vecs[j]) if vecs[i] and vecs[j] else 0.0
                    local.append(Pair(a=cluster[i], b=cluster[j], topic=subq, similarity=sim))
            local.sort(key=lambda p: p.similarity, reverse=True)
            out.extend(local[:per_cluster])
    out.sort(key=lambda p: p.similarity, reverse=True)
    if len(out) > global_cap:
        log.info(f"[conflict] 待比对 {len(out)} 对，超过全局上限 {global_cap}，按相似度截断")
        out = out[:global_cap]
    log.info(f"[conflict] 受限配对完成：{len(out)} 对（每簇上限 {per_cluster} / 全局上限 {global_cap}）")
    return out


def pairs_as_ids(pairs: Sequence[Pair]) -> list[tuple[str, str, str]]:
    return [(p.a.citation_id, p.b.citation_id, p.topic) for p in pairs]
