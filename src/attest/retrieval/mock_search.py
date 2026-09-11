"""T2.1 · 离线检索 Adapter（fixture 回放）。

按任务的硬要求：**开发调试与测试默认走回放，只有录 fixture 与验收才打真实 API**
（额度瓶颈在开发期，不在演示期）。所以这个 Adapter 不是"测试补丁"，是默认路径。

打分方式刻意做得**确定且可解释**：关键词命中数 → 基础分，便于单测断言与失败复现。

⚠️ **多主题隔离（T7.1 起必需）**：fixture 从 1 个主题扩到 4 个后，出现了跨主题污染——
「规模」「市场」这类通用 keyword 会让知识库 query 命中新能源汽车的记录，
进而污染证据集（凭空造矛盾、把真缺口误判为已覆盖）。
因此这里加了一层**主题路由**：query 命中某主题的 `topic_keywords` 时，
只在该主题的 records 内检索。识别不到主题（或命中多个主题）**不静默**——
回退全库并打 warning，让"路由失效"这件事可见。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..logging import get_logger
from .ports import SearchResult

log = get_logger(__name__)


@dataclass
class FixtureSearchClient:
    fixture_dir: Path
    min_fallback: int = 2
    #: 是否启用主题路由（关闭则退回"全库混合检索"的旧行为，供对照与排查）
    topic_routing: bool = True
    #: 一个 query 命中超过这么多个主题时，视为无法判别 → 回退全库
    max_topics_per_query: int = 1

    _records: list[dict] = field(default_factory=list, repr=False)
    #: topic → 该主题的 records
    _by_topic: dict[str, list[dict]] = field(default_factory=dict, repr=False)
    #: topic → 特征词
    _topic_kw: dict[str, list[str]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for path in sorted(self.fixture_dir.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                log.warning(f"[fixture] 跳过无法解析的文件 {path.name}: {exc}")
                continue

            meta = payload.get("_meta") or {}
            topic = str(meta.get("topic") or "").strip()
            topic_kws = [str(k) for k in (meta.get("topic_keywords") or []) if k]

            recs = []
            for rec in payload.get("records", []):
                rec["_file"] = path.name
                rec["_topic"] = topic
                recs.append(rec)
                self._records.append(rec)

            if topic:
                self._by_topic.setdefault(topic, []).extend(recs)
                if topic_kws:
                    self._topic_kw.setdefault(topic, []).extend(topic_kws)

        n_topic = len(self._by_topic)
        log.info(
            f"[fixture] 载入 {len(self._records)} 条离线检索记录，"
            f"{n_topic} 个主题，来源目录 {self.fixture_dir}"
        )
        if self.topic_routing and n_topic and not self._topic_kw:
            log.warning("[fixture] 启用了主题路由但没有任何 topic_keywords，路由将始终回退全库")

    # ---------------------------------------------------------------- 主题路由

    def _route(self, query: str) -> tuple[list[dict], str]:
        """按 query 选择候选记录集。返回 (候选 records, 路由说明)。"""
        if not self.topic_routing or not self._topic_kw:
            return self._records, "全库（未启用主题路由）"

        hit_topics = [
            t for t, kws in self._topic_kw.items() if any(kw in query for kw in kws)
        ]
        if len(hit_topics) == 1:
            return self._by_topic[hit_topics[0]], f"主题={hit_topics[0]}"
        if not hit_topics:
            # 不静默：路由失败必须可见，否则会以为隔离生效了
            log.warning(f"[fixture] 未识别出主题，回退全库检索（可能导致跨主题污染）：{query!r}")
            return self._records, "全库（未识别主题）"
        if len(hit_topics) > self.max_topics_per_query:
            log.warning(
                f"[fixture] query 命中多个主题 {hit_topics}，无法判别，回退全库：{query!r}"
            )
            return self._records, f"全库（命中多主题 {hit_topics}）"
        return self._by_topic[hit_topics[0]], f"主题={hit_topics[0]}"

    # ---------------------------------------------------------------- 检索

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        pool, route_note = self._route(query)
        ranked: list[tuple[int, float, dict]] = []
        for rec in pool:
            hits = sum(1 for kw in rec.get("keywords", []) if kw and kw in query)
            ranked.append((hits, float(rec.get("score", 0.0)), rec))
        ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)

        real = [t for t in ranked if t[0] > 0][:max_results]
        chosen = list(real)
        fallback_ids: set[int] = set()
        if len(chosen) < self.min_fallback:
            # 兜底：只保证链路不断（演示友好），但**必须让下游知道这是兜底**——
            # 否则不相关文档会污染证据集：凭空造出"矛盾"、把真实缺口误判成"已覆盖"。
            picked = {id(t[2]) for t in chosen}
            for t in ranked:
                if id(t[2]) in picked:
                    continue
                chosen.append(t)
                picked.add(id(t[2]))
                fallback_ids.add(id(t[2]))
                if len(chosen) >= max(self.min_fallback, 1):
                    break
            log.warning(
                f"[fixture] 真实命中 {len(real)} 条，不足 min_fallback={self.min_fallback}，"
                f"兜底补齐 {len(fallback_ids)} 条（标 score=0）：{query!r}"
            )

        return [
            SearchResult(
                source="web",
                title=rec.get("title", "(无标题)"),
                url=rec.get("url", ""),
                content=rec.get("content", ""),
                # 兜底条目标 0 分：下游（矛盾比对 / 缺口覆盖）据此排除，不假装命中
                score=0.0 if id(rec) in fallback_ids else round(rec.get("score", 0.0), 4),
            )
            for _, _, rec in chosen
        ]

