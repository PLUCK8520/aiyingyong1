"""T2.1 · 离线检索 Adapter（fixture 回放）。

按任务的硬要求：**开发调试与测试默认走回放，只有录 fixture 与验收才打真实 API**
（额度瓶颈在开发期，不在演示期）。所以这个 Adapter 不是"测试补丁"，是默认路径。

打分方式刻意做得**确定且可解释**：关键词命中数 → 基础分，便于单测断言与失败复现。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..logging import get_logger
from .ports import SearchResult

log = get_logger(__name__)


@dataclass
class FixtureSearchClient:
    fixture_dir: Path
    min_fallback: int = 2

    def __post_init__(self) -> None:
        self._records: list[dict] = []
        for path in sorted(self.fixture_dir.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                log.warning(f"[fixture] 跳过无法解析的文件 {path.name}: {exc}")
                continue
            for rec in payload.get("records", []):
                rec["_file"] = path.name
                self._records.append(rec)
        log.info(f"[fixture] 载入 {len(self._records)} 条离线检索记录，来源目录 {self.fixture_dir}")

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        ranked: list[tuple[int, float, dict]] = []
        for rec in self._records:
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
