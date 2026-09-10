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

        hits_only = [t for t in ranked if t[0] > 0]
        chosen = hits_only[:max_results]
        if len(chosen) < self.min_fallback:
            # 兜底：保证链路不断（演示友好），并把不足如实记日志——不假装命中
            picked = {id(t[2]) for t in chosen}
            for t in ranked:
                if id(t[2]) in picked:
                    continue
                chosen.append(t)
                if len(chosen) >= max(self.min_fallback, 1):
                    break
            log.debug(f"[fixture] 查询命中不足，兜底补齐至 {len(chosen)} 条：{query!r}")

        return [
            SearchResult(
                source="web",
                title=rec.get("title", "(无标题)"),
                url=rec.get("url", ""),
                content=rec.get("content", ""),
                score=round(rec.get("score", 0.0), 4),
            )
            for _, _, rec in chosen
        ]
