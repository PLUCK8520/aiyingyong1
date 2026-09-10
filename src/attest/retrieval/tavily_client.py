"""T2.1 · Tavily 检索 Adapter（含录制/回放）。

额度保护线的落点：默认 `replay`（命中本地缓存才返回），只有显式 `record` 或缓存未命中且
`auto` 时才打真实 API。真实响应落盘到 `data/cache/`，之后开发调试全部回放。

⚠️ Tavily 免费额度 **1000 credits/月**（basic 1 credit/次）——额度瓶颈在开发期，不在演示期。
   该额度数字**本轮未复核成功**（审查报告 F-4），开工后请在控制台确认。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from ..logging import get_logger
from .ports import SearchResult

log = get_logger(__name__)

API_URL = "https://api.tavily.com/search"
Mode = Literal["auto", "replay", "record"]


@dataclass
class TavilyClient:
    api_key: str
    cache_dir: Path
    mode: Mode = "auto"
    timeout: float = 30.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, query: str, max_results: int) -> Path:
        key = hashlib.sha1(f"{query}|{max_results}".encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{key}.json"

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        path = self._cache_path(query, max_results)

        if self.mode in ("auto", "replay") and path.exists():
            log.debug(f"[tavily] 回放缓存 {path.name} <- {query!r}")
            return self._parse(json.loads(path.read_text(encoding="utf-8")))

        if self.mode == "replay":
            log.warning(f"[tavily] replay 模式下缓存未命中，返回空：{query!r}")
            return []

        payload = json.loads(self._request(query, max_results))
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(f"[tavily] 已录制 {path.name} <- {query!r}（消耗 1 credit）")
        return self._parse(payload)

    def _request(self, query: str, max_results: int) -> str:
        body: dict[str, Any] = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "include_raw_content": True,
            "search_depth": "basic",
        }
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = httpx.post(API_URL, json=body, timeout=self.timeout)
                resp.raise_for_status()
                return resp.text
            except (httpx.HTTPError, httpx.TimeoutException) as exc:  # noqa: PERF203
                last = exc
                log.warning(f"[tavily] 第 {attempt + 1} 次失败：{exc}")
        raise RuntimeError(f"Tavily 检索失败：{last}") from last

    @staticmethod
    def _parse(payload: dict[str, Any]) -> list[SearchResult]:
        out: list[SearchResult] = []
        for item in payload.get("results", []):
            out.append(
                SearchResult(
                    source="web",
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("raw_content") or item.get("content") or "",
                    score=float(item.get("score", 0.0) or 0.0),
                )
            )
        return out
