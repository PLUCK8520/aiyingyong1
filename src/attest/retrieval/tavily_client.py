"""T2.1 · Tavily 检索 Adapter（含录制/回放）。

**额度保护线**：免费档 1000 credits/月、basic 1 credit/次（①额度数字待复核，见审查报告 F-4），
而开发期一轮调研要 3~5 次检索——额度瓶颈在**开发期**，不在演示期。所以本 Adapter 有三种模式：

  - `replay`（**默认**）：只读 `data/cache/` 下的录制；命中即返回，**未命中返回空、绝不联网**。
    这是默认值不是洁癖：2026-09-15 复核发现，此前 `mode` 的 dataclass 默认是 `auto` 且
    `graph/build.py` 不传该参数，等于"未命中就联网"——配上 key 后跑一轮就静默扣额度，
    与本模块文档自称的"默认 replay"**相反**。已修为配置项 `ATTEST_TAVILY_MODE` 显式传入。
  - `record`：强制走真实 API 并把响应落盘（消耗 1 credit/次）。录新 query 时显式设它。
  - `auto`：有缓存用缓存、没缓存就联网。方便，但**会悄悄烧额度**——排查/演示时慎用。

⚠️ **score 契约风险（接真实检索后必然要面对，先记在这里）**：
    Tavily 返回的 `score` 是 0~1 的**相似度**，**可能为 0**；而 `agents/analyst.py` 的
    T7.10 证据充足性闸门用 `score > 0` 作为"**真实命中**（非兜底填充）"的判据——
    那个语义来自离线 fixture：`mock_search` 把"主题未识别时的兜底填充"标成 `score=0`。
    两者语义不同却共用同一字段。后果：**若 Tavily 对某条结果给了 0 分而其余字段正常，
    该条会被闸门判成"填充物"，极端情况下整份报告被判"证据不足"而拒编。**
    本模块**不伪造分数**（把 0 改成 0.01 属于造假），只如实映射；接上真实 key 后
    应先用一轮实跑确认 Tavily 的 score 分布，再决定是改闸门判据（按 source 区分）
    还是改 Adapter 映射。测试 `test_tavily_zero_score_is_preserved_not_faked` 钉住了当前行为。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx

from ..logging import get_logger
from .ports import SearchResult

log = get_logger(__name__)

API_URL = "https://api.tavily.com/search"
Mode = Literal["auto", "replay", "record"]

#: 默认原文截断上限（字符）。Tavily 的 `raw_content` 是整页正文，动辄数万字符——
#: 不截会顶爆 LLM 上下文与成本预算。调用方通常传 `settings.context_truncate_chars`。
DEFAULT_MAX_CONTENT_CHARS = 4000
_TRUNCATE_SUFFIX = "…（原文过长，已截断）"


@dataclass
class TavilyClient:
    api_key: str
    cache_dir: Path
    #: ⚠️ 默认 `replay`——额度安全优先。见模块文档串的额度保护线说明。
    mode: Mode = "replay"
    timeout: float = 30.0
    max_retries: int = 2
    #: 单条结果的正文截断上限（字符）。0 或负数 = 不截断（不建议：整页正文会顶爆上下文）。
    max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS
    #: 重试退避基数（秒）：第 n 次重试前等待 base * 2^(n-1)。对 429/5xx 有效。
    retry_backoff_s: float = 0.5
    #: 真实联网次数计数（仅用于日志与自检，**不参与任何业务判断**）。
    live_requests: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, query: str, max_results: int) -> Path:
        key = hashlib.sha1(f"{query}|{max_results}".encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{key}.json"

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        path = self._cache_path(query, max_results)

        if self.mode in ("auto", "replay") and path.exists():
            log.debug(f"[tavily] 回放缓存 {path.name} <- {query!r}")
            return self._parse(self._read_cache(path))

        if self.mode == "replay":
            # **静默返回空是最糟的选择**：上游只会看到"这次检索没结果"，无从知道
            # "其实是因为没录过这个 query"。所以这里给一条可执行的警告。
            log.warning(
                f"[tavily] replay 模式缓存未命中，返回空（**未联网、未消耗额度**）：{query!r}\n"
                f"  → 要录这条 query：设 ATTEST_TAVILY_MODE=record 后重跑一次；"
                f"或改用 ATTEST_SEARCH_MODE=mock 走离线 fixture。"
            )
            return []

        payload = self._loads(self._request(query, max_results))
        self._write_cache(path, query, payload)
        log.info(f"[tavily] 已录制 {path.name} <- {query!r}（消耗 1 credit）")
        return self._parse(payload)

    # ---------------- 缓存读写 ----------------

    @staticmethod
    def _loads(text: str) -> dict[str, Any]:
        """把 HTTP 响应体解析成 dict；非 JSON 时给出**可定位**的错误（而不是裸 JSONDecodeError）。

        真实 API 出错时返回的可能是 HTML 错误页（网关 502、CDN 拦截页），
        裸 `JSONDecodeError: Expecting value` 完全看不出发生了什么。
        """
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            head = (text or "")[:200].replace("\n", " ")
            raise RuntimeError(f"Tavily 响应不是合法 JSON（前 200 字符：{head!r}）") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Tavily 响应不是 JSON 对象，而是 {type(payload).__name__}")
        return payload

    @staticmethod
    def _read_cache(path: Path) -> dict[str, Any]:
        """读缓存。录制文件带 `_meta` 但不依赖它——旧格式（纯响应体）同样可读。"""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(f"[tavily] 缓存文件损坏，按未命中处理：{path.name}（{exc}）")
            return {"results": []}
        return payload if isinstance(payload, dict) else {"results": []}

    @staticmethod
    def _write_cache(path: Path, query: str, payload: dict[str, Any]) -> None:
        """落盘录制。包一层 `_meta` 记下 query 与时间——排查"这条缓存是哪次录的"必需，
        且**不破坏 `_parse`**（它只读 `results`）。"""
        body = {
            "_meta": {
                "query": query,
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
                "api_url": API_URL,
            },
            **payload,
        }
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 网络 ----------------

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
            if attempt:
                # 退避：429（rate limit）与 5xx 立刻重试基本必然再失败，白等不如等一会儿。
                delay = self.retry_backoff_s * (2 ** (attempt - 1))
                log.debug(f"[tavily] 第 {attempt + 1} 次尝试前退避 {delay:.1f}s")
                import time as _time

                _time.sleep(delay)
            try:
                self.live_requests += 1
                resp = httpx.post(API_URL, json=body, timeout=self.timeout)
                resp.raise_for_status()
                return resp.text
            except (httpx.HTTPError, httpx.TimeoutException) as exc:  # noqa: PERF203
                last = exc
                log.warning(f"[tavily] 第 {attempt + 1} 次失败：{type(exc).__name__}: {exc}")
        raise RuntimeError(f"Tavily 检索失败（已重试 {self.max_retries} 次）：{last}") from last

    # ---------------- 解析 ----------------

    def _truncate(self, text: str) -> str:
        limit = int(self.max_content_chars or 0)
        if limit > 0 and len(text) > limit:
            return text[:limit].rstrip() + _TRUNCATE_SUFFIX
        return text

    def _parse(self, payload: dict[str, Any]) -> list[SearchResult]:
        out: list[SearchResult] = []
        for item in payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            raw = item.get("raw_content") or item.get("content") or ""
            try:
                score = float(item.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            out.append(
                SearchResult(
                    source="web",
                    title=str(item.get("title") or ""),
                    url=str(item.get("url") or ""),
                    content=self._truncate(str(raw)),
                    score=score,
                )
            )
        return out
