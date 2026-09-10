"""T0.5 · Trace：JSONL 事件流。

三个数据源解耦（架构设计 §7）：
  - `trace.jsonl`  → 审计与评测的数据源（后端写）
  - SSE           → 展示（P6）
  - 评测报告      → 聚合产物（P7）
**前端不直接读 trace 文件。**

事件成对：每个节点的 `node_start` / `node_end` 必须成对，这是冒烟清单第 3 条。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceWriter:
    path: Path
    run_id: str = "local"
    echo: bool = False

    _events: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _fh: Any = field(default=None, init=False, repr=False)

    # ---------------- 写 ----------------
    def emit(self, event: str, **data: Any) -> dict[str, Any]:
        record = {"ts": round(time.time(), 4), "run_id": self.run_id, "event": event, **data}
        with self._lock:
            self._events.append(record)
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("a", encoding="utf-8")
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()
        return record

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    # ---------------- 读 ----------------
    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def of(self, event: str) -> list[dict[str, Any]]:
        return [e for e in self._events if e["event"] == event]

    # ---------------- 汇总 ----------------
    def summary(self) -> dict[str, Any]:
        """冒烟清单要看的：节点成对、token、成本、耗时。"""
        starts = {e.get("node") for e in self.of("node_start")}
        ends = {e.get("node") for e in self.of("node_end")}
        llm_calls = self.of("llm_call")
        # node_start 可能因并行重复出现，用计数比对
        start_counts: dict[str, int] = {}
        end_counts: dict[str, int] = {}
        for e in self.of("node_start"):
            start_counts[e.get("node")] = start_counts.get(e.get("node"), 0) + 1
        for e in self.of("node_end"):
            end_counts[e.get("node")] = end_counts.get(e.get("node"), 0) + 1
        unmatched = {n: (start_counts[n], end_counts.get(n, 0)) for n in start_counts if start_counts[n] != end_counts.get(n, 0)}
        return {
            "run_id": self.run_id,
            "nodes": sorted(starts | ends),
            "node_pairs_ok": not unmatched,
            "unmatched_nodes": unmatched,
            "llm_calls": len(llm_calls),
            "input_tokens": sum(c.get("input_tokens", 0) for c in llm_calls),
            "output_tokens": sum(c.get("output_tokens", 0) for c in llm_calls),
            "total_tokens": sum(c.get("total_tokens", 0) for c in llm_calls),
            "cost_cny": round(sum(c.get("cost_cny", 0.0) for c in llm_calls), 6),
            "duration_ms": round(
                (self._events[-1]["ts"] - self._events[0]["ts"]) * 1000, 1
            )
            if self._events
            else 0.0,
        }


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：CJK 按 1 字 1 token，其余按 4 字符 1 token。

    真实调用优先用 API 返回的 usage；只有在 mock / 无 usage 时才用它，
    并在 trace 里标 `token_estimate=true`，避免把估算当账单。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + max(1, other // 4)
