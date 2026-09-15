"""T2.1 · Tavily 客户端：额度保护线 + 解析 + 重试（全程离线，不联网）。

**为什么值得单独一组测试**：这条路径在接上真实 key 之前从未被跑过（项目里没有 tavily 录制资产），
而它直接连着**真金白银的额度**（1000 credits/月，1 credit/次）。所以本组钉住三件事：

1. **`replay` 模式绝不联网**——这是额度保护线的全部意义。曾经 `graph/build.py` 不传 mode，
   吃 dataclass 默认的 `auto`，等于"没缓存就联网"：配上 key 跑一轮就静默扣额度。
2. **解析与截断正确**——`raw_content` 是整页正文，不截会顶爆 LLM 上下文与预算。
3. **失败可诊断**——非 JSON 响应（网关 HTML 错误页）要给出可定位的报错，而不是裸
   `JSONDecodeError`；重试要有退避。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from attest.retrieval.tavily_client import API_URL, TavilyClient

SAMPLE = {
    "results": [
        {
            "title": "示例研究院 · 知识库 Agent 市场概览",
            "url": "https://example.com/a",
            "raw_content": "市场规模约 180 亿元，私有化部署占比约 55%。",
            "content": "（截断版）市场规模约 180 亿元。",
            "score": 0.93,
        },
        {
            "title": "示例证券 · 厂商格局",
            "url": "https://example.com/b",
            "content": "三类厂商：云厂商系 / 独立 AI 创业公司 / 传统软件商转型。",
            "score": 0.71,
        },
    ]
}


class _Resp:
    def __init__(self, text: str, status: int = 200) -> None:
        self.text = text
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad", request=None, response=None)  # type: ignore[arg-type]


def _patch_post(monkeypatch: pytest.MonkeyPatch, responses: list[Any]) -> list[dict[str, Any]]:
    """把 httpx.post 换成按序返回的桩；返回调用记录列表。"""
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, *, json: dict[str, Any], timeout: float) -> Any:
        calls.append({"url": url, "body": json, "timeout": timeout})
        item = responses[min(len(calls) - 1, len(responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls


def _client(tmp_path: Path, **kw: Any) -> TavilyClient:
    kw.setdefault("retry_backoff_s", 0.0)  # 测试不真睡
    return TavilyClient("fake-key", tmp_path / "cache", **kw)


# ------------------------------------------------------- 1. 额度保护线（核心）

def test_default_mode_is_replay() -> None:
    """**默认必须 replay**：防回归——默认值回到 auto 就等于"配 key 即烧额度"。"""
    import inspect

    sig = inspect.signature(TavilyClient)
    assert sig.parameters["mode"].default == "replay"


def test_replay_never_touches_network_when_cache_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_post(monkeypatch, [_Resp(json.dumps(SAMPLE))])
    c = _client(tmp_path, mode="replay")
    assert c.search("没录过的 query") == []
    assert calls == [], "replay 模式绝不允许联网"
    assert c.live_requests == 0


def test_replay_returns_cached_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_post(monkeypatch, [_Resp(json.dumps(SAMPLE))])
    c = _client(tmp_path, mode="replay")
    # 先手工落一份录制（模拟"以前录过"）
    path = c._cache_path("已录过的 query", 5)
    path.write_text(json.dumps(SAMPLE, ensure_ascii=False), encoding="utf-8")

    out = c.search("已录过的 query")
    assert len(out) == 2
    assert calls == [], "命中缓存时不该联网"
    assert out[0].url == "https://example.com/a"


def test_record_hits_api_and_writes_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_post(monkeypatch, [_Resp(json.dumps(SAMPLE))])
    c = _client(tmp_path, mode="record")
    out = c.search("新 query")
    assert len(calls) == 1 and calls[0]["url"] == API_URL
    assert calls[0]["body"]["api_key"] == "fake-key"
    assert c.live_requests == 1
    assert len(out) == 2
    # 落盘 + 带 _meta（可追溯"哪次录的"）
    path = c._cache_path("新 query", 5)
    assert path.exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["_meta"]["query"] == "新 query"
    assert "recorded_at" in saved["_meta"]
    assert len(saved["results"]) == 2

    # 第二遍：换 replay 客户端应直接吃缓存（不再联网）
    c2 = _client(tmp_path, mode="replay")
    assert len(c2.search("新 query")) == 2
    assert c2.live_requests == 0


def test_auto_prefers_cache_then_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_post(monkeypatch, [_Resp(json.dumps(SAMPLE))])
    c = _client(tmp_path, mode="auto")
    c.search("q1")  # 未命中 → 联网
    assert len(calls) == 1
    c.search("q1")  # 命中 → 不联网
    assert len(calls) == 1


# ------------------------------------------------------- 2. 解析与截断

def test_parse_prefers_raw_content_and_falls_back_to_content(tmp_path: Path) -> None:
    out = _client(tmp_path)._parse(SAMPLE)
    assert out[0].content.startswith("市场规模约 180 亿元，私有化部署")
    assert "截断版" not in out[0].content, "有 raw_content 时不用 content"
    assert out[1].content.startswith("三类厂商"), "缺 raw_content 时回落到 content"
    assert out[0].source == "web"
    assert out[0].score == pytest.approx(0.93)


def test_parse_tolerates_missing_fields_and_bad_items(tmp_path: Path) -> None:
    payload = {"results": [{"title": None, "url": None}, "非字典项", {"score": "bad", "content": "x"}]}
    out = _client(tmp_path)._parse(payload)
    assert len(out) == 2, "非字典项被跳过"
    assert out[0].title == "" and out[0].url == ""
    assert out[1].score == 0.0, "score 非法时归零而不是抛错"


def test_long_raw_content_is_truncated(tmp_path: Path) -> None:
    payload = {"results": [{"title": "t", "url": "u", "raw_content": "字" * 500, "score": 1.0}]}
    out = _client(tmp_path, max_content_chars=100)._parse(payload)
    assert len(out[0].content) < 200
    assert out[0].content.endswith("（原文过长，已截断）")


def test_truncation_disabled_when_limit_zero(tmp_path: Path) -> None:
    payload = {"results": [{"title": "t", "url": "u", "raw_content": "字" * 500}]}
    out = _client(tmp_path, max_content_chars=0)._parse(payload)
    assert out[0].content == "字" * 500


def test_tavily_zero_score_is_preserved_not_faked(tmp_path: Path) -> None:
    """**契约风险钉桩**：Tavily 的 score 可能为 0，而 T7.10 闸门用 `score > 0` 判"真实命中"。

    本 Adapter 的立场是**如实映射、不伪造分数**（把 0 改成 0.01 属于造假）。
    这条测试把当前行为钉住：score=0 原样保留 —— 若日后决定改闸门判据（按 source 区分），
    改的是 `agents/analyst.py`，不是这里。
    """
    payload = {"results": [{"title": "t", "url": "u", "content": "有效正文", "score": 0}]}
    out = _client(tmp_path)._parse(payload)
    assert out[0].score == 0.0
    assert out[0].content == "有效正文", "score 为 0 不代表内容不可用"


# ------------------------------------------------------- 3. 失败可诊断 + 重试

def test_retries_then_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_post(monkeypatch, [httpx.ConnectError("boom"), _Resp(json.dumps(SAMPLE))])
    c = _client(tmp_path, mode="record", max_retries=2)
    out = c.search("退避重试")
    assert len(out) == 2
    assert len(calls) == 2, "第一次失败后应重试成功"


def test_raises_runtime_error_after_exhausting_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_post(monkeypatch, [httpx.ConnectError("boom")])
    c = _client(tmp_path, mode="record", max_retries=2)
    with pytest.raises(RuntimeError, match="已重试 2 次"):
        c.search("永远失败")
    assert len(calls) == 3, "初试 + 2 次重试"


def test_non_json_response_raises_actionable_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """网关 502 返回 HTML 时，报错要能看出发生了什么（裸 JSONDecodeError 看不出）。"""
    calls = _patch_post(monkeypatch, [_Resp("<html><body>502 Bad Gateway</body></html>")])
    c = _client(tmp_path, mode="record")
    with pytest.raises(RuntimeError, match="不是合法 JSON"):
        c.search("网关挂了")
    assert len(calls) == 1


def test_corrupted_cache_treated_as_miss(tmp_path: Path) -> None:
    c = _client(tmp_path, mode="replay")
    c._cache_path("坏缓存", 5).write_text("{ 这不是 json", encoding="utf-8")
    assert c.search("坏缓存") == [], "缓存损坏按未命中处理，不抛错、不联网"
    assert c.live_requests == 0


def test_legacy_cache_without_meta_is_readable(tmp_path: Path) -> None:
    """兼容旧格式（纯响应体、无 `_meta`）——否则升级后历史录制全部失效。"""
    c = _client(tmp_path, mode="replay")
    c._cache_path("老格式", 5).write_text(json.dumps(SAMPLE, ensure_ascii=False), encoding="utf-8")
    assert len(c.search("老格式")) == 2
