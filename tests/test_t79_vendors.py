"""T7.9 · 多厂商接入（硅基流动 / OpenAI 兼容）回归测试。

钉住这次踩过的三个坑，避免以后再犯：
  ① **key 与厂商必须对得上**——把 A 家的 key 填进 B 家的配置，报错是 401「key 无效」，
     看着像"key 坏了"，实际是走错门（这次真实踩过：硅基流动的 key 打百炼 → 401）；
  ② **免费托管的模型要计 token 但不计费**，且不能与"本地模型"混为一类（语义不同）；
  ③ **rerank 的响应结构各家不同**（硅基流动是 `results`，百炼是 `output.results`），
     按索引取、不假设返回顺序。

全部离线：httpx 一律打桩，不发真实请求。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from attest.config import Settings
from attest.llm.gateway import build_provider
from attest.llm.providers import (
    DashScopeProvider,
    OpenAICompatProvider,
    SiliconFlowProvider,
)
from attest.retrieval.rerank import SiliconFlowReranker
from attest.trace.pricing import (
    FREE_EMBED_MODELS,
    FREE_HOSTED_MODELS,
    LOCAL_MODELS,
    UnknownModelError,
    compute_cost,
    compute_embed_cost,
)


def _settings(**kw) -> Settings:
    base = dict(llm_mode="mock", search_mode="mock")
    base.update(kw)
    return Settings(**base)


# ---------------------------------------------------------------- ① 厂商 / 域名 / key


def test_provider_name_flows_into_logs_and_meta() -> None:
    """厂商名要能区分——否则排查"这次是谁在跑"只能靠猜。"""
    assert OpenAICompatProvider("k", "https://x/v1").name == "openai-compat"
    assert DashScopeProvider("k", "https://x/v1").name == "dashscope"
    assert SiliconFlowProvider("k", "https://x/v1").name == "siliconflow"


def test_build_provider_selects_siliconflow() -> None:
    p = build_provider(_settings(llm_mode="siliconflow", siliconflow_api_key="sk-test"))
    assert isinstance(p, SiliconFlowProvider)
    assert p.name == "siliconflow"


def test_siliconflow_without_key_fails_with_actionable_hint() -> None:
    """缺 key 要给出可执行的中文提示，并且明确警告"别把别家 key 填进来"。"""
    with pytest.raises(ValueError) as ei:
        _settings(llm_mode="siliconflow")
    msg = str(ei.value)
    assert "SILICONFLOW_API_KEY" in msg
    assert "走错门" in msg, "这次真实踩过的坑必须写在报错里"


def test_siliconflow_rejects_foreign_domain() -> None:
    """把百炼的域名配给 siliconflow → 直接拦下（防止把 A 家 key 打到 B 家）。"""
    with pytest.raises(ValueError) as ei:
        _settings(
            llm_mode="siliconflow",
            siliconflow_api_key="sk-test",
            siliconflow_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    assert "api.siliconflow.cn" in str(ei.value)


def test_siliconflow_default_domain_passes_validation() -> None:
    s = _settings(llm_mode="siliconflow", siliconflow_api_key="sk-test")
    assert s.siliconflow_base_url == "https://api.siliconflow.cn/v1"


# ---------------------------------------------------------------- ② 免费模型计费语义


def test_free_hosted_models_cost_zero_but_track_tokens() -> None:
    """免费托管的模型：费用 0（不是报错，也不是漏记）。"""
    for model in FREE_HOSTED_MODELS:
        assert compute_cost(model, 1000, 500) == 0.0


def test_free_embed_models_cost_zero() -> None:
    for model in FREE_EMBED_MODELS:
        assert compute_embed_cost(model, 10_000) == 0.0


def test_free_hosted_and_local_are_distinct_concepts() -> None:
    """两类的语义不同（厂商送的额度 vs 跑在本机），不能混为一个集合。"""
    assert not (FREE_HOSTED_MODELS & LOCAL_MODELS)


def test_unknown_model_still_raises() -> None:
    """免费清单不能变成"什么都能过"——未知模型仍必须报错（FR-20）。"""
    with pytest.raises(UnknownModelError):
        compute_cost("some/unknown-model", 10, 10)
    with pytest.raises(UnknownModelError):
        compute_embed_cost("some/unknown-embed", 10)


def test_siliconflow_prices_present_and_ordered() -> None:
    """硅基流动的价必须是分档升序，且与官方页一致（①一手，2026-09-12）。"""
    from attest.trace.pricing import PRICING

    p = PRICING["Qwen/Qwen3.5-35B-A3B"]
    assert p.tier_for(10_000).input_cny_per_million == 0.40
    assert p.tier_for(200_000).input_cny_per_million == 1.60
    assert [t.max_context_tokens for t in p.tiers] == sorted(t.max_context_tokens for t in p.tiers)

    # 分时段定价取白天高价档：宁可高估不可低估
    assert PRICING["deepseek-ai/DeepSeek-V4-Flash"].tier_for(1).input_cny_per_million == 3.00


# ---------------------------------------------------------------- ③ rerank 契约


def test_siliconflow_reranker_parses_its_own_response_shape(monkeypatch) -> None:
    """硅基流动是 `results`，不是百炼的 `output.results`——结构混用会静默返回空。"""
    captured: dict = {}

    class _Resp:
        def raise_for_status(self) -> None: ...
        def json(self) -> dict:
            # 故意打乱顺序：必须按 index 取，不能假设第 i 个就是文档 i
            return {
                "results": [
                    {"index": 2, "relevance_score": 0.11},
                    {"index": 0, "relevance_score": 0.97},
                ]
            }

    def _fake_post(url, *, headers, json, timeout):  # noqa: A002 - 对齐 httpx 形参名
        captured.update(url=url, headers=headers, payload=json)
        return _Resp()

    monkeypatch.setattr("attest.retrieval.rerank.httpx.post", _fake_post)

    r = SiliconFlowReranker("sk-test")
    out = r.rerank("市场规模", ["doc0", "doc1", "doc2"], top_n=2)

    assert captured["url"] == "https://api.siliconflow.cn/v1/rerank"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["payload"]["model"] == "BAAI/bge-reranker-v2-m3"
    assert captured["payload"]["documents"] == ["doc0", "doc1", "doc2"]
    assert out == [(2, 0.11), (0, 0.97)], "必须按响应里的 index，而不是列表位置"


def test_rerank_model_is_configurable(tmp_path: Path) -> None:
    s = _settings(llm_mode="siliconflow", siliconflow_api_key="sk-test")
    assert s.model_rerank == "BAAI/bge-reranker-v2-m3"


# ---------------------------------------------------------------- 装配：mode → reranker


def test_make_reranker_picks_siliconflow_when_mode_matches(tmp_path: Path) -> None:
    from attest.graph.build import make_reranker

    s = _settings(llm_mode="siliconflow", siliconflow_api_key="sk-test")
    r = make_reranker(s)
    assert isinstance(r, SiliconFlowReranker)


def test_make_reranker_falls_back_to_lexical_in_mock() -> None:
    from attest.graph.build import make_reranker
    from attest.retrieval.rerank import LexicalReranker

    assert isinstance(make_reranker(_settings()), LexicalReranker)


# ---------------------------------------------------------------- 通用兼容档（T7.9b）


def test_openai_compat_requires_key_and_base_url() -> None:
    """通用档必须同时给 key 与 base_url，且报错要指明"域名不对会伪装成 key 无效"。"""
    with pytest.raises(ValueError) as ei:
        _settings(llm_mode="openai_compat")
    msg = str(ei.value)
    assert "ATTEST_COMPAT_API_KEY" in msg and "ATTEST_COMPAT_BASE_URL" in msg
    assert "官方域名一律 401" in msg, "这条实测教训必须在报错里，否则下次还会踩"


def test_openai_compat_base_url_must_end_with_v1() -> None:
    """漏 /v1 是最常见的 404 成因，启动就拦下来。"""
    with pytest.raises(ValueError) as ei:
        _settings(
            llm_mode="openai_compat",
            compat_api_key="sk-x",
            compat_base_url="https://api.moonshot.cn",
        )
    assert "/v1" in str(ei.value)


def test_openai_compat_build_provider_uses_label() -> None:
    """标签只进日志/meta，不参与逻辑——但存在感很重要：排查时要知道是谁在跑。"""
    s = _settings(
        llm_mode="openai_compat",
        compat_api_key="sk-x",
        compat_base_url="https://api.moonshot.cn/v1",
        compat_label="kimi",
    )
    p = build_provider(s)
    assert isinstance(p, OpenAICompatProvider)
    assert p.name == "kimi"


def test_openai_compat_default_label() -> None:
    s = _settings(
        llm_mode="openai_compat",
        compat_api_key="sk-x",
        compat_base_url="https://api.example.com/v1",
    )
    assert build_provider(s).name == "openai-compat"


# ---------------------------------------------------------------- 通用兼容档（T7.9b）


def test_openai_compat_requires_key_and_base_url() -> None:
    """通用档必须同时给 key 与 base_url，且报错要指明"域名不对会伪装成 key 无效"。"""
    with pytest.raises(ValueError) as ei:
        _settings(llm_mode="openai_compat")
    msg = str(ei.value)
    assert "ATTEST_COMPAT_API_KEY" in msg and "ATTEST_COMPAT_BASE_URL" in msg
    assert "官方域名一律 401" in msg, "这条实测教训必须在报错里，否则下次还会踩"


def test_openai_compat_base_url_must_end_with_v1() -> None:
    """漏 /v1 是最常见的 404 成因，启动就拦下来。"""
    with pytest.raises(ValueError) as ei:
        _settings(
            llm_mode="openai_compat",
            compat_api_key="sk-x",
            compat_base_url="https://api.moonshot.cn",
        )
    assert "/v1" in str(ei.value)


def test_openai_compat_build_provider_uses_label() -> None:
    """标签只进日志/meta，不参与逻辑——但存在感很重要：排查时要知道是谁在跑。"""
    s = _settings(
        llm_mode="openai_compat",
        compat_api_key="sk-x",
        compat_base_url="https://api.moonshot.cn/v1",
        compat_label="kimi",
    )
    p = build_provider(s)
    assert isinstance(p, OpenAICompatProvider)
    assert p.name == "kimi"


def test_openai_compat_default_label() -> None:
    s = _settings(
        llm_mode="openai_compat",
        compat_api_key="sk-x",
        compat_base_url="https://api.example.com/v1",
    )
    assert build_provider(s).name == "openai-compat"
