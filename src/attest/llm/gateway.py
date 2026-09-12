"""T0.3 · LLM 网关：统一出口 + **唯一记账点**。

约定（架构不变式）：
  - 节点不自己算钱、不自己调 SDK；所有 LLM/Embedding 调用必经此处。
  - 每次调用都 (a) 写一条 `llm_call` trace、(b) 返回 `cost_cny` 与 `total_tokens`，
    由节点放进 state 的 `cost_incurred` / `tokens_incurred`（皆有累加 reducer）。
  - **绕过网关 = 漏账**，预算熔断即失效。
  - 结构化输出校验失败自动重试（≤2 次），重试的成本同样记账。

模型路由（任务 → 模型）**不在本文件**：策略表与熔断降级在 `llm/router.py`（T4.4），
网关只负责记账与调用。本文件通过 `self.router` 委托选型，并接受节点传入的 `fuse_level`。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import Settings
from ..logging import get_logger
from ..trace.events import TraceWriter, estimate_tokens
from ..trace.pricing import compute_cost, compute_embed_cost
from .providers import (
    DashScopeProvider,
    MockProvider,
    OllamaProvider,
    Provider,
    ProviderResult,
    SiliconFlowProvider,
)
from .router import ModelRouter

log = get_logger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> str:
    """容忍模型把 JSON 包在 ```json 围栏里，或前后带解释性文字。"""
    m = _FENCE_RE.search(text or "")
    if m:
        return m.group(1).strip()
    s = (text or "").strip()
    start = min((i for i in (s.find("{"), s.find("[")) if i != -1), default=-1)
    if start > 0:
        end = max(s.rfind("}"), s.rfind("]"))
        if end > start:
            return s[start : end + 1]
    return s


@dataclass
class LLMResponse:
    text: str
    model: str
    task: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_cny: float
    latency_ms: float
    token_estimate: bool
    reasoning: str | None = None
    attempts: int = 1
    parsed: BaseModel | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def build_provider(settings: Settings) -> Provider:
    mode = settings.llm_mode
    if mode == "mock":
        log.info("[gateway] LLM 走离线 mock（无 key 也能跑通链路；输出会在报告页脚标注）")
        return MockProvider(settings)
    if mode == "dashscope":
        assert settings.dashscope_api_key  # config 已校验
        log.info(f"[gateway] LLM 走 dashscope，base_url={settings.dashscope_base_url}")
        return DashScopeProvider(settings.dashscope_api_key, settings.dashscope_base_url)
    if mode == "ollama":
        log.info(f"[gateway] LLM 走 ollama，base_url={settings.ollama_base_url}")
        return OllamaProvider(settings.ollama_base_url)
    if mode == "siliconflow":
        assert settings.siliconflow_api_key  # config 已校验
        log.info(f"[gateway] LLM 走 siliconflow，base_url={settings.siliconflow_base_url}")
        return SiliconFlowProvider(settings.siliconflow_api_key, settings.siliconflow_base_url)
    raise ValueError(f"未知 llm_mode：{mode}")


@dataclass
class LLMGateway:
    settings: Settings
    provider: Provider = field(default=None)  # type: ignore[assignment]
    trace: TraceWriter | None = None

    def __post_init__(self) -> None:
        if self.provider is None:
            self.provider = build_provider(self.settings)
        #: T4.4 模型路由器（无状态，可安全复用）。策略表不在网关里。
        self.router = ModelRouter(self.settings)

    # ---------------- 模型路由（委托 llm/router.py，T4.4）----------------
    def model_for(self, task: str, *, fuse_level: int = 0) -> str:
        return self.router.route(task, fuse_level=fuse_level)

    # ---------------- 对话 ----------------
    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        task: str,
        model: str | None = None,
        temperature: float = 0.2,
        response_model: type[TModel] | None = None,
        max_retries: int = 2,
        fuse_level: int = 0,
    ) -> LLMResponse:
        model = model or self.model_for(task, fuse_level=fuse_level)
        convo = list(messages)
        total_in = total_out = 0
        cost = 0.0
        attempt = 0
        last_error: BaseException | None = None

        for attempt in range(max_retries + 1):
            started = time.perf_counter()
            result: ProviderResult = self.provider.chat(
                convo,
                model=model,
                temperature=temperature,
                json_schema=response_model,
                task=task,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            c = compute_cost(
                model,
                result.input_tokens,
                result.output_tokens,
                context_tokens=result.input_tokens,
            )
            total_in += result.input_tokens
            total_out += result.output_tokens
            cost += c
            self._emit_call(task, model, result, c, latency_ms, attempt + 1, fuse_level)

            if response_model is None:
                return LLMResponse(
                    text=result.text,
                    model=model,
                    task=task,
                    provider=self.provider.name,
                    input_tokens=total_in,
                    output_tokens=total_out,
                    cost_cny=round(cost, 8),
                    latency_ms=round(latency_ms, 2),
                    token_estimate=result.token_estimate,
                    reasoning=result.reasoning,
                    attempts=attempt + 1,
                )

            try:
                parsed = response_model.model_validate_json(_extract_json(result.text))
            except (ValidationError, ValueError) as exc:
                last_error = exc
                log.warning(
                    f"[gateway] {task} 结构化输出校验失败（第 {attempt + 1} 次）：{str(exc)[:200]}"
                )
                if attempt < max_retries:
                    convo = list(messages) + [
                        {"role": "assistant", "content": result.text},
                        {
                            "role": "user",
                            "content": (
                                f"你上一次的输出不满足 JSON Schema，校验错误：{str(exc)[:300]}。"
                                "请只输出修正后的合法 JSON，不要任何解释。"
                            ),
                        },
                    ]
                    continue
                raise

            return LLMResponse(
                text=result.text,
                model=model,
                task=task,
                provider=self.provider.name,
                input_tokens=total_in,
                output_tokens=total_out,
                cost_cny=round(cost, 8),
                latency_ms=round(latency_ms, 2),
                token_estimate=result.token_estimate,
                reasoning=result.reasoning,
                attempts=attempt + 1,
                parsed=parsed,
            )

        raise RuntimeError(f"结构化输出在 {max_retries + 1} 次内仍未通过校验：{last_error}")

    # ---------------- 向量 ----------------
    def embed(self, texts: Sequence[str], *, task: str = "embed") -> tuple[list[list[float]], float, int]:
        """返回 (向量, 费用, token 数)。"""
        model = self.settings.model_embed
        vectors = self.provider.embed(texts, model=model)
        tokens = sum(estimate_tokens(t) for t in texts)
        cost = compute_embed_cost(model, tokens) if self.provider.name != "mock" else 0.0
        if self.trace is not None:
            self.trace.emit(
                "embed_call",
                task=task,
                model=model,
                provider=self.provider.name,
                texts=len(texts),
                tokens=tokens,
                cost_cny=round(cost, 8),
                token_estimate=True,
            )
        return vectors, cost, tokens

    # ---------------- 内部 ----------------
    def _emit_call(
        self,
        task: str,
        model: str,
        result: ProviderResult,
        cost: float,
        latency_ms: float,
        attempt: int,
        fuse_level: int = 0,
    ) -> None:
        if self.trace is None:
            return
        self.trace.emit(
            "llm_call",
            task=task,
            model=model,
            provider=self.provider.name,
            tier=self.router.tier(task, fuse_level=fuse_level),
            fuse_level=fuse_level,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.input_tokens + result.output_tokens,
            cost_cny=round(cost, 8),
            latency_ms=round(latency_ms, 2),
            token_estimate=result.token_estimate,
            attempt=attempt,
        )
