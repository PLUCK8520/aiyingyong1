"""T0.2 · 静态配置（pydantic-settings）。

设计要点：
  - **默认离线**：`llm_mode` / `search_mode` 默认 `mock`，保证无任何 key 也能跑通整条链路。
  - **base_url 锁华北2(北京)**：百炼免费额度只认北京地域实时推理调用，配成 intl 域名会静默按量扣费
    （审查报告 F 节，①一手官方）。启动即校验，不合法直接报中文错。
  - 缺 key 时给出**可执行的中文提示**，不是 pydantic 的英文堆栈。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"

# 免费额度只认华北2(北京)地域（①一手官方，2026-09-10 复核）
BEIJING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BEIJING_HOST = "dashscope.aliyuncs.com"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ---------------- 模式开关（默认离线）----------------
    llm_mode: Literal["mock", "dashscope", "ollama"] = Field("mock", alias="ATTEST_LLM_MODE")
    search_mode: Literal["mock", "tavily"] = Field("mock", alias="ATTEST_SEARCH_MODE")

    # ---------------- 密钥 / 端点 ----------------
    dashscope_api_key: str | None = Field(None, alias="DASHSCOPE_API_KEY")
    dashscope_base_url: str = Field(BEIJING_BASE_URL, alias="DASHSCOPE_BASE_URL")
    tavily_api_key: str | None = Field(None, alias="TAVILY_API_KEY")
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field("qwen3:4b", alias="OLLAMA_MODEL")

    # ---------------- 模型映射：任务类型 → 模型（T4.4 策略表，先静态配置化）----------------
    model_intent: str = Field("qwen3.7-flash", alias="ATTEST_MODEL_INTENT")
    model_direct: str = Field("qwen3.8-flash", alias="ATTEST_MODEL_DIRECT")
    model_planner: str = Field("qwen3.8-max", alias="ATTEST_MODEL_PLANNER")
    model_judge: str = Field("qwen3.7-flash", alias="ATTEST_MODEL_JUDGE")
    model_analyst: str = Field("qwen3.8-max", alias="ATTEST_MODEL_ANALYST")
    model_audit_fast: str = Field("qwen3.7-flash", alias="ATTEST_MODEL_AUDIT_FAST")
    model_audit_strong: str = Field("qwen3.8-max", alias="ATTEST_MODEL_AUDIT_STRONG")
    model_embed: str = Field("text-embedding-v4", alias="ATTEST_MODEL_EMBED")

    # ---------------- 预算（派生式，ADR-02）----------------
    budget_total_cny: float = Field(2.0, alias="ATTEST_BUDGET_CNY")
    budget_total_tokens: int = Field(300_000, alias="ATTEST_BUDGET_TOKENS")
    fuse_warn_ratio: float = Field(0.70, alias="ATTEST_FUSE_WARN")
    fuse_hard_ratio: float = Field(0.90, alias="ATTEST_FUSE_HARD")

    # ---------------- 运行参数 ----------------
    log_level: str = Field("INFO", alias="ATTEST_LOG_LEVEL")
    trace_dir: Path = Field(DATA_DIR / "trace", alias="ATTEST_TRACE_DIR")
    report_dir: Path = Field(DATA_DIR / "reports", alias="ATTEST_REPORT_DIR")
    fixture_dir: Path = Field(DATA_DIR / "fixtures", alias="ATTEST_FIXTURE_DIR")

    context_truncate_chars: int = Field(4000, alias="ATTEST_CTX_TRUNCATE")
    search_concurrency: int = Field(3, alias="ATTEST_SEARCH_CONCURRENCY")
    max_reflect_rounds: int = Field(2, alias="ATTEST_MAX_REFLECT_ROUNDS")

    # ---------------- 校验 ----------------
    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if self.llm_mode == "dashscope":
            if not self.dashscope_api_key:
                raise ValueError(
                    "LLM 模式为 dashscope 但没有 DASHSCOPE_API_KEY。\n"
                    "  解决：① 复制 .env.example 为 .env 并填入 key；或\n"
                    "        ② 改回离线模式 ATTEST_LLM_MODE=mock（无需任何 key）。"
                )
            if BEIJING_HOST not in self.dashscope_base_url or "intl" in self.dashscope_base_url:
                raise ValueError(
                    f"DASHSCOPE_BASE_URL 不是华北2(北京)地域：{self.dashscope_base_url}\n"
                    f"  免费额度只支持北京地域实时推理，用 intl/其他域名会静默按量扣费。\n"
                    f"  正确值：{BEIJING_BASE_URL}"
                )
        if self.search_mode == "tavily" and not self.tavily_api_key:
            raise ValueError(
                "检索模式为 tavily 但没有 TAVILY_API_KEY。\n"
                "  解决：① 在 .env 填入 key；或 ② 用 ATTEST_SEARCH_MODE=mock 走离线 fixture。"
            )
        if not 0 < self.fuse_warn_ratio < self.fuse_hard_ratio <= 1:
            raise ValueError("熔断阈值必须满足 0 < warn < hard <= 1。")
        return self

    def ensure_dirs(self) -> None:
        for p in (self.trace_dir, self.report_dir, self.fixture_dir):
            p.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """唯一入口：读 .env + 环境变量，返回校验过的 Settings。"""
    return Settings()
