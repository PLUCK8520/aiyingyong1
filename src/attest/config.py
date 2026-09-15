"""T0.2 · 静态配置（pydantic-settings）。

设计要点：
  - **默认离线**：`llm_mode` / `search_mode` 默认 `mock`，保证无任何 key 也能跑通整条链路。
  - **base_url 锁华北2(北京)**：百炼免费额度只认北京地域实时推理调用，配成 intl 域名会静默按量扣费
    （审查报告 F 节，①一手官方）。启动即校验，不合法直接报中文错。
  - 缺 key 时给出**可执行的中文提示**，不是 pydantic 的英文堆栈。
"""

from __future__ import annotations

import re
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
# 硅基流动（OpenAI 兼容；国内直连，无需代理）
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_HOST = "api.siliconflow.cn"
# 智谱（OpenAI 兼容，但版本段是 /v4 而不是 /v1——校验不能假设单一厂商的约定）
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

#: OpenAI 兼容端点必须以**版本段**结尾。哪些写法见过：
#:   /v1（Kimi、DeepSeek、硅基流动）· /v4（智谱）· /compatible-mode/v1（百炼）
#: 只认 /v1 会把智谱误判成非法配置（2026-09-13 实测踩到）。
_OPENAI_BASE_RE = re.compile(r"/v\d+/?$")


def _looks_like_openai_base(url: str) -> bool:
    return bool(_OPENAI_BASE_RE.search((url or "").strip()))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ---------------- 模式开关（默认离线）----------------
    llm_mode: Literal["mock", "dashscope", "siliconflow", "openai_compat", "ollama"] = Field(
        "mock", alias="ATTEST_LLM_MODE"
    )
    search_mode: Literal["mock", "tavily"] = Field("mock", alias="ATTEST_SEARCH_MODE")

    # ---------------- 密钥 / 端点 ----------------
    dashscope_api_key: str | None = Field(None, alias="DASHSCOPE_API_KEY")
    dashscope_base_url: str = Field(BEIJING_BASE_URL, alias="DASHSCOPE_BASE_URL")
    #: 硅基流动（T7.9）。与百炼同为 OpenAI 兼容协议，差别只有域名/key/模型名。
    #: 域名校验的理由与百炼相反：这里**不需要**地域限制，但必须防止把别家的 key 打到它家去
    #: （实测教训：把硅基流动的 key 填进 DASHSCOPE_API_KEY，百炼回 401 invalid_api_key，
    #:  报错文案是"key 无效"，很容易被误判成"key 坏了"，其实是走错门）。
    siliconflow_api_key: str | None = Field(None, alias="SILICONFLOW_API_KEY")
    siliconflow_base_url: str = Field(SILICONFLOW_BASE_URL, alias="SILICONFLOW_BASE_URL")
    #: **通用 OpenAI 兼容档**（T7.9b）：只给 base_url + key 就能接任何兼容厂商
    #: （Kimi / DeepSeek / 各类中转聚合站 / 自建网关……）。
    #: 加这一档的动机：用户连着换了三家 key，每换一家就要改一次代码——那是设计缺陷。
    #: ⚠️ 中转/聚合站的 key **只在它自己的域名有效**，官方域名一律 401；所以 base_url 必须给对。
    compat_api_key: str | None = Field(None, alias="ATTEST_COMPAT_API_KEY")
    compat_base_url: str | None = Field(None, alias="ATTEST_COMPAT_BASE_URL")
    #: 仅用于日志与 trace 的可读标签（如 "kimi" / "中转站A"），不参与任何逻辑。
    compat_label: str = Field("openai-compat", alias="ATTEST_COMPAT_LABEL")
    #: **没有可用 embedding 时的兜底策略**（2026-09-13 实测需要）：
    #:   none    —— 如实失败（默认，保持"不假装"的原则）；
    #:   hashing —— 退回离线散列向量（CJK bigram 哈希词袋，256 维，确定性、零成本）。
    #: 什么时候需要：部分厂商的免费档**不含 embedding**（智谱免费档就是：
    #: embedding-3/2 都回「无可用资源包」），而本地知识库检索与研究闭环都要 embedding。
    #: 用 hashing 兜底能保住这两条链路，但**语义检索质量弱于真实向量模型**，
    #: 故 trace 里会把 provider 标成 `mock-hashing(兜底)`——不许悄悄冒充真向量。
    embed_fallback: Literal["none", "hashing"] = Field("none", alias="ATTEST_EMBED_FALLBACK")
    tavily_api_key: str | None = Field(None, alias="TAVILY_API_KEY")
    #: T2.1 额度保护线：Tavily 检索的录制/回放模式。
    #:   `replay`（**默认**）——只读本地缓存；缓存未命中返回空，**永不联网、永不消耗额度**；
    #:   `record` —— 强制打真实 API 并把响应落盘（每次 1 credit）；
    #:   `auto`   —— 有缓存用缓存，没缓存就打真实 API（**会悄悄烧额度**，排查/演示慎用）。
    #: ⚠️ 默认必须是 `replay`：Tavily 免费档 1000 credits/月（①额度数字待复核，见审查报告 F-4），
    #:    而开发期一轮调研要 3~5 次检索——默认 `auto` 会让"跑一轮看看"直接扣额度，
    #:    与"额度保护线"的设计意图相反。要录新 query 时显式设成 `record`。
    tavily_mode: Literal["auto", "replay", "record"] = Field("replay", alias="ATTEST_TAVILY_MODE")
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field("qwen3:4b", alias="OLLAMA_MODEL")

    # ---------------- 模型映射：任务类型 → 模型（T4.4 策略表，先静态配置化）----------------
    model_intent: str = Field("qwen3.7-flash", alias="ATTEST_MODEL_INTENT")
    model_direct: str = Field("qwen3.8-flash", alias="ATTEST_MODEL_DIRECT")
    model_planner: str = Field("qwen3.8-max", alias="ATTEST_MODEL_PLANNER")
    model_judge: str = Field("qwen3.7-flash", alias="ATTEST_MODEL_JUDGE")
    #: T3.7：矛盾检测是 O(对数) 的成本敏感任务，用 fast 档；仅在需要强推理时才升级
    model_conflict: str = Field("qwen3.7-flash", alias="ATTEST_MODEL_CONFLICT")
    model_analyst: str = Field("qwen3.8-max", alias="ATTEST_MODEL_ANALYST")
    model_audit_fast: str = Field("qwen3.7-flash", alias="ATTEST_MODEL_AUDIT_FAST")
    model_audit_strong: str = Field("qwen3.8-max", alias="ATTEST_MODEL_AUDIT_STRONG")
    model_embed: str = Field("text-embedding-v4", alias="ATTEST_MODEL_EMBED")
    #: T7.9：精排模型。硅基流动的 bge-reranker-v2-m3 官方标免费；
    #: 换 embedding/rerank 模型都要重建索引并重标矛盾阈值（见 T7.4 说明）。
    model_rerank: str = Field("BAAI/bge-reranker-v2-m3", alias="ATTEST_MODEL_RERANK")
    #: T4.4 / 熔断 L1：轻任务（intent / direct / judge / conflict / audit_fast）降级目标。
    #: 默认取单价表里最便宜的对话模型（qwen3.7-flash，0.3/1.2 元每百万）。
    model_fuse_light: str = Field("qwen3.7-flash", alias="ATTEST_MODEL_FUSE_LIGHT")

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
    #: T9.1：analyst 结构化输出协议开关。开（默认）：模型按 <<CHAPTER>>/<<CITE>>/<<TEXT>>
    #: 分章输出并**声明本章依据的证据编号**，由代码确定性拼装进正文（章末依据行）——
    #: 编号不再依赖弱模型"句末挂编号"的服从性（2026-09-13/14 实测 glm-4-flash 免费档
    #: 四次整篇零编号，审计网因此整体失效）。模型不服从格式时自动降级回纯文本路径
    #: （原 T8.6 纠偏重试逻辑不变）。关掉则完全回到 P8 及以前的行为，便于对照。
    analyst_structured: bool = Field(True, alias="ATTEST_ANALYST_STRUCTURED")
    #: 单次 LLM 调用的读超时（秒）。实测教训（2026-09-13）：证据判别要一次送 20+ 条证据，
    #: 免费档模型 60s 内回不来 → 三次超时把整轮拖垮。真实档给足余量。
    llm_timeout_s: float = Field(120.0, alias="ATTEST_LLM_TIMEOUT")
    #: T4.3 / 熔断 L2：正文证据按相关性截断到 top-k（设计 §6.5「上下文截断（证据按相关性取 top-k）」）
    fuse_ctx_top_k: int = Field(6, alias="ATTEST_FUSE_CTX_TOP_K")
    search_concurrency: int = Field(3, alias="ATTEST_SEARCH_CONCURRENCY")
    max_reflect_rounds: int = Field(2, alias="ATTEST_MAX_REFLECT_ROUNDS")

    # ---------------- P3：本地知识库（T3.1 / T3.2）----------------
    local_enabled: bool = Field(True, alias="ATTEST_LOCAL_ENABLED")
    local_docs_dir: Path = Field(DATA_DIR / "fixtures" / "local", alias="ATTEST_LOCAL_DOCS")
    local_chroma_dir: Path = Field(DATA_DIR / "index" / "chroma", alias="ATTEST_CHROMA_DIR")
    #: numpy=内存余弦（离线默认，快且无锁）/ chroma=持久化（验收档，先跑 ingest_local.py）
    local_store: Literal["numpy", "chroma"] = Field("numpy", alias="ATTEST_LOCAL_STORE")
    local_chunk_chars: int = Field(800, alias="ATTEST_CHUNK_CHARS")
    local_chunk_overlap: int = Field(120, alias="ATTEST_CHUNK_OVERLAP")
    #: 精排候选数（融合后送 rerank 的条数）
    rerank_candidate_n: int = Field(20, alias="ATTEST_RERANK_CANDIDATES")

    # ---------------- P3：矛盾检测（T3.7，《功能设计》§6.3）----------------
    #: 证据相似度聚类阈值（同子问题内先聚类，再簇内两两比对）。
    #: ⚠️ 该默认值是 **0.25**，由 `scripts/calibrate_conflict.py` 在**离线词法向量**上实测标定
    #: （同子问题内相似度实测区间约 -0.01~0.56，取 0.25 可覆盖真实同口径证据对）。
    #: **换成 text-embedding-v4 语义向量后必须重新标定**——两套向量空间的相似度分布完全不同。
    conflict_cluster_threshold: float = Field(0.25, alias="ATTEST_CONFLICT_THRESHOLD")
    #: 每簇最多比对数
    conflict_pairs_per_cluster: int = Field(6, alias="ATTEST_CONFLICT_PAIRS_CLUSTER")
    #: 全局比对数上限（超出按簇内相似度降序截断）——成本上限必须写进配置
    conflict_pairs_global: int = Field(30, alias="ATTEST_CONFLICT_PAIRS_GLOBAL")
    #: 数值口径差异达到该倍数才判为冲突（离线启发式用；真实模型由提示词约束）
    conflict_min_ratio: float = Field(1.5, alias="ATTEST_CONFLICT_MIN_RATIO")

    # ---------------- P4：引用审计（T4.1 / T4.2，《功能设计》FR-17）----------------
    #: 审计开关。离线（mock）走规则版；dashscope 走「flash 初筛 + 强模型复核」两段式。
    audit_enabled: bool = Field(True, alias="ATTEST_AUDIT_ENABLED")
    #: 单章节 unsupported 占比超过该值 → 触发该章节重写（T4.2b，限 audit_max_rewrites 次）
    audit_rewrite_ratio: float = Field(0.5, alias="ATTEST_AUDIT_REWRITE_RATIO")
    audit_max_rewrites: int = Field(1, alias="ATTEST_AUDIT_MAX_REWRITES")

    # ---------------- P5：记忆与协同（T5.1 ~ T5.5，《功能设计》§4.3 / §7）----------------
    #: 检查点开关。关掉则退化为"无状态单跑"（P1~P4 的行为），便于对照与排查。
    checkpoint_enabled: bool = Field(True, alias="ATTEST_CHECKPOINT_ENABLED")
    #: 检查点 SQLite 路径。CLI 用**同步** `SqliteSaver`；P6 的 FastAPI(async) **必须换**
    #: `AsyncSqliteSaver`（aiosqlite）——同步 saver 在 async 事件循环里会阻塞（T5.1 硬约束）。
    checkpoint_db: Path = Field(DATA_DIR / "checkpoints.sqlite", alias="ATTEST_CHECKPOINT_DB")
    #: 用户画像 SQLite 路径（T5.2）
    profile_db: Path = Field(DATA_DIR / "profile.sqlite", alias="ATTEST_PROFILE_DB")
    #: 人工确认大纲开关（T5.4）。关掉 = 直接跳过确认（等价于用户选「跳过」）。
    human_confirm_enabled: bool = Field(False, alias="ATTEST_HUMAN_CONFIRM")
    #: 用户画像注入开关（T5.2）
    profile_enabled: bool = Field(True, alias="ATTEST_PROFILE_ENABLED")

    # ---------------- T7.4：研究闭环（research_memory）----------------
    #: 研究沉淀开关。关掉则 `memory_writer` 如实留痕跳过、`scout_local` 不检索历史结论，
    #: 等价于 P6 及以前的行为——便于对照"闭环有没有实际收益"。
    research_memory_enabled: bool = Field(True, alias="ATTEST_RESEARCH_MEMORY_ENABLED")
    #: 沉淀向量集合的持久化目录。**必须与 `local_chroma_dir` 分开**：
    #: docs 是用户资料、research_memory 是我们自己产出的结论，混在一起就分不清"谁说的"。
    research_memory_dir: Path = Field(
        DATA_DIR / "index" / "research_memory", alias="ATTEST_RESEARCH_MEMORY_DIR"
    )
    #: 每次调研从历史结论里检索的条数上限（只做提示，不取代正文引用）
    research_memory_top_k: int = Field(3, alias="ATTEST_RESEARCH_MEMORY_TOP_K")

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
        if self.llm_mode == "openai_compat":
            if not (self.compat_api_key and self.compat_base_url):
                raise ValueError(
                    "LLM 模式为 openai_compat 但缺少 ATTEST_COMPAT_API_KEY / ATTEST_COMPAT_BASE_URL。\n"
                    "  解决：在 .env 里补齐这两项（base_url 填**该 key 所属平台**的地址，含 /v1）。\n"
                    "  ⚠️ 中转/聚合站的 key 只在它自己的域名有效，官方域名一律 401「Invalid Authentication」——\n"
                    "     报错看着像 key 废了，其实是域名不对。不确定 key 属于谁："
                    "scripts/probe_which_vendor.py --key \"sk-...\""
                )
            if not _looks_like_openai_base(self.compat_base_url):
                raise ValueError(
                    f"ATTEST_COMPAT_BASE_URL 看起来不像 OpenAI 兼容端点：{self.compat_base_url}\n"
                    "  正确形态：https://<host>/v1 或 https://<host>/api/paas/v4 —— 即"
                    "**必须以版本段结尾**（缺版本段的 URL 请求会 404）。\n"
                    "  实测参考：智谱 = https://open.bigmodel.cn/api/paas/v4（不是 /v1！）；\n"
                    "           硅基流动 = https://api.siliconflow.cn/v1；Kimi = https://api.moonshot.cn/v1"
                )
        if self.llm_mode == "siliconflow":
            if not self.siliconflow_api_key:
                raise ValueError(
                    "LLM 模式为 siliconflow 但没有 SILICONFLOW_API_KEY。\n"
                    "  解决：① 在 .env 填入 SILICONFLOW_API_KEY（https://cloud.siliconflow.cn 创建）；或\n"
                    "        ② 改回离线模式 ATTEST_LLM_MODE=mock。\n"
                    "  ⚠️ 别把别家的 key 填进来：不同平台 key 不通用，报错是 401「key 无效」，"
                    "看着像 key 坏了，其实是走错门。"
                )
            if SILICONFLOW_HOST not in self.siliconflow_base_url:
                raise ValueError(
                    f"SILICONFLOW_BASE_URL 域名不对：{self.siliconflow_base_url}\n"
                    f"  正确值：{SILICONFLOW_BASE_URL}"
                )
        if not 0 < self.fuse_warn_ratio < self.fuse_hard_ratio <= 1:
            raise ValueError("熔断阈值必须满足 0 < warn < hard <= 1。")
        return self

    def ensure_dirs(self) -> None:
        for p in (self.trace_dir, self.report_dir, self.fixture_dir):
            p.mkdir(parents=True, exist_ok=True)
        # 检查点 / 画像的父目录（SQLite 文件本身由 sqlite 驱动创建）
        for p in (self.checkpoint_db.parent, self.profile_db.parent):
            p.mkdir(parents=True, exist_ok=True)
        if self.research_memory_enabled:
            self.research_memory_dir.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """唯一入口：读 .env + 环境变量，返回校验过的 Settings。"""
    return Settings()
