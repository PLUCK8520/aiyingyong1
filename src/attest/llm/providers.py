"""T0.3 · LLM Provider 实现（mock / dashscope / ollama）。

三者的关系：**mock 是默认档**（无 key 也能跑通整条链路），dashscope 是生产档，
ollama 是"断网时的可选项"（审查报告建议：熔断 L1 应切 qwen3.7-flash，而非本地 Ollama，
因为本地小模型的延迟/质量折损未量化，且同机跑 Ollama + Chroma 内存吃紧）。

Provider 只负责"拿回文本和用量"，**不负责记账**——记账是网关的职责。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import httpx

from ..logging import get_logger
from ..retrieval.citations import (
    parse_conflicts_block,
    parse_evidence_block,
    parse_pairs_block,
)
from ..trace.events import estimate_tokens
from .prompts import parse_objective, parse_outline, parse_rewrite_target, parse_subqs

log = get_logger(__name__)

#: 离线伪向量维度。真实语义向量用 text-embedding-v4（百炼，需 key）。
MOCK_EMBED_DIM = 256


@dataclass
class ProviderResult:
    text: str
    reasoning: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    # True = token 数由本地估算（mock / 无 usage 字段），trace 里要如实标注
    token_estimate: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    name: str

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.2,
        json_schema: type | None = None,
        task: str = "generic",
        timeout: float = 60.0,
    ) -> ProviderResult: ...

    def embed(self, texts: Sequence[str], *, model: str, timeout: float = 60.0) -> list[list[float]]: ...


def _flatten(messages: Sequence[dict[str, str]]) -> str:
    return "\n".join(m.get("content", "") for m in messages)


def _last_user(messages: Sequence[dict[str, str]]) -> str:
    for m in reversed(list(messages)):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


# ==================================================================== mock


_RESEARCH_HINTS = (
    "调研", "研究", "分析", "市场", "报告", "对比", "竞品", "收费", "定价",
    "模式", "趋势", "规模", "格局", "梳理", "深度", "报告",
)


class MockProvider:
    """离线 Provider：**由脚本按任务类型生成确定性输出**。

    诚实声明：这不是模型。它只保证"链路可跑、结构合法、引用可回查"，
    便于在无任何 API key 时开发与回归。所有 mock 产出的正文都由网关打上
    `provider=mock` 标记，报告页脚也会显式声明——不允许把 mock 输出当真实结论。
    """

    name = "mock"

    def __init__(self, settings: Any = None) -> None:
        # 冲突判定的数值倍数门槛与配置同源：配置是唯一真值来源，避免两处硬编码漂移
        self._min_ratio = float(getattr(settings, "conflict_min_ratio", 1.5) or 1.5)

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.2,
        json_schema: type | None = None,
        task: str = "generic",
        timeout: float = 60.0,
    ) -> ProviderResult:
        user = _last_user(messages)
        handler = getattr(self, f"_t_{task}", None)
        text = handler(user) if handler is not None else self._t_generic(user)
        prompt_text = _flatten(messages)
        return ProviderResult(
            text=text,
            input_tokens=estimate_tokens(prompt_text),
            output_tokens=estimate_tokens(text),
            token_estimate=True,
            meta={"provider": "mock", "task": task},
        )

    def embed(self, texts: Sequence[str], *, model: str, timeout: float = 60.0) -> list[list[float]]:
        """离线 embedding = CJK bigram 哈希词袋，L2 归一化，**维度固定 256**。

        ⚠️ 这是**词法**向量（相似度≈字面重叠），不是语义向量。它足够支撑混合检索/RRF/
        聚类这些工程路径的离线验证，但**不能**替代 text-embedding-v4 的语义召回能力。
        真实语义向量需 `ATTEST_LLM_MODE=dashscope` + DASHSCOPE_API_KEY。
        """
        return [_hashing_embed(t) for t in texts]

    # ---------------- 各任务脚本 ----------------

    def _t_generic(self, user: str) -> str:
        return f"[mock] 未指定任务类型，收到输入：{user[:80]}"

    def _t_intent(self, user: str) -> str:
        is_research = any(h in user for h in _RESEARCH_HINTS) or len(user) >= 30
        route = "research" if is_research else "direct"
        reason = (
            "命中调研类关键词/问题较长，需检索与多源综合"
            if is_research
            else "可直接回答的短问题，无需检索"
        )
        return json.dumps({"route": route, "reason": reason}, ensure_ascii=False)

    def _t_direct(self, user: str) -> str:
        return (
            f"（离线 mock 答复）关于「{user.strip()}」：当前处于离线演示模式，"
            "这条回答由脚本生成，用于验证 direct 分流分支是否连通。"
            "接入真实模型（ATTEST_LLM_MODE=dashscope）后，此处即为模型正文。"
        )

    def _t_planner(self, user: str) -> str:
        subject = _extract_subject(user)
        parts = _extract_parts(user)
        if parts:
            sub_qs = [f"{subject}——{p}" for p in parts]
            outlines = parts
        else:
            sub_qs = [
                f"{subject}的市场规模与增长情况",
                f"{subject}的主要参与方与竞争格局",
                f"{subject}的收费模式与落地成本",
            ]
            outlines = ["市场规模", "竞争格局", "收费模式与落地成本"]
        return json.dumps(
            {
                "objective": f"调研「{subject}」，输出可溯源的结构化报告",
                "sub_questions": sub_qs,
                "outlines": outlines,
                "requires_data": True,
            },
            ensure_ascii=False,
        )

    def _t_judge(self, user: str) -> str:
        records = parse_evidence_block(user)
        expected = parse_subqs(user)
        judgments: list[dict[str, Any]] = []
        covered: set[str] = set()
        for rec in records:
            sq = rec.get("sub_question", "")
            content = rec.get("content", "")
            title = rec.get("title", "")
            covered.add(sq)
            toks = [t for t in re.split(r"[——/、，,与和的\s]+", sq) if len(t) >= 2]
            tail = _distinctive_tokens(sq)
            # 只共享"主题词"不算相关：必须命中子问题的**区别性部分**（`——` 之后那段）。
            # 实测教训（eval M4）：不设这道闸，"量子计算专利布局"会被一篇讲市场规模的文章
            # 判成高相关，真实缺口被淹没成"已覆盖"，Reflect 补检永不触发。
            if tail and not any(t in content or t in title for t in tail):
                hit = 0
                relevance = 2
            else:
                hit = sum(1 for t in toks if t in content or t in title)
                relevance = min(5, 2 + hit) if hit else 2
            confidence = 5 if len(content) >= 120 else (4 if len(content) >= 60 else 3)
            judgments.append(
                {
                    "citation_id": rec.get("citation_id", ""),
                    "sub_question": sq,
                    "relevance": relevance,
                    "confidence": confidence,
                    "note": "离线 mock 判定（按关键词覆盖与正文长度启发式）",
                }
            )
        gaps = [f"缺少「{s}」的直接证据" for s in expected if s not in covered]
        return json.dumps({"judgments": judgments, "gaps": gaps}, ensure_ascii=False)

    def _t_analyst(self, user: str) -> str:
        objective = parse_objective(user) or "调研报告"
        outlines = parse_outline(user)
        records = _dedupe_by_url(parse_evidence_block(user))
        if not outlines:
            outlines = _unique(r.get("sub_question", "") for r in records) or ["调研发现"]

        lines: list[str] = [f"# {objective}", "", "## 核心摘要", ""]
        top = sorted(records, key=lambda r: len(r.get("content", "")), reverse=True)[:3]
        if top:
            for r in top:
                lines.append(f"- {_snippet(r.get('content', ''), 70)} {r['citation_id']}")
        else:
            lines.append("- （证据不足）本轮未检索到可用证据。")
        lines.append("")

        for chapter in outlines:
            lines.append(f"## {chapter}")
            lines.append("")
            picked = _rank_for_chapter(chapter, records)[:3]
            if not picked:
                lines.append("- （证据不足）未检索到与该章节直接相关的证据。")
                lines.append("")
                continue
            for r in picked:
                lines.append(f"- {_snippet(r.get('content', ''), 90)} {r['citation_id']}")
            lines.append("")

        conflicts = parse_conflicts_block(user)
        if conflicts:
            lines.append("## 争议与分歧")
            lines.append("")
            for c in conflicts:
                lines.append(
                    f"- **{c['topic']}**：一方口径为「{c['claim_a']}」（{c['id_a']}），"
                    f"另一方口径为「{c['claim_b']}」（{c['id_b']}）。"
                    "两者不可并存，本报告并列呈现，不做调和。"
                )
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _t_analyst_rewrite(self, user: str) -> str:
        """T4.2b 离线章节重写：只回抄**本节所引证据**的片段（带编号）。

        这正是重写的语义——把"证据支持不了的句子"换成"证据里真有的内容"。
        离线档没有模型可造句，所以退化为"证据摘录"；但结构合法、引用可回查，
        足以验证 T4.2b 的「重写 → 复检 → 收口」这条链路确实接通。
        """
        _title, _old = parse_rewrite_target(user)
        records = _dedupe_by_url(parse_evidence_block(user))
        if not records:
            return "- （证据不足）本节未能找到可支撑的结论。"
        return "\n".join(
            f"- {_snippet(r.get('content', ''), 90)} {r['citation_id']}" for r in records[:3]
        )

    def _t_conflict(self, user: str) -> str:
        """离线矛盾检测：同一指标、数值差异 ≥ 阈值的倍数 → 判冲突。

        刻意**保守**：只认"可比的同一单位族数值"，且差异不足阈值就不报。
        宁缺毋滥——把正常口径差异渲染成冲突，比漏报更伤报告可信度。

        **两道闸门**（缺一不可）：
          ① 同单位族：`万辆` ≠ `亿`，不可比就不判；
          ② 数值倍数 ≥ `conflict_min_ratio`（配置项，默认 1.5）。

        ⚠️ **已知局限（T7.2 评测排查结论，不要试图用启发式修补）**：
        planner 在离线档产出的是**与主题无关的通用子问题**（"市场规模与增长情况" /
        "主要参与方与竞争格局" / "收费模式与落地成本"），而 `sub_question` 是按**扇出分支**
        贴的标签，不是按正文内容判的。结果是：词法检索把同主题不同侧面的证据塞进同一分支，
        `select_pairs` 照样两两比对。两个后果——
          · 假阳性：某题的冲突对恰好落在"市场规模"组，即使该题问的是厂商差异也会报冲突；
          · 漏报：正文讲"降幅"的证据被贴成"市场规模"标签，数值差异再大也不在该题的比对范围。
        试过"正文必须命中子问题侧词"的第三道闸：**假阳性没清干净，真阳性反而被一起拦掉**
        （侧词是通用词，与真实冲突内容对不上）。所以撤掉了——这里是**离线 mock 的保真度上限**，
        不是可以通过调参解决的缺陷。真实路径（dashscope）由 LLM 判定，不依赖启发式。
        """
        records = {r["citation_id"]: r for r in parse_evidence_block(user)}
        pairs = parse_pairs_block(user)
        conflicts: list[dict[str, Any]] = []
        for p in pairs:
            a, b = records.get(p["id_a"]), records.get(p["id_b"])
            if not a or not b:
                continue
            na = _primary_number(a.get("content", ""))
            nb = _primary_number(b.get("content", ""))
            # 闸门①：单位族不同不可比
            if not na or not nb or na[1] != nb[1]:
                continue
            lo, hi = sorted((na[0], nb[0]))
            if lo <= 0:
                continue
            ratio = hi / lo
            # 闸门②：差异倍数不足阈值
            if ratio < self._min_ratio:
                continue
            severity = "high" if ratio >= 3 else ("medium" if ratio >= 2 else "low")
            conflicts.append(
                {
                    "sub_question": p["topic"],
                    "topic": _topic_label(p["topic"]),
                    "claim_a": na[2],
                    "source_a": p["id_a"],
                    "claim_b": nb[2],
                    "source_b": p["id_b"],
                    "severity": severity,
                    "summary": (
                        f"同一指标出现约 {ratio:.1f} 倍的数值差异（同为 {na[1]} 口径），"
                        "两者不可并存，需并列呈现。"
                    ),
                }
            )
        return json.dumps({"conflicts": conflicts}, ensure_ascii=False)


def _hashing_embed(text: str, dim: int = MOCK_EMBED_DIM) -> list[float]:
    """CJK bigram + ASCII 词的哈希词袋向量（L2 归一化）。确定性：同文本必得同向量。"""
    s = (text or "").lower()
    toks: list[str] = []
    cjk = [ch for ch in s if "\u4e00" <= ch <= "\u9fff"]
    toks.extend(cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1))
    toks.extend(re.findall(r"[a-z0-9]{2,}", s))
    if not toks:
        toks = [s[:4] or "_"]
    vec = [0.0] * dim
    for t in toks:
        h = hashlib.md5(t.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        vec[idx] += 1.0 if h[4] % 2 == 0 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(亿元|万元|亿|万|%|％)")
_UNIT_BASE: dict[str, tuple[str, float]] = {
    "亿元": ("亿", 1e8),
    "亿": ("亿", 1e8),
    "万元": ("万", 1e4),
    "万": ("万", 1e4),
    "%": ("%", 1.0),
    "％": ("%", 1.0),
}


def _primary_number(text: str) -> tuple[float, str, str] | None:
    """取正文主数值 → (归一化值, 单位族, 含该数值的原句)。

    **取正文中出现的第一个数值**（而非量级最大者）——这是 F1/F3/D1 实测校准后的结论：

    反驳"取最大"的教训：口径对比类资料里，第二份资料几乎总会**引述第一份的数字**做对照
    （"……仅 620 亿元，远低于含交付口径的 1800 亿元"）。若按量级最大取，两条证据会抽到
    **同一个数**，比值恒为 1.0，矛盾检测静默失效——实测 F1/F3/D1 三例全因此漏报。
    而"主张"通常在句首、引述在后，取首个数才能各自抽到本方口径。

    这是启发式取舍，不是金融口径判断；离线判别器而已，不替代 LLM 判定。
    """
    m = _NUM_RE.search(text or "")
    if not m:
        return None
    fam, scale = _UNIT_BASE[m.group(2)]
    return (float(m.group(1)) * scale, fam, _sentence_with(text, m.group(0)))


_HEADING_MARK_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)


def _plain(text: str) -> str:
    """剥掉 markdown 标记，供**展示**用（摘要/引用里的句子要是人话，不是 `# 标题 **重点**`）。"""
    s = _HEADING_MARK_RE.sub("", text or "")
    return s.replace("**", "").replace("__", "").replace("`", "")


def _sentence_with(text: str, needle: str, limit: int = 60) -> str:
    for seg in re.split(r"[。；;！!\n]", _plain(text)):
        if needle in seg:
            s = re.sub(r"\s+", " ", seg).strip()
            return s if len(s) <= limit else s[:limit] + "…"
    return needle


_TAIL_SEP_RE = re.compile(r"[—\-]{2,}|[:：]")


def _distinctive_tokens(sub_question: str) -> list[str]:
    """子问题的"区别性部分"：`——`（或冒号）之后那段的分词。

    planner 生成的子问题是「主题——侧面」结构（如「企业知识库 Agent 平台——市场规模」），
    主题词在所有子问题里都出现，只有侧面能区分证据是否真的切题。
    """
    parts = [p for p in _TAIL_SEP_RE.split(sub_question or "") if p.strip()]
    tail = parts[-1] if len(parts) > 1 else (sub_question or "")
    return [t for t in re.split(r"[——/、，,与和的\s]+", tail) if len(t) >= 2]


def _claim_tokens(text: str) -> list[str]:
    """比较两句是否在谈同一指标用的实词（剔除纯数字与单字）。"""
    out = []
    for t in re.split(r"[^\w\u4e00-\u9fff]+", text or ""):
        if len(t) >= 2 and not t.isdigit():
            out.append(t)
    return out


def _topic_label(sub_question: str) -> str:
    s = (sub_question or "").strip()
    for sep in ("——", "：", ":", "·"):
        if sep in s:
            s = s.split(sep)[-1].strip()
    return s[:20] or "未标注主题"


def _dedupe_by_url(records: list[dict[str, str]]) -> list[dict[str, str]]:
    """同一 URL 只保留一条（编号含子问题序号，跨子问题重复检索会拿到多个编号）。"""
    seen: dict[str, dict[str, str]] = {}
    for r in records:
        seen.setdefault(r.get("url") or r.get("title", ""), r)
    return list(seen.values())


def _unique(items) -> list[str]:
    seen: dict[str, None] = {}
    for i in items:
        if i:
            seen.setdefault(i, None)
    return list(seen)


def _snippet(text: str, n: int) -> str:
    s = re.sub(r"\s+", " ", _plain(text).strip())
    return s if len(s) <= n else s[:n].rstrip() + "…"


def _rank_for_chapter(chapter: str, records: list[dict[str, str]]) -> list[dict[str, str]]:
    """给章节挑证据。

    优先级：① 章节名出现在证据所属子问题里（planner 常把大纲项写进子问题）；
    ② 章节实词与证据正文/标题重叠。**无命中就返回空**——宁可显式标注"证据不足"，
    也不拿一篇不相关的长文去凑数（那正是报告可信度崩塌的开始）。
    """
    ch = (chapter or "").strip()
    if ch:
        by_subq = [r for r in records if ch in r.get("sub_question", "")]
        if by_subq:
            return by_subq

    toks = [t for t in re.split(r"[——/、，,与和的\s]+", ch) if len(t) >= 2]
    if not toks:
        return []
    scored: list[tuple[int, int, dict[str, str]]] = []
    for r in records:
        blob = r.get("title", "") + r.get("content", "")
        hit = sum(1 for t in toks if t in blob)
        scored.append((hit, len(r.get("content", "")), r))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [r for hit, _, r in scored if hit > 0]


_SUBJECT_RE = re.compile(r"[「『\"']([^」』\"']{2,40})[」』\"']")
_PART_RE = re.compile(r"按(.{1,40}?)(?:三|四|五|两)?(?:个)?(?:部分|方面|章节)")


def _extract_subject(query: str) -> str:
    m = _SUBJECT_RE.search(query)
    if m:
        return m.group(1)
    cleaned = re.sub(r"^(请|帮我|帮忙)?(调研|研究|分析|查一下|看看)", "", query.strip())
    cleaned = re.sub(r"[，,。；;].*$", "", cleaned)
    return cleaned[:30] or "该主题"


def _extract_parts(query: str) -> list[str]:
    m = _PART_RE.search(query)
    if not m:
        return []
    raw = m.group(1)
    parts = [p.strip() for p in re.split(r"[、/／,，和与及]", raw) if p.strip()]
    return parts if len(parts) >= 2 else []


# ==================================================================== dashscope


class DashScopeProvider:
    """生产 Provider：OpenAI 兼容模式直连（httpx，不额外引入 openai SDK）。"""

    name = "dashscope"

    def __init__(self, api_key: str, base_url: str, *, max_retries: int = 3) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        url = f"{self._base_url}{path}"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        last: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"可重试状态码 {resp.status_code}", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, httpx.TimeoutException) as exc:  # noqa: PERF203
                last = exc
                backoff = 2**attempt
                log.warning(f"[dashscope] 第 {attempt + 1} 次失败：{exc}；{backoff}s 后重试")
                time.sleep(backoff)
        raise RuntimeError(f"DashScope 调用失败（已重试 {self._max_retries} 次）：{last}") from last

    def chat(self, messages, *, model, temperature=0.2, json_schema=None, task="generic", timeout=60.0):
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if json_schema is not None:
            payload["response_format"] = {"type": "json_object"}
        data = self._post("/chat/completions", payload, timeout)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        usage = data.get("usage", {}) or {}
        return ProviderResult(
            text=msg.get("content") or "",
            reasoning=msg.get("reasoning_content"),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            token_estimate=False,
            meta={"provider": "dashscope", "finish_reason": choice.get("finish_reason")},
        )

    def embed(self, texts, *, model, timeout=60.0):
        data = self._post("/embeddings", {"model": model, "input": list(texts)}, timeout)
        items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        return [item.get("embedding", []) for item in items]


# ==================================================================== ollama


class OllamaProvider:
    """离线兜底（仅断网时可用）。原生 /api/chat。"""

    name = "ollama"

    def __init__(self, base_url: str, *, max_retries: int = 2) -> None:
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries

    def chat(self, messages, *, model, temperature=0.2, json_schema=None, task="generic", timeout=120.0):
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_schema is not None:
            payload["format"] = "json"
        last: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = httpx.post(f"{self._base_url}/api/chat", json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                return ProviderResult(
                    text=(data.get("message") or {}).get("content", ""),
                    input_tokens=int(data.get("prompt_eval_count", 0)),
                    output_tokens=int(data.get("eval_count", 0)),
                    token_estimate=False,
                    meta={"provider": "ollama"},
                )
            except (httpx.HTTPError, httpx.TimeoutException) as exc:  # noqa: PERF203
                last = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"Ollama 调用失败：{last}") from last

    def embed(self, texts, *, model, timeout=60.0):
        out = []
        for t in texts:
            resp = httpx.post(
                f"{self._base_url}/api/embeddings", json={"model": model, "prompt": t}, timeout=timeout
            )
            resp.raise_for_status()
            out.append(resp.json().get("embedding", []))
        return out
