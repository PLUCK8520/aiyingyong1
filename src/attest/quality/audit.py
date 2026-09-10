"""T3.10 · 审计判定**单点原型**（P3 末闸门）。

任务清单把这条定成"闸门"是有道理的：引用忠实性审计是项目的头号亮点（L1-1，★★★），
但它**自身会不会误判**从来没被验证过。所以本节先造一个最小判定器，在 20 条已知答案的
样本对上**脱离完整链路**单独跑一次，先拿到准确率数字，再决定 P4 要不要换策略。

实现选择：**规则版**（实词覆盖 + 数值一致性），不是 LLM 版。理由：
  - 单点验证要的是"可复现的准确率基线"，规则版零成本、零随机性，一天能跑一千次；
  - LLM 版（分组批量核验，FR-17）属 P4 T4.1，届时换的是 `verify_claim` 的实现，
    调用方与评测口径都不动——这就是"先立尺子、再换零件"。

判据（三档）：
  supported   : 关键实词覆盖率 ≥ 80% **且** claim 里的数值全部能在证据中找到
  partial     : 覆盖率 ≥ 45%，或无关键数值但要素不全
  unsupported : 覆盖率过低，**或** claim 的数值在证据中不存在（数值不一致是硬否决）

数值一致性的硬否决是刻意的：引用审计最危险的不是"没找到"，而是"数字被改过却看起来对"。
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from ..logging import get_logger
from ..retrieval.hybrid import tokenize
from ..schemas import AuditItem, AuditVerdict

log = get_logger(__name__)

SUPPORTED_RATIO = 0.80
PARTIAL_RATIO = 0.45

_UNIT_FAMILY: dict[str, tuple[str, float]] = {
    "亿元": ("yuan", 1e8),
    "亿": ("yuan", 1e8),
    "万元": ("yuan", 1e4),
    "万": ("yuan", 1e4),
    "元": ("yuan", 1.0),
    "%": ("pct", 1.0),
    "％": ("pct", 1.0),
    "倍": ("times", 1.0),
    "家": ("count", 1.0),
    "个": ("count", 1.0),
}
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(亿元|亿|万元|万|元|%|％|倍|家|个)")


def _numbers(text: str) -> set[tuple[str, float]]:
    """抽出 (单位族, 归一化值)。按族比较，避免拿 62 亿去匹配 62%。"""
    out: set[tuple[str, float]] = set()
    for m in _NUM_RE.finditer(text or ""):
        fam, scale = _UNIT_FAMILY[m.group(2)]
        out.add((fam, round(float(m.group(1)) * scale, 4)))
    return out


def _content_tokens(text: str) -> list[str]:
    """审计用实词：只保留 ≥2 字的词，单字噪声太大（"的""是"已由分词器过滤）。"""
    return [t for t in tokenize(text) if len(t) >= 2]


def verify_claim(
    claim: str,
    evidence: str,
    *,
    supported_ratio: float = SUPPORTED_RATIO,
    partial_ratio: float = PARTIAL_RATIO,
) -> tuple[AuditVerdict, str]:
    """判定单条 claim 是否被证据支持。返回 (裁决, 理由)。"""
    claim = (claim or "").strip()
    if not claim:
        return "unsupported", "空 claim，无法判定"
    uniq = list(dict.fromkeys(_content_tokens(claim)))
    if not uniq:
        return "partial", "claim 里没有可判定的实词（疑似纯指代句）"

    ev_tokens = set(_content_tokens(evidence))
    hit = [t for t in uniq if t in ev_tokens]
    ratio = len(hit) / len(uniq)

    claim_nums = _numbers(claim)
    if claim_nums:
        missing = claim_nums - _numbers(evidence)
        if missing:
            shown = "、".join(f"{v:g}({fam})" for fam, v in sorted(missing, key=lambda x: -x[1]))
            return "unsupported", f"证据中不存在 claim 的关键数值：{shown}"

    if ratio >= supported_ratio:
        return "supported", f"关键实词覆盖率 {ratio:.0%}，数值一致"
    if ratio >= partial_ratio:
        return "partial", f"仅部分覆盖（{ratio:.0%}），存在证据未提及的要素"
    return "unsupported", f"关键实词覆盖率过低（{ratio:.0%}）"


_CLAIM_SPLIT_RE = re.compile(r"(?<=[。；;！!？?])|\n+")


def extract_claims(report: str, citation_ids: Iterable[str]) -> list[tuple[str, str]]:
    """从报告里抽出「带引用的句子」→ [(句子, 引用编号)]。审计只关心有引用的句子。"""
    known = set(citation_ids)
    out: list[tuple[str, str]] = []
    for raw in _CLAIM_SPLIT_RE.split(report or ""):
        sent = (raw or "").strip().lstrip("-*# ").strip()
        if not sent:
            continue
        ids = [i for i in re.findall(r"\[(?:WEB|LOC)\d+-\d+-\d+\]", sent) if i in known]
        for cid in dict.fromkeys(ids):
            out.append((sent, cid))
    return out


def audit_claims(
    claims: Sequence[tuple[str, str]], evidence_by_id: dict[str, str]
) -> list[AuditItem]:
    """逐条判定（真实版应按 citation_id 分组批量调用，见 FR-17；此处是同口径的规则实现）。"""
    items: list[AuditItem] = []
    for sent, cid in claims:
        evidence = evidence_by_id.get(cid, "")
        verdict, reason = verify_claim(sent, evidence)
        items.append(AuditItem(citation_id=cid, verdict=verdict, reason=reason))
    return items
