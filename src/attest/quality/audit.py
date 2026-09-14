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
from ..retrieval.citations import parse_basis_line
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
#: 引用编号不是 claim 的内容——判定前必须先剥掉。实测踩过坑：碎片句「… [WEB1-1-1]」里的
#: "WEB1" 被当成实词，覆盖率算成 0% → 误判 unsupported → 对正确的报告触发假降级。
_CITE_STRIP_RE = re.compile(r"\[(?:WEB|LOC)\d+-\d+-\d+\]")


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
    # 先剥掉引用编号再判定：编号（如 WEB1-1-1）不是 claim 的内容，
    # 留着会把"含编号的碎片句"误判为 unsupported（见 _CITE_STRIP_RE 注释）。
    text = _CITE_STRIP_RE.sub("", claim)
    if not (evidence or "").strip():
        # 证据为空 = 这个编号在证据集里**根本不存在**（模型臆造）。必须判 unsupported 而不是
        # 走下面的覆盖率分支：理由要让人一眼看出是"幻觉编号"，而不是"覆盖率低"。
        return "unsupported", "该编号在证据集中不存在（疑似模型臆造编号），无证据可核验"
    uniq = list(dict.fromkeys(_content_tokens(text)))
    if not uniq:
        return "partial", "claim 里没有可判定的实词（疑似纯指代/被截断的占位句）"

    ev_tokens = set(_content_tokens(evidence))
    hit = [t for t in uniq if t in ev_tokens]
    ratio = len(hit) / len(uniq)

    claim_nums = _numbers(text)
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
    """从报告里抽出「带引用的句子」→ [(句子, 引用编号)]。审计只关心有引用的句子。

    ⚠️ **不过滤"未知编号"**（2026-09-13 真实运行暴露的漏洞，务必保持）：
    原实现是 `ids = [i for i in findall(...) if i in known]`，于是**引用了不存在编号的
    句子被静默跳过**——而这恰恰是最该被质证的一类：模型臆造编号。
    实测案例：第 2 轮只补检了子问题 6–9，模型却在正文里写了 `[WEB2-2-2]`——该编号
    在证据集里根本不存在。过滤掉等于给幻觉开后门：正文留着编号、参考资料里查无此条、
    审计也不吭声，而 `CitationIndex` 只报出一个"未解析"，没人知道是哪句话的问题。
    现在未知编号一并送审，`verify_claim` 会因证据为空判 `unsupported`，
    再由 `degrade_report` 去掉编号并标注「（未证实）」——这才是"逐句质证"的完整闭环。

    `citation_ids` 参数保留：调用方仍在传"已知编号集合"，但实现上**不再用于过滤**。
    留着是为了不破坏既有调用方与测试签名；语义已改为仅供调用方自述。
    """
    out: list[tuple[str, str]] = []
    for raw in _CLAIM_SPLIT_RE.split(report or ""):
        sent = (raw or "").strip().lstrip("-*# ").strip()
        if not sent:
            continue
        ids = dict.fromkeys(_CITE_RE.findall(sent))
        for cid in ids:
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


# ------------------------------------------------------------------ T4.1/T4.2：分节 + 定位 + 降级

REF_HEADING = "## 参考资料"
UNVERIFIED_MARK = "（未证实）"
_CITE_RE = re.compile(r"\[(?:WEB|LOC)\d+-\d+-\d+\]")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")


def report_body(report: str) -> str:
    """截掉文末「参考资料」章节——审计只针对正文（参考行不含正文引用语义）。"""
    return (report or "").split(REF_HEADING, 1)[0]


def iter_sections(report: str) -> list[tuple[str, str]]:
    """把正文按 markdown 标题切成 [(章节标题, 章节正文)]。首个无标题段落标题为空串。"""
    sections: list[tuple[str, str]] = []
    title = ""
    buf: list[str] = []
    for line in report_body(report).splitlines():
        m = _HEADING_RE.match(line)
        if m:
            if buf or title:
                sections.append((title, "\n".join(buf)))
            title = m.group(1).strip()
            buf = []
        else:
            buf.append(line)
    if buf or title:
        sections.append((title, "\n".join(buf)))
    return sections


def extract_claims_sectioned(
    report: str, citation_ids: Iterable[str]
) -> list[tuple[str, str, str]]:
    """在 `extract_claims` 基础上带上所属章节 → [(句子, 引用编号, 章节标题)]。

    同样**不过滤未知编号**（理由见 `extract_claims` 的 docstring）——这是审计器
    `citation_auditor` 实际调用的入口，过滤掉的话幻觉编号就彻底没人管了。

    **T9.1 章节级依据行的两条规则**（与 `retrieval.citations.SECTION_BASIS_MARK` 契约配套）：
      1. 依据行本身**不当 claim**：它是"本章论述依据这些来源"的元信息，不是事实性结论，
         拿它去核对证据必判 unsupported，会被降级器剥掉编号——恰好杀死它要保护的引用；
      2. 但本章内**无编号的句子**会关联到本章声明的编号集合**逐句受审**——
         弱模型给不出句级编号（2026-09-13 实测四次整篇零编号），章节级声明是它
         能服从的替代；判定走"该句所引来源的并集"，与多源句子同口径，
         凭空数值 / 凭空要素照样被 `verify_claim` 硬否决。
    """
    out: list[tuple[str, str, str]] = []
    for title, body in iter_sections(report):
        sents: list[str] = []
        basis_ids: list[str] = []
        for raw in _CLAIM_SPLIT_RE.split(body):
            sent = (raw or "").strip().lstrip("-*# ").strip()
            if not sent:
                continue
            ids_in_basis = parse_basis_line(sent)
            if ids_in_basis is not None:
                basis_ids = ids_in_basis
                continue  # 规则 1：依据行是元信息，不送审
            sents.append(sent)
        for sent in sents:
            own = list(dict.fromkeys(_CITE_RE.findall(sent)))
            for cid in (own if own else basis_ids):  # 规则 2：无编号句关联章节声明
                out.append((sent, cid, title))
    return out


def group_claims_by_citation(
    claims: Iterable[tuple[str, str]]
) -> dict[str, list[str]]:
    """按 citation_id 分组（FR-17）：同一引用的多个句子合并为一次核验调用。"""
    grouped: dict[str, list[str]] = {}
    for sent, cid in claims:
        bucket = grouped.setdefault(cid, [])
        if sent not in bucket:
            bucket.append(sent)
    return grouped


def sentence_citation_map(claims: Iterable[tuple[str, str, str]]) -> dict[str, list[str]]:
    """句子 → 该句引用的全部编号（保持出现顺序、去重）。

    为什么需要它：一句话可能**同时引用两个来源**（尤其「争议与分歧」里并列双方口径的句子）。
    此时按单条来源核对数值必然"缺数"，会把正确的句子误判为 unsupported——实测踩到过。
    所以判定要按**该句所引来源的并集**，而不是只看其中一条。
    """
    out: dict[str, list[str]] = {}
    for sent, cid, _ in claims:
        bucket = out.setdefault(sent, [])
        if cid not in bucket:
            bucket.append(cid)
    return out


def union_evidence_text(sent_cids: Iterable[str], evidence_by_id: dict[str, str]) -> str:
    """把一句话所引来源的证据正文拼成并集文本（去重、保持顺序）。"""
    seen: dict[str, None] = {}
    for cid in sent_cids:
        text = evidence_by_id.get(cid)
        if text:
            seen.setdefault(text, None)
    return "\n".join(seen)


def degrade_report(report: str, items: Sequence[AuditItem]) -> tuple[str, dict[str, object]]:
    """T4.2a 降级动作：unsupported 的句子 → **去掉其引用编号**并标注「（未证实）」。

    为什么"去编号 + 标注"而不是"整句删除"：Attest 的立场是**不静默删证据**——
    读者应当看到"这里原本有个结论，但它没通过核验"，而不是被无声抹掉。

    只在正文里替换；参考资料由调用方用同一索引重渲染，保证编号不悬空。
    """
    bad: dict[str, set[str]] = {}
    for it in items:
        if it.verdict == "unsupported" and it.sentence:
            bad.setdefault(it.sentence, set()).add(it.citation_id)
    if not bad:
        return report, {"degraded": 0, "removed_citations": [], "sections": []}

    out = report
    removed: list[str] = []
    sections: set[str] = set()
    degraded = 0
    for sent, cids in bad.items():
        if sent not in out:
            continue  # 句子已被前一次替换改写，跳过（幂等）
        new_sent = sent
        for cid in cids:
            new_sent = new_sent.replace(cid, "")
            removed.append(cid)
        # 去掉编号后常残留"词 空格 标点"的缝隙，顺手清理，避免出现"42% 。"这类观感问题
        new_sent = re.sub(r"\s{2,}", " ", new_sent)
        new_sent = re.sub(r"\s+([。；;！!？?，,、）)])", r"\1", new_sent)
        new_sent = re.sub(r"[（(]\s*[）)]", "", new_sent)  # 编号被移走后可能留下空括号
        new_sent = new_sent.strip()
        out = out.replace(sent, f"{new_sent}{UNVERIFIED_MARK}")
        degraded += 1
    for it in items:
        if it.verdict == "unsupported" and it.section:
            sections.add(it.section)
    return out, {
        "degraded": degraded,
        "removed_citations": sorted(set(removed)),
        "sections": sorted(sections),
    }


def section_failure_ratios(items: Sequence[AuditItem]) -> dict[str, float]:
    """按章节汇总 unsupported 占比（T4.2b 判定"该章节要不要重写"用）。"""
    total: dict[str, int] = {}
    bad: dict[str, int] = {}
    for it in items:
        sec = it.section or "(无章节)"
        total[sec] = total.get(sec, 0) + 1
        if it.verdict == "unsupported":
            bad[sec] = bad.get(sec, 0) + 1
    return {s: round(bad.get(s, 0) / n, 4) for s, n in total.items() if n}


# ------------------------------------------------------------- T4.2b：章节重写

#: 无标题段没有锚点，无法定位重写——只能靠"降级标注"兜底
PLACEHOLDER_SECTION = "(无章节)"


def rewrite_targets(
    items: Sequence[AuditItem], threshold: float
) -> dict[str, float]:
    """需要重写的章节 → 失败率（严格大于阈值才重写）。

    无标题段（`PLACEHOLDER_SECTION`）**排除在外**：没有标题就没有替换锚点，
    硬重写会写错位置。这类段落由 `degrade_report` 逐句标注「未证实」兜底。
    """
    out: dict[str, float] = {}
    for sec, ratio in section_failure_ratios(items).items():
        if not sec or sec == PLACEHOLDER_SECTION:
            continue
        if ratio > threshold:
            out[sec] = ratio
    return out


def section_body(report: str, title: str) -> str:
    """取指定章节的正文（不含标题行，已 strip）。找不到返回空串。"""
    for sec_title, body in iter_sections(report):
        if sec_title == title:
            return body.strip()
    return ""


def citations_in_section(report: str, title: str) -> list[str]:
    """该章节正文里出现过的引用编号（按出现顺序去重）——重写时只喂这些证据。"""
    return list(dict.fromkeys(_CITE_RE.findall(section_body(report, title))))


def replace_section(report: str, title: str, new_body: str) -> str:
    """把指定章节的正文整体替换（保留标题行与文末「参考资料」）。找不到章节则原样返回。

    只动目标章节的行区间：从它的标题行起，到下一个标题行前止。这样顶层标题（`# 报告`）
    与相邻章节都不受影响——避免"重写一节、错搬一片"。
    """
    body = report_body(report)
    tail = report[len(body) :]  # 以「## 参考资料」开头（可能为空）
    lines = body.splitlines()
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(lines):
        line = lines[i]
        m = _HEADING_RE.match(line)
        if m and m.group(1).strip() == title:
            out.append(line)
            out.append("")
            out.append(new_body.strip())
            i += 1
            while i < len(lines) and not _HEADING_RE.match(lines[i]):
                i += 1
            replaced = True
            continue
        out.append(line)
        i += 1
    if not replaced:
        return report
    return "\n".join(out).rstrip() + "\n" + tail
