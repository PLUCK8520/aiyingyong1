"""提示词模板与消息构造（④ 服务层）。

为什么放 llm/ 而不是 agents/：提示词是**面向模型**的产物，属服务层；
③ agents 依赖 ④ llm 是合规的向下依赖，反之则是架构违规。

机器可读标记（`<<OBJECTIVE>>` / `<<OUTLINE>>`）是给离线 mock 解析用的，
提示词与 mock 共用同一份定义，避免两边各写一套解析。
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from ..retrieval.citations import (
    Evidence,
    render_conflicts_block,
    render_evidence_block,
    render_pairs_block,
)

OBJ_BEGIN = "<<OBJECTIVE>>"
OBJ_END = "<<END>>"
OUTLINE_BEGIN = "<<OUTLINE>>"
OUTLINE_END = "<<END>>"
SUBQ_BEGIN = "<<SUBQ>>"
SUBQ_END = "<<END>>"

_OBJ_RE = re.compile(re.escape(OBJ_BEGIN) + r"(.*?)" + re.escape(OBJ_END), re.DOTALL)
_OUTLINE_BLOCK_RE = re.compile(
    re.escape(OUTLINE_BEGIN) + r"(.*?)" + re.escape(OUTLINE_END), re.DOTALL
)
_SUBQ_BLOCK_RE = re.compile(re.escape(SUBQ_BEGIN) + r"(.*?)" + re.escape(SUBQ_END), re.DOTALL)
_NUMBERED_RE = re.compile(r"^\s*(?:\d+[.、)]|-)\s*(.+?)\s*$", re.MULTILINE)


def compact_block(text: str) -> str:
    return (text or "").strip()


def objective_block(objective: str) -> str:
    return f"{OBJ_BEGIN}{compact_block(objective)}{OBJ_END}"


def outline_block(outlines: Sequence[str]) -> str:
    body = "\n".join(f"{i}. {o}" for i, o in enumerate(outlines, start=1))
    return f"{OUTLINE_BEGIN}\n{body}\n{OUTLINE_END}"


def subq_block(sub_questions: Sequence[str]) -> str:
    body = "\n".join(f"{i}. {o}" for i, o in enumerate(sub_questions, start=1))
    return f"{SUBQ_BEGIN}\n{body}\n{SUBQ_END}"


def parse_objective(text: str) -> str:
    m = _OBJ_RE.search(text or "")
    return m.group(1).strip() if m else ""


def parse_outline(text: str) -> list[str]:
    m = _OUTLINE_BLOCK_RE.search(text or "")
    body = m.group(1) if m else (text or "")
    return [hit.group(1) for hit in _NUMBERED_RE.finditer(body)]


def parse_subqs(text: str) -> list[str]:
    m = _SUBQ_BLOCK_RE.search(text or "")
    if not m:
        return []
    return [hit.group(1) for hit in _NUMBERED_RE.finditer(m.group(1))]


# ------------------------------------------------------------------ 消息构造

INTENT_SYSTEM = (
    "你是调研系统的意图分流器。判断用户问题该走哪条路：\n"
    "- direct：闲聊、定义、简单常识、可直接回答的问题；\n"
    "- research：需要检索多源信息、拆解子问题、综合成文的市场/行业/技术调研。\n"
    '只输出 JSON：{"route": "direct"|"research", "reason": "一句话理由"}。'
)

DIRECT_SYSTEM = (
    "你是调研助手。这是一个可直接回答的问题，用 3 句以内简洁回答，不要编造数据；"
    "拿不准就说明不确定性。"
)

PLANNER_SYSTEM = (
    "你是调研规划器。把用户问题拆成 3–5 个可独立检索的子问题，并给出报告大纲。\n"
    "要求：\n"
    "1. 子问题之间不重叠，且**每个子问题都必须能靠公开资料检索到证据**"
    "（网页、行业报告、公司公告、新闻、政策文件）；\n"
    "2. **不要输出只能通过企业内部调研才能回答的问题**（如用户满意度问卷、内部访谈、"
    "员工调研、客户 NPS）——这类问题公开信源里没有，会留下空章节；\n"
    "3. 子问题要落在**可查证**的维度上（市场规模与增速、主要参与方与竞争格局、产品与技术路线、"
    "收费模式与落地成本、政策与合规环境等）；\n"
    "4. 大纲是报告章节名，与子问题不必一一对应；章节名要具体，不要\"其他\"\"补充\"这类泛称。\n"
    '只输出 JSON：{"objective": str, "sub_questions": [str], "outlines": [str], "requires_data": bool}。'
)

JUDGE_SYSTEM = (
    "你是证据判别器。逐条评估证据与所属子问题的相关度（relevance 1-5）与内容可信度（confidence 1-5），"
    "并指出证据不足的缺口（gaps）。\n"
    '只输出 JSON：{"judgments": [{"citation_id": str, "sub_question": str, "relevance": int, '
    '"confidence": int, "note": str}], "gaps": [str]}。'
)

CONFLICT_SYSTEM = (
    "你是矛盾检测器。给定若干**证据对**（同一子问题下、主题相近的两条证据），"
    "判断它们是否构成真实冲突。规则：\n"
    "1. **口径不同不算冲突**。同一指标因统计范围不同而数值不同，只有在无法并存时才判冲突；\n"
    "2. 只有同时满足「指向同一指标」且「数值或结论互斥」才输出 conflict；\n"
    "3. claim_a / claim_b 必须是**证据原文里的原话片段**（≤60 字），不要改写、不要推断；\n"
    "4. 严重度按差异量级：数值差 ≥3 倍 high、≥1.5 倍 medium、其余 low；\n"
    "5. 不构成冲突的证据对**不要输出**。宁缺毋滥。\n"
    '只输出 JSON：{"conflicts": [{"sub_question": str, "topic": str, "claim_a": str, '
    '"source_a": str, "claim_b": str, "source_b": str, "severity": "low"|"medium"|"high", '
    '"summary": str}]}，其中 source_* 用给定证据编号（如 [WEB1-1-1]）。'
)

ANALYST_SYSTEM = (
    "你是调研撰稿人。基于给定证据撰写结构化中文调研报告。\n"
    "**写法（正文是分析论述，不是证据罗列）**：\n"
    "1. 用**自己的话**归纳观点——对比、归因、趋势、对读者的含义；**禁止整段照抄证据原文**，"
    "证据只在需要支撑结论时以短语形式带入；\n"
    "2. 引用编号置于**结论句末**（如「…复合增速约 42% [WEB1-1-1]」），"
    "**不要放在句首**当分隔符——句首挂编号正是「证据堆砌」的典型形态；\n"
    "3. 同一份证据**只在最相关的章节引用一次**，不要在每章复述同一段证据；\n"
    "4. 每章至少一句**分析性判断**（谁更强 / 为什么 / 意味着什么），不能只有事实堆叠；\n"
    "5. 章节内容必须扣住该章标题；某章无对应证据时，直接写「（证据不足）」并说明缺什么，"
    "**不要拿别的章节的证据来填充**。\n"
    "**铁律**：\n"
    "1. 每一句涉及事实 / 数据的结论，必须紧跟其证据编号，如 [WEB1-1-1]；\n"
    "2. 只能使用给定证据里的信息，**禁止补充外部数据或常识推断**；\n"
    "3. 证据互相矛盾时，单列「争议与分歧」小节，摆明双方口径，**不要强行调和**；\n"
    "4. **不要输出「参考资料」章节**，系统会按正文引用顺序自动追加，避免重复与编号漂移。\n"
    "输出 Markdown：一级标题为报告名，随后是「核心摘要」与按大纲分章正文。"
)

AUDIT_SYSTEM = (
    "你是引用审计员。给定若干「引用编号 + 该编号的证据原文 + 引用了该编号的句子」，"
    "逐句判断句子是否被**它所引用的那条证据**支持。三档判定：\n"
    "- supported  ：证据直接支持该句的关键事实与数值；\n"
    "- partial    ：证据只部分支持，句子里还有证据未提及的要素；\n"
    "- unsupported：证据不支持，**或句子里的关键数值在证据里不存在**（数值不一致是硬否决）。\n"
    "铁律：\n"
    "1. 只依据给定证据判断，**不得引入外部知识或常识**；\n"
    "2. 数值必须逐字核对——引用审计最危险的不是'没找到'，而是'数字被改过却看起来对'；\n"
    '3. 只输出 JSON：{"items": [{"citation_id": str, "sentence": str, '
    '"verdict": "supported"|"partial"|"unsupported", "reason": str}]}；\n'
    "4. sentence 必须原样回抄待核验句子，不要改写，以便系统回定位。"
)


def build_audit_messages(
    objective: str,
    groups: Sequence[tuple[str, str, Sequence[str]]],
) -> list[dict[str, str]]:
    """引用审计消息（T4.1 / FR-17）。

    `groups` 为 `(citation_id, 证据原文, [引用该编号的句子, ...])` 序列——
    **按 citation_id 分组**是本任务的关键成本约束：若逐句一次调用，调用数会随报告
    长度线性爆炸，审计成本盖过主链路。分组后一组一次，调用数 = 被引用编号数。
    """
    blocks: list[str] = []
    for cid, evidence, sentences in groups:
        lines = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))
        blocks.append(
            f"引用编号 {cid}\n证据原文：{compact_block(evidence)}\n"
            f"待核验句子（共 {len(sentences)} 条）：\n{lines}"
        )
    user = (
        f"调研目标：{objective_block(objective)}\n\n"
        + "\n\n".join(blocks)
        + "\n\n请逐句判定，只输出 JSON。"
    )
    return [
        {"role": "system", "content": AUDIT_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_intent_messages(query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": INTENT_SYSTEM},
        {"role": "user", "content": query},
    ]


def build_direct_messages(query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": query},
    ]


def _append_profile(system: str, profile_block: str) -> str:
    """T5.2：把用户画像片段拼到系统提示词末尾。

    **留空则原样返回**——绝不为空画像插入一个空的"用户偏好"段落，那会污染提示词、
    让模型以为"用户没有偏好"（与"不知道"是不同语义）。
    """
    if not profile_block:
        return system
    return f"{system}\n\n{profile_block}"


def build_planner_messages(query: str, *, profile_block: str = "") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _append_profile(PLANNER_SYSTEM, profile_block)},
        {"role": "user", "content": query},
    ]


def build_judge_messages(
    objective: str, sub_questions: Sequence[str], evidence: Iterable[Evidence]
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"调研目标：{objective_block(objective)}\n\n"
                f"子问题清单：\n{subq_block(sub_questions)}\n\n"
                f"待判别证据：\n{render_evidence_block(evidence)}"
            ),
        },
    ]


def build_conflict_messages(
    objective: str,
    evidence: Iterable[Evidence],
    pairs: Sequence[tuple[str, str, str]],
) -> list[dict[str, str]]:
    """矛盾检测消息：一份证据块 + 一份**受限**证据对清单（成本闸门在调用方）。

    `pairs` 为 (id_a, id_b, topic) 序列——只传编号对，正文由证据块提供，不重复送上下文。
    """
    user = (
        f"调研目标：{objective_block(objective)}\n\n"
        f"证据原文：\n{render_evidence_block(evidence)}\n\n"
        f"待比对证据对（`id_a || id_b || 主题`）：\n{render_pairs_block(pairs)}\n\n"
        "请逐对判断是否构成矛盾，只输出构成冲突的项。"
    )
    return [
        {"role": "system", "content": CONFLICT_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_analyst_messages(
    objective: str,
    outlines: Sequence[str],
    evidence: Iterable[Evidence],
    conflicts: Sequence[object] = (),
    *,
    profile_block: str = "",
) -> list[dict[str, str]]:
    conflict_part = ""
    if conflicts:
        conflict_part = (
            "\n\n已检出的**矛盾清单**（请在「争议与分歧」章节逐一呈现，并列双方口径与编号，"
            "**不要调和、不要取平均、不要选边**）：\n"
            f"{render_conflicts_block(conflicts)}"
        )
    user = (
        f"{objective_block(objective)}\n\n"
        f"{outline_block(outlines)}\n\n"
        f"可用证据：\n{render_evidence_block(evidence)}"
        f"{conflict_part}\n\n"
        "请按大纲撰写报告。"
    )
    return [
        {"role": "system", "content": _append_profile(ANALYST_SYSTEM, profile_block)},
        {"role": "user", "content": user},
    ]


# ------------------------------------------------------------------ T4.2b 章节重写

REWRITE_BEGIN = "<<REWRITE>>"
SECBODY_BEGIN = "<<SECTION_BODY>>"
REWRITE_END_MARK = "<<END>>"

REWRITE_SYSTEM = (
    "你是调研报告的**章节重写员**。上一轮写出的某章节，其句子被引用审计判为"
    "「证据不支持」——它在证据里找不到对应的事实或数值。现在请你只重写这一节。铁律：\n"
    "1. **只保留证据能支持的结论**；证据没提到的数字、比例、结论，一律不得出现；\n"
    "2. 每一句涉及事实/数据的结论，必须紧跟其证据编号，如 [WEB1-1-1]；\n"
    "3. 只能使用给定证据，**禁止补充外部数据或常识推断**；\n"
    "4. 若给定证据不足以支撑该章节的主题，就**如实写「（证据不足）」**，不要用空话凑数；\n"
    "5. 只输出重写后的**章节正文**，不要重复章节标题，也不要输出「参考资料」。\n"
    "6. 输出格式：把正文放在 "
    f"{SECBODY_BEGIN} 与 {REWRITE_END_MARK} 之间；不要输出其它标记或解释。"
)

_SECTION_BODY_RE = re.compile(
    re.escape(SECBODY_BEGIN) + r"(.*?)" + re.escape(REWRITE_END_MARK), re.DOTALL
)
_REWRITE_BLOCK_RE = re.compile(
    re.escape(REWRITE_BEGIN) + r"(.*?)" + re.escape(REWRITE_END_MARK), re.DOTALL
)


def _rewrite_block(title: str, body: str) -> str:
    return f"{REWRITE_BEGIN}{compact_block(title)}{REWRITE_END_MARK}\n{compact_block(body)}"


def parse_rewrite_target(text: str) -> tuple[str, str]:
    """从重写提示词里取回 (章节标题, 原正文)。供离线 mock 解析——提示词与 mock 共用一份定义。"""
    m = _REWRITE_BLOCK_RE.search(text or "")
    if not m:
        return "", ""
    block = m.group(1)
    if REWRITE_END_MARK in block:
        title, body = block.split(REWRITE_END_MARK, 1)
        return title.strip(), body.strip()
    return block.strip(), ""


def parse_rewrite_output(text: str) -> str:
    """取模型输出的章节正文；没按标记输出时退化为整段（宽松解析，不中断链路）。"""
    m = _SECTION_BODY_RE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


def build_rewrite_messages(
    objective: str,
    title: str,
    body: str,
    evidence: Iterable[Evidence],
) -> list[dict[str, str]]:
    user = (
        f"调研目标：{objective_block(objective)}\n\n"
        f"待重写章节标题：{title}\n"
        f"该章节当前正文：\n{_rewrite_block(title, body)}\n\n"
        f"本节可用证据（编号 → 原文）：\n{render_evidence_block(evidence)}\n\n"
        "请只重写这一节的正文，按第 6 条格式输出。"
    )
    return [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": user},
    ]
