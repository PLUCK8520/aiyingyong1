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
    '只输出 JSON：{"objective": str, "sub_questions": [str], "outlines": [str], "requires_data": bool}。\n'
    "要求：子问题之间不重叠；大纲与子问题不必一一对应。"
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
    "你是调研撰稿人。基于给定证据撰写结构化中文调研报告。铁律：\n"
    "1. 每一句涉及事实/数据的结论，必须紧跟其证据编号，如 [WEB1-1-1]；\n"
    "2. 只能使用给定证据里的信息，**禁止补充外部数据或常识推断**；\n"
    "3. 证据互相矛盾时，单列「争议与分歧」小节，摆明双方口径，**不要强行调和**；\n"
    "4. 证据不足的结论要显式标注「（证据不足）」；\n"
    "5. **不要输出「参考资料」章节**，系统会按正文引用顺序自动追加，避免重复与编号漂移。\n"
    "输出 Markdown：一级标题为报告名，随后是「核心摘要」与按大纲分章正文。"
)


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


def build_planner_messages(query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PLANNER_SYSTEM},
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
        {"role": "system", "content": ANALYST_SYSTEM},
        {"role": "user", "content": user},
    ]
