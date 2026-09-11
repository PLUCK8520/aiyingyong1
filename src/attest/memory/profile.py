"""T5.2 · 用户画像（跨会话偏好记忆）。

**存储**：SQLite 单表 `profile(key, value, updated_at)`（《功能设计》§7 数据设计）。
**注入**：每轮调研前由 `memory_loader` 节点读出全量画像 → 注入 planner / analyst 的提示词。
**更新**：识别「记住 xx」类指令，写入画像。

⚠️ **画像 vs research_memory 是两回事**（别混）：
    - **画像**（本模块）：**用户偏好**（"输出用中文""偏好表格""关注成本"）——小、键值对、可覆盖；
    - **research_memory**（T7.4）：**研究结论**沉淀——大、向量库、只收 `supported` 结论。
    两者受众与生命周期完全不同，故分模块、分存储。

⚠️ **只存偏好，不存隐私**：画像会进提示词、可能进报告、可能被检查点落盘。
    不写手机号/密钥/身份信息；"记住"的应是**偏好与工作方式**，不是个人数据。
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..logging import get_logger

log = get_logger(__name__)

#: 单条 value 的长度上限——防止把整篇文档塞进画像（它会进提示词，直接烧 token）
MAX_VALUE_CHARS = 500
#: 画像总条数上限——注入提示词的画像必须有硬上限，否则长年累积会挤爆上下文
MAX_ENTRIES = 50

_SCHEMA = """
CREATE TABLE IF NOT EXISTS profile (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


@dataclass(frozen=True)
class ProfileItem:
    key: str
    value: str
    updated_at: float


class ProfileStore:
    """画像存储。轻量、无并发写（CLI 单进程），不引入 ORM。"""

    def __init__(self, db_path: Path | str, *, enabled: bool = True) -> None:
        self.db_path = Path(db_path)
        self.enabled = enabled
        self._ensure()

    def _ensure(self) -> None:
        if not self.enabled:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    # ---------------- 读 ----------------

    def all(self) -> list[ProfileItem]:
        """全量读出（按更新时间倒序）——注入提示词的入口。"""
        if not self.enabled:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value, updated_at FROM profile ORDER BY updated_at DESC LIMIT ?",
                (MAX_ENTRIES,),
            ).fetchall()
        return [ProfileItem(k, v, t) for k, v, t in rows]

    def as_prompt_block(self) -> str:
        """把画像渲染成提示词片段。**空画像返回空串**（不注入空段落，避免污染提示词）。"""
        items = self.all()
        if not items:
            return ""
        lines = [f"- {it.key}：{it.value}" for it in items]
        return "【用户偏好（跨会话记忆）】\n" + "\n".join(lines)

    # ---------------- 写 ----------------

    def set(self, key: str, value: str) -> ProfileItem:
        """写入/覆盖一条偏好。长度超限则**截断并留痕**（不静默丢内容）。"""
        if not self.enabled:
            raise RuntimeError("画像存储已关闭（ATTEST_PROFILE_ENABLED=0）")
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError("画像 key 不能为空")
        if len(value) > MAX_VALUE_CHARS:
            log.warning(
                f"[profile] 画像值过长（{len(value)} > {MAX_VALUE_CHARS}），已截断：key={key!r}"
            )
            value = value[:MAX_VALUE_CHARS]
        ts = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO profile(key, value, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, ts),
            )
        log.info(f"[profile] 记住：{key} = {value[:40]}{'…' if len(value) > 40 else ''}")
        return ProfileItem(key, value, ts)

    def delete(self, key: str) -> bool:
        if not self.enabled:
            return False
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM profile WHERE key = ?", (key.strip(),))
        return cur.rowcount > 0

    def clear(self) -> int:
        if not self.enabled:
            return 0
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM profile")
        return cur.rowcount


# ------------------------------------------------------------------ 指令解析

#: 「记住 xx」类指令的触发词。**刻意保守**：只在句首/明确祈使时触发，避免把
#: "记住这个结论"这类调研内容误判成偏好写入。
_REMEMBER_PREFIXES: tuple[str, ...] = (
    "记住",
    "请记住",
    "我要你记住",
    "以后",
    "以后都",
    "今后",
    "下次",
    "remember",
)


def parse_remember_directive(text: str) -> tuple[str, str] | None:
    """从用户输入里识别「记住 xx」指令。

    返回 `(key, value)`；不是指令则返回 None。

    规则（刻意简单、可解释、可测）：
      1. 必须以触发词开头（strip 后）；
      2. 支持 `记住：键=值` / `记住 键=值` 显式键值；
      3. 否则把整句作为 **value**，key 取 `偏好N`（调用方用时间戳补全更稳）。
    """
    t = text.strip()
    if not t:
        return None
    matched = next((p for p in _REMEMBER_PREFIXES if t.lower().startswith(p.lower())), None)
    if matched is None:
        return None
    body = t[len(matched):].lstrip("：: ，,、").strip()
    if not body:
        return None
    # 显式键值：`键 = 值` / `键：值`
    for sep in ("=", "：", ":"):
        if sep in body:
            k, _, v = body.partition(sep)
            k, v = k.strip(), v.strip()
            if k and v:
                return k, v
    return "偏好", body


def format_profile_for_prompt(items: Iterable[ProfileItem]) -> str:
    """供测试与调试：把已读出的条目渲染成提示词片段（与 `as_prompt_block` 同格式）。"""
    items = list(items)
    if not items:
        return ""
    lines = [f"- {it.key}：{it.value}" for it in items]
    return "【用户偏好（跨会话记忆）】\n" + "\n".join(lines)
