"""T9.5 · 冷启动会话恢复的测试。

**为什么值得单独一组**：这条路径修的是"数据健在、界面却显示'还没有会话'"——
故障表现是**静默的**（没有报错，只是列表空），所以必须有测试钉住它，
否则下次谁动了 `_sessions` / `list_sessions()` 都不会有人发现入口又断了。

不依赖真实图运行：直接往临时 sqlite 里塞**与 LangGraph 同形**的检查点行。
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app import session as sess


# ----------------------------------------------------------------- 夹具

@pytest.fixture(autouse=True)
def _clean_registry():
    """`_sessions` 是模块级全局——不在每个用例前后清干净，用例之间会互相污染。"""
    sess._sessions.clear()
    yield
    sess._sessions.clear()


def _mkdb(path: Path) -> None:
    """建一张与 LangGraph 同形的 checkpoints 表。"""
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE checkpoints ("
        " thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL DEFAULT '',"
        " checkpoint_id TEXT NOT NULL, parent_checkpoint_id TEXT,"
        " type TEXT, checkpoint BLOB, metadata BLOB)"
    )
    con.commit()
    con.close()


def _add(path: Path, thread_id: str, state: dict, ts: str = "2026-09-13T02:00:00+00:00") -> None:
    """写入一条检查点（走真 serde，保证二进制形态与线上一致）。"""
    serde = JsonPlusSerializer()
    ck = {"channel_values": state, "channel_versions": {}, "id": "c1", "ts": ts, "v": 1}
    typ, blob = serde.dumps_typed(ck)
    con = sqlite3.connect(path)
    con.execute(
        "INSERT INTO checkpoints VALUES (?, '', ?, NULL, ?, ?, ?)",
        (thread_id, "cp-%s" % thread_id, typ, blob, b"{}"),
    )
    con.commit()
    con.close()


def _mgr(db: Path) -> sess.SessionManager:
    return sess.SessionManager(SimpleNamespace(checkpoint_db=db))


# ----------------------------------------------------------------- 状态推断

def test_infer_status_report_means_done():
    assert sess._infer_status({"report": "# 报告\n正文"}) == "done"


def test_infer_status_direct_answer_means_done():
    assert sess._infer_status({"direct_answer": "一句话回答"}) == "done"


def test_infer_status_errors_means_error():
    assert sess._infer_status({"errors": [{"node": "analyst", "msg": "boom"}]}) == "error"


def test_infer_status_pending_confirm():
    """有 plan 但没 approved → T5.4 静态断点挂起。"""
    assert sess._infer_status({"plan": {"chapters": []}, "plan_approved": False}) == "awaiting_confirm"


def test_infer_status_halfway_is_aborted_not_done():
    """关键：跑到一半的会话**不许**被美化成 done——推断值要偏保守。"""
    assert sess._infer_status({"evidence": [1, 2, 3], "plan": {"x": 1}, "plan_approved": True}) == "aborted"


def test_infer_status_empty_is_aborted():
    assert sess._infer_status({}) == "aborted"


# ----------------------------------------------------------------- 扫描

def test_recover_reads_query_and_status(tmp_path: Path):
    db = tmp_path / "ck.sqlite"
    _mkdb(db)
    _add(db, "web-1", {"query": "调研企业知识库 Agent 平台", "report": "# 报告"})
    _add(db, "web-2", {"query": "调研国产数据库替换", "errors": [{"msg": "x"}]})

    rows = sess._recover_sessions_sync(db)
    by_id = {r[0]: r for r in rows}

    assert set(by_id) == {"web-1", "web-2"}
    assert by_id["web-1"][1] == "调研企业知识库 Agent 平台"
    assert by_id["web-1"][3] == "done"
    assert by_id["web-2"][3] == "error"
    # 时间戳要真的解出来（2026-09-13，不是退回的当前时间）
    assert by_id["web-1"][2] > 1_700_000_000


def test_recover_takes_last_checkpoint_per_thread(tmp_path: Path):
    """同一个 thread 有多条时，只认最后一条（后面那条才有终态产物）。"""
    db = tmp_path / "ck.sqlite"
    _mkdb(db)
    _add(db, "web-1", {"query": "问题", "plan": {"a": 1}})
    _add(db, "web-1", {"query": "问题", "report": "# 完成"})

    rows = sess._recover_sessions_sync(db)
    assert len(rows) == 1
    assert rows[0][3] == "done"


def test_recover_skips_broken_row_without_losing_others(tmp_path: Path):
    """一条坏数据不许拖垮整张列表——其余会话要照常恢复。"""
    db = tmp_path / "ck.sqlite"
    _mkdb(db)
    _add(db, "good-1", {"query": "好会话", "report": "# r"})
    # 手工塞一条无法反解的垃圾
    con = sqlite3.connect(db)
    con.execute(
        "INSERT INTO checkpoints VALUES ('bad-1', '', 'cp', NULL, 'msgpack', ?, NULL)",
        (b"\x81garbage",),
    )
    con.commit()
    con.close()
    _add(db, "good-2", {"query": "另一个好会话", "report": "# r"})

    rows = sess._recover_sessions_sync(db)
    ids = {r[0] for r in rows}
    assert ids == {"good-1", "good-2"}, "坏行应被跳过，好行不受影响"


# ----------------------------------------------------------------- 接入

def test_recover_from_checkpoints_fills_list(tmp_path: Path):
    db = tmp_path / "ck.sqlite"
    _mkdb(db)
    _add(db, "web-1", {"query": "调研新能源汽车出海", "report": "# r"})

    n = asyncio.run(_mgr(db).recover_from_checkpoints())
    assert n == 1
    listed = asyncio.run(sess.SessionManager.list_sessions())
    assert [s.thread_id for s in listed] == ["web-1"]
    assert listed[0].query == "调研新能源汽车出海"
    assert listed[0].status == "done"


def test_recover_keeps_existing_in_memory_session(tmp_path: Path):
    """本进程内已建的会话优先——恢复不许覆盖活着的内存态。"""
    db = tmp_path / "ck.sqlite"
    _mkdb(db)
    _add(db, "web-1", {"query": "旧问题", "report": "# r"})

    async def scenario():
        await sess.SessionManager.create_session("web-1", "新问题（内存里的）")
        return await _mgr(db).recover_from_checkpoints()

    n = asyncio.run(scenario())
    assert n == 0
    s = asyncio.run(sess.SessionManager.get("web-1"))
    assert s is not None and s.query == "新问题（内存里的）"


def test_recover_marks_missing_query_honestly(tmp_path: Path):
    """state 里没有 query 时，如实标注"问题未记录"，不伪造一个像样的标题。"""
    db = tmp_path / "ck.sqlite"
    _mkdb(db)
    _add(db, "web-none", {"report": "# r"})

    asyncio.run(_mgr(db).recover_from_checkpoints())
    s = asyncio.run(sess.SessionManager.get("web-none"))
    assert s is not None
    assert "未记录" in s.query


def test_recover_missing_db_is_noop(tmp_path: Path):
    """库不存在（全新环境）不能抛——服务照常启动。"""
    assert asyncio.run(_mgr(tmp_path / "nope.sqlite").recover_from_checkpoints()) == 0
