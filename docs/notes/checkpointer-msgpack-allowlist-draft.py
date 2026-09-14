# ============================================================================
# 存档说明（2026-09-14 阿布补）
# 本文件是 2026-09-13 针对 LangGraph msgpack 反序列化前向风险写的 checkpointer.py
# 改进草案，此前散落在工作区根目录（_checkpointer_new.py），今移入本目录存档。
#
# 实测结论（langgraph 1.2.11 / langgraph-checkpoint 4.2.0，详见
# .workbuddy/memory/2026-09-14.md「整理收尾」）：
#   - 告警真实存在：Deserializing unregistered type attest.schemas.Plan ...
#     will be blocked in a future version
#   - 写法无误：allowed_msgpack_modules 接受类对象，归一成 (module, name) 元组
#   - 但默认 serde 的 _allowed_msgpack_modules=True（允许全部），显式白名单
#     范围更窄，不消除告警；价值仅在于为未来严格模式预登记，非修当前的坏
#   - 附带的 aiosqlite + asynccontextmanager 改动属新功能，落地前需单独评审
# 当前未落地（src/attest/memory/checkpointer.py 无 serde 参数），也不阻塞。
# ============================================================================

"""T5.1 · 检查点（checkpointer）装配。

**职责单一**：只负责"给图配一个可持久化的检查点后端"，不含业务逻辑。

⚠️ **同步 vs 异步（T5.1 硬约束，P6 已接线）**：
    - CLI / 脚本（同步调用栈）→ 用 `SqliteSaver`（`make_checkpointer`）；
    - **P6 的 FastAPI（async 事件循环）必须用 `AsyncSqliteSaver`（`aiosqlite`）**——
      同步 saver 在 async 事件循环里执行阻塞式 SQLite I/O，会阻塞整个事件循环，
      表现为"单请求卡死全部并发"，且**不会报错**（最难查的一类）。
    - `make_async_checkpointer()` 返回 **async 上下文管理器**（不是 saver 本身），
      由 `app/session.py::SessionManager` 在 FastAPI lifespan 里 `__aenter__` / `__aexit__`。

为什么用 `contextmanager` 而不是直接返回 saver：
    `SqliteSaver.from_conn_string()` 返回的是**上下文管理器**，连接在其 `__exit__` 关闭。
    若直接 `with` 退出后再编译/调用图，连接已关闭 → 运行期才炸。故用 `contextlib` 包一层，
    把"连接生命周期"与"图使用期"绑定。
"""

from __future__ import annotations

import aiosqlite
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..logging import get_logger

log = get_logger(__name__)


def _make_serde() -> Any:
    """显式注册 attest 自定义类型到 msgpack 白名单。

    背景（2026-09-13 真实运行实测）：LangGraph checkpoint 默认宽松模式，
    反序列化未知类型只告警不阻断。未来版本会默认阻断，届时旧 checkpoint
    的 state 反序列化静默退化为 dict——续跑读到脏数据但无任何报错。

    修复：saver 构造时传 serde=JsonPlusSerializer(allowed_msgpack_modules=[...])，
    把 attest 包里所有会进 checkpoint 的类型显式注册进白名单。
    传类对象最稳妥——_normalize_module_keys 精确转成
    (cls.__module__, cls.__name__) 二元组，与 _check_allowed 的 key 格式完全一致。

    注意：不要设 LANGGRAPH_STRICT_MSGPACK=true——方向相反，
    它会立即阻断所有未注册类型，等于提前引爆前向兼容风险。
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from .. import schemas
    from ..retrieval import ports

    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            schemas.RouteDecision,
            schemas.Plan,
            schemas.Judgment,
            schemas.JudgeResult,
            schemas.Conflict,
            schemas.ConflictResult,
            schemas.AuditItem,
            schemas.AuditResult,
            ports.SearchResult,
            ports.Evidence,
            ports.Hit,
        ],
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def make_checkpointer(db_path: Path | str) -> Iterator[Any]:
    """同步检查点：SQLite 文件持久化（CLI / 脚本用）。

    用法：
        with make_checkpointer(path) as ckpt:
            app = build_graph(ctx, checkpointer=ckpt)
            app.invoke(state, config={"configurable": {"thread_id": tid}})

    ⚠️ `SqliteSaver` 默认 `check_same_thread=False`，且内部用一把连接锁——
    **单进程多线程 OK，多进程并发写同一 SQLite 文件会锁冲突**（与 Chroma 同理）。
    CLI 单进程无碍；P6 若起多 worker，需换 AsyncSqliteSaver 或独立 DB。
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = Path(db_path)
    _ensure_parent(db_path)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        saver = SqliteSaver(conn, serde=_make_serde())
        log.info(f"[memory] 检查点已接入（同步 SqliteSaver）：{db_path}")
        yield saver
    finally:
        conn.close()


def make_async_checkpointer(db_path: Path | str) -> Any:
    """异步检查点：`AsyncSqliteSaver`（P6 接线，2026-09-11 实跑验证）。

    **返回的不是 saver，而是 async 上下文管理器**——这是 LangGraph 的设计，
    与同步版一致（连接生命周期必须由调用方绑定）。用法：

        cm = make_async_checkpointer(path)
        saver = await cm.__aenter__()        # FastAPI lifespan 启动
        app = build_graph(ctx, checkpointer=saver)
        ...
        await cm.__aexit__(None, None, None) # lifespan 关闭

    或直接用 `async with`：

        async with make_async_checkpointer(path) as saver:
            ...

    为什么必须用它（而不是同步 `SqliteSaver`）：
        同步 saver 在 async 事件循环里执行**阻塞式** SQLite I/O，会卡住整个事件循环，
        表现为"一个请求把全部并发拖死"，且**不会抛异常**（最难查的一类故障）。

    验证层级：`scripts/p6_probe.py` 4 项全通过（可导入 / 上下文可用 / async 静态断点挂起 /
    async 续跑出报告）；`tests/test_p6_session.py` 覆盖同一路径。**离线 mock 模式**。
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    @asynccontextmanager
    async def _saver_cm(path: Path):
        async with aiosqlite.connect(str(path)) as conn:
            yield AsyncSqliteSaver(conn, serde=_make_serde())

    db_path = Path(db_path)
    _ensure_parent(db_path)
    log.info(f"[memory] 异步检查点已接入（AsyncSqliteSaver）：{db_path}")
    return _saver_cm(db_path)
