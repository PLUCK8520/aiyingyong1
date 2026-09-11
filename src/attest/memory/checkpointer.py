"""T5.1 · 检查点（checkpointer）装配。

**职责单一**：只负责"给图配一个可持久化的检查点后端"，不含业务逻辑。

⚠️ **同步 vs 异步（T5.1 硬约束，P6 必踩）**：
    - CLI / 脚本（同步调用栈）→ 用 `SqliteSaver`（本模块默认，`langgraph-checkpoint-sqlite`）；
    - **P6 的 FastAPI（async 事件循环）必须用 `AsyncSqliteSaver`（`aiosqlite`）**——
      同步 saver 在 async 事件循环里执行阻塞式 SQLite I/O，会阻塞整个事件循环，
      表现为"单请求卡死全部并发"，且**不会报错**（最难查的一类）。
    - 本模块预留 `make_async_checkpointer()` 占位，P6 直接接线即可。

为什么用 `contextmanager` 而不是直接返回 saver：
    `SqliteSaver.from_conn_string()` 返回的是**上下文管理器**，连接在其 `__exit__` 关闭。
    若直接 `with` 退出后再编译/调用图，连接已关闭 → 运行期才炸。故用 `contextlib` 包一层，
    把"连接生命周期"与"图使用期"绑定。
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..logging import get_logger

log = get_logger(__name__)


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
        saver = SqliteSaver(conn)
        log.info(f"[memory] 检查点已接入（同步 SqliteSaver）：{db_path}")
        yield saver
    finally:
        conn.close()


def make_async_checkpointer(db_path: Path | str) -> Any:
    """异步检查点占位（P6 接线）。

    P6 的 FastAPI 必须走这里——但 async 版是 `async with` 语义，装配方式与同步不同，
    等到 P6 真正需要时再实现，避免现在写一个**没跑过**的 async 分支（未验证即等于没有）。
    """
    raise NotImplementedError(
        "P6 接线：FastAPI(async) 需用 `AsyncSqliteSaver`（aiosqlite）。\n"
        "  原因：同步 SqliteSaver 在 async 事件循环里会阻塞（详见本模块文档串）。\n"
        "  实现要点：`AsyncSqliteSaver.from_conn_string(path)` 是 async 上下文管理器，"
        "需在 lifespan 里 `await __aenter__` 并挂到 app.state。"
    )
