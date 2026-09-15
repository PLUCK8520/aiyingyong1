"""T6.1 · P6 会话层：async 图生命周期 + 每会话运行状态。

**这一层解决什么**：① FastAPI 的 async 事件循环里必须用 `AsyncSqliteSaver`
（同步 saver 会阻塞整个事件循环，**且不报错**——见 `memory/checkpointer.py` 模块文档串）；
② 一次调研是长任务，进程内要能拿到"某个 thread 跑到哪了 / 事件流在哪 / 成本多少"。

**分层归属**：① 接入层（架构设计 §2）。只做"协议适配 + 生命周期"，**不含任何业务逻辑**，
不调模型、不改 state 语义——业务全在 `attest.graph` 里。

⚠️ **明确的设计限制（不掩饰）**：
    会话元数据（`_sessions`）放在**进程内存 dict** 里，不是数据库。
    - 后果一：**多 worker 下不共享**（uvicorn `--workers 2` 会各看各的内存）→ 故启动脚本
      固定单 worker；真要扩容得把元数据落 SQLite（见下方 `TODO(scale)`）。
    - 后果二：进程重启后内存态丢失。**但调研状态本身不丢**——它在 `checkpoints.sqlite` 里，
      重启后用同一 `thread_id` 走 `GET /api/session/{id}` 就能看到"待确认/可续跑"。
      这是"检查点持久 / 元数据易失"的**有意分层**：审计与续跑靠检查点，实时展示靠内存。
    - ⚠️ **T9.5 补的第三条（上面那条分层自己挖的坑）**：既然列表也读内存，重启后列表就是空的——
      而**列表是唯一入口**，拿不到 `thread_id` 就走不了上面那条恢复路径。
      分层的初衷没错，但它漏了"入口本身也得能恢复"这一步，实际效果是
      **数据一直健在、界面上却是"还没有会话"**（在用户眼里与数据丢失无异）。
      故 `start()` 现在会调 `recover_from_checkpoints()` 扫库把历史会话恢复进列表；
      恢复出来的 `status` 是**推断值**（见 `_infer_status`），只在能确证时才宣称终态。
    - `TODO(scale)`：多 worker 时把 `list_sessions()` / 事件缓冲换成 SQLite 表或 Redis。

⚠️ **事件缓冲 vs 订阅队列（关键区别，别混）**：
    - `events`（list）：**已发生事件的环形缓冲**，给"迟到订阅者"重放用（SSE 断线重连靠它）。
    - `subscribers`（asyncio.Queue）：**实时投递通道**，给"此刻正在听的连接"用。
    两者都要写：只写队列 → 重连的客户端丢历史；只写缓冲 → 实时性没了（要轮询）。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from attest.agents.human_confirm import confirm_payload
from attest.config import Settings
from attest.graph.build import build_context, build_graph, graph_node_sequence, initial_state
from attest.logging import get_logger
from attest.memory.checkpointer import make_async_checkpointer
from attest.reporting import export_report
from attest.trace.events import TraceWriter

log = get_logger(__name__)

#: 会话状态机的取值（前端侧栏状态点用它）
SessionStatus = Literal["idle", "running", "awaiting_confirm", "done", "error", "aborted"]

#: 事件环形缓冲上限。长任务事件数有限（节点级 + 阶段级，非 token 级），
#: 3000 足够覆盖一次完整调研；超出则丢最旧的（只影响"重连重放"，不影响落盘的 trace）。
EVENT_BUFFER_MAX = 3000

#: 进程内会话登记表。key = thread_id。
_sessions: dict[str, "Session"] = {}
_sessions_lock = asyncio.Lock()


@dataclass
class Session:
    """单个会话线程的运行期元数据（**非**持久层；见模块文档串的设计限制）。"""

    thread_id: str
    query: str
    created_at: float = field(default_factory=time.time)
    status: SessionStatus = "idle"
    #: 当前正在执行的节点名（SSE agent_start/agent_end 用）
    current_node: str | None = None
    #: 已发生事件（重连重放的数据源）
    events: list[dict[str, Any]] = field(default_factory=list)
    #: 实时订阅者队列（此刻在听的 SSE 连接）
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    #: 待人工确认的大纲载荷（静态断点挂起时有值；确认后清空）
    pending_confirm: dict[str, Any] | None = None
    #: 节点真实耗时镜像（node → duration_ms），由本 run 的 trace 同步进来。
    #: ⚠️ 只做"读 trace.summary() 的公开字段"，不解析文件格式——前端契约仍是 SSE。
    trace_durations: dict[str, float] = field(default_factory=dict)
    #: 未闭合的任务行：`task_id → (时间线行号, 节点名)`。
    #: debug 流的 `task` / `task_result` 用它配对——**不能按节点名配对**，
    #: 因为 Send 扇出会让同名节点并行且乱序结束（见 `_handle_debug`）。
    task_rows: dict[str, tuple[int, str]] = field(default_factory=dict)
    #: 最终产物（报告 / 直答 / 引用校验 / 审计摘要）
    result: dict[str, Any] = field(default_factory=dict)
    #: 错误信息（status=error 时有值）
    error: str | None = None
    #: 后台运行任务的句柄（用于判断"是否已在跑"，防重复提交）
    task: asyncio.Task | None = field(default=None, repr=False)

    # ---------------- 事件 ----------------
    def push(self, event: dict[str, Any]) -> None:
        """写缓冲 + fanout 给所有活跃订阅者。**同步方法**——调用点已在事件循环内。"""
        self.events.append(event)
        if len(self.events) > EVENT_BUFFER_MAX:
            del self.events[: len(self.events) - EVENT_BUFFER_MAX]
        for q in list(self.subscribers):
            # put_nowait：满队列不能阻塞发布方（丢展示事件 << 卡住调研任务）
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.warning(f"[session] 订阅者队列已满，丢弃事件 {event.get('event')}")

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self.subscribers:
            self.subscribers.remove(q)

    def snapshot(self) -> dict[str, Any]:
        """给 `GET /api/session/{id}` 的可序列化视图（不含队列/任务句柄等运行期对象）。"""
        return {
            "thread_id": self.thread_id,
            "query": self.query,
            "status": self.status,
            "current_node": self.current_node,
            "pending_confirm": self.pending_confirm,
            "event_count": len(self.events),
            "result_keys": sorted(self.result.keys()),
            "error": self.error,
            "created_at": self.created_at,
        }


# ============================================================ 冷启动恢复（T9.5）

def _infer_status(cv: dict[str, Any]) -> SessionStatus:
    """从检查点里的 state 反推会话状态。

    ⚠️ **这是推断值，不是原始值**——原状态随内存丢失了。所以规则刻意保守：
    只在能**确证**时才宣称终态，其余一律落到 `aborted`（"跑过但没跑到终态"），
    绝不把"跑到一半"美化成 `done`。

    顺序即优先级：
      1. 有 `report` / `direct_answer` → `done`（终态产物在手，确证）
      2. 有 `errors` → `error`
      3. 有 `plan` 但 `plan_approved` 为假 → `awaiting_confirm`（T5.4 静态断点挂起）
      4. 其余 → `aborted`
    """
    if str(cv.get("report") or "").strip() or str(cv.get("direct_answer") or "").strip():
        return "done"
    if cv.get("errors"):
        return "error"
    if cv.get("plan") and not cv.get("plan_approved"):
        return "awaiting_confirm"
    return "aborted"


def _recover_sessions_sync(db_path: Path) -> list[tuple[str, str, float, SessionStatus]]:
    """同步扫检查点库，返回 [(thread_id, query, created_at, status)]。

    **为什么必须有这个函数**（T9.5 修的确实是个真缺口）：
      `_sessions` 是进程内存 dict，进程一重启列表就空。而**列表是唯一的入口**——
      拿不到 thread_id 就走不了 `GET /api/session/{id}` 的恢复路径（那条路径 P6 就做好了，
      本来就是为重启准备的）。结果：**数据一直在检查点里，用户在界面上却是"还没有会话"。**
      本模块文档串承诺的"调研状态不丢"因此落空——不是数据丢了，是**不可达**。

    **实现要点**：
      - 只读 URI 打开，不与运行中的 `AsyncSqliteSaver` 争用；
      - 每个 thread 只反解**最后一个**检查点（`MAX(rowid)`，LangGraph 顺序写入）；
      - serde 用**默认宽容模式**（不设 `allowed_msgpack_modules`）——恢复历史数据时
        宁可多认一个类型，也不要因白名单漏项而整条会话消失；
      - 单条失败只跳过该条并留 warning，**不让一个坏检查点毁掉整个列表**。
    """
    import sqlite3

    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    serde = JsonPlusSerializer()
    uri = "file:%s?mode=ro" % str(db_path).replace("\\", "/")
    out: list[tuple[str, str, float, SessionStatus]] = []
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            "SELECT c.thread_id, c.type, c.checkpoint FROM checkpoints c "
            "WHERE c.rowid = (SELECT MAX(rowid) FROM checkpoints "
            "                 WHERE thread_id = c.thread_id)"
        ).fetchall()
    finally:
        con.close()

    for tid, typ, blob in rows:
        try:
            ck = serde.loads_typed((typ, blob))
        except Exception as e:  # noqa: BLE001 —— 单条坏数据不该拖垮整张列表
            log.warning(f"[session] 检查点反解失败，跳过 thread={tid}: {type(e).__name__}: {e}")
            continue
        # ⚠️ 「能反解」不等于「是检查点」（T9.5 写测试时实测到的边界）：
        # 畸形 blob 可能反解成别的东西而不抛异常——例如 `b"\x81garbage"` 会被 msgpack
        # 认成一个 map。不认结构就跳过，否则侧栏里会冒出没有内容的僵尸会话。
        if not isinstance(ck, dict) or "channel_values" not in ck:
            log.warning(f"[session] 检查点结构异常，跳过 thread={tid}（type={typ}）")
            continue
        cv = ck.get("channel_values") or {}
        query = str(cv.get("query") or "").strip()
        created = _ckpt_ts(ck)
        out.append((tid, query, created, _infer_status(cv)))
    return out


def _ckpt_ts(ck: dict[str, Any]) -> float:
    """取检查点时间戳（LangGraph 存的是 ISO8601 字符串）。缺了就退回当前时间。"""
    from datetime import datetime

    ts = ck.get("ts")
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str) and ts:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return time.time()


class SessionManager:
    """图的生命周期持有者（FastAPI lifespan 创建一次，挂到 `app.state`）。

    **为什么把 saver 与图挂在管理器上而不是每次请求现建**：
      `AsyncSqliteSaver.from_conn_string()` 是 async 上下文管理器，进出成本高；
      且 LangGraph 编译期要解析图结构。一次装配、全生命周期复用是正解。
      代价：单进程单 Saver → 与"单 worker"约束一致（见模块文档串）。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._saver_cm: Any = None
        self._saver: Any = None
        self._app: Any = None
        self._started = False

    # ---------------- 生命周期 ----------------
    async def start(self) -> None:
        """装配 async 检查点 + 编译图。在 FastAPI lifespan 启动时调用一次。"""
        self.settings.ensure_dirs()
        db_path = Path(self.settings.checkpoint_db)
        # trace 文件仍写盘（审计源），但 run_id 由每会话的 TraceWriter 覆盖写入同一文件。
        self._saver_cm = make_async_checkpointer(db_path)
        self._saver = await self._saver_cm.__aenter__()
        # 注意：这里用一个"模板" ctx 只为编译图结构；每次运行会用**独立的 trace** 另建 ctx。
        # 若共用同一个 ctx，则 trace 会把所有会话混在一个文件里、无法按 run 分离。
        self._app = None  # 图延迟到首次运行前编译（见 _ensure_graph）
        self._started = True
        log.info(f"[session] SessionManager 已启动 | checkpointer=AsyncSqliteSaver | db={db_path}")
        # T9.5：把检查点里的历史会话恢复进列表。
        # 不做这一步，重启后侧栏永远是"还没有会话"——而列表是唯一入口，
        # 用户就再也够不到那些其实健在的调研记录（数据没丢，只是不可达）。
        n_recovered = await self.recover_from_checkpoints()
        log.info(f"[session] 冷启动恢复：从检查点载入 {n_recovered} 个历史会话")

    async def stop(self) -> None:
        """关闭检查点连接。进程退出时调用。"""
        for s in _sessions.values():
            if s.task and not s.task.done():
                s.task.cancel()
        if self._saver_cm is not None:
            await self._saver_cm.__aexit__(None, None, None)
            self._saver_cm = None
            self._saver = None
        self._started = False
        log.info("[session] SessionManager 已停止")

    @property
    def started(self) -> bool:
        return self._started

    def _build_app(self, trace: TraceWriter, run_id: str) -> Any:
        """编译一张**绑定了本 run trace** 的图。

        ⚠️ 为什么每次运行都重新 `build_graph`：`build_context(trace=...)` 是**节点闭包捕获**的，
        换 trace 必须换 ctx、换图。编译开销在毫秒级，相比一次调研可忽略；
        换来的是"每个会话的 trace 严格隔离"，这比省下编译更值。
        """
        ctx = build_context(self.settings, trace=trace, run_id=run_id)
        return build_graph(ctx, checkpointer=self._saver)

    # ---------------- 会话 CRUD ----------------
    @staticmethod
    async def create_session(thread_id: str, query: str) -> Session:
        async with _sessions_lock:
            if thread_id in _sessions:
                # 复用已存在的会话（幂等），但把 query 更新为本次的——用户可能改了问题
                s = _sessions[thread_id]
                s.query = query
                return s
            s = Session(thread_id=thread_id, query=query)
            _sessions[thread_id] = s
            return s

    @staticmethod
    async def get(thread_id: str) -> Session | None:
        return _sessions.get(thread_id)

    @staticmethod
    async def list_sessions() -> list[Session]:
        return sorted(_sessions.values(), key=lambda s: s.created_at, reverse=True)

    async def recover_from_checkpoints(self) -> int:
        """从检查点库重建会话列表（冷启动恢复）。返回恢复的条数。

        ⚠️ **为什么这件事值得单独做**：`list_sessions()` 读的是内存 dict，
        而它是**唯一入口**。列表空的 → 前端拿不到任何 thread_id →
        `GET /api/session/{id}` 的恢复路径（P6 已做好的那条）永远走不到。
        "数据没丢、只是不可达"在用户眼里和"数据丢了"没有区别。

        同步 IO + msgpack 反解放 `asyncio.to_thread`：不在事件循环里做阻塞活
        （启动期数据量小，但这属于既定红线）。恢复失败**不拦服务启动**——
        宁可少一个历史列表，也不要让整个后端起不来。
        """
        db_path = Path(self.settings.checkpoint_db)
        if not db_path.exists():
            return 0
        try:
            rows = await asyncio.to_thread(_recover_sessions_sync, db_path)
        except Exception as e:  # noqa: BLE001
            log.warning(f"[session] 冷启动恢复失败（服务照常启动）: {type(e).__name__}: {e}")
            return 0

        added = 0
        async with _sessions_lock:
            for tid, query, created, status in rows:
                if tid in _sessions:
                    continue  # 本进程内已建的会话优先，不覆盖内存态
                _sessions[tid] = Session(
                    thread_id=tid,
                    # 空问题**如实标注**，不要伪造一个看起来正常的标题
                    query=query or "（历史会话 · 问题未记录）",
                    created_at=created,
                    status=status,
                )
                added += 1
        return added

    # ---------------- 运行 ----------------
    async def run(self, session: Session, *, decision: dict[str, Any] | None = None) -> None:
        """跑一次调研（首跑 or 断点续跑）。

        Args:
            session: 会话。
            decision: 人工确认决策（续跑时传）；`None` = 首跑。

        **SSE 事件映射（T6.2 协议）**：
            - `phase`        → 阶段切换（route 分流结果：direct / research）
            - `agent_start`  → 节点开始（带 node + 序号）
            - `agent_end`    → 节点结束（带 node + 耗时 + 输出字段）
            - `report_ready` → 终态（带报告 / 直答 / 审计摘要 / 成本）
            - `awaiting_confirm` → 静态断点挂起，前端应弹大纲确认
            - `error`        → 异常

        ⚠️ **数据源是 `astream(stream_mode=["updates","values","debug"])`，不是 trace.jsonl**。
        架构设计 §7 的硬决策：前端不读 trace 文件（展示契约与审计格式解耦）。
        其中 `debug` 流用于**实时**的 `agent_start`（节点开始执行时即发出）、
        `updates` 流只用于同步 trace 的真实耗时、`values` 流提供终态。
        """
        run_id = f"{session.thread_id}-{int(time.time())}"
        trace = TraceWriter(path=self.settings.trace_dir / "trace.jsonl", run_id=run_id)
        # 每会话独立图：保证 trace 不串（见 `_build_app` 文档串）
        app = self._build_app(trace, run_id)
        cfg = {"configurable": {"thread_id": session.thread_id}}
        session.status = "running"
        session.error = None

        try:
            trace.emit("run_start", query=session.query, thread_id=session.thread_id,
                       llm_mode=self.settings.llm_mode, search_mode=self.settings.search_mode)
            session.push({"event": "run_start", "thread_id": session.thread_id,
                          "query": session.query, "llm_mode": self.settings.llm_mode})

            # ---------- 首跑：先落初始 state；续跑：投递决策后以 None 推进 ----------
            if decision is None:
                input_value: Any = initial_state(session.query, thread_id=session.thread_id)
            else:
                # 静态断点续跑两步法（T5.4 实测语义，`Command(resume=...)` 在此无处投递）：
                # 1. update_state 把决策落进 state；2. 以 None 推进过断点。
                await app.aupdate_state(cfg, {"plan_approval": decision}, as_node="planner")
                session.pending_confirm = None
                input_value = None

            final: dict[str, Any] = {}
            async for chunk in app.astream(
                input_value, cfg, stream_mode=["updates", "values", "debug"]
            ):
                mode, payload = chunk  # type: ignore[misc]
                if mode == "debug":
                    # debug 流在节点**开始执行时**就发 `task`（结束发 `task_result`）——
                    # 这是"实时进度"的唯一可靠来源：`updates` 只在节点跑完后才到，
                    # 长节点（实测 evidence_judge 单次 188s）期间界面会长时间零更新，
                    # 用户以为卡死。详见 `_handle_debug`。
                    self._handle_debug(session, payload, trace)
                elif mode == "updates":
                    # 只用于同步真实耗时（此刻该节点的 trace.node_end 已写入）。
                    self._sync_durations(session, trace)
                else:
                    final = payload  # type: ignore[assignment]
            # 防御：异常/中断可能留下未闭合的任务行，补终态（耗时 None → 前端显示"—"）
            self._close_open_rows(session)

            # ---------- 静态断点判定：停在 human_confirm 前 ----------
            snap = await app.aget_state(cfg)
            if snap.next:
                if "human_confirm" in snap.next:
                    plan = (snap.values or {}).get("plan") or {}
                    payload = confirm_payload(plan)
                    session.pending_confirm = payload
                    session.status = "awaiting_confirm"
                    session.current_node = None
                    session.push({"event": "awaiting_confirm", **payload})
                    trace.emit("run_paused", node="human_confirm", thread_id=session.thread_id)
                    log.info(f"[session] thread={session.thread_id} 挂在人工确认断点")
                    return
                # 非预期断点：如实报错，不猜
                session.status = "error"
                session.error = f"停在非预期断点：{list(snap.next)}"
                session.push({"event": "error", "message": session.error})
                return

            # ---------- 终态 ----------
            session.result = {
                "query": session.query,
                "route": final.get("route"),
                "direct_answer": final.get("direct_answer") or "",
                "report": final.get("report") or "",
                "reference_list": final.get("reference_list") or "",
                "citation_check": final.get("citation_check") or {},
                "audit_summary": final.get("audit_summary") or {},
                "audit_items": final.get("audit_items") or [],
                "profile": final.get("profile") or {},
                "plan": final.get("plan") or {},
                # T7.10：证据充足性评估。`sufficient=False` 表示本轮**拒编**，
                # report 是一页如实说明而非调研结论——前端据此显示提示条。
                "evidence_sufficiency": final.get("evidence_sufficiency") or {},
                "cost_incurred": round(float(final.get("cost_incurred") or 0.0), 6),
                "tokens_incurred": int(final.get("tokens_incurred") or 0),
                "references": self._reference_index(final),
            }
            # T7.7：跑完即落盘 md（交付动作）。失败只记日志，不影响会话状态。
            md_path = export_report(
                session.result, self.settings.report_dir, thread_id=session.thread_id
            )
            session.result["report_path"] = str(md_path) if md_path else ""
            session.status = "done"
            session.current_node = None
            summary = trace.summary()
            session.push({
                "event": "report_ready",
                "route": session.result["route"],
                "cost_cny": session.result["cost_incurred"],
                "tokens": session.result["tokens_incurred"],
                "audit_summary": session.result["audit_summary"],
                "citation_check": session.result["citation_check"],
                "evidence_sufficiency": session.result["evidence_sufficiency"],
                "llm_calls": summary.get("llm_calls", 0),
                "node_pairs_ok": summary.get("node_pairs_ok", False),
                "mock": self.settings.llm_mode == "mock",
            })
            trace.emit("run_end", cost_cny=session.result["cost_incurred"],
                       tokens=session.result["tokens_incurred"])
        except asyncio.CancelledError:
            session.status = "aborted"
            session.push({"event": "error", "message": "运行被取消"})
            trace.emit("run_aborted")
            raise
        except Exception as exc:  # noqa: BLE001 - 会话边界必须兜住，否则前端看不到任何线索
            session.status = "error"
            session.error = f"{type(exc).__name__}: {exc}"
            session.push({"event": "error", "message": session.error})
            trace.emit("run_error", error=session.error)
            log.error(f"[session] thread={session.thread_id} 运行失败：{session.error}")
        finally:
            trace.close()

    # ---------------- 内部 ----------------
    @staticmethod
    def _sync_durations(session: Session, trace: TraceWriter) -> None:
        """把 trace 里已记录的 `node_end.duration_ms` 同步到会话（`node_start` 无耗时，跳过）。

        用 `trace.of("node_end")` 这个**公开读接口**，而不是解析 jsonl 文件——
        这样 trace 内部结构变了也不会静默影响前端（I-01 的同类约束：不越过接口拿内部状态）。
        同名节点取**最近一次**（补检轮次里 scout 会跑多轮，最后一次更有代表性）。
        """
        for e in trace.of("node_end"):
            node = e.get("node")
            if node and e.get("duration_ms") is not None:
                session.trace_durations[str(node)] = float(e["duration_ms"])

    #: 图中真实存在的节点名。debug 流里还会出现 LangGraph 的内部任务，
    #: 只对业务节点发进度事件，避免时间线里混入无意义的行。
    _KNOWN_NODES: frozenset[str] = frozenset(graph_node_sequence())

    def _handle_debug(self, session: Session, payload: Any, trace: TraceWriter) -> None:
        """把 LangGraph 的 debug 事件翻译成 `agent_start` / `agent_end`（T6.2 协议）。

        **为什么要改用 debug 流**（2026-09-13）：
            `updates` 模式只在节点**跑完之后**产出一个 chunk，所以在它上面打点，
            只能得到"节点结束的那一刻同时报开始和结束"——时间线里每行都是瞬间闭合的，
            长节点期间界面毫无变化。实测 `evidence_judge` 单次 188s / 148s，
            用户等待的 8 分钟里约 6 分钟屏幕是静止的，看起来就是卡死。
            debug 流的 `task` 事件在节点**开始执行时**发出，`task_result` 在结束时发出，
            两者带**同一个任务 id**，这正是"实时进度 + 可靠配对"需要的东西。

        **为什么按 `task_id` 配对、而不是按节点名**：
            Send 扇出会让**同名节点并行**（`scout_web` ×N），且实测**结束顺序与开始顺序不同**
            （3 路 scout 同时开始、乱序结束）。按名配对会把它们的起止接错，算出荒谬的耗时。
            `task_id` 是 debug 流里唯一的实例标识，用它配对才正确。

        耗时仍取 **trace 的真实 `node_end.duration_ms`**（`bind` 打点），不从事件间隔估算——
        两个时间戳都是"事件到达时刻"，差值不是节点真实耗时。
        """
        if not isinstance(payload, dict):
            return
        kind = payload.get("type")
        p = payload.get("payload") or {}
        name = str(p.get("name") or "")
        task_id = str(p.get("id") or "")
        if not name or not task_id or name not in self._KNOWN_NODES:
            return

        now = time.time()
        if kind == "task":
            index = sum(1 for e in session.events if e.get("event") == "agent_start") + 1
            session.task_rows[task_id] = (index, name)
            session.current_node = name
            session.push({
                "event": "agent_start",
                "node": name,
                "task_id": task_id,
                "index": index,
                "ts": now,
                "outputs": [],
            })
        elif kind == "task_result":
            row = session.task_rows.pop(task_id, None)
            if row is None:
                return
            index, _node = row
            result = p.get("result")
            # 该节点的 trace.node_end 此刻已写入，同步过来再发结束事件（耗时才是真的）
            self._sync_durations(session, trace)
            session.push({
                "event": "agent_end",
                "node": name,
                "task_id": task_id,
                "index": index,
                "ts": now,
                "duration_ms": session.trace_durations.get(name),
                # 产出字段名（来自节点真实返回值）——起止事件分离后，outputs 只能在结束时拿到
                "outputs": sorted(k for k in (result or {}) if not str(k).startswith("_"))
                if isinstance(result, dict) else [],
                "brief": _brief(name, result),
            })

    @staticmethod
    def _close_open_rows(session: Session) -> None:
        """把未闭合的任务行补上终态。

        正常路径下每个 `task` 都有配对的 `task_result`（已实测：15 个任务 15 对），
        这里是**防御**：异常/中断时宁可给一行 `duration_ms=None`（前端显示"—"），
        也不要让进度列表里永远挂着一个转圈的"进行中"。
        """
        for task_id, (index, name) in list(session.task_rows.items()):
            session.push({
                "event": "agent_end",
                "node": name,
                "task_id": task_id,
                "index": index,
                "ts": time.time(),
                "duration_ms": None,
                "brief": "（未正常结束）",
            })
        session.task_rows.clear()

    @staticmethod
    def _reference_index(final: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """建立 `引用编号 → 证据` 的映射，供报告双栏的悬浮卡用（T6.5 的数据基础）。

        编号体系见 `retrieval/citations.py`：WEB/LOC + 轮次 + 序号。
        这里**只做索引，不重编号**——I-06：引用编号分配后不可回收、不可重排。
        """
        idx: dict[str, dict[str, Any]] = {}
        for ev in final.get("evidence") or []:
            cid = getattr(ev, "citation_id", None) or getattr(ev, "id", None)
            if not cid:
                continue
            idx[str(cid)] = {
                "title": getattr(ev, "title", "") or "",
                "url": getattr(ev, "url", None),
                "source": getattr(ev, "source", None),
                "snippet": (getattr(ev, "content", "") or "")[:500],
            }
        return idx


#: 节点完成时给人看的一句话（与 CLI `_brief` 同源语义，但这里要 JSON 安全）
def _brief(node: str, increment: Any) -> str:
    if not isinstance(increment, dict):
        return ""
    if node == "memory_loader":
        return f"注入画像 {len(increment.get('profile') or {})} 条"
    if node == "intent_router":
        return f"route={increment.get('route')}"
    if node == "planner":
        plan = increment.get("plan") or {}
        return f"子问题 {len(plan.get('sub_questions', []))} 个 / 大纲 {len(plan.get('outlines', []))} 章"
    if node == "human_confirm":
        ap = increment.get("plan_approval") or {}
        return f"决策={ap.get('action', '-')}"
    if node in ("scout_web", "scout_local"):
        return f"命中 {len(increment.get('evidence', []))} 条证据"
    if node == "evidence_judge":
        return f"判别 {len(increment.get('judgments', []))} 条 / 缺口 {len(increment.get('gaps', []))} 项"
    if node == "reflect":
        return f"补检 {len(increment.get('reflect_targets') or [])} 路"
    if node == "analyst":
        suff = increment.get("evidence_sufficiency") or {}
        if suff and suff.get("sufficient") is False:
            return f"证据不足，拒编（可用 0 / 原始 {suff.get('n_evidence', 0)} 条）"
        check = increment.get("citation_check") or {}
        return f"引用 {check.get('referenced', 0)} 处"
    if node == "auditor":
        s = increment.get("audit_summary") or {}
        return f"判定 {s.get('total', 0)} 句 | 未证实 {s.get('unsupported', 0)}"
    if node == "direct_responder":
        return f"{len(increment.get('direct_answer', ''))} 字"
    return ""


async def stream_events(session: Session, *, last_index: int = 0) -> AsyncIterator[dict[str, Any]]:
    """产出某会话的事件流（SSE 生成器的数据源）。

    Args:
        session: 会话。
        last_index: 客户端已收到的最后事件序号（**断线重连靠它**）。
            先重放 `events[last_index:]`，再挂到实时队列继续推。

    ⚠️ **为什么重连能补齐历史**：事件先写 `session.events`（环形缓冲）再 fanout；
    重连时按 `last_index` 从缓冲重放，天然不漏。这正是架构 §5 要求的
    "前端故障不能传导到后端任务 / 事件可从状态重放"。

    ⚠️ **终态后必须收尾**：会话进入 `done/error/aborted` 且事件重放完毕 → 结束生成器，
    否则 SSE 连接会挂住（浏览器一直转圈）。
    """
    # 1) 先补历史
    for ev in list(session.events)[last_index:]:
        yield ev

    # 2) 已终态？直接收尾，不留悬挂连接
    if session.status in ("done", "error", "aborted"):
        yield {"event": "stream_end", "status": session.status}
        return

    # 3) 挂实时队列
    q = session.subscribe()
    try:
        while True:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=15.0)
            except asyncio.TimeoutError:
                # 心跳：SSE 连接在长任务中必须保活（中间代理会掐断静默连接）
                yield {"event": "ping", "ts": time.time()}
                if session.status in ("done", "error", "aborted"):
                    break
                continue
            yield ev
            if ev.get("event") in ("report_ready", "error", "awaiting_confirm"):
                # awaiting_confirm 后前端要弹模态，连接可保留（用户确认后前端再发 /api/confirm）；
                # 为简单起见这里也收尾，前端重连时靠 last_index 补齐——重连是廉价且必需的能力。
                yield {"event": "stream_end", "status": session.status}
                return
    finally:
        session.unsubscribe(q)
