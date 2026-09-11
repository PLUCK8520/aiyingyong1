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
from attest.graph.build import build_context, build_graph, initial_state
from attest.logging import get_logger
from attest.memory.checkpointer import make_async_checkpointer
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

        ⚠️ **数据源是 `astream(stream_mode=["updates","values"])`，不是 trace.jsonl**。
        架构设计 §7 的硬决策：前端不读 trace 文件（展示契约与审计格式解耦）。
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
                input_value, cfg, stream_mode=["updates", "values"]
            ):
                mode, payload = chunk  # type: ignore[misc]
                if mode == "updates":
                    for node, increment in (payload or {}).items():
                        if str(node).startswith("__"):
                            continue
                        # 先把本节点耗时从 trace 同步过来（`updates` 到达时节点已跑完，
                        # trace 的 node_end 已写入），再发事件。见 `_emit_node_events`。
                        self._sync_durations(session, trace)
                        self._emit_node_events(session, str(node), increment)
                else:
                    final = payload  # type: ignore[assignment]

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
                "cost_incurred": round(float(final.get("cost_incurred") or 0.0), 6),
                "tokens_incurred": int(final.get("tokens_incurred") or 0),
                "references": self._reference_index(final),
            }
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

    def _emit_node_events(self, session: Session, node: str, increment: Any) -> None:
        """把 `updates` chunk 翻译成前端协议事件（T6.2）。

        ⚠️ **耗时从哪来**：`updates` 模式只在节点**完成后**产出一个 chunk，拿到它时节点已经跑完，
        所以"在 chunk 到达时先后打两个时间戳"必然得到 0.0ms——那是假数据，不如不给。
        真实耗时的唯一可靠来源是 trace 的 `node_end.duration_ms`（`agents/base.py::bind` 打点）。

        ⚠️ **两条流水线的关系**（别搞混）：
            - trace.jsonl → 审计/评测（含真实耗时、token、成本）；
            - SSE → 前端展示契约（架构 §7）。
        本函数从 `session.trace_durations`（由 trace 在无耦合前提下同步过来的一份镜像）取耗时，
        **不是**去解析 trace 文件格式——前端仍只认 SSE 协议。
        """
        now = time.time()
        session.current_node = node
        node_index = sum(1 for e in session.events if e.get("event") == "agent_start") + 1
        outputs = (
            sorted(k for k in (increment or {}) if not str(k).startswith("_"))
            if isinstance(increment, dict) else []
        )
        session.push({
            "event": "agent_start",
            "node": node,
            "index": node_index,
            "ts": now,
            "outputs": outputs,
        })
        session.push({
            "event": "agent_end",
            "node": node,
            "index": node_index,
            "ts": now,
            # 真实耗时（毫秒）；trace 里没记到则为 None，前端显示"—"而不是编一个 0
            "duration_ms": session.trace_durations.get(node),
            "brief": _brief(node, increment),
        })

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
