"""T6.1 / T6.2 · P6 接入层：FastAPI + SSE。

**分层归属**：① 接入层（架构设计 §2）。只做"协议适配 + 请求编排 + 事件推送"，
**不含任何业务逻辑、不调任何模型**——全部委托给 `app/session.py` → `attest.graph`。

端点清单（任务清单 T6.1 的 4 个 + 静态断点必需的 1 个）：

    POST /api/session              新建/复用会话
    GET  /api/session              列出会话（侧栏）
    GET  /api/session/{thread_id}  查单个会话状态（含"待确认/可续跑"）
    POST /api/chat                 SSE 流：跑一次调研（或续跑）
    POST /api/confirm              投递人工确认决策（静态断点必需，清单未列）
    GET  /api/report/{thread_id}   取报告正文 + 引用索引（T6.5 双栏数据源）
    GET  /api/trace/{thread_id}    取本次运行的事件摘要（审计视图）

⚠️ **为什么必须有 `/api/confirm`**（任务清单 T6.1 只写了 4 个端点）：
    P5 确定的 T5.4 语义是**编译期静态断点**，续跑必须
    `update_state(plan_approval)` + `invoke(None)`（`Command(resume=...)` 在静态断点下无处投递）。
    这就需要一个**独立入口**来投递决策——SSE 是单向推送通道，投不进去。

⚠️ **`/api/chat` 为什么不 `await` 到跑完**：
    一次调研是长任务（真实模型下分钟级）。若在请求协程里 await，则 SSE 连接与任务生命周期
    强绑定，**前端一断线就取消任务**——违反架构 §5 "前端故障不能传导到后端任务"。
    故：任务用 `asyncio.create_task` 起在后台，SSE 只订阅事件；断线只是取消订阅，
    任务照跑，重连时靠 `last_event_id` 从缓冲重放补齐。
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from attest.config import load_settings
from attest.logging import get_logger, setup_logging

from .session import SessionManager, stream_events

log = get_logger(__name__)


# ============================================================ 请求 / 响应模型


class CreateSessionIn(BaseModel):
    thread_id: str | None = Field(None, description="留空则服务端生成")
    query: str = Field("", description="初始问题（可后续在 chat 时再给）")


class ChatIn(BaseModel):
    session_id: str = Field(..., description="thread_id")
    query: str = Field(..., description="调研问题")
    resume: bool = Field(False, description="true = 从静态断点续跑（此时忽略 query 的业务语义）")


class ConfirmIn(BaseModel):
    session_id: str
    action: str = Field("approve", description="approve | skip | edit | abort")
    #: action=edit 时的新大纲
    plan: dict[str, Any] | None = None


# ============================================================ 应用


def _sse(event: dict[str, Any], seq: int | None = None) -> str:
    """把事件字典序列化成 SSE 帧。

    ⚠️ **`id:` 字段是断线重连的关键**：浏览器 `EventSource` 会自动把最后收到的 `id`
    放进 `Last-Event-ID` 请求头，重连时带回来。我们用它做"从哪条开始重放"的游标。
    没有它，重连必然丢失中间事件。

    ⚠️ 事件名放在 `payload.event` 内部，而 SSE 的 `event:` 字段统一用 `message`：
    因为一个会话里事件类型很多，前端用 `JSON.parse(e.data).event` 分派更灵活，
    也避免前端要 `addEventListener` 注册一堆具名事件（协议更易演进）。
    """
    data = json.dumps(event, ensure_ascii=False)
    lines = []
    if seq is not None:
        lines.append(f"id: {seq}")
    lines.append("event: message")
    lines.append(f"data: {data}")
    lines.append("")  # 空行结束一帧
    lines.append("")
    return "\n".join(lines)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """装配 / 释放 `SessionManager`（async 检查点连接绑定在这里，见 T5.1 硬约束）。"""
    settings = load_settings()
    setup_logging(settings.log_level)
    settings.ensure_dirs()
    mgr = SessionManager(settings)
    await mgr.start()
    app.state.manager = mgr
    log.info(
        f"[api] 就绪 | llm_mode={settings.llm_mode} | search_mode={settings.search_mode} "
        f"| 人工确认={settings.human_confirm_enabled}"
    )
    try:
        yield
    finally:
        await mgr.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Attest（质证）· Web 工作台 API",
        version="0.6.0",
        description="逐句质证的多 Agent 行业深度调研系统 · P6 接入层",
        lifespan=lifespan,
    )
    # 开发期跨域：Vite dev server (5173) → API (8000)。生产由 Vite 打包后同源托管，无需 CORS。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173", "http://127.0.0.1:5173",
            "http://localhost:8000", "http://127.0.0.1:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _register_routes(app)
    return app


def _mgr(request: Request) -> SessionManager:
    mgr: SessionManager | None = getattr(request.app.state, "manager", None)
    if mgr is None or not mgr.started:
        raise HTTPException(status_code=503, detail="服务尚未就绪（SessionManager 未启动）")
    return mgr


def _register_routes(app: FastAPI) -> None:
    # ---------------------------------------------------------- 会话
    @app.post("/api/session")
    async def create_session(request: Request, body: CreateSessionIn) -> dict[str, Any]:
        """新建（或幂等复用）一个会话。"""
        mgr = _mgr(request)
        thread_id = body.thread_id or f"web-{int(time.time() * 1000)}"
        session = await mgr.create_session(thread_id, body.query)
        return {"ok": True, "session": session.snapshot()}

    @app.get("/api/session")
    async def list_sessions(request: Request) -> dict[str, Any]:
        """会话列表（侧栏数据源）。"""
        mgr = _mgr(request)
        items = await mgr.list_sessions()
        return {"ok": True, "sessions": [s.snapshot() for s in items]}

    @app.get("/api/session/{thread_id}")
    async def get_session(request: Request, thread_id: str) -> dict[str, Any]:
        """单个会话状态。

        ⚠️ 内存里没有 ≠ 不存在：进程重启后内存态丢失，但**检查点还在**。
        这里会去检查点问一次"这个 thread 是否有待执行任务"，据此把 status 标为
        `awaiting_confirm`（可续跑）——这是 P5 "断点续跑"能力在 Web 侧的暴露点。
        """
        mgr = _mgr(request)
        session = await mgr.get(thread_id)
        if session is not None:
            return {"ok": True, "session": session.snapshot()}

        # 内存无记录 → 查检查点（冷启动恢复路径）
        ckpt = await _checkpoint_view(mgr, thread_id)
        if ckpt is None:
            raise HTTPException(status_code=404, detail=f"会话不存在：{thread_id}")
        return {"ok": True, "session": ckpt}

    @app.get("/api/sessions/{thread_id}/checkpoint")
    async def checkpoint_state(request: Request, thread_id: str) -> dict[str, Any]:
        """直接读检查点里的待执行任务与 state 摘要（排查 / 冷启动恢复用）。"""
        mgr = _mgr(request)
        view = await _checkpoint_view(mgr, thread_id)
        if view is None:
            raise HTTPException(status_code=404, detail=f"无检查点：{thread_id}")
        return {"ok": True, "checkpoint": view}

    # ---------------------------------------------------------- SSE 主链路
    @app.post("/api/chat")
    async def chat(request: Request, body: ChatIn) -> StreamingResponse:
        """跑一次调研并**以 SSE 流式推送进度**。

        协议（T6.2）：`run_start` / `agent_start` / `agent_end` / `awaiting_confirm`
        / `report_ready` / `error` / `ping` / `stream_end`。
        """
        mgr = _mgr(request)
        session = await mgr.get(body.session_id)
        if session is None:
            # 允许直接用新 thread 起跑（前端"新建会话 + 提问"一步到位）
            session = await mgr.create_session(body.session_id, body.query)
        if body.query:
            session.query = body.query

        # 已在跑就别重复起任务（防前端双击 / 重复提交导致状态互踩）
        running = session.task is not None and not session.task.done()
        if running:
            log.warning(f"[api] thread={session.thread_id} 已有任务在跑，本次仅订阅事件流")
        else:
            decision = {"action": "approve"} if body.resume else None
            session.task = asyncio.create_task(mgr.run(session, decision=decision))

        # 重连游标：浏览器自动带回 Last-Event-ID
        last_index = _parse_last_event_id(request)

        async def gen() -> AsyncIterator[str]:
            seq = last_index
            try:
                async for ev in stream_events(session, last_index=last_index):
                    yield _sse(ev, seq=seq)
                    seq += 1
            except asyncio.CancelledError:
                # 客户端断开：**不取消后台任务**（架构 §5 硬要求）
                log.info(f"[api] SSE 订阅者断开 thread={session.thread_id}（任务继续）")
                raise

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # 关掉 nginx 类反代的缓冲，否则事件会被攒着一起发（SSE 就白做了）
                "X-Accel-Buffering": "no",
            },
        )

    # ---------------------------------------------------------- 人工确认
    @app.post("/api/confirm")
    async def confirm(request: Request, body: ConfirmIn) -> dict[str, Any]:
        """投递人工确认决策 → 触发（或继续）调研。

        静态断点续跑两步法（T5.4 实测语义）：`update_state(plan_approval)` + `invoke(None)`，
        已在 `SessionManager.run(decision=...)` 里实现。
        """
        mgr = _mgr(request)
        session = await mgr.get(body.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"会话不存在：{body.session_id}")
        if session.status != "awaiting_confirm":
            raise HTTPException(
                status_code=409,
                detail=f"会话不处于待确认状态（当前 {session.status}）——确认只能在静态断点上做",
            )
        if body.action not in ("approve", "skip", "edit", "abort"):
            raise HTTPException(status_code=422, detail=f"非法 action：{body.action}")

        decision: dict[str, Any] = {"action": body.action}
        if body.action == "edit":
            if not body.plan or not isinstance(body.plan.get("outlines"), list):
                raise HTTPException(
                    status_code=422,
                    detail="action=edit 必须带 plan.outlines（字符串数组）",
                )
            outlines = [str(x).strip() for x in body.plan["outlines"] if str(x).strip()]
            if not outlines:
                raise HTTPException(status_code=422, detail="编辑后的大纲不能为空")
            decision["plan"] = {"outlines": outlines}
        if body.action == "abort":
            session.status = "aborted"
            session.push({"event": "error", "message": "用户中止"})
            return {"ok": True, "status": session.status, "resumed": False}

        session.status = "running"
        session.pending_confirm = None
        session.task = asyncio.create_task(mgr.run(session, decision=decision))
        return {"ok": True, "status": session.status, "resumed": True}

    # ---------------------------------------------------------- 产物
    @app.get("/api/report/{thread_id}")
    async def get_report(request: Request, thread_id: str) -> dict[str, Any]:
        """报告正文 + 引用索引（T6.5 报告双栏 / 悬浮卡的数据源）。"""
        mgr = _mgr(request)
        session = await mgr.get(thread_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"会话不存在：{thread_id}")
        if not session.result:
            raise HTTPException(
                status_code=409,
                detail=f"会话尚未产出结果（当前 {session.status}）",
            )
        return {"ok": True, "report": session.result}

    @app.get("/api/trace/{thread_id}")
    async def get_trace(
        request: Request,
        thread_id: str,
        limit: int = Query(500, ge=1, le=EVENT_LIMIT_MAX),
    ) -> dict[str, Any]:
        """本次运行的**事件流摘要**（执行时间线 / 成本面板数据源，T6.7）。

        ⚠️ 这里返回的是 **SSE 事件缓冲**，不是 `trace.jsonl`。
        架构设计 §7 的硬决策：前端不直接读 trace 文件（否则改 trace 格式要改前端）。
        `trace.jsonl` 仍是审计与评测的数据源，属另一条链路。
        """
        mgr = _mgr(request)
        session = await mgr.get(thread_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"会话不存在：{thread_id}")
        events = session.events[-limit:]
        return {
            "ok": True,
            "thread_id": thread_id,
            "status": session.status,
            "count": len(session.events),
            "events": events,
            "timeline": _timeline(events),
        }

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, Any]:
        """健康检查（启动脚本探活用）。"""
        mgr = _mgr(request)
        return {
            "ok": True,
            "started": mgr.started,
            "llm_mode": mgr.settings.llm_mode,
            "search_mode": mgr.settings.search_mode,
            "human_confirm": mgr.settings.human_confirm_enabled,
            "sessions": len(await mgr.list_sessions()),
        }


#: `/api/trace` 单次最多返回的事件数
EVENT_LIMIT_MAX = 5000


def _parse_last_event_id(request: Request) -> int:
    """从 `Last-Event-ID` 头解析重连游标（解析失败按 0 处理 = 全量重放）。

    浏览器 `EventSource` 断线自动重连时会带上这个头；它对应我们 `_sse(seq=...)` 里发出的 `id:`。
    """
    raw = request.headers.get("last-event-id")
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _timeline(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 `agent_start` / `agent_end` 配对成时间线行（T6.7 执行时间线）。

    ⚠️ 配对按 **`index`**（节点实例序号）而不是节点名——`scout_web` 会扇出多路、
    补检还会多跑一轮，同名节点出现多次；按名配对会把它们的耗时混成一条。

    ⚠️ 耗时取 `agent_end.duration_ms`（服务端从 trace 镜像过来的**真实**耗时）。
    可能在两种情况下为 `None`，前端应显示"—"而**不是** 0：
      1. trace 里该节点没有 `node_end`（异常中断）；
      2. 节点还在跑（未闭合的 open row）。
    """
    open_rows: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for ev in events:
        idx = ev.get("index")
        if idx is None:
            continue
        if ev.get("event") == "agent_start":
            open_rows[idx] = {
                "index": idx,
                "node": ev.get("node"),
                "start": ev.get("ts"),
                "outputs": ev.get("outputs") or [],
            }
        elif ev.get("event") == "agent_end" and idx in open_rows:
            row = open_rows.pop(idx)
            row["end"] = ev.get("ts")
            row["duration_ms"] = ev.get("duration_ms")
            row["brief"] = ev.get("brief", "")
            rows.append(row)
    # 未闭合的（正在跑 / 被中断）也给出，前端显示"进行中"
    for row in open_rows.values():
        row["duration_ms"] = None
        row["brief"] = "（进行中）"
        rows.append(row)
    rows.sort(key=lambda r: r.get("index", 0))
    return rows


def _checkpoint_view_sync(mgr: SessionManager, thread_id: str) -> dict[str, Any] | None:
    """同步读取检查点（由 `asyncio.to_thread` 包着跑，避免阻塞事件循环）。

    ⚠️ 这里**故意用同步 SqliteSaver 独立开一个只读连接**，而不是复用 async saver：
    只读探测走独立短连接最简单，也避免与运行中的 async 连接争用同一事务状态。
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    db = Path(mgr.settings.checkpoint_db)
    if not db.exists():
        return None
    with SqliteSaver.from_conn_string(str(db)) as cp:
        try:
            from attest.graph.build import build_context, build_graph
            # 一张"轻量图"：只需能读出检查点即可，节点是否真能跑不影响 get_state
            ctx = build_context(mgr.settings, run_id="probe")
            app = build_graph(ctx, checkpointer=cp)
            snap = app.get_state({"configurable": {"thread_id": thread_id}})
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[api] 检查点探测失败 thread={thread_id}：{type(exc).__name__}: {exc}")
            return None
        values = snap.values or {}
        if not values:
            return None
        return {
            "thread_id": thread_id,
            "query": values.get("query", ""),
            "status": "awaiting_confirm" if snap.next else "done",
            "next": list(snap.next or ()),
            "has_report": bool(values.get("report")),
            "cost_incurred": round(float(values.get("cost_incurred") or 0.0), 6),
            "tokens_incurred": int(values.get("tokens_incurred") or 0),
            "source": "checkpoint",  # 标记来源，前端可区分"内存态"vs"冷启动恢复"
        }


async def _checkpoint_view(mgr: SessionManager, thread_id: str) -> dict[str, Any] | None:
    return await asyncio.to_thread(_checkpoint_view_sync, mgr, thread_id)


#: `uvicorn app.main:app` 的入口
app = create_app()
