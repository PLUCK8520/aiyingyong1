"""T6.1 / T6.2 · P6 会话层与 SSE 协议回归测试（离线 mock，不消耗任何 API 额度）。

**这一组测试要证明的**（而不是"跑过就算"）：
  1. async 检查点真的装配成功（`AsyncSqliteSaver`，不是同步 saver 冒充）；
  2. `SessionManager.run` 在真实图上跑到底，产出报告，且 `status=done`；
  3. **静态断点真挂起**：`human_confirm_enabled=True` 时停在 `awaiting_confirm`，
     且此时**没有报告**（证明断点在检索之前）；
  4. **续跑两步法生效**：`decision=approve` → 跑完出报告；
  5. **编辑大纲真的生效**（不是静默沿用旧 plan）——这是 P5 用真实图 A/B 对照抓出来的坑，
     P6 必须同样用真实图验证，不能只测 `human_confirm.run` 这个纯函数；
  6. SSE 事件协议字段齐全：`run_start`/`agent_start`/`agent_end`/`report_ready`，
     且 agent_start/agent_end **成对**；
  7. **SSE 断线不影响后端任务**（架构不变式：前端故障不能传导到后端）。
     —— 现在有实现可测了，P5 留的这条只能"纸上声明"，这里补上真断言。

⚠️ 全部离线：`llm_mode=mock` + `search_mode=mock`（fixture 回放），每用例独立临时目录。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from attest.config import Settings  # noqa: E402


def make_settings(tmp_path: Path, *, human_confirm: bool = False) -> Settings:
    s = Settings(
        llm_mode="mock",
        search_mode="mock",
        trace_dir=tmp_path / "trace",
        report_dir=tmp_path / "reports",
        fixture_dir=ROOT / "data" / "fixtures",
        checkpoint_db=tmp_path / "cp.sqlite",
        profile_db=tmp_path / "profile.sqlite",
        local_docs_dir=tmp_path / "nodocs",  # 不启用本地库，避免受 fixtures 变化影响
        human_confirm_enabled=human_confirm,
    )
    s.ensure_dirs()
    return s


def run(coro):
    """跑一个协程（测试内没有 pytest-asyncio 依赖，自己起 loop）。"""
    return asyncio.run(coro)


# ============================================================ 1. 异步检查点


def test_async_checkpointer_is_actually_async(tmp_path: Path) -> None:
    """`make_async_checkpointer` 必须返回 async 上下文管理器，且产出 `AsyncSqliteSaver`。

    防的是"照抄同步实现"——同步 saver 在 async 事件循环里会阻塞且不报错。
    """
    from attest.memory.checkpointer import make_async_checkpointer

    async def _check() -> None:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        cm = make_async_checkpointer(tmp_path / "nested" / "cp.sqlite")
        # 关键：不是 saver 本身，而是上下文管理器（有 __aenter__）
        assert hasattr(cm, "__aenter__"), "必须返回 async 上下文管理器"
        saver = await cm.__aenter__()
        try:
            assert isinstance(saver, AsyncSqliteSaver)
            # 父目录被自动创建（_ensure_parent）
            assert (tmp_path / "nested").exists()
        finally:
            await cm.__aexit__(None, None, None)

    run(_check())


# ============================================================ 2~3. 首跑到底 / 断点挂起


def test_run_to_completion_produces_report(tmp_path: Path) -> None:
    """无人确认（断点关）时，一次跑到底并产出报告。"""
    from app.session import SessionManager

    settings = make_settings(tmp_path, human_confirm=False)

    async def _check() -> None:
        mgr = SessionManager(settings)
        await mgr.start()
        try:
            s = await mgr.create_session("t-done", "调研'企业知识库 Agent 平台'市场")
            await mgr.run(s)  # 直接 await，跑完为止
            assert s.status == "done", f"status={s.status} error={s.error}"
            assert s.result.get("report"), "必须有报告正文"
            assert s.result.get("tokens_incurred", 0) > 0, "mock 也要有 token 记账"
            # 事件协议：run_start / report_ready 必须存在
            kinds = [e.get("event") for e in s.events]
            assert "run_start" in kinds
            assert "report_ready" in kinds
            assert kinds[-1] == "report_ready", "最后一条应是终态事件"
            # node_start/end 在 trace 里成对（审计源），这里顺带核一下 run 没吞异常
            assert s.result.get("citation_check") is not None
        finally:
            await mgr.stop()

    run(_check())


def test_static_breakpoint_pauses_before_retrieval(tmp_path: Path) -> None:
    """**核心用例**：断点开启时停在 `awaiting_confirm`，且断点前**没有检索、没有报告**。

    "断点前无报告" 是 P5 定的硬断言（`verify_resume.py` 同款），这里在 Web 链路上复验：
    它证明断点真的在 `human_confirm` 之前，而不是"跑完了却说自己停着"。
    """
    from app.session import SessionManager

    settings = make_settings(tmp_path, human_confirm=True)

    async def _check() -> None:
        mgr = SessionManager(settings)
        await mgr.start()
        try:
            s = await mgr.create_session("t-pause", "调研'企业知识库 Agent 平台'市场")
            await mgr.run(s)

            assert s.status == "awaiting_confirm", f"status={s.status} error={s.error}"
            assert s.pending_confirm is not None, "必须给出待确认载荷"
            assert s.pending_confirm.get("outlines"), "载荷里必须有大纲"
            assert not s.result.get("report"), "断点前不得有报告"

            kinds = [e.get("event") for e in s.events]
            assert "awaiting_confirm" in kinds
            # 断点前不得出现检索节点事件——这是"断点位置正确"的实证
            started = [e.get("node") for e in s.events if e.get("event") == "agent_start"]
            assert not any(str(n).startswith("scout_") for n in started), (
                f"断点前不应检索，实际已跑：{started}"
            )
        finally:
            await mgr.stop()

    run(_check())


# ============================================================ 4~5. 续跑 / 编辑生效


def test_resume_after_approve_completes(tmp_path: Path) -> None:
    """同意后续跑到底（`update_state` + `invoke(None)` 两步法在 async 下可行）。"""
    from app.session import SessionManager

    settings = make_settings(tmp_path, human_confirm=True)

    async def _check() -> None:
        mgr = SessionManager(settings)
        await mgr.start()
        try:
            s = await mgr.create_session("t-resume", "调研'企业知识库 Agent 平台'市场")
            await mgr.run(s)
            assert s.status == "awaiting_confirm"

            await mgr.run(s, decision={"action": "approve"})
            assert s.status == "done", f"status={s.status} error={s.error}"
            assert s.result.get("report")
            # 续跑后事件缓冲里应有两段 run_start（首跑 + 续跑）
            assert sum(1 for e in s.events if e.get("event") == "run_start") == 2
            # 检索节点现在必须跑过（与 pause 用例形成 A/B 对照）
            started = [e.get("node") for e in s.events if e.get("event") == "agent_start"]
            assert any(str(n).startswith("scout_") for n in started)
        finally:
            await mgr.stop()

    run(_check())


def test_edited_outlines_reach_the_graph(tmp_path: Path) -> None:
    """**编辑大纲必须真的生效**（用真实图 + 检查点，不是只测纯函数）。

    这是 P5 踩过的坑：静态断点下 `Command(resume=...)` 的 payload 无处投递，
    会**静默沿用旧 plan、不报错**。所以必须看续跑后 state 里的 `plan.outlines`
    是否等于编辑后的值——只看"跑完了"会被静默失效骗过。
    """
    from app.session import SessionManager

    settings = make_settings(tmp_path, human_confirm=True)
    NEW_OUTLINES = ["市场规模与增速（编辑）", "竞品格局（编辑）", "收费模式（编辑）"]

    async def _check() -> None:
        mgr = SessionManager(settings)
        await mgr.start()
        try:
            s = await mgr.create_session("t-edit", "调研'企业知识库 Agent 平台'市场")
            await mgr.run(s)
            assert s.status == "awaiting_confirm"
            before = list((s.pending_confirm or {}).get("outlines") or [])
            assert before != NEW_OUTLINES, "编辑前后必须不同，否则用例无意义"

            await mgr.run(s, decision={"action": "edit", "plan": {"outlines": NEW_OUTLINES}})
            assert s.status == "done", f"status={s.status} error={s.error}"

            plan = s.result.get("plan") or {}
            assert list(plan.get("outlines") or []) == NEW_OUTLINES, (
                f"编辑未生效！实际 outlines={plan.get('outlines')}"
            )
        finally:
            await mgr.stop()

    run(_check())


# ============================================================ 6. SSE 协议


def test_sse_event_protocol_pairs_agent_start_end(tmp_path: Path) -> None:
    """SSE 事件协议：agent_start / agent_end 成对，且字段齐全（T6.2）。"""
    from app.session import SessionManager

    settings = make_settings(tmp_path, human_confirm=False)

    async def _check() -> None:
        mgr = SessionManager(settings)
        await mgr.start()
        try:
            s = await mgr.create_session("t-proto", "调研'企业知识库 Agent 平台'市场")
            await mgr.run(s)

            starts = [e for e in s.events if e.get("event") == "agent_start"]
            ends = [e for e in s.events if e.get("event") == "agent_end"]
            assert starts, "必须有 agent_start"
            assert len(starts) == len(ends), (
                f"agent_start/agent_end 必须成对：{len(starts)} vs {len(ends)}"
            )
            # index 是配对键，不能重复
            idx_s = [e["index"] for e in starts]
            assert len(idx_s) == len(set(idx_s)), "agent_start 的 index 必须唯一"
            # 字段契约
            for e in starts:
                assert {"node", "index", "ts"} <= set(e)
            for e in ends:
                assert {"node", "index", "ts", "brief"} <= set(e)
            rr = [e for e in s.events if e.get("event") == "report_ready"]
            assert len(rr) == 1
            assert {"cost_cny", "tokens", "audit_summary", "mock"} <= set(rr[0])
        finally:
            await mgr.stop()

    run(_check())


def test_sse_stream_replays_from_cursor(tmp_path: Path) -> None:
    """断线重连：`last_index` 之后的事件可补齐（架构 §5「事件可重放」）。

    这是"SSE 断线可重连"的实现证据——重连靠 `Last-Event-ID` → `last_index` → 缓冲重放，
    而不是靠"重跑一遍"。
    """
    from app.session import SessionManager, stream_events

    settings = make_settings(tmp_path, human_confirm=False)

    async def _check() -> None:
        mgr = SessionManager(settings)
        await mgr.start()
        try:
            s = await mgr.create_session("t-replay", "调研'企业知识库 Agent 平台'市场")
            await mgr.run(s)
            total = len(s.events)

            # 从头订阅：应收到全部历史 + 一个 stream_end（因为已终态）
            got_head = [ev async for ev in stream_events(s, last_index=0)]
            assert len(got_head) == total + 1, f"{len(got_head)} != {total}+1"
            assert got_head[-1]["event"] == "stream_end"

            # 从中间订阅：只应是尾部
            cut = total // 2
            got_tail = [ev async for ev in stream_events(s, last_index=cut)]
            assert len(got_tail) == total - cut + 1
            assert got_tail[0] == s.events[cut], "重放起点必须精确对齐 last_index"
        finally:
            await mgr.stop()

    run(_check())


# ============================================================ 7. 断开不传导


def test_client_disconnect_does_not_cancel_task(tmp_path: Path) -> None:
    """**架构不变式**：SSE 订阅者断开，后端任务必须继续跑完。

    架构设计 §5：「前端故障不能传导到后端任务」。P5 时这条只能纸面声明，
    现在有 SSE 生成器可测了，补上真断言。

    手法：起任务 → 订阅一次立刻主动 break（模拟客户端断开）→ 等任务结束 → 会话仍 done。
    """
    from app.session import SessionManager, stream_events

    settings = make_settings(tmp_path, human_confirm=False)

    async def _check() -> None:
        mgr = SessionManager(settings)
        await mgr.start()
        try:
            s = await mgr.create_session("t-disc", "调研'企业知识库 Agent 平台'市场")
            s.task = asyncio.create_task(mgr.run(s))

            # 订阅后立刻"断开"（只取一条就退出生成器）
            agen = stream_events(s, last_index=0)
            first = await agen.__anext__()
            assert first is not None
            await agen.aclose()  # 模拟连接关闭

            assert len(s.subscribers) == 0, "断开后必须从订阅者列表移除（防泄漏）"

            # 等后台任务跑完
            await s.task
            assert s.status == "done", f"断开后任务被取消了？status={s.status}"
            assert s.result.get("report"), "断开不应影响产出"
        finally:
            await mgr.stop()

    run(_check())


# ============================================================ 8. 引用索引


def test_reference_index_is_built_for_report_view(tmp_path: Path) -> None:
    """报告双栏要用 `引用编号 → 证据` 映射（T6.5 数据基础）。

    只断言"索引非空 + 值为 dict + 含标题字段"，不硬断言编号格式——
    编号格式是 `retrieval/citations.py` 的责任，那里已有单测，重复断言会增加脆性。
    """
    from app.session import SessionManager

    settings = make_settings(tmp_path, human_confirm=False)

    async def _check() -> None:
        mgr = SessionManager(settings)
        await mgr.start()
        try:
            s = await mgr.create_session("t-ref", "调研'企业知识库 Agent 平台'市场")
            await mgr.run(s)
            refs = s.result.get("references") or {}
            assert refs, "引用索引不得为空（证据存在时必须建索引）"
            for cid, item in refs.items():
                assert isinstance(item, dict), f"{cid} 的值必须是 dict"
                assert "title" in item and "snippet" in item
                assert len(item["snippet"]) <= 500, "snippet 必须截断（否则前端卡）"
        finally:
            await mgr.stop()

    run(_check())


# ============================================================ 9. 并发保护


def test_second_run_while_running_does_not_clobber(tmp_path: Path) -> None:
    """重复提交保护：任务在跑时再次 `run` 应被拒绝（返回/抛出可识别信号）。

    ⚠️ 当前实现的口径：`run` 本身不做拦截（拦在 API 层 `chat` 的 `session.task.done()` 判断里），
    但**必须有可判断的句柄**。这条用例锁定"句柄可用性"这个契约——
    如果以后有人把 task 句柄去掉，这条会红。
    """
    from app.session import SessionManager

    settings = make_settings(tmp_path, human_confirm=False)

    async def _check() -> None:
        mgr = SessionManager(settings)
        await mgr.start()
        try:
            s = await mgr.create_session("t-guard", "调研'企业知识库 Agent 平台'市场")
            s.task = asyncio.create_task(mgr.run(s))
            assert s.task is not None and not s.task.done(), "必须暴露可判断的运行句柄"
            await s.task
            assert s.task.done(), "跑完必须置为 done（API 层据此判断可否再次提交）"
        finally:
            await mgr.stop()

    run(_check())


# ============================================================ 11. 冷启动恢复（P6 待核实 #4）


def test_rehydrate_session_from_checkpoint_enables_resume(tmp_path: Path) -> None:
    """冷启动：内存态丢失后，`_rehydrate_session` 必须能从检查点重建**可续跑**的会话。

    背景（2026-09-13 跨进程探针实测，见 `scripts/p6_coldstart_probe.py`）：
    修复前 `POST /api/confirm` 只认内存态 → 进程重启后返回 404，
    用户在 UI 上表现为"看得到待确认会话、点不动"。
    本用例用**真实检查点 + 真实图**验证重建逻辑；
    完整跨进程链路（两个独立 OS 进程）由那个探针脚本负责。
    """
    from app import main as api_main
    from app.session import SessionManager, _sessions

    settings = make_settings(tmp_path, human_confirm=True)

    async def _check() -> None:
        mgr = SessionManager(settings)
        await mgr.start()
        try:
            s = await mgr.create_session("t-cold", "调研'企业知识库 Agent 平台'市场")
            await mgr.run(s)
            assert s.status == "awaiting_confirm", f"应先挂在断点：{s.status}"

            # 模拟进程重启：内存登记表清空（检查点仍在磁盘）
            _sessions.clear()

            revived = await api_main._rehydrate_session(mgr, "t-cold")
            assert revived is not None, "停在 human_confirm 断点的会话必须能重建"
            assert revived.status == "awaiting_confirm"
            assert (revived.pending_confirm or {}).get("outlines"), "应带回大纲供前端弹窗"

            # 重建后能续跑，才算"恢复"完整（不然只是只读展示）
            await mgr.run(revived, decision={"action": "approve"})
            assert revived.status == "done", f"续跑应到 done：{revived.status} {revived.error}"
            assert revived.result.get("report"), "冷启动续跑必须产出报告"
        finally:
            _sessions.clear()
            await mgr.stop()

    run(_check())


def test_rehydrate_returns_none_for_completed_thread(tmp_path: Path) -> None:
    """已完成（done）的会话**不重建**——防"重启后随便点一下就重跑一遍调研"。"""
    from app import main as api_main
    from app.session import SessionManager, _sessions

    settings = make_settings(tmp_path, human_confirm=False)

    async def _check() -> None:
        mgr = SessionManager(settings)
        await mgr.start()
        try:
            s = await mgr.create_session("t-done2", "调研'企业知识库 Agent 平台'市场")
            await mgr.run(s)
            assert s.status == "done"
            _sessions.clear()
            assert await api_main._rehydrate_session(mgr, "t-done2") is None, (
                "已完成的会话没有可推进的动作，不应重建"
            )
        finally:
            _sessions.clear()
            await mgr.stop()

    run(_check())
