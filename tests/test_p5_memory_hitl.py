"""P5 回归测试：记忆与协同（T5.1 / T5.2 / T5.4 / T5.5）。

把这四条从"代码写了"变成"改坏了立刻红"：
    - T5.1  检查点：有 checkpointer 时可暂停-恢复，无 checkpointer 时行为退化但不报错
    - T5.2  画像：写入 → `memory_loader` 注入 state → 跨 thread 仍生效（模板 demo 7）
    - T5.4  人工确认：**静态断点**真的挂起在 `human_confirm` 之前；放行后可跑完；编辑大纲生效
    - T5.5  断点续跑：冷启动恢复后**不重复检索**（重复检索 = 重复扣额度）

⚠️ 为什么 T5.4 必须用静态断点（`interrupt_before`）而不是节点内 `interrupt()`：
    实测（2026-09-11）节点内 `interrupt()` 在 SqliteSaver 下会被**静默放行**——
    `interrupt()` 抛异常前写入的 `RESUME` 空值残留 + 通道版本单调递增，
    导致节点重跑时 `get_null_resume()` 拿到空值直接返回，`__interrupt__` 不再出现。
    详见 `src/attest/agents/human_confirm.py` 模块文档串。
    本文件的 `test_node_level_interrupt_is_silently_skipped` 把这个坑**钉成回归断言**。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from attest.config import Settings
from attest.graph.build import build_context, build_graph, initial_state
from attest.memory.profile import ProfileStore, parse_remember_directive

REPO = Path(__file__).resolve().parents[1]
RESEARCH_QUERY = "调研'企业知识库 Agent 平台'市场，按市场规模/竞品/收费模式三部分输出，附溯源链接"


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_mode="mock",
        search_mode="mock",
        trace_dir=tmp_path / "trace",
        report_dir=tmp_path / "reports",
        fixture_dir=REPO / "data" / "fixtures",
        checkpoint_db=tmp_path / "checkpoints.sqlite",
        profile_db=tmp_path / "profile.sqlite",
    )


def _ctx(settings: Settings, run_id: str):
    return build_context(settings, run_id=run_id)


def _scout_starts(ctx) -> int:
    return sum(
        1
        for e in ctx.trace.of("node_start")
        if str(e.get("node", "")).startswith("scout_")
    )


# --------------------------------------------------------------- T5.1 检查点


def test_checkpointer_persists_thread_state(settings: Settings) -> None:
    """T5.1：接了 checkpointer 后，同一 thread 的 state 可被再次读回。"""
    with SqliteSaver.from_conn_string(str(settings.checkpoint_db)) as cp:
        ctx = _ctx(settings, "t51")
        app = build_graph(ctx, checkpointer=cp)
        cfg = {"configurable": {"thread_id": "t51"}}

        app.invoke(initial_state(RESEARCH_QUERY, thread_id="t51"), cfg)
        snap = app.get_state(cfg)
        assert snap.values, "检查点里没有 state——持久化没生效"
        assert snap.values.get("report"), "报告未落盘到检查点"


def test_graph_without_checkpointer_still_runs(settings: Settings) -> None:
    """T5.1：不传 checkpointer 时退化为无状态单跑（P1~P4 行为），不能报错。"""
    ctx = _ctx(settings, "t51-nocp")
    app = build_graph(ctx)
    out = app.invoke(initial_state(RESEARCH_QUERY, thread_id="none"))
    assert out.get("report")


# ----------------------------------------------------------------- T5.2 画像


def test_profile_roundtrip_and_prompt_block(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "p.sqlite")
    assert store.as_prompt_block() == "", "空画像不应产出提示词段落"

    store.set("输出语言", "中文")
    store.set("偏好格式", "表格优先")
    block = store.as_prompt_block()
    assert "中文" in block and "表格优先" in block
    assert len(store.all()) == 2


def test_profile_overwrite_same_key(tmp_path: Path) -> None:
    """同 key 覆盖，不产生重复条目（画像按 key 去重）。"""
    store = ProfileStore(tmp_path / "p.sqlite")
    store.set("输出语言", "中文")
    store.set("输出语言", "英文")
    items = store.all()
    assert len(items) == 1
    assert items[0].value == "英文"


def test_profile_injected_into_state_across_threads(settings: Settings) -> None:
    """T5.2 + T5.3：换 thread 偏好仍生效（复刻模板截图 7 的跨会话 demo）。"""
    store = ProfileStore(settings.profile_db)
    store.set("输出语言", "中文")
    store.set("关注点", "成本控制")

    for thread in ("sess-a", "sess-b"):
        ctx = _ctx(settings, thread)
        app = build_graph(ctx)
        out = app.invoke(initial_state(RESEARCH_QUERY, thread_id=thread))
        assert out.get("profile"), f"thread={thread} 未注入画像"
        assert out["profile"].get("输出语言") == "中文"
        assert out["profile"].get("关注点") == "成本控制"


def test_profile_disabled_returns_empty(settings: Settings) -> None:
    """画像关闭时 `memory_loader` 如实返回空画像，而不是假装读过。"""
    ctx = _ctx(settings, "t52-off")
    ctx.profile = None
    app = build_graph(ctx)
    out = app.invoke(initial_state(RESEARCH_QUERY, thread_id="off"))
    assert out.get("profile") == {}


def test_parse_remember_directive_forms() -> None:
    assert parse_remember_directive("记住：输出语言=中文") == ("输出语言", "中文")
    assert parse_remember_directive("记住 偏好表格") == ("偏好", "偏好表格")
    assert parse_remember_directive("这次调研的结论是 180 亿") is None


# ------------------------------------------------- T5.4 人工确认（静态断点）


def test_static_interrupt_pauses_before_human_confirm(settings: Settings) -> None:
    """T5.4：开启确认后，图必须**停在 `human_confirm` 之前**，且一次检索都没发生。

    这条是 T5.5 的前提——如果这里 `snap.next` 为空，说明断点没生效，
    整个"人工确认"特性就是摆设（09-10 那版正是如此，且不报错）。
    """
    settings.human_confirm_enabled = True
    with SqliteSaver.from_conn_string(str(settings.checkpoint_db)) as cp:
        ctx = _ctx(settings, "t54")
        app = build_graph(ctx, checkpointer=cp)
        cfg = {"configurable": {"thread_id": "t54"}}

        app.invoke(initial_state(RESEARCH_QUERY, thread_id="t54"), cfg)
        snap = app.get_state(cfg)

        assert "human_confirm" in (snap.next or ()), f"未停在断点：next={snap.next}"
        assert not snap.values.get("report"), "断点前不该有报告"
        assert _scout_starts(ctx) == 0, "断点前不该发生检索（会白烧额度）"

        app.update_state(cfg, {"plan_approval": {"action": "approve"}}, as_node="planner")
        resumed = app.invoke(None, cfg)
        assert resumed.get("plan_approved") is True
        assert resumed.get("report"), "放行后应产出报告"
        assert _scout_starts(ctx) > 0, "放行后才该检索"


def test_no_interrupt_when_confirm_disabled(settings: Settings) -> None:
    """默认（未开启确认）时图**一次跑完**，不产生断点。"""
    settings.human_confirm_enabled = False
    with SqliteSaver.from_conn_string(str(settings.checkpoint_db)) as cp:
        ctx = _ctx(settings, "t54-off")
        app = build_graph(ctx, checkpointer=cp)
        cfg = {"configurable": {"thread_id": "t54-off"}}

        app.invoke(initial_state(RESEARCH_QUERY, thread_id="t54-off"), cfg)
        snap = app.get_state(cfg)
        assert not snap.next, f"不该有断点：next={snap.next}"
        assert snap.values.get("report")


def test_edit_outline_on_resume_takes_effect(settings: Settings) -> None:
    """T5.4：resume 时传新大纲 → 必须用新大纲继续执行（验收口径）。

    ⚠️ 静态断点的续跑方式是 `update_state(plan_approval) + invoke(None)`，
    **不是** `Command(resume=...)`——后者在静态断点下没有 `interrupt()` 消费点，
    resume 值无处投递，实测会**静默沿用旧 plan**（编辑不生效，且不报错）。
    """
    settings.human_confirm_enabled = True
    with SqliteSaver.from_conn_string(str(settings.checkpoint_db)) as cp:
        ctx = _ctx(settings, "t54-edit")
        app = build_graph(ctx, checkpointer=cp)
        cfg = {"configurable": {"thread_id": "t54-edit"}}

        app.invoke(initial_state(RESEARCH_QUERY, thread_id="t54-edit"), cfg)
        snap = app.get_state(cfg)
        assert "human_confirm" in (snap.next or ())

        new_outlines = ["市场规模", "竞品格局", "收费模式", "风险提示"]
        app.update_state(
            cfg,
            {"plan_approval": {"action": "edit", "plan": {"outlines": new_outlines}}},
            as_node="planner",
        )
        out = app.invoke(None, cfg)

        assert out["plan"]["outlines"] == new_outlines
        assert out.get("report")


def test_approve_via_update_state_completes(settings: Settings) -> None:
    """T5.4：放行路径（`update_state(approve)` + `invoke(None)`）也能跑完并产出报告。"""
    settings.human_confirm_enabled = True
    with SqliteSaver.from_conn_string(str(settings.checkpoint_db)) as cp:
        ctx = _ctx(settings, "t54-ok")
        app = build_graph(ctx, checkpointer=cp)
        cfg = {"configurable": {"thread_id": "t54-ok"}}

        app.invoke(initial_state(RESEARCH_QUERY, thread_id="t54-ok"), cfg)
        assert "human_confirm" in (app.get_state(cfg).next or ())

        app.update_state(cfg, {"plan_approval": {"action": "approve"}}, as_node="planner")
        out = app.invoke(None, cfg)
        assert out.get("plan_approved") is True
        assert out.get("report")


# --------------------------------------------------------------- T5.5 续跑


def test_resume_across_fresh_app_does_not_refetch(settings: Settings) -> None:
    """T5.5：**新进程/新 app** 从同一 SQLite 恢复 → 放行后检索只发生一次。

    模拟"杀进程 → 重启"：第一次 `with` 退出后连接关闭，第二次重新建 saver + 编译图，
    只靠 SQLite 恢复。断言 resume 前检索数为 0、resume 后正好等于首轮扇出量。
    """
    settings.human_confirm_enabled = True
    cfg = {"configurable": {"thread_id": "t55"}}

    with SqliteSaver.from_conn_string(str(settings.checkpoint_db)) as cp:
        ctx1 = _ctx(settings, "t55-a")
        app1 = build_graph(ctx1, checkpointer=cp)
        app1.invoke(initial_state(RESEARCH_QUERY, thread_id="t55"), cfg)
        snap1 = app1.get_state(cfg)
        assert "human_confirm" in (snap1.next or ()), "第一阶段未挂起"
        assert _scout_starts(ctx1) == 0, "挂起前不该检索"

    # ---- 以上进程"结束"；以下冷启动，只靠 SQLite ----
    with SqliteSaver.from_conn_string(str(settings.checkpoint_db)) as cp:
        ctx2 = _ctx(settings, "t55-b")
        app2 = build_graph(ctx2, checkpointer=cp)
        snap2 = app2.get_state(cfg)

        recovered = snap2.values or {}
        assert recovered.get("plan"), "冷启动未从 SQLite 恢复出 plan"
        assert recovered.get("plan", {}).get("outlines"), "恢复的大纲为空"
        assert "human_confirm" in (snap2.next or ())

        app2.update_state(cfg, {"plan_approval": {"action": "approve"}}, as_node="planner")
        out = app2.invoke(None, cfg)
        assert out.get("report")

        n_sub = len(recovered["plan"].get("sub_questions") or [])
        expected = max(n_sub, 1) * 2  # 子问题数 × 2 源（web/local）
        assert _scout_starts(ctx2) == expected, (
            f"检索次数异常：{_scout_starts(ctx2)} != {expected}"
            f"（重复检索 = 重复扣额度，FR-24）"
        )


# ------------------------------------------- 钉死 09-10 的坑：节点内 interrupt


def test_node_level_interrupt_is_silently_skipped_on_real_graph(settings: Settings) -> None:
    """回归断言：**在真实图上**，节点内 `interrupt()` 会被静默放行（不挂起）。

    这条是 09-10 那版实现被废弃的**实测依据**，必须钉住——否则有人会照旧写法重做一遍。

    ⚠️ 为什么必须在**真实图**上做对照，而不是造一个极简图：
    极简图（单节点 + `interrupt()` 打头）**能正常挂起**——本文件最初那版就是这么写的，
    结果给出反例。说明该行为依赖真实链路的**版本递增时序**（`planner → human_confirm`
    之前有多个 super-step），简化模型无法复现。**用一个过度简化的模型断言真实行为，
    得到的结论不可信**——这正是本项目"零造假"要防的事。

    对照方式：同一份图、同一个 checkpointer、同一个问题，
    只切换 `interrupt_before` 来对比"静态断点"与"节点内 `interrupt()`"：
      - 静态断点 → `next=('human_confirm',)`，`scout=0`，无报告；
      - 节点内 `interrupt()` → `next=()`，`__interrupt__` 不出现，`scout=6`，报告已产出。
    若将来上游修好该行为（B 组也挂起），本条会红——那时再评估是否切回节点内写法。
    """
    settings.human_confirm_enabled = True

    # ---- A 组：静态断点（当前实现）----
    with SqliteSaver.from_conn_string(str(settings.checkpoint_db)) as cp:
        ctx_a = _ctx(settings, "t54-a")
        app_a = build_graph(ctx_a, checkpointer=cp)
        cfg_a = {"configurable": {"thread_id": "cmp-a"}}
        app_a.invoke(initial_state(RESEARCH_QUERY, thread_id="cmp-a"), cfg_a)
        snap_a = app_a.get_state(cfg_a)

    # ---- B 组：关掉静态断点，走节点内 interrupt()（09-10 旧实现的行为）----
    with SqliteSaver.from_conn_string(str(settings.checkpoint_db)) as cp:
        ctx_b = _ctx(settings, "t54-b")
        app_b = build_graph(ctx_b, checkpointer=cp, interrupt_before=[])
        cfg_b = {"configurable": {"thread_id": "cmp-b"}}
        out_b = app_b.invoke(initial_state(RESEARCH_QUERY, thread_id="cmp-b"), cfg_b)
        snap_b = app_b.get_state(cfg_b)

    # A 组：真的挂起
    assert "human_confirm" in (snap_a.next or ()), "静态断点未挂起"
    assert _scout_starts(ctx_a) == 0, "静态断点下不该检索"
    assert not (snap_a.values or {}).get("report"), "静态断点下不该有报告"

    # B 组：被静默放行（已知上游行为）
    assert not snap_b.next, "上游行为已变：节点内 interrupt() 现在会挂起"
    assert "__interrupt__" not in out_b, "上游行为已变：interrupt 信号出现了"
    assert _scout_starts(ctx_b) > 0, "B 组应被静默放行并跑完全链路"
    assert out_b.get("report"), "B 组应已产出报告"
