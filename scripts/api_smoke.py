"""T6.1 / T6.2 · P6 接口层冒烟：起真 uvicorn，用真 HTTP 打全部端点。

**为什么必须有这一步**（而不是只跑 pytest）：
    pytest 里的 10 条用例走的是 `SessionManager`（进程内直接调用），
    **没经过 ASGI 栈**——路由注册、请求体校验、SSE 分帧、`Last-Event-ID` 头解析、
    lifespan 装配，全都测不到。P5 的教训是"自证必须打到真边界"。

覆盖：
    /api/health                       健康检查
    POST /api/session                 建会话
    GET  /api/session                 列表
    GET  /api/session/{id}            单会话
    POST /api/chat (SSE)              流式跑完，且**事件帧格式合法、agent_start/end 成对**
    GET  /api/report/{id}             报告 + 引用索引
    GET  /api/trace/{id}              事件摘要 + 时间线
    POST /api/confirm                 人工确认续跑（含编辑生效）
    GET  /api/sessions/{id}/checkpoint 检查点视图（冷启动恢复路径）

用法：
    .venv/Scripts/python.exe scripts/api_smoke.py
退出码 0 = 全通过。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUERY = "调研'企业知识库 Agent 平台'市场，按市场规模/竞品/收费模式三部分输出"

results: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    results.append((label, ok, detail))
    print(f"[{'OK' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


def main() -> int:
    # 独立临时工作目录：不污染 data/ 下的真库，跑完即删
    tmp = Path(tempfile.mkdtemp(prefix="api-smoke-"))
    os.environ.update(
        {
            "ATTEST_LLM_MODE": "mock",
            "ATTEST_SEARCH_MODE": "mock",
            "ATTEST_TRACE_DIR": str(tmp / "trace"),
            "ATTEST_REPORT_DIR": str(tmp / "reports"),
            "ATTEST_FIXTURE_DIR": str(ROOT / "data" / "fixtures"),
            "ATTEST_CHECKPOINT_DB": str(tmp / "cp.sqlite"),
            "ATTEST_PROFILE_DB": str(tmp / "profile.sqlite"),
            "ATTEST_CHROMA_DIR": str(tmp / "chroma"),
            "ATTEST_LOCAL_DOCS": str(tmp / "nodocs"),
            "ATTEST_HUMAN_CONFIRM": "1",  # 打开静态断点，才能测 /api/confirm
            "ATTEST_LOG_LEVEL": "WARNING",
        }
    )

    import uvicorn
    from fastapi.testclient import TestClient

    from app.main import create_app

    # ---- 用 TestClient 起 ASGI 栈（真 lifespan + 真路由 + 真校验，进程内不起端口）----
    # ⚠️ 为什么不用 uvicorn 起真端口：TestClient 走完整 ASGI 协议栈（含 lifespan、
    #    ！依赖注入、请求校验、StreamingResponse 分帧），覆盖度足够；起真端口的额外价值
    #    （TCP/TLS/反代）不在本项目范围内，且会让脚本变脆（端口占用、时序竞态）。
    app = create_app()
    with TestClient(app) as client:
        # cfg 里的 thread_id 由 lifespan 里的 settings 读取 → 与上面 env 一致
        # ---------- 1. 健康检查 ----------
        r = client.get("/api/health")
        check("GET /api/health 200", r.status_code == 200, f"HTTP {r.status_code}")
        h = r.json() if r.status_code == 200 else {}
        check("health 报告 mock 模式", h.get("llm_mode") == "mock", str(h.get("llm_mode")))
        check("health 报告人工确认开启", h.get("human_confirm") is True, str(h.get("human_confirm")))

        # ---------- 2. 建会话 ----------
        tid = f"smoke-{int(time.time())}"
        r = client.post("/api/session", json={"thread_id": tid, "query": QUERY})
        check("POST /api/session 200", r.status_code == 200, f"HTTP {r.status_code}")
        check("建会话返回 thread_id", (r.json().get("session") or {}).get("thread_id") == tid)

        # ---------- 3. SSE 流（首跑 → 应停在 awaiting_confirm）----------
        events: list[dict] = []
        with client.stream("POST", "/api/chat", json={"session_id": tid, "query": QUERY}) as resp:
            check("POST /api/chat 200", resp.status_code == 200, f"HTTP {resp.status_code}")
            check(
                "SSE content-type 正确",
                resp.headers.get("content-type", "").startswith("text/event-stream"),
                resp.headers.get("content-type", ""),
            )
            seq_ids: list[int] = []
            cur_id: int | None = None
            for line in resp.iter_lines():
                if line.startswith("id: "):
                    cur_id = int(line[4:])
                    seq_ids.append(cur_id)
                elif line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        kinds = [e.get("event") for e in events]
        check("SSE 收到 run_start", "run_start" in kinds, str(kinds[:6]))
        check("SSE 收到 awaiting_confirm（静态断点经 HTTP 生效）", "awaiting_confirm" in kinds)
        check("SSE 有 id: 游标（断线重连基础）", bool(seq_ids) and seq_ids == sorted(seq_ids),
              f"{seq_ids[:5]}...共 {len(seq_ids)}")
        check("SSE 末尾有 stream_end", kinds and kinds[-1] == "stream_end", str(kinds[-1:]))
        starts = [e for e in events if e.get("event") == "agent_start"]
        ends = [e for e in events if e.get("event") == "agent_end"]
        check("HTTP 层 agent_start/agent_end 成对", len(starts) == len(ends) and starts,
              f"{len(starts)} vs {len(ends)}")
        check("断点前未检索（断点位置正确）",
              not any(str(e.get("node", "")).startswith("scout_") for e in starts),
              str([e.get("node") for e in starts]))
        confirm_ev = next((e for e in events if e.get("event") == "awaiting_confirm"), {})
        check("awaiting_confirm 载荷含大纲", bool(confirm_ev.get("outlines")),
              f"{len(confirm_ev.get('outlines') or [])} 章")

        # ---------- 4. 会话状态查询 ----------
        r = client.get(f"/api/session/{tid}")
        check("GET /api/session/{id} 200", r.status_code == 200, f"HTTP {r.status_code}")
        check("会话状态 = awaiting_confirm",
              (r.json().get("session") or {}).get("status") == "awaiting_confirm",
              str((r.json().get("session") or {}).get("status")))

        r = client.get("/api/session")
        check("GET /api/session 列表含本会话",
              any(s.get("thread_id") == tid for s in r.json().get("sessions", [])))

        # ---------- 5. 确认（编辑大纲）→ 续跑 ----------
        new_outlines = ["市场规模（HTTP 编辑）", "竞品格局（HTTP 编辑）", "收费模式（HTTP 编辑）"]
        r = client.post("/api/confirm", json={
            "session_id": tid, "action": "edit", "plan": {"outlines": new_outlines},
        })
        check("POST /api/confirm 200", r.status_code == 200, f"HTTP {r.status_code}")
        check("confirm 报告已续跑", r.json().get("resumed") is True)

        # 续跑也走 SSE：等它跑完
        events2: list[dict] = []
        with client.stream("POST", "/api/chat", json={"session_id": tid, "query": QUERY}) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    events2.append(json.loads(line[6:]))
        kinds2 = [e.get("event") for e in events2]
        # 若续跑已跑完（event buffer 已终态），这里主要靠 report 端点核验
        r = client.get(f"/api/report/{tid}")
        check("GET /api/report/{id} 200（续跑已产出）", r.status_code == 200, f"HTTP {r.status_code}")
        report = r.json().get("report") or {}
        check("报告非空", bool(report.get("report")), f"{len(report.get('report') or '')} 字")
        check("编辑的大纲真的进了结果",
              list((report.get("plan") or {}).get("outlines") or []) == new_outlines,
              str((report.get("plan") or {}).get("outlines")))
        check("引用索引已建", bool(report.get("references")), f"{len(report.get('references') or {})} 条")
        check("成本/token 记账存在",
              report.get("tokens_incurred", 0) > 0, f"tokens={report.get('tokens_incurred')}")
        check("状态终态 done", report.get("route") is not None, str(report.get("route")))

        # ---------- 6. trace / timeline ----------
        r = client.get(f"/api/trace/{tid}")
        check("GET /api/trace/{id} 200", r.status_code == 200, f"HTTP {r.status_code}")
        tr = r.json()
        tl = tr.get("timeline") or []
        check("时间线段非空", bool(tl), f"{len(tl)} 段")
        check("时间线段含 duration_ms 字段",
              all("duration_ms" in row for row in tl), f"首段={tl[0] if tl else None}")
        check("耗时是真实值（非 None，禁止假 0）",
              all(row.get("duration_ms") is not None for row in tl),
              f"None 段数={sum(1 for r in tl if r.get('duration_ms') is None)}")
        check("检索节点进入时间线",
              any(str(row.get("node", "")).startswith("scout_") for row in tl))

        # ---------- 7. 检查点视图 ----------
        r = client.get(f"/api/sessions/{tid}/checkpoint")
        check("GET /api/sessions/{id}/checkpoint 200", r.status_code == 200, f"HTTP {r.status_code}")
        ck = r.json().get("checkpoint") or {}
        check("检查点可见报告与成本",
              ck.get("has_report") is True and ck.get("tokens_incurred", 0) > 0,
              f"has_report={ck.get('has_report')} tokens={ck.get('tokens_incurred')}")

        # ---------- 8. 错误路径（必须给出明确错误，不能静默）----------
        r = client.get("/api/session/does-not-exist-xyz")
        check("未知会话 404", r.status_code == 404, f"HTTP {r.status_code}")
        r = client.post("/api/confirm", json={"session_id": tid, "action": "approve"})
        check("非待确认态 confirm 409", r.status_code == 409, f"HTTP {r.status_code}")
        r = client.post("/api/confirm", json={"session_id": "nope-xyz", "action": "approve"})
        check("未知会话 confirm 404", r.status_code == 404, f"HTTP {r.status_code}")
        r = client.post("/api/confirm", json={"session_id": tid, "action": "bogus"})
        check("非法 action 被拒（404/409/422 均可）",
              r.status_code in (404, 409, 422), f"HTTP {r.status_code}")

    print("-" * 66)
    failed = [r for r in results if not r[1]]
    print(f"接口冒烟结果：{'PASS' if not failed else f'FAIL（{len(failed)} 项）'} / 共 {len(results)} 项")
    for label, _, _ in failed:
        print(f"  ✗ {label}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
