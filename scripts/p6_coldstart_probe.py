"""P6 冷启动恢复探针：**跨进程**验证"进程重启后靠检查点恢复会话"。

补 P6 待核实清单第 4 条：
> `GET /api/session/{id}` 的冷启动恢复路径有实现但缺端到端实测（进程重启后回查检查点）。
> —— 单测覆盖的是"内存命中"分支，重启场景需一次真手测。

`scripts/api_smoke.py` 覆盖的是**同进程内存命中**分支；本探针补**跨进程**：
`start` 与 `recover` 必须是**两个独立进程**——只跑一个进程测不出"重启"，
因为 `_sessions` 只是进程内存 dict（设计如此，见 `app/session.py` 模块文档串）。

用法：
    ATTEST_HUMAN_CONFIRM=1 python scripts/p6_coldstart_probe.py start   <db路径> <thread_id>
    ATTEST_HUMAN_CONFIRM=1 python scripts/p6_coldstart_probe.py recover <db路径> <thread_id>

`start`   ：建会话 → chat 跑到 awaiting_confirm（写下检查点）→ 进程退出（内存态随之消失）
`recover` ：全新进程（`_sessions` 必为空）→ 查 /api/session/{id}，必须走检查点恢复路径

退出码 0 = 本阶段全部断言通过。
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

HOST = "127.0.0.1"
PORT = int(os.environ.get("ATTEST_PROBE_PORT", "8130"))
QUERY = "调研企业知识库 Agent 平台的市场规模与竞品格局，按市场规模/竞品/收费模式三部分输出"

_results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, extra: str = "") -> None:
    _results.append((ok, label))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {extra}" if extra else ""), flush=True)


def observe(label: str, extra: str = "") -> None:
    """观察项：记录事实，不参与 PASS/FAIL（用于"尚未定论"的行为）。"""
    print(f"[OBS ] {label}" + (f"  -- {extra}" if extra else ""), flush=True)


def req(method, path, body=None, *, stream=False):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=180)
    payload = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if stream:
        return resp.status, resp
    raw = resp.read()
    conn.close()
    try:
        return resp.status, json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return resp.status, raw.decode("utf-8", "replace")


def parse_sse(resp) -> list[dict]:
    events, buf = [], b""
    while True:
        chunk = resp.read(2048)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            data = [ln.decode("utf-8", "replace")[5:].strip()
                    for ln in frame.split(b"\n") if ln.startswith(b"data:")]
            if data:
                try:
                    events.append(json.loads("\n".join(data)))
                except json.JSONDecodeError:
                    pass
    return events


def start_server() -> None:
    import uvicorn

    from app.main import create_app

    server = uvicorn.Server(uvicorn.Config(
        create_app(), host=HOST, port=PORT, log_level="warning", lifespan="on"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            c = http.client.HTTPConnection(HOST, PORT, timeout=2)
            c.request("GET", "/api/health")
            r = c.getresponse(); r.read(); c.close()
            if r.status == 200:
                return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("uvicorn 未能在 60s 内起来")


def phase_start(tid: str) -> int:
    print(f"== [start] 起服务 {HOST}:{PORT}（human_confirm=1，全新进程）==", flush=True)
    start_server()

    status, body = req("GET", "/api/health")
    check(status == 200 and body.get("human_confirm") is True,
          "健康检查 human_confirm=True", f"hc={body.get('human_confirm') if isinstance(body, dict) else '?'}")

    status, body = req("POST", "/api/session", {"thread_id": tid, "query": QUERY})
    check(status == 200, "建会话", f"HTTP {status}")

    status, resp = req("POST", "/api/chat", {"session_id": tid, "query": QUERY}, stream=True)
    events = parse_sse(resp)
    kinds = [e.get("event") for e in events]
    check("awaiting_confirm" in kinds, "chat 挂起在 awaiting_confirm", f"tail={kinds[-3:]}")

    status, body = req("GET", f"/api/session/{tid}")
    sess = body.get("session") if isinstance(body, dict) else {}
    check(sess.get("status") == "awaiting_confirm", "会话状态 = awaiting_confirm",
          f"status={sess.get('status')} source={sess.get('source')}")

    print(f"\n== [start] 完成：检查点已写盘，进程即将退出（内存态随之消失）==", flush=True)
    return summarize("start")


def phase_recover(tid: str) -> int:
    print(f"== [recover] 全新进程起服务 {HOST}:{PORT}（_sessions 必为空）==", flush=True)
    start_server()

    # ---- 断言 1：内存里没有（证明这是真·冷启动，不是内存命中）----
    from app.session import _sessions as live_sessions
    check(tid not in live_sessions, "新进程内存里查无此会话（确为冷启动）",
          f"live_sessions={list(live_sessions)}")

    # ---- 断言 2：读路径 —— 靠检查点恢复会话视图 ----
    status, body = req("GET", f"/api/session/{tid}")
    sess = body.get("session") if isinstance(body, dict) else {}
    check(status == 200, "GET /api/session/{id} 200（内存未命中→查检查点）", f"HTTP {status}")
    check(sess.get("source") == "checkpoint", "会话来源标记 = checkpoint（走了恢复路径）",
          f"source={sess.get('source')}")
    check(sess.get("status") == "awaiting_confirm" and "human_confirm" in (sess.get("next") or []),
          "状态与待执行节点正确（awaiting_confirm / next=human_confirm）",
          f"status={sess.get('status')} next={sess.get('next')}")
    check(bool(sess.get("query")), "检查点里带回了原始 query", f"query={str(sess.get('query'))[:30]}")

    # ---- 断言 3：检查点视图端点 ----
    status, body = req("GET", f"/api/sessions/{tid}/checkpoint")
    check(status == 200, "GET /api/sessions/{id}/checkpoint 200", f"HTTP {status}")

    # ---- 断言 4：写路径 —— 冷启动后必须能**续跑**（这才是"恢复"的完整含义）----
    # 修复前实测：HTTP 404「会话不存在」——读得到、点不动。见 app/main.py `_rehydrate_session`。
    status, body = req("POST", "/api/confirm", {"session_id": tid, "action": "approve"})
    check(status == 200 and isinstance(body, dict) and body.get("resumed") is True,
          "冷启动后 POST /api/confirm 可续跑（不再 404）",
          f"HTTP {status} body={str(body)[:120]}")

    if status == 200:
        status, resp = req("POST", "/api/chat", {"session_id": tid, "query": QUERY}, stream=True)
        kinds = [e.get("event") for e in parse_sse(resp)]
        check("report_ready" in kinds, "冷启动续跑跑到 report_ready", f"tail={kinds[-3:]} n={len(kinds)}")

        status, body = req("GET", f"/api/report/{tid}")
        rep = (body.get("report") or {}) if isinstance(body, dict) else {}
        txt = (rep.get("report") or "").strip()
        check(status == 200 and bool(txt), "冷启动续跑产出报告",
              f"HTTP {status} chars={len(txt)}")

        status, body = req("GET", f"/api/session/{tid}")
        sess2 = body.get("session") if isinstance(body, dict) else {}
        check(sess2.get("status") == "done", "续跑后状态回到 done", f"status={sess2.get('status')}")

    return summarize("recover")


def summarize(stage: str) -> int:
    total, passed = len(_results), sum(1 for ok, _ in _results if ok)
    print(f"\n==== [{stage}] {passed}/{total} PASS ====", flush=True)
    for ok, label in _results:
        if not ok:
            print(f"  FAIL: {label}", flush=True)
    return 0 if passed == total else 1


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    _, phase, db_path, tid = sys.argv
    if not os.environ.get("ATTEST_HUMAN_CONFIRM"):
        print("⚠️  需要 ATTEST_HUMAN_CONFIRM=1，否则图不含断点，本探针无意义。")
        return 2
    # 必须在 import app.main 之前设置（create_app 在模块底部执行）——本文件靠 start_server 内延迟 import 达成
    os.environ["ATTEST_CHECKPOINT_DB"] = db_path
    print(f"（检查点库 = {db_path}）", flush=True)
    if phase == "start":
        return phase_start(tid)
    if phase == "recover":
        return phase_recover(tid)
    print(f"未知阶段：{phase}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
