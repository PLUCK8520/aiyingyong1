"""P6 端到端探针：进程内起真 uvicorn（后台线程），用 HTTP 直连跑完整链路。

为什么不用 Bash 的 & —— 本环境 Bash 后台进程在工具返回后会被杀，端口拒连。
所以这里在同一个 Python 进程里用 uvicorn.Server + threading 起服务，能真占端口。

用法：
    python scripts/p6_e2e_probe.py

退出码 0 = 全链路通过；非 0 = 有断言失败（见 stdout 的 FAIL 行）。
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import threading
import time

# 允许脚本直接跑：把仓库根加进 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HOST = "127.0.0.1"
PORT = int(os.environ.get("ATTEST_PROBE_PORT", "8123"))

_results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, extra: str = "") -> None:
    _results.append((ok, label))
    mark = "PASS" if ok else "FAIL"
    line = f"[{mark}] {label}"
    if extra:
        line += f"  -- {extra}"
    print(line, flush=True)


def req(
    method: str,
    path: str,
    body: dict | None = None,
    *,
    stream: bool = False,
    headers: dict | None = None,
):
    """HTTP 直连（绕开沙箱代理）。stream=True 时返回 (status, headers, raw_reader)。"""
    conn = http.client.HTTPConnection(HOST, PORT, timeout=60)
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    payload = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=payload, headers=hdrs)
    resp = conn.getresponse()
    if stream:
        return resp.status, dict(resp.getheaders()), resp
    raw = resp.read()
    conn.close()
    try:
        return resp.status, json.loads(raw.decode("utf-8")), dict(resp.getheaders())
    except json.JSONDecodeError:
        return resp.status, raw.decode("utf-8", "replace"), dict(resp.getheaders())


def start_server() -> threading.Thread:
    import uvicorn

    from app.main import create_app

    config = uvicorn.Config(
        create_app(),
        host=HOST,
        port=PORT,
        log_level="warning",
        lifespan="on",
    )
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    # 等端口可连（最多 60s：启动要装配 Chroma + 检查点）
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection(HOST, PORT, timeout=2)
            conn.request("GET", "/api/health")
            r = conn.getresponse()
            r.read()
            conn.close()
            if r.status == 200:
                return t
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("uvicorn 未能在 60s 内起来")


def parse_sse(resp) -> list[dict]:
    """读 SSE 流，按帧解析成事件列表。"""
    events: list[dict] = []
    buf = b""
    while True:
        chunk = resp.read(1024)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            data_lines = []
            for line in frame.split(b"\n"):
                line = line.decode("utf-8", "replace")
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            if data_lines:
                try:
                    events.append(json.loads("\n".join(data_lines)))
                except json.JSONDecodeError:
                    pass
    return events


def main() -> int:
    print(f"== 起服务 {HOST}:{PORT} ==", flush=True)
    start_server()
    print("== 服务就绪 ==", flush=True)

    # 0. 健康检查
    status, data, _ = req("GET", "/api/health")
    check(status == 200, "GET /api/health 返回 200", f"status={status}")

    # 1. 建会话（响应形状：{"ok": true, "session": {...}}）
    # 用带"深度调研"意图的问题，确保走 research 全链路（检索→写作→审计），
    # 而不是被 intent_router 判为 direct 直答（那样测不到报告链路）。
    QUERY = "深度调研：量子计算在金融风控领域的应用现状与主要挑战"
    status, body, _ = req("POST", "/api/session", {"query": QUERY})
    sess = body.get("session") if isinstance(body, dict) else None
    ok = status == 200 and isinstance(sess, dict) and sess.get("thread_id")
    check(ok, "POST /api/session 建会话", f"status={status} thread_id={sess.get('thread_id') if isinstance(sess, dict) else body}")
    if not ok:
        return summarize()
    tid = sess["thread_id"]

    # 2. 列表（响应形状：{"ok": true, "sessions": [...]}）
    status, body, _ = req("GET", "/api/session")
    lst = body.get("sessions") if isinstance(body, dict) else None
    ok = status == 200 and isinstance(lst, list) and any(s.get("thread_id") == tid for s in lst)
    check(ok, "GET /api/session 列表含新会话", f"status={status} n={len(lst) if isinstance(lst, list) else '?'}")

    # 3. chat SSE（真跑一次调研；不开 human_confirm 时一路到底）
    # 请求体形状（app/main.py ChatIn）：session_id + query 均必填
    status, headers, resp = req(
        "POST", "/api/chat",
        {"session_id": tid, "query": QUERY},
        stream=True,
    )
    ctype = headers.get("content-type", "")
    check(status == 200 and "text/event-stream" in ctype, "POST /api/chat 返回 SSE 流", f"status={status} ctype={ctype}")
    events = parse_sse(resp)
    kinds = [e.get("event") for e in events]
    check(len(events) > 0, "SSE 收到事件", f"n={len(events)} kinds={sorted(set(kinds))}")
    check("agent_start" in kinds, "收到 agent_start（节点级进度）")
    check("agent_end" in kinds, "收到 agent_end（节点耗时）")
    check("report_ready" in kinds or "awaiting_confirm" in kinds, "终态事件到达", f"tail={kinds[-4:]}")

    # agent_end 的 duration_ms 必须非 None 且 > 0（P6 修掉的假 0.0ms 回归防线）
    dur = [e.get("duration_ms") for e in events if e.get("event") == "agent_end"]
    check(bool(dur) and all(isinstance(d, (int, float)) and d > 0 for d in dur),
          "agent_end.duration_ms 全为正值（防假 0.0ms 回归）", f"durations={dur[:6]}")

    # 4. 时间线（响应形状：{"ok": true, "thread_id": ..., "timeline": [...]}）
    status, body, _ = req("GET", f"/api/trace/{tid}")
    tl = body.get("timeline") if isinstance(body, dict) else None
    if tl is None and isinstance(body, dict):
        # 兼容直接返回 list 的实现
        tl = body
    check(status == 200 and isinstance(tl, list) and len(tl) > 0, "GET /api/trace 返回时间线", f"status={status} keys={list(body.keys()) if isinstance(body, dict) else type(body).__name__} n={len(tl) if isinstance(tl, list) else '?'}")

    if "report_ready" in kinds:
        # 5. 报告（响应形状：{"ok": true, "report": {...}}）
        # 字段语义：route=research 时正文在 report；route=direct 时正文在 direct_answer。
        # 两者任一非空即算链路通（具体走哪条由 intent_router 决定，不是缺陷）。
        status, body, _ = req("GET", f"/api/report/{tid}")
        rep = body.get("report") if isinstance(body, dict) else None
        rep = rep or {}
        main_text = (rep.get("report") or "").strip() or (rep.get("direct_answer") or "").strip()
        ok = status == 200 and isinstance(rep, dict) and bool(main_text)
        check(
            ok,
            "GET /api/report 拿到正文",
            f"status={status} route={rep.get('route')} chars={len(main_text)} refs={len(rep.get('references', []))}",
        )
    else:
        # 断点路径：确认接口协议正确（无断点时 404 是**预期**行为，不算失败）
        status, body, _ = req("GET", f"/api/sessions/{tid}/checkpoint")
        ok = status in (200, 404)
        note = "有断点" if status == 200 else "无断点（404，预期）"
        check(ok, f"断点态 GET checkpoint 协议正确", f"status={status} {note}")

    return summarize()


def summarize() -> int:
    total = len(_results)
    passed = sum(1 for ok, _ in _results if ok)
    print(f"\n==== {passed}/{total} PASS ====", flush=True)
    if passed != total:
        print("FAILED:", flush=True)
        for ok, label in _results:
            if not ok:
                print(f"  - {label}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
