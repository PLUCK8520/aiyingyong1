"""P6 断点路径探针：验证 human_confirm 静态断点的挂起 → 确认 → 续跑。

与 p6_e2e_probe.py 分开，是因为它需要 ATTEST_HUMAN_CONFIRM=1 环境变量，
而默认（无该变量）运行的图不含断点。混在一个脚本里会互相干扰。

用法：
    ATTEST_HUMAN_CONFIRM=1 python scripts/p6_confirm_probe.py
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HOST = "127.0.0.1"
PORT = int(os.environ.get("ATTEST_PROBE_PORT", "8124"))

_results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, extra: str = "") -> None:
    _results.append((ok, label))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {extra}" if extra else ""), flush=True)


def req(method, path, body=None, *, stream=False, headers=None):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=120)
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


def parse_sse(resp) -> list[dict]:
    events, buf = [], b""
    while True:
        chunk = resp.read(1024)
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


def main() -> int:
    if not os.environ.get("ATTEST_HUMAN_CONFIRM"):
        print("⚠️  需要 ATTEST_HUMAN_CONFIRM=1，否则图不含断点，本探针无意义。")
        return 2

    print(f"== 起服务 {HOST}:{PORT}（human_confirm 开启）==", flush=True)
    start_server()
    print("== 服务就绪 ==", flush=True)

    status, body, _ = req("GET", "/api/health")
    check(status == 200 and body.get("human_confirm") is True,
          "健康检查确认 human_confirm=True", f"status={status} hc={body.get('human_confirm') if isinstance(body, dict) else '?'}")

    QUERY = "深度调研：量子计算在金融风控领域的应用现状与主要挑战"
    status, body, _ = req("POST", "/api/session", {"query": QUERY})
    tid = body["session"]["thread_id"] if isinstance(body, dict) and body.get("session") else None
    check(bool(tid), "建会话", f"thread_id={tid}")
    if not tid:
        return summarize()

    # 第一次 chat：应挂起在 awaiting_confirm
    status, headers, resp = req("POST", "/api/chat", {"session_id": tid, "query": QUERY}, stream=True)
    events = parse_sse(resp)
    kinds = [e.get("event") for e in events]
    check("awaiting_confirm" in kinds, "chat 挂起在 awaiting_confirm", f"tail={kinds[-3:]}")
    check("report_ready" not in kinds, "挂起时**未**产出报告（断点确实拦住了）")

    aw = next((e for e in events if e.get("event") == "awaiting_confirm"), {})
    # 结构：{"event":"awaiting_confirm", "node", "objective", "outlines", "sub_questions"}
    # （confirm_payload 平铺，不嵌套 plan；前端 ConfirmModal 即按此读）
    outlines = aw.get("outlines") or []
    sub_qs = aw.get("sub_questions") or []
    check(bool(outlines), "挂起事件带 outlines（前端弹窗数据源）",
          f"n_outlines={len(outlines)} outlines={outlines[:3]}")
    check(bool(sub_qs), "挂起事件带 sub_questions", f"n={len(sub_qs)}")

    # 会话状态应为 awaiting_confirm
    status, body, _ = req("GET", f"/api/session/{tid}")
    sess = body.get("session") if isinstance(body, dict) else {}
    check(sess.get("status") == "awaiting_confirm",
          "GET session 状态 = awaiting_confirm", f"status_field={sess.get('status')}")

    # 校验：action=edit 但 plan 为空 → 422
    status, _, _ = req("POST", "/api/confirm", {"session_id": tid, "action": "edit", "plan": {}})
    check(status == 422, "action=edit 缺 outlines → 422（校验生效）", f"status={status}")

    # 正常确认：approve（返回 JSON，不流式；续跑在后台 task 里）
    status, body, _ = req("POST", "/api/confirm", {"session_id": tid, "action": "approve"})
    check(status == 200 and body.get("resumed") is True,
          "POST /api/confirm approve 接受并续跑", f"status={status} body={body}")

    # 前端契约：confirm 后**重新订阅 /api/chat** 拿后续事件（POST /api/chat 会复用已有任务）
    status, headers, resp = req("POST", "/api/chat", {"session_id": tid, "query": QUERY}, stream=True)
    events2 = parse_sse(resp)
    kinds2 = [e.get("event") for e in events2]
    check("report_ready" in kinds2, "重订阅后跑到 report_ready", f"tail={kinds2[-3:]} n={len(events2)}")

    # 会话应到 done
    status, body, _ = req("GET", f"/api/session/{tid}")
    sess2 = body.get("session") if isinstance(body, dict) else {}
    check(sess2.get("status") == "done", "续跑后会话状态 = done", f"status={sess2.get('status')}")

    # 最终报告
    status, body, _ = req("GET", f"/api/report/{tid}")
    rep = (body.get("report") if isinstance(body, dict) else None) or {}
    txt = (rep.get("report") or "").strip()
    check(status == 200 and bool(txt),
          "续跑后拿到完整报告", f"status={status} route={rep.get('route')} chars={len(txt)}")

    return summarize()


def summarize() -> int:
    total, passed = len(_results), sum(1 for ok, _ in _results if ok)
    print(f"\n==== {passed}/{total} PASS ====", flush=True)
    for ok, label in _results:
        if not ok:
            print(f"  FAIL: {label}", flush=True)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
