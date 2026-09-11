"""P6 前端契约探针：验证 Vite dev server 能正确提供前端 + 代理 /api 到后端。

为什么需要它：本环境的 Chromium 起不来（沙箱拦），所以跑不了真浏览器截图。
但"前端能不能工作"里有很大一部分是 **HTTP 层可验证的**：
  1. dev server 返回 HTML，且注入了 React 入口 / Vite 客户端；
  2. 源码模块能被 Vite 编译成可加载的 ESM（不是 404 / 编译错）；
  3. `/api/*` 请求经 Vite 代理**真转发到后端**（返回后端的 JSON 而非 404）；
  4. SSE 流经代理后**逐帧到达**（不是被攒到最后一起给）。

这些都是"打开浏览器才看得见的东西"背后的必要条件。视觉/交互仍需用户亲自看，
本探针不声称覆盖那部分。

用法：
    python scripts/p6_frontend_probe.py     # 需先起后端 8000 + 前端 5173
"""

from __future__ import annotations

import http.client
import json
import sys
import time

FE = ("127.0.0.1", 5173)
BE = ("127.0.0.1", 8000)

_results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, extra: str = "") -> None:
    _results.append((ok, label))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {extra}" if extra else ""), flush=True)


def get(host_port, path, timeout=10):
    c = http.client.HTTPConnection(*host_port, timeout=timeout)
    c.request("GET", path)
    r = c.getresponse()
    body = r.read()
    hdrs = dict(r.getheaders())
    c.close()
    return r.status, body, hdrs


def post(host_port, path, body, timeout=120):
    c = http.client.HTTPConnection(*host_port, timeout=timeout)
    c.request("POST", path, body=json.dumps(body).encode(),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    raw = r.read()
    hdrs = dict(r.getheaders())
    c.close()
    try:
        return r.status, json.loads(raw.decode("utf-8")), hdrs
    except json.JSONDecodeError:
        return r.status, raw.decode("utf-8", "replace"), hdrs


def main() -> int:
    # 0. 两个服务都在
    try:
        s_be, _, _ = get(BE, "/api/health")
        check(s_be == 200, "后端 8000 /api/health 200", f"status={s_be}")
    except Exception as e:
        check(False, "后端 8000 可达", f"{type(e).__name__}: {e}")
        return summarize()

    try:
        s_fe, html, _ = get(FE, "/")
    except Exception as e:
        check(False, "前端 5173 可达", f"{type(e).__name__}: {e}")
        return summarize()

    # 1. dev server 返回 HTML 且注入 React 入口
    text = html.decode("utf-8", "replace")
    check(s_fe == 200 and "<div id=\"root\"" in text, "前端返回 HTML 且含 #root 挂载点", f"status={s_fe}")
    check("/@vite/client" in text, "含 Vite HMR 客户端脚本（dev server 正常）")
    check("src/main.tsx" in text, "含 React 入口 src/main.tsx")
    check("lang=\"zh-CN\"" in text, "html lang=zh-CN（中文页面）")

    # 2. 入口模块能被 Vite 编译（不是 404 / 500）
    s, body, _ = get(FE, "/src/main.tsx")
    ok = s == 200 and "createRoot" in body.decode("utf-8", "replace")
    check(ok, "/src/main.tsx 被 Vite 编译为可加载 ESM", f"status={s}")

    # 3. API 客户端模块能编译（含手写 SSE 解析）
    s, body, _ = get(FE, "/src/api.ts")
    txt = body.decode("utf-8", "replace")
    check(s == 200 and "streamChat" in txt, "/src/api.ts 编译通过且含 streamChat", f"status={s}")

    # 4. 代理真转发：经 5173 打 /api/health，应拿到后端 JSON
    s, body, _ = get(FE, "/api/health")
    ok = False
    try:
        j = json.loads(body.decode("utf-8"))
        ok = s == 200 and j.get("ok") is True
    except Exception:
        j = body[:120]
    check(ok, "Vite 代理 /api/health → 后端（真转发）", f"status={s} body={j}")

    # 5. 经代理建会话（证明 POST + JSON body 也转发）
    s, j, _ = post(FE, "/api/session", {"query": "深度调研：量子计算在金融风控领域的应用现状与主要挑战"})
    tid = (j.get("session") or {}).get("thread_id") if isinstance(j, dict) else None
    check(s == 200 and bool(tid), "经代理 POST /api/session 成功", f"status={s} thread_id={tid}")

    # 6. SSE 经代理逐帧到达（关键：代理没把流攒起来）
    if tid:
        c = http.client.HTTPConnection(*FE, timeout=180)
        c.request("POST", "/api/chat",
                  body=json.dumps({"session_id": tid, "query": "深度调研：量子计算在金融风控领域的应用现状与主要挑战"}).encode(),
                  headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        r = c.getresponse()
        ctype = r.getheader("content-type") or ""
        check(r.status == 200 and "text/event-stream" in ctype,
              "经代理 POST /api/chat 返回 SSE", f"status={r.status} ctype={ctype}")

        # 逐帧计时：若代理缓冲，第一帧会等到很晚才到
        t0 = time.time()
        first_byte_at = None
        frames = 0
        kinds = set()
        buf = b""
        while True:
            chunk = r.read(1024)
            if not chunk:
                break
            if first_byte_at is None:
                first_byte_at = time.time() - t0
            buf += chunk
            while b"\n\n" in buf:
                frame, buf = buf.split(b"\n\n", 1)
                frames += 1
                for ln in frame.split(b"\n"):
                    if ln.startswith(b"data:"):
                        try:
                            kinds.add(json.loads(ln[5:].strip().decode()).get("event"))
                        except Exception:
                            pass
        c.close()
        check(frames > 0, "经代理收到 SSE 帧", f"frames={frames} kinds={sorted(k for k in kinds if k)}")
        check(first_byte_at is not None and first_byte_at < 10.0,
              "首帧 <10s 到达（代理未缓冲，流式有效）",
              f"first_byte={first_byte_at:.2f}s" if first_byte_at is not None else "无首帧")
        check("report_ready" in kinds or "awaiting_confirm" in kinds,
              "经代理最终收到终态事件")

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
