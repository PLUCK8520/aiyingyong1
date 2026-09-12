"""真实 API 连通性探针（最小调用，防烧额度）。

用途：`ATTEST_LLM_MODE` 切真实档之前，先确认三件事——
  ① key 有效（否则 401）；
  ② 地域正确（百炼免费额度只认华北2(北京)，intl 域名会静默按量扣费）；
  ③ 账号实际能调**哪些模型**（配置里的模型名如果不在名单里，跑起来才会 400，白折腾）。

只发 3 个请求：GET /models（免费）+ 1 次最小 chat + 1 次最小 embedding。
用法：
    .venv/Scripts/python.exe scripts/probe_dashscope.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from attest.config import load_settings  # noqa: E402

TIMEOUT = 60


def _req(url: str, *, key: str, payload: dict | None = None) -> tuple[int, dict | str]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # noqa: S310 - 固定官方域名
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        try:
            return exc.code, json.loads(body)
        except Exception:  # noqa: BLE001
            return exc.code, body
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


def main() -> int:
    s = load_settings()
    key = s.dashscope_api_key or ""
    if not key:
        print("✗ DASHSCOPE_API_KEY 为空——先填 .env（或确认 .env 在 repo 根目录）")
        return 2

    base = s.dashscope_base_url.rstrip("/")
    host = base.split("//")[-1].split("/")[0]
    print(f"端点：{base}")
    print(f"地域：{'华北2(北京) ✓' if host == 'dashscope.aliyuncs.com' else f'⚠️ 非北京域名：{host}'}")
    print(f"key ：{key[:6]}…{key[-4:]}（{len(key)} 字符）")
    print()

    ok = True

    # ---------- ① 模型清单 ----------
    status, body = _req(f"{base}/models", key=key)
    if status == 200 and isinstance(body, dict):
        ids = sorted(m.get("id", "") for m in body.get("data", []))
        print(f"① GET /models → 200，可用模型 {len(ids)} 个")
        for name in ("qwen3.8-max", "qwen3.7-flash", "qwen3.8-flash", "text-embedding-v4"):
            mark = "✓" if name in ids else "✗ 不在名单"
            print(f"    配置里在用：{name:20s} {mark}")
        if len(ids) <= 40:
            print("    完整名单：" + ", ".join(ids))
    else:
        ok = False
        print(f"① GET /models → {status}：{str(body)[:300]}")

    # ---------- ② 最小 chat ----------
    model = s.model_intent
    status, body = _req(
        f"{base}/chat/completions",
        key=key,
        payload={"model": model, "messages": [{"role": "user", "content": "回复两个字：收到"}], "max_tokens": 16},
    )
    if status == 200 and isinstance(body, dict):
        usage = body.get("usage") or {}
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
        print(f"② chat[{model}] → 200，回复「{text.strip()[:20]}」，token={usage.get('total_tokens')}")
    else:
        ok = False
        print(f"② chat[{model}] → {status}：{str(body)[:300]}")

    # ---------- ③ 最小 embedding ----------
    status, body = _req(
        f"{base}/embeddings",
        key=key,
        payload={"model": s.model_embed, "input": ["连通性探针"]},
    )
    if status == 200 and isinstance(body, dict):
        vec = (body.get("data") or [{}])[0].get("embedding") or []
        usage = body.get("usage") or {}
        print(f"③ embed[{s.model_embed}] → 200，维度={len(vec)}，token={usage.get('total_tokens')}")
    else:
        ok = False
        print(f"③ embed[{s.model_embed}] → {status}：{str(body)[:300]}")

    print()
    print("结论：" + ("三个探针全通，可以切真实档 ✓" if ok else "有探针失败，先别切真实档 ✗"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
