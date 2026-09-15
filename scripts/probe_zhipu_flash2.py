# -*- coding: utf-8 -*-
"""复验第二轮：glm-4.7-flash 重试是否可用 + glm-4.5-flash 正常长度下能否出内容。

第一轮结论：glm-4.7-flash 报 429「该模型当前访问量过大」（≠ 付费档的「余额不足」），
疑似**有权限但繁忙**。本脚本重试 4 次（间隔 5s）确认；并用正常 max_tokens 验证
glm-4.5-flash 不是空壳。
"""
from __future__ import annotations

import time
from pathlib import Path

import httpx

REPO = Path(r"D:\aiyingyong1\attest")
cfg: dict[str, str] = {}
for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()

key = cfg.get("ATTEST_COMPAT_API_KEY") or ""
url = (cfg.get("ATTEST_COMPAT_BASE_URL") or "").rstrip("/") + "/chat/completions"
hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def call(model: str, prompt: str, max_tokens: int) -> tuple[int, str]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    try:
        r = httpx.post(url, headers=hdr, json=body, timeout=90.0)
    except Exception as e:  # noqa: BLE001
        return -1, f"异常 {type(e).__name__}: {str(e)[:100]}"
    if r.status_code == 200:
        try:
            d = r.json()
            msg = (d.get("choices") or [{}])[0].get("message", {})
            txt = (msg.get("content") or "").strip()
            reason = msg.get("reasoning_content")
            u = d.get("usage") or {}
            return 200, f"len={len(txt)} content={txt[:60]!r} reasoning={'有' if reason else '无'} tokens={u.get('total_tokens')}"
        except Exception as e:  # noqa: BLE001
            return 200, f"解析失败 {e}"
    try:
        m = (r.json().get("error") or {}).get("message") or r.text[:120]
    except Exception:  # noqa: BLE001
        m = r.text[:120]
    return r.status_code, str(m)[:140]


print("=== A. glm-4.7-flash 重试 4 次（间隔 5s）===")
ok_47 = False
for i in range(4):
    code, note = call("glm-4.7-flash", "用一句话说明什么是向量检索。", 64)
    print(f"  第{i + 1}次: {code} | {note}")
    if code == 200:
        ok_47 = True
        break
    if i < 3:
        time.sleep(5)

print()
print("=== B. glm-4.5-flash 正常长度出文（确认不是空壳）===")
code, note = call("glm-4.5-flash", "用两句话说明什么是向量检索，以及它的典型用途。", 256)
print(f"  {code} | {note}")

print()
print("=== C. 对照：当前在用的 glm-4-flash ===")
code, note = call("glm-4-flash", "用两句话说明什么是向量检索，以及它的典型用途。", 256)
print(f"  {code} | {note}")

print()
print("=== 结论 ===")
print("  glm-4.7-flash 可用:", "是" if ok_47 else "否（繁忙，需稍后重试）")
