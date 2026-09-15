# -*- coding: utf-8 -*-
"""探测智谱各 Flash 模型在**当前账号**下是否可用（免费档）。

背景：用户以为"要花钱才能换好模型"，但第三方汇总（③证据）称智谱
GLM-4.7-Flash / GLM-4.5-Flash 等 Flash 系列**永久免费、无需绑卡**。
本脚本用 .env 里的现有 key 实测，把证据等级抬到 ①（自己账号的真实响应）。

只发最小请求（max_tokens 极小），不消耗有意义额度；key 不落盘、不回显。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

REPO = Path(r"D:\aiyingyong1\attest")
ENV = REPO / ".env"

# 读 .env（只取需要的两项）
cfg: dict[str, str] = {}
for line in ENV.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    cfg[k.strip()] = v.strip()

key = cfg.get("ATTEST_COMPAT_API_KEY") or ""
base = cfg.get("ATTEST_COMPAT_BASE_URL") or ""
print("base_url =", base)
print("key 长度 =", len(key), "前缀 =", key[:6] + "..." if key else "(空)")
if not key:
    sys.exit("没有 key，无法探测")

# 候选模型：智谱 Flash 系列（免费档）+ 当前在用的作对照 + 若干付费档作对照
CANDIDATES = [
    "glm-4-flash",          # 当前在用
    "glm-4-flash-250414",   # Flash 的新版本号形态
    "glm-4.5-flash",
    "glm-4.7-flash",
    "glm-4.6-flash",
    "glm-4-air",            # 已知 429（付费档对照）
    "glm-4.7",              # 付费档对照
]

url = base.rstrip("/") + "/chat/completions"
print()
print("=== 逐个探测（max_tokens=8）===")
for m in CANDIDATES:
    body = {
        "model": m,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "temperature": 0,
    }
    try:
        r = httpx.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
            timeout=45.0,
        )
        code = r.status_code
        note = ""
        if code == 200:
            try:
                d = r.json()
                note = "OK | 回复=" + repr((d.get("choices") or [{}])[0].get("message", {}).get("content", ""))[:40]
                u = d.get("usage") or {}
                note += f" | tokens={u.get('total_tokens')}"
            except Exception as e:  # noqa: BLE001
                note = f"200 但解析失败 {e}"
        else:
            try:
                err = r.json()
                msg = (err.get("error") or {}).get("message") or str(err)
            except Exception:  # noqa: BLE001
                msg = r.text[:160]
            note = f"{code} | {msg[:140]}"
        print(f"  {m:22s} -> {note}")
    except Exception as e:  # noqa: BLE001
        print(f"  {m:22s} -> 异常 {type(e).__name__}: {str(e)[:120]}")
