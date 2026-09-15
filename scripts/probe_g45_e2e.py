# -*- coding: utf-8 -*-
"""实测 glm-4.5-flash（免费档推理模型）端到端跑一轮，与 glm-4-flash 基线对比。

基线（2026-09-15 A/B 评测，glm-4-flash，各 6 轮均值）：
  引用 14.5 处 / 审计 62.2 句 / 未证实 19.3 句 / 降级 8.5 句
  未证实占审计比 ≈ 31%
本脚本用同一主题、同一 thread 命名规则跑一轮 glm-4.5-flash，输出可比的指标。

**不改 .env**：通过子进程环境变量覆盖全部模型映射（pydantic-settings 下 env 优先于 .env 文件）。
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

REPO = Path(r"D:\aiyingyong1\attest")
VENV = REPO / ".venv" / "Scripts" / "python.exe"
MODEL = "glm-4.5-flash"
TOPIC = "调研企业知识库 Agent 平台的市场规模与竞争格局"

ENV_KEYS = [
    "ATTEST_MODEL_INTENT", "ATTEST_MODEL_DIRECT", "ATTEST_MODEL_PLANNER",
    "ATTEST_MODEL_JUDGE", "ATTEST_MODEL_CONFLICT", "ATTEST_MODEL_ANALYST",
    "ATTEST_MODEL_AUDIT_FAST", "ATTEST_MODEL_AUDIT_STRONG", "ATTEST_MODEL_FUSE_LIGHT",
]

env = dict(os.environ)
for k in ENV_KEYS:
    env[k] = MODEL
# reasoning 模型 output 侧 token 多，给足预算避免误熔断（观察用）
env["ATTEST_BUDGET_TOKENS"] = "600000"

thread = f"g45-{int(time.time())}"
t0 = time.time()
r = subprocess.run(
    [str(VENV), "scripts/chat.py", "--thread", thread, TOPIC],
    cwd=str(REPO), env=env, capture_output=True, text=True,
    encoding="utf-8", errors="replace", timeout=1800,
)
dur = time.time() - t0
out = (r.stdout or "") + "\n" + (r.stderr or "")

print("模型:", MODEL, "| thread:", thread, "| rc:", r.returncode, "| 耗时: %.1fs" % dur)
print()
for pat, label in [
    (r"引用[:：].*", "引用行"),
    (r"审计[:：].*", "审计行"),
    (r"trace:.*", "trace 摘要"),
    (r"报告已保存[:：].*", "报告路径"),
]:
    m = re.search(pat, out)
    print(f"{label}: {m.group(0) if m else '（未找到）'}")

saved = re.search(r"报告已保存[:：]\s*(.+)", out)
if saved:
    p = Path(saved.group(1).strip())
    if p.exists():
        txt = p.read_text(encoding="utf-8", errors="replace")
        print()
        print("报告字符数:", len(txt))
        print("依据行数:", txt.count("本节论述依据："))
        print("未回查横幅:", "有" if "未通过逐句回查" in txt else "无")
        # 熔断 / 降级痕迹
        for k in ("降级提示", "证据不足", "未证实"):
            print(f"  含「{k}」:", txt.count(k), "处")
        print()
        print("--- 报告前 900 字 ---")
        print(txt[:900])

if r.returncode != 0:
    print()
    print("=== stderr 尾部 ===")
    print((r.stderr or "")[-1500:])
