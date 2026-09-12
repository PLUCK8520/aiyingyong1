"""T7.9 · 切真实档的一键流程（余额/实名到位后跑这一条）。

做的四件事（顺序有讲究，不能颠倒）：
  ① **探针**：先花 4 次小调用确认 key 能用（免费 chat + 付费 chat + embed + rerank）。
     不通就直接停——否则后面每一步都会以"网络错误"的面目出现，白折腾。
  ② **清索引**：换 embedding 模型必须重建。mock 是 256 维、bge-m3 是 1024 维，
     不清就会拿两套向量空间的分数混算（NFR-11 明令禁止，Chroma 档会直接抛错）。
  ③ **标定矛盾阈值**：`conflict_cluster_threshold=0.25` 是在**离线词法向量**上标的，
     换语义向量后分布完全不同，必须重标（这一步输出报告，需要人看分布定值，不自动改配置）。
  ④ **真跑一次完整调研**：拿到真实模型下的报告、耗时与成本——这才是"接上了"的证据。

用法：
    .venv/Scripts/python.exe scripts/go_real.py
    .venv/Scripts/python.exe scripts/go_real.py --skip-calibrate   # 只探针 + 真跑
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from attest.config import load_settings  # noqa: E402

QUERY = "调研'企业知识库 Agent 平台'市场，按市场规模/竞品/收费模式三部分输出，附溯源链接"


def _call(path: str, key: str, base: str, payload: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        f"{base.rstrip('/')}{path}",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:  # noqa: S310 - 固定官方域名
            return r.status, json.dumps(json.loads(r.read().decode()))[:160]
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()[:200]
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


def step1_probe(settings) -> bool:
    key = settings.siliconflow_api_key or ""
    base = settings.siliconflow_base_url
    if not key:
        print("✗ SILICONFLOW_API_KEY 为空：先填 .env")
        return False

    print(f"① 探针（端点 {base}）")
    probes = [
        ("免费 chat", "/chat/completions", {"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}),
        ("重活 chat", "/chat/completions", {"model": settings.model_planner, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}),
        ("embedding", "/embeddings", {"model": settings.model_embed, "input": ["探针"]}),
        ("rerank", "/rerank", {"model": settings.model_rerank, "query": "市场规模", "documents": ["a", "b"], "top_n": 2}),
    ]
    ok = True
    for label, path, payload in probes:
        status, detail = _call(path, key, base, payload)
        flag = "✓" if status == 200 else "✗"
        print(f"   {flag} {label:10s} → {status} {detail[:110]}")
        if status != 200:
            ok = False
    if not ok:
        print(
            "\n✗ 探针未全通，已停在第一步（后面的步骤都会以'网络错误'的面目出现，跑也是白跑）。\n"
            "  账户余额不足（402 / code 30001）时：到控制台完成**实名认证** →\n"
            "  活动中心「认证专享礼」领券（或充值）。免费模型同样受余额/状态门禁，不是代码问题。"
        )
    return ok


def step2_reset_index() -> None:
    idx = REPO / "data" / "index"
    if idx.exists():
        shutil.rmtree(idx, ignore_errors=True)
        print(f"② 已清索引 {idx}（换 embedding 必须重建）")
    else:
        print("② 索引目录本就不存在，跳过")


def step3_calibrate() -> None:
    print("③ 标定矛盾聚类阈值（换 embedding 后必做）")
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "calibrate_conflict.py")],
        cwd=str(REPO),
        check=False,
    )
    print("   ↑ 看相似度分布定 ATTEST_CONFLICT_THRESHOLD（现配置见上方末行）")


def step4_run_real() -> None:
    print("④ 真跑一次完整调研（真实模型 + 离线证据）")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "chat.py"), "--llm", "siliconflow", "--quiet", QUERY],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    dt = time.perf_counter() - t0
    out = (proc.stdout or "") + (proc.stderr or "")
    # chat.py 会把报告落盘并打印路径与成本；这里只回显关键行，避免刷屏
    for line in out.splitlines():
        if any(k in line for k in ("报告", "成本", "token", "引用", "token/", "¥")):
            print("   " + line.strip()[:160])
    print(f"   退出码 {proc.returncode}，墙钟 {dt:.1f}s")
    if proc.returncode != 0:
        print("   ⚠️ 真跑失败——把上面输出发我，这就是真实分支第一次暴露 bug 的地方（正常，不丢人）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="切真实档的一键流程")
    ap.add_argument("--skip-calibrate", action="store_true", help="跳过阈值标定")
    args = ap.parse_args(argv)

    settings = load_settings()
    print(f"llm_mode = {settings.llm_mode}（本脚本强制走 siliconflow 探针，与 .env 无关）\n")
    if not step1_probe(settings):
        return 1
    step2_reset_index()
    if not args.skip_calibrate:
        step3_calibrate()
    step4_run_real()
    print("\n完成。真实档数字与 mock 档分开放：mock 是回归基准，真实档是能力证据，别混着比。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
