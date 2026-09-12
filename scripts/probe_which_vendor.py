"""判断一个 `sk-` 开头的 key 属于哪家平台（只打 GET /models，每个候选 1 个请求）。

**为什么需要**：国内多家平台的 key 都是 `sk-` 前缀，光看字符串分不出是谁家的；
而"填错平台"的报错统一是 401「key 无效」，极易被误判成"key 坏了"。
（2026-09-12 真实踩过：硅基流动的 key 填进 DASHSCOPE_API_KEY，百炼回
 `Incorrect API key provided`，看着像 key 废了，其实只是走错门。）

用法：
    .venv/Scripts/python.exe scripts/probe_which_vendor.py --key "sk-...."
    .venv/Scripts/python.exe scripts/probe_which_vendor.py          # 用 .env 里已有的 key
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CANDIDATES: list[tuple[str, str]] = [
    ("阿里云百炼 · 华北2北京(免费额度地域)", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    ("阿里云百炼 · 国际站(新加坡)", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    ("硅基流动 SiliconFlow", "https://api.siliconflow.cn/v1"),
    ("DeepSeek", "https://api.deepseek.com/v1"),
    ("Moonshot Kimi", "https://api.moonshot.cn/v1"),
    ("智谱 GLM", "https://open.bigmodel.cn/api/paas/v4"),
    ("火山方舟 Ark", "https://ark.cn-beijing.volces.com/api/v3"),
]


def resolve_key() -> str:
    """key 从命令行传入；不传则读 .env——不把 key 写死在脚本里。"""
    ap = argparse.ArgumentParser(description="判断一个 sk- key 属于哪家平台")
    ap.add_argument("--key", help="待判断的 key；不传则用 .env 里已有的")
    args = ap.parse_args()
    if args.key:
        return args.key.strip()
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    from attest.config import load_settings  # noqa: PLC0415 - 仅在需要时导入

    s = load_settings()
    return (s.siliconflow_api_key or s.dashscope_api_key or "").strip()


def probe(url: str, key: str) -> tuple[int, str]:
    req = urllib.request.Request(
        f"{url}/models",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - 固定候选域名
            body = json.loads(resp.read().decode())
            ids = [m.get("id", "") for m in (body.get("data") or [])]
            return resp.status, f"{len(ids)} 个模型：{', '.join(sorted(ids)[:6])}"
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            err = json.loads(raw).get("error") or {}
            msg = err.get("message") or err.get("code") or raw
        except Exception:  # noqa: BLE001
            msg = raw
        return exc.code, str(msg)[:120]
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


def main() -> int:
    key = resolve_key()
    if not key:
        print("✗ 没有可用的 key：用 --key 传入，或在 .env 里填 SILICONFLOW_API_KEY")
        return 2
    print(f"待判断 key：{key[:6]}…{key[-4:]}（{len(key)} 字符）\n")

    hits: list[str] = []
    for name, url in CANDIDATES:
        status, detail = probe(url, key)
        print(f"{status:>5}  {name:38s} {detail} {'✅ 命中' if status == 200 else ''}")
        if status == 200:
            hits.append(name)

    print()
    if hits:
        print("归属：" + "、".join(hits))
    else:
        print("归属：以上候选全部拒绝。请按这个顺序排查——")
        print("  1) **复制不完整/手打错字符**（最常见）：控制台的 key 是打码显示的（sk-abc****xyz），")
        print("     完整值**只在创建那一刻显示一次**。请用「复制」按钮取，别照着截图手打。")
        print("  2) **刚被重置**：重置会让旧 key 立刻失效，用新生成的那个。")
        print("  3) **属于其他平台**：把平台名告诉我，我加进候选列表再判断。")
        print("  对照：硅基流动 code 30014 = Token is invalid（key 问题）；")
        print("        30001 = balance insufficient（key 没问题，是账户没钱）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
