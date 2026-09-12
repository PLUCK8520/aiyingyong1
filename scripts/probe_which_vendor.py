"""判断一个 sk- key 属于哪家平台（只打 GET /models，每个候选 1 个请求）。

为什么需要：国内多家平台的 key 都是 `sk-` 前缀，光看字符串分不出是谁家的；
而"填错平台"的报错统一是 401，很容易被误判成"key 失效"。
本脚本逐个候选端点做**只读**探测，谁回 200 就是谁。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

KEY = "sk-ncpwaglsizquaevriedbtibgizjlkppaarbuyjagtssdbwge"

CANDIDATES: list[tuple[str, str]] = [
    ("阿里云百炼 · 华北2北京(免费额度地域)", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    ("阿里云百炼 · 国际站(新加坡)", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    ("硅基流动 SiliconFlow", "https://api.siliconflow.cn/v1"),
    ("DeepSeek", "https://api.deepseek.com/v1"),
    ("Moonshot Kimi", "https://api.moonshot.cn/v1"),
    ("智谱 GLM", "https://open.bigmodel.cn/api/paas/v4"),
    ("火山方舟 Ark", "https://ark.cn-beijing.volces.com/api/v3"),
]


def probe(url: str) -> tuple[int, str]:
    req = urllib.request.Request(
        f"{url}/models",
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
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
    hits = []
    for name, url in CANDIDATES:
        status, detail = probe(url)
        flag = "✅ 命中" if status == 200 else ""
        print(f"{status:>5}  {name:38s} {detail} {flag}")
        if status == 200:
            hits.append(name)
    print()
    print("归属：" + ("、".join(hits) if hits else "以上候选全部拒绝（key 可能已作废、复制有误，或属于其他平台）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
