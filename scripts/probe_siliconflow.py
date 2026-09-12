"""硅基流动：确认可用模型 + 实测免费档能不能撑这个项目。

只发 4 个请求：GET /models + 1 次免费档 chat + 1 次 embedding + 1 次 rerank。
目的是拿到**实测**结论而不是"听说免费"：
  - 免费档小模型能不能稳定吐结构化 JSON（planner 靠它）
  - embedding 维度是多少（换维度必须重建索引 + 重标矛盾阈值）
  - rerank 模型在不在（在的话 T3.2 的精排就从"离线词典兜底"升级成真模型）
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from attest.config import load_settings  # noqa: E402

BASE = "https://api.siliconflow.cn/v1"
#: 免费档候选（第三方资料口径，此处用**实测**验证到底通不通）
FREE_CHAT = "Qwen/Qwen2.5-7B-Instruct"
FREE_CHAT_ALT = "THUDM/glm-4-9b-chat"
EMBED_CANDIDATES = ["BAAI/bge-m3", "BAAI/bge-large-zh-v1.5", "Qwen/Qwen3-Embedding-0.6B"]
RERANK_CANDIDATE = "BAAI/bge-reranker-v2-m3"


def call(path: str, key: str, payload: dict | None = None, timeout: int = 90):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode() if payload else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST" if payload else "GET",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode()), time.perf_counter() - t0
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            err = json.loads(raw)
        except Exception:  # noqa: BLE001
            err = raw
        return exc.code, err, time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0


def main() -> int:
    key = load_settings().dashscope_api_key or ""
    if not key:
        print("✗ key 为空")
        return 2

    status, body, dt = call("/models", key)
    ids = [m.get("id", "") for m in (body.get("data") or [])] if isinstance(body, dict) else []
    print(f"① GET /models → {status}（{dt:.1f}s），{len(ids)} 个模型")
    for probe in [FREE_CHAT, FREE_CHAT_ALT, RERANK_CANDIDATE, *EMBED_CANDIDATES, "deepseek-ai/DeepSeek-V3", "Qwen/Qwen3-235B-A22B-Instruct-2507"]:
        print(f"    {'✓' if probe in ids else '✗'} {probe}")
    if ids:
        print("    全部：")
        for i in range(0, len(ids), 4):
            print("      " + ", ".join(ids[i : i + 4]))

    # ② 免费档 chat：要它吐 JSON（planner 的关键能力）
    status, body, dt = call(
        "/chat/completions",
        key,
        {
            "model": FREE_CHAT,
            "messages": [
                {"role": "system", "content": "你是规划器，只输出 JSON，格式：{\"outlines\":[\"...\"]}，不要额外文字。"},
                {"role": "user", "content": "调研企业知识库 Agent 平台市场，给出 3 个大纲章节名。"},
            ],
            "max_tokens": 300,
        },
    )
    if status == 200 and isinstance(body, dict):
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
        usage = body.get("usage") or {}
        print(f"\n② chat[{FREE_CHAT}] → 200（{dt:.1f}s，token={usage.get('total_tokens')}）")
        print("    原始回复：" + text.strip().replace("\n", " ")[:200])
        try:
            obj = json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
            print(f"    结构化解析：✓ {list(obj.keys())}")
        except Exception as exc:  # noqa: BLE001
            print(f"    结构化解析：✗ {type(exc).__name__}（真实调用时 planner 需要兜底解析）")
    else:
        print(f"\n② chat[{FREE_CHAT}] → {status}：{str(body)[:200]}")

    # ③ embedding
    for model in EMBED_CANDIDATES:
        status, body, dt = call("/embeddings", key, {"model": model, "input": ["连通性探针"]})
        if status == 200 and isinstance(body, dict):
            vec = (body.get("data") or [{}])[0].get("embedding") or []
            print(f"③ embed[{model}] → 200（{dt:.1f}s），维度={len(vec)}")
            break
        print(f"③ embed[{model}] → {status}：{str(body)[:120]}")

    # ④ rerank
    status, body, dt = call(
        "/rerank",
        key,
        {"model": RERANK_CANDIDATE, "query": "市场规模", "documents": ["2025 年市场规模约 180 亿元", "今天天气不错"], "top_n": 2},
    )
    if status == 200:
        print(f"④ rerank[{RERANK_CANDIDATE}] → 200（{dt:.1f}s）：{str(body)[:160]}")
    else:
        print(f"④ rerank[{RERANK_CANDIDATE}] → {status}：{str(body)[:160]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
