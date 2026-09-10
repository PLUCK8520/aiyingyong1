"""P-1 / T-1.3 Chroma 冒烟：验证"自传向量"路径不触发 200MB 静默下载。

要点（对应任务清单 T3.1 的三个静默坑）：
  1. 显式 embedding_function=None，自己传向量 —— 否则首调会静默拉 onnxruntime 模型；
  2. PersistentClient 单例（本脚本只建一个 client）；
  3. ANONYMIZED_TELEMETRY=False 关掉默认外发。

用法：
    .venv/Scripts/python.exe scripts/chroma_smoke.py
退出码：通过 0 / 失败 1。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# 必须在 import chromadb 之前关掉埋点
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402

SMOKE_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_smoke"


def onnx_cache_dir() -> Path:
    """Chroma 默认 embedding 的模型缓存目录（用于判断是否发生静默下载）。"""
    return Path.home() / ".cache" / "chroma" / "onnx_models"


def main() -> int:
    print(f"chromadb: {chromadb.__version__}")

    cache = onnx_cache_dir()
    before = cache.exists()

    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    SMOKE_DIR.parent.mkdir(parents=True, exist_ok=True)

    # 单例 client；关闭 telemetry
    client = chromadb.PersistentClient(
        path=str(SMOKE_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )

    # 关键：embedding_function=None，向量全部自传
    col = client.get_or_create_collection(
        name="smoke",
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )

    col.add(
        ids=["d1", "d2", "d3"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
        documents=["苹果", "香蕉", "红富士"],
        metadatas=[{"kind": "fruit"}, {"kind": "fruit"}, {"kind": "fruit"}],
    )

    res = col.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=2, include=["documents", "distances"])
    ids = res["ids"][0]
    docs = res["documents"][0]
    print(f"count      : {col.count()}")
    print(f"query top2 : {list(zip(ids, docs))}")

    after = cache.exists()
    downloaded = (not before) and after
    print(f"onnx cache : {cache} (before={before}, after={after})")
    if downloaded:
        print("[WARN] 检测到 chroma onnx 模型缓存目录被创建，可能发生了静默下载！")

    ok = ids[0] == "d1" and col.count() == 3
    print("RESULT     :", "PASS" if ok and not downloaded else "FAIL")
    return 0 if (ok and not downloaded) else 1


if __name__ == "__main__":
    sys.exit(main())
