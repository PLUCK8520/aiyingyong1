"""T3.1 · 本地知识库导入：md / txt / pdf → 语义分块 → embedding → **Chroma docs collection**。

三个 Chroma 静默坑（P-1 实测①，任务清单 T3.1）在本脚本里全部落地：
  ① 显式 `embedding_function=None` + 自传向量（否则首调静默下载 ~200MB onnx 模型）；
  ② 单例 `PersistentClient`（不支持多进程并发）；
  ③ `ANONYMIZED_TELEMETRY=False`。

`embedder` + `dim` 写进 collection 元数据：**换 embedding 模型必须全量重建索引**，
不一致时如实拒绝并提示 `--rebuild`（NFR-11）。脚本末尾做**回环校验**：
拿一个 chunk 的正文片段当查询，看它自己能否被召回——这是索引可用性的最低证据。

用法：
    .venv/Scripts/python.exe scripts/ingest_local.py                 # 增量（库已存在则校验）
    .venv/Scripts/python.exe scripts/ingest_local.py --rebuild       # 换了 embedder 后强制重建
    .venv/Scripts/python.exe scripts/ingest_local.py --docs D:/kb --chroma data/index/chroma
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from attest.config import load_settings  # noqa: E402
from attest.llm.gateway import LLMGateway  # noqa: E402
from attest.logging import setup_logging  # noqa: E402
from attest.retrieval.local_search import LocalSearchClient, build_local_index  # noqa: E402
from attest.retrieval.rerank import LexicalReranker  # noqa: E402
from attest.retrieval.stores import ChromaVectorStore  # noqa: E402
from attest.retrieval.text import iter_documents  # noqa: E402
from attest.trace.events import TraceWriter  # noqa: E402


def onnx_cache_dir() -> Path:
    return Path.home() / ".cache" / "chroma" / "onnx_models"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Attest（质证）· 本地知识库导入（Chroma）")
    ap.add_argument("--docs", help="文档目录（默认取配置 ATTEST_LOCAL_DOCS）")
    ap.add_argument("--chroma", help="Chroma 持久目录（默认取配置 ATTEST_CHROMA_DIR）")
    ap.add_argument("--rebuild", action="store_true", help="删除已有索引后全量重建")
    args = ap.parse_args(argv)

    setup_logging("INFO")
    settings = load_settings()
    docs_dir = Path(args.docs) if args.docs else settings.local_docs_dir
    chroma_dir = Path(args.chroma) if args.chroma else settings.local_chroma_dir

    files = iter_documents(docs_dir)
    if not files:
        print(f"[失败] 文档目录里没有可导入的文件（md/txt/pdf）：{docs_dir}")
        return 1

    print("=" * 78)
    print(f"Attest（质证）· 本地知识库导入（Chroma）")
    print(f"文档目录：{docs_dir}（{len(files)} 个文件）")
    print(f"索引目录：{chroma_dir}")
    print(f"embedder：{settings.model_embed} | LLM 模式：{settings.llm_mode}")
    print("=" * 78)

    if args.rebuild and chroma_dir.exists():
        shutil.rmtree(chroma_dir)
        print("[重建] 已删除旧索引目录")

    cache_before = onnx_cache_dir().exists()

    trace = TraceWriter(path=settings.trace_dir / "trace.jsonl", run_id="ingest")
    gateway = LLMGateway(settings=settings, trace=trace)

    probe, _, _ = gateway.embed(["dimension probe"], task="dim_probe")
    dim = len(probe[0]) if probe else 0
    embedder = "mock-hashing" if settings.llm_mode == "mock" else settings.model_embed
    print(f"embedding 维度（实测）：{dim}")

    store = ChromaVectorStore(
        path=chroma_dir, collection="docs", embedder=embedder, dim=dim, rebuild=args.rebuild
    )
    before = store.count()

    index = build_local_index(
        docs_dir,
        lambda texts: gateway.embed(list(texts), task="ingest")[0],
        store=store,
        embedder=embedder,
        dim=dim,
        chunk_chars=settings.local_chunk_chars,
        chunk_overlap=settings.local_chunk_overlap,
    )
    trace.close()

    # ---------------- 回环校验：chunk 能否被自己的正文召回 ----------------
    client = LocalSearchClient(index=index, embed_fn=lambda t: gateway.embed(list(t), task="verify")[0],
                              reranker=LexicalReranker())
    probe_chunk = index.documents[0] if index.documents else ""
    probe_query = probe_chunk[:40]
    hits = client.search(probe_query, max_results=3)
    hit_ids = [h.url.rsplit("#", 1)[-1] for h in hits]
    self_hit = index.ids[0] in hit_ids if index.ids else False

    cache_after = onnx_cache_dir().exists()
    downloaded = (not cache_before) and cache_after

    print("-" * 78)
    print(f"导入前 collection 条目：{before} → 导入后：{store.count()}（本次 chunk：{len(index.ids)}）")
    print(f"回环校验：查询 {probe_query!r}")
    print(f"  top3 = {hit_ids}")
    print(f"  自身命中：{'是' if self_hit else '否'}")
    print(f"onnx 缓存目录：{onnx_cache_dir()}（before={cache_before} after={cache_after}）")
    if downloaded:
        print("  [警告] 出现了 onnx 模型缓存目录 → 可能触发了静默下载！")
    print("=" * 78)

    ok = store.count() > 0 and self_hit and not downloaded
    print("RESULT:", "PASS" if ok else "FAIL")
    if not ok:
        if not self_hit:
            print("  失败原因：回环校验未召回自身——索引或检索链路有问题")
        if downloaded:
            print("  失败原因：疑似触发了 Chroma 默认 embedding 的静默下载")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
