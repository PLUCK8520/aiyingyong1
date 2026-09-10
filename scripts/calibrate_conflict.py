"""T3.7 · 矛盾检测聚类阈值的**数据标定**脚本。

任务清单里写的是"矛盾检测聚类阈值未定 → P3 定阈值并写入配置"。这个脚本就是"定"的依据：
把同子问题内的证据两两相似度打出来，看分布，再选一个能覆盖真实同口径证据对、
又不会把无关证据拉进同一个簇的阈值。

⚠️ 关键前提：**阈值与 embedding 模型强绑定**。离线词法向量（mock-hashing）与
text-embedding-v4 的相似度分布完全不同，换模型必须重跑本脚本（NFR-11）。

用法：
    .venv/Scripts/python.exe scripts/calibrate_conflict.py
    .venv/Scripts/python.exe scripts/calibrate_conflict.py --query "调研……"
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from attest.config import load_settings  # noqa: E402
from attest.graph.build import build_context, build_graph, initial_state  # noqa: E402
from attest.logging import setup_logging  # noqa: E402
from attest.quality.conflict import _cosine, group_by_sub_question  # noqa: E402
from attest.retrieval.local_search import clear_local_cache  # noqa: E402

DEFAULT_QUERY = "调研'企业知识库 Agent 平台'市场，按市场规模/竞品/收费模式三部分输出，附溯源链接"

CANDIDATES = (0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="矛盾检测聚类阈值标定")
    ap.add_argument("query", nargs="*", help="用于取证据的调研问题")
    args = ap.parse_args(argv)

    setup_logging("WARNING")
    settings = load_settings()
    settings.ensure_dirs()
    clear_local_cache()

    query = " ".join(args.query).strip() or DEFAULT_QUERY
    ctx = build_context(settings, run_id="calibrate")
    app = build_graph(ctx)
    final = app.invoke(initial_state(query, thread_id="calibrate"))
    ctx.trace.close()

    evidence = list(final.get("evidence") or [])
    if len(evidence) < 2:
        print("证据不足 2 条，无法标定。")
        return 1

    embedder = "mock-hashing" if settings.llm_mode == "mock" else settings.model_embed
    print(f"embedder = {embedder} | 证据 {len(evidence)} 条 | 问题：{query}\n")

    sims: list[float] = []
    for subq, items in group_by_sub_question(evidence).items():
        if len(items) < 2:
            continue
        vecs = ctx.gateway.embed([e.content for e in items], task="calibrate")[0]
        print(f"=== {subq}（{len(items)} 条）===")
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                sim = _cosine(vecs[i], vecs[j])
                sims.append(sim)
                print(f"  {items[i].citation_id} ~ {items[j].citation_id}: {sim:+.3f}")
        print()

    if not sims:
        print("没有可比的证据对。")
        return 1

    print("相似度分布：")
    print(f"  最小 {min(sims):+.3f} / 中位 {statistics.median(sims):+.3f} / 最大 {max(sims):+.3f}")
    print(f"  均值 {statistics.fmean(sims):+.3f} / 标准差 {statistics.pstdev(sims):.3f}\n")

    print("候选阈值下的**单簇首次聚合数**（就地把同一子问题证据按贪心聚类，统计被并进簇的对数）：")
    for th in CANDIDATES:
        kept = sum(1 for s in sims if s >= th)
        print(f"  τ={th:.2f} → 相似度达标的对 {kept}/{len(sims)}")

    print(
        f"\n当前配置 ATTEST_CONFLICT_THRESHOLD = {settings.conflict_cluster_threshold}\n"
        "建议：τ 取到「能覆盖真实同口径证据对」的最小值；过高会漏检矛盾，过低会把无关证据拉进同簇、\n"
        "浪费比对配额（配额本身有 per-cluster / global 上限兜底，所以宁可略低）。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
