"""T7.1 · 语料覆盖自检：`python -m eval.coverage_check`

**为什么需要它**：离线 fixture 是 mock 检索的唯一数据源。fixture 的 keywords 与
评测题 query 用词一旦不匹配，检索就会走到"兜底补齐"分支——链路不断，但
**证据集里混入了不相关文档**，后果是：凭空造出矛盾、把真实缺口误判成已覆盖。
这类问题**不会报错**，只会让指标悄悄失真。

所以这里把"query ↔ 语料"的匹配情况显式量出来：
  - 主题路由结果（是否识别出主题，还是回退全库）
  - 真实命中条数 / 兜底条数（兜底 = 该 query 在语料里覆盖不足）
  - 零分条数（零分是被下游排除的，等于该题证据不足）

退出码：无 WARN 及以上 0；有覆盖不足 1（便于 CI / tag 前拦住）。

用法：
    .venv/Scripts/python.exe -m eval.coverage_check
    .venv/Scripts/python.exe -m eval.coverage_check --dataset eval/datasets.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from attest.logging import setup_logging  # noqa: E402
from attest.retrieval.mock_search import FixtureSearchClient  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent


def check_query(client: FixtureSearchClient, query: str, *, max_results: int = 5) -> dict:
    """对单条 query 跑检索，返回覆盖情况。"""
    _pool, route_note, matched = client._route(query)
    results = client.search(query, max_results=max_results)
    real = [r for r in results if r.score > 0]
    zero = [r for r in results if r.score == 0]
    return {
        "route": route_note,
        "returned": len(results),
        "real": len(real),
        "fallback": len(zero),
        # matched=False 表示"主题无法判别" → search 会把整池结果都置 0 分（不可信）
        "degraded": (not matched) or route_note.startswith("全库"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="评测集 ↔ 离线语料 覆盖自检")
    ap.add_argument("--dataset", default=str(EVAL_DIR / "datasets.yaml"))
    ap.add_argument("--fixtures", default=None, help="fixture 目录（默认取 settings.fixture_dir/web）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    setup_logging("WARNING")  # 走查时不需要节点日志刷屏

    if args.fixtures:
        fixture_dir = Path(args.fixtures)
    else:
        from attest.config import Settings

        fixture_dir = Settings().fixture_dir / "web"

    client = FixtureSearchClient(fixture_dir=fixture_dir)
    data = yaml.safe_load(Path(args.dataset).read_text(encoding="utf-8"))
    cases = data.get("cases") or []

    print("=" * 92)
    print(f"语料覆盖自检 · {len(cases)} 题 · fixture={fixture_dir}")
    print("=" * 92)
    print(f"{'id':<5}{'kind':<9}{'route':<24}{'命中':>4}{'兜底':>5}  判定")
    print("-" * 92)

    problems: list[tuple[str, str]] = []
    direct_ids = set((data.get("meta") or {}).get("direct_cases") or [])

    for case in cases:
        cid = case["id"]
        kind = case.get("kind", "?")
        # 直答题不检索，跳过（否则会被误报为覆盖不足）
        if cid in direct_ids or case.get("expect_route") == "direct":
            print(f"{cid:<5}{kind:<9}{'（直答，不检索）':<24}{'-':>4}{'-':>5}  SKIP")
            continue

        info = check_query(client, case["query"])
        if info["degraded"]:
            verdict = "⚠️ 主题路由回退全库"
            problems.append((cid, verdict))
        elif info["fallback"] > 0:
            verdict = f"⚠️ 覆盖不足（兜底 {info['fallback']} 条）"
            problems.append((cid, verdict))
        elif info["real"] < 2:
            verdict = "⚠️ 真实命中偏少（<2）"
            problems.append((cid, verdict))
        else:
            verdict = "OK"

        print(
            f"{cid:<5}{kind:<9}{info['route']:<24}{info['real']:>4}{info['fallback']:>5}  {verdict}"
        )

    print("-" * 92)
    if problems:
        print(f"发现 {len(problems)} 处覆盖问题：")
        for cid, msg in problems:
            print(f"  {cid}: {msg}")
        print(
            "\n修法：给对应 fixture 记录的 keywords 补上该 query 的自然用词"
            "（真实检索正文本就含查询词），或调整 query 表述。不要靠调检索阈值绕过。"
        )
        return 1

    print("全部题目语料覆盖充足（无兜底、主题路由正常）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
