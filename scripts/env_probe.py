"""P-1 / T-1.1 依赖探针：逐包 import 并打印实际版本。

用法：
    .venv/Scripts/python.exe scripts/env_probe.py

退出码：全部通过 0；有失败项 1（失败项会在末尾汇总，供 env-report.md 记录）。
"""

from __future__ import annotations

import importlib.metadata as md
import sys
import traceback

# (import 名, distribution 名，可能不同)
PACKAGES: list[tuple[str, str]] = [
    ("langgraph", "langgraph"),
    ("langgraph.checkpoint.sqlite", "langgraph-checkpoint-sqlite"),
    ("chromadb", "chromadb"),
    ("jieba", "jieba"),
    ("rank_bm25", "rank-bm25"),
    ("aiosqlite", "aiosqlite"),
    ("pypdf", "pypdf"),
    ("fastapi", "fastapi"),
    ("pydantic_settings", "pydantic-settings"),
]


def main() -> int:
    print(f"python  : {sys.version}")
    print(f"exe     : {sys.executable}")
    print("-" * 72)

    ok: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []

    for module_name, dist_name in PACKAGES:
        try:
            __import__(module_name)
            try:
                version = md.version(dist_name)
            except md.PackageNotFoundError:
                version = "(未注册版本)"
            ok.append((dist_name, version))
            print(f"[OK]   {dist_name:<32} {version}")
        except Exception as exc:  # noqa: BLE001 - 探针要吞掉任何异常并记录
            failed.append((dist_name, f"{type(exc).__name__}: {exc}"))
            print(f"[FAIL] {dist_name:<32} {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)

    print("-" * 72)
    print(f"通过 {len(ok)}/{len(PACKAGES)}")
    if failed:
        print("失败项：")
        for name, reason in failed:
            print(f"  - {name}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
