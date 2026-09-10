"""T3.10 · 审计判定**单点验证**入口（P3 末闸门）。

用法：
    .venv/Scripts/python.exe -m eval.audit_accuracy

判据（任务清单 T3.10）：准确率 **≥ 70%** 才允许带着该判定策略进 P4；否则必须换混合判定策略。
退出码：达标 0 / 未达标 1（可直接接进 CI 当闸门）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from attest.quality.audit import verify_claim  # noqa: E402

GATE = 0.70
PAIRS_PATH = Path(__file__).resolve().parent / "audit_pairs.yaml"

_LABEL = {"supported": "支持", "partial": "部分", "unsupported": "不支撑"}


def main() -> int:
    data = yaml.safe_load(PAIRS_PATH.read_text(encoding="utf-8"))
    pairs = data.get("pairs") or []
    if not pairs:
        print("验证集为空。")
        return 1

    rows: list[tuple[str, str, str, bool, str]] = []
    for item in pairs:
        got, reason = verify_claim(item["claim"], item["evidence"])
        want = item["verdict"]
        rows.append((item["id"], want, got, got == want, reason))

    ok = sum(1 for _, _, _, hit, _ in rows if hit)
    acc = ok / len(rows)

    print("=" * 78)
    print(f"审计判定单点验证 · {PAIRS_PATH.name} · {len(rows)} 条样本对")
    print("=" * 78)
    print(f"{'ID':<5}{'标注':<8}{'判定':<8}{'一致':<6}理由")
    for pid, want, got, hit, reason in rows:
        print(f"{pid:<5}{_LABEL[want]:<8}{_LABEL[got]:<8}{'✓' if hit else '✗':<6}{reason[:60]}")

    errors = [(pid, want, got) for pid, want, got, hit, _ in rows if not hit]
    print("-" * 78)
    print(f"准确率 = {ok}/{len(rows)} = {acc:.1%}   （闸门 ≥ {GATE:.0%}）")
    if errors:
        print(f"误判 {len(errors)} 条：" + "、".join(f"{p}(标{_LABEL[w]}→判{_LABEL[g]})" for p, w, g in errors))
    print(
        "\n⚠️ 本集 claim/evidence/标签**均由作者构造**，是自洽基线而非独立基准；\n"
        "   真实场景准确率会低于此值。P7 T7.1 的人工标注集才是校准基准。"
    )

    if acc < GATE:
        print(f"\n[未达标] {acc:.1%} < {GATE:.0%} → 按 T3.10 必须更换混合判定策略后再进 P4。")
        return 1
    print(f"\n[达标] {acc:.1%} ≥ {GATE:.0%}：该判定策略可作为 P4 审计器的对照基线继续推进。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
