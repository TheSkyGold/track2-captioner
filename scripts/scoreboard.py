"""Rank local judge score files.

Usage:
    python scripts/scoreboard.py
    python scripts/scoreboard.py --glob "eval/scores_stress*.json"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _score_file(path: Path) -> tuple[str, int, float, float, float] | None:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(rows, list) or not rows:
        return None
    try:
        acc = sum(float(row.get("accuracy", 0)) for row in rows) / len(rows)
        style = sum(float(row.get("style_match", 0)) for row in rows) / len(rows)
    except (TypeError, ValueError):
        return None
    return (str(path), len(rows), acc, style, (acc + style) / 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="eval/scores*.json")
    args = parser.parse_args()

    rows = [
        scored
        for path in Path().glob(args.glob)
        if (scored := _score_file(path)) is not None
    ]
    rows.sort(key=lambda row: row[4], reverse=True)

    print("file\tn\taccuracy\tstyle\tfinal")
    for path, count, acc, style, final in rows:
        print(f"{path}\t{count}\t{acc:.3f}\t{style:.3f}\t{final:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
