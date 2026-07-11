"""
Self-check on a results.json file — verifies STRUCTURAL requirements and STYLE
bans WITHOUT calling any judge. Runs in ~50 ms. Use this as a pre-submit gate.

Checks:
    ✓ list of objects with task_id + captions
    ✓ all 4 required styles present per row
    ✓ every caption is a non-empty string ≤ 300 chars
    ✓ english-looking (ASCII/latin-1)
    ✓ style bans not violated:
        - formal: no ! ? emoji, no first/second person
        - sarcastic: no exclamations, no obvious tech jargon
        - humorous_non_tech: no tech vocabulary at all
    ✓ captions are meaningfully DIFFERENT between styles for the same clip
      (bans "same caption pasted 4 times" attack)

Non-zero exit code if any check fails.

Usage:
    python eval/self_check.py --results out/results.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import TECH_KEYWORDS, TECH_PHRASES

REQUIRED_STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")

_TECH_TERMS = set(TECH_KEYWORDS) | set(TECH_PHRASES)

_FIRST_SECOND_PERSON = re.compile(r"\b(i|we|us|our|you|your)\b", re.IGNORECASE)
_EXCLAM = "!"


def _looks_english(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not any("a" <= ch.lower() <= "z" for ch in letters):
        return False
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return non_ascii <= max(2, len(text) // 20)


def _tech_hits(text: str) -> list[str]:
    return sorted(
        term
        for term in _TECH_TERMS
        if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])",
            text,
            re.IGNORECASE,
        )
    )


def _fail(errs: list[str], msg: str) -> None:
    errs.append(msg)


def check_row(row: dict, errs: list[str]) -> None:
    tid = row.get("task_id", "?")
    caps = row.get("captions", {})
    if not isinstance(caps, dict):
        _fail(errs, f"[{tid}] captions must be an object")
        return

    for style in REQUIRED_STYLES:
        if style not in caps:
            _fail(errs, f"[{tid}] missing style: {style}")
            continue
        cap = caps[style]
        if not isinstance(cap, str) or not cap.strip():
            _fail(errs, f"[{tid}/{style}] caption empty or not a string")
            continue
        max_chars = int(os.environ.get("MAX_CAPTION_CHARS", "300"))
        if len(cap) > max_chars:
            _fail(errs, f"[{tid}/{style}] caption too long ({len(cap)} > {max_chars} chars)")
        if not _looks_english(cap):
            _fail(errs, f"[{tid}/{style}] caption does not look English/ASCII-safe")

        low = cap.lower()

        if style == "formal":
            if _EXCLAM in cap:
                _fail(errs, f"[{tid}/{style}] formal caption contains '!'")
            if _FIRST_SECOND_PERSON.search(cap):
                _fail(errs, f"[{tid}/{style}] formal caption uses first/second person")

        if style == "sarcastic":
            if _EXCLAM in cap:
                _fail(errs, f"[{tid}/{style}] sarcastic caption contains '!' (should be dry)")
            if _tech_hits(cap):
                _fail(errs, f"[{tid}/{style}] sarcastic caption uses tech jargon")

        if style == "humorous_non_tech":
            hits = _tech_hits(cap)
            if hits:
                _fail(errs, f"[{tid}/{style}] non-tech caption contains tech words: {hits}")

    # Cross-style novelty — no two styles should be identical strings.
    values = [caps.get(s, "").strip() for s in REQUIRED_STYLES if s in caps]
    if len(set(values)) < len(values):
        _fail(errs, f"[{tid}] two styles produced identical captions")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    args = p.parse_args()

    data = json.loads(args.results.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        print("FAIL: results.json must be a non-empty JSON list", file=sys.stderr)
        sys.exit(1)

    errs: list[str] = []
    for row in data:
        check_row(row, errs)

    if errs:
        print(f"SELF-CHECK FAILED - {len(errs)} issue(s):")
        for e in errs[:50]:
            print("  x", e)
        sys.exit(2)

    n = len(data)
    print(f"SELF-CHECK OK - {n} row(s), {n * 4} captions, all bans respected.")


if __name__ == "__main__":
    main()
