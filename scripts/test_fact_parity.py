"""Regression guards for the v19-density fact-parity experiment."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import ensemble  # noqa: E402


def main() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    # v32 is a one-lever experiment on the best official v19 baseline. These
    # settings must not silently drift back to the score-losing v30 profile.
    assert "FACT_PARITY=1" in dockerfile
    assert "MAX_CAPTION_CHARS=1600" in dockerfile
    assert "NUM_FRAMES=10" in dockerfile
    assert "FRAME_MAX_EDGE=896" in dockerfile
    assert (
        "ENSEMBLE_OBSERVERS=openai/gpt-5.5,google/gemini-3.1-pro-preview,"
        "anthropic/claude-opus-4.5"
    ) in dockerfile
    assert "ENSEMBLE_WRITER=anthropic/claude-opus-4.5" in dockerfile

    old_parity = ensemble.FACT_PARITY
    old_concise = ensemble.CONCISE
    old_exemplars = ensemble.EXEMPLARS
    try:
        ensemble.FACT_PARITY = True
        ensemble.CONCISE = False
        ensemble.EXEMPLARS = True
        prompt = ensemble._writer_system_prompt()
    finally:
        ensemble.FACT_PARITY = old_parity
        ensemble.CONCISE = old_concise
        ensemble.EXEMPLARS = old_exemplars

    required = (
        "CONSENSUS FACT SPINE",
        "at least five",
        "not a hard maximum",
        "at least two independent observation lists",
        "same ordered spine facts",
        "one-for-one coverage audit",
        "literal visible technology remains factual scene content",
        "visible subject, state, object, or action",
        "one clearly non-literal",
        "never mention absent or not-visible entities",
        "55-90 words",
    )
    for phrase in required:
        assert phrase.lower() in prompt.lower(), phrase

    assert "ZERO technology words" not in prompt
    assert "40-60 words" not in prompt
    assert "TONE EXAMPLES" in prompt

    ensemble.FACT_PARITY = False
    try:
        control_prompt = ensemble._writer_system_prompt()
    finally:
        ensemble.FACT_PARITY = old_parity
    assert "CONSENSUS FACT SPINE" not in control_prompt

    print("FACT-SPINE TEST OK - v19 profile preserved; confidence-gated parity enabled")


if __name__ == "__main__":
    main()
