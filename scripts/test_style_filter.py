"""Guard: common English words must not be mistaken for tech jargon.

Runs standalone: `python scripts/test_style_filter.py`. Fails loudly if a
regression re-adds a top-frequency word (like the pronoun "it") to the tech
filter, which would silently reject good sarcastic / humorous_non_tech captions.
"""
from __future__ import annotations

from app.models import (
    _has_tech_jargon,
    _has_tech_reference,
    caption_passes_style_filter,
    fallback_caption,
    normalize_captions,
)

# Everyday captions that legitimately use words a bad tech filter over-matches.
CLEAN = [
    "A mesmerizing display of urban efficiency, conveniently sped up to avoid experiencing it.",
    "The cat inspects it carefully before deciding the leaf is beneath its notice.",
    "She loves it here, where the afternoon light does all the work.",
    # substring regressions: 'rat' in laboratory/rather/operation, 'thin' in within
    "A gloved hand transfers green liquid into a multi-well plate within a laboratory.",
    "One appreciates the rather decorative pavement and the orderly vehicle storage on display.",
    # 'thin' as an object adjective must pass (branches, cables, strips)
    "A ginger kitten advances through green bushes and thin branches toward the camera.",
]

# 'race condition' is legit tech humour — bare 'race' must not trip the
# sensitive-appearance filter (racial/ethnicity still do).
RACE_OK = "The hand lifted the cup, triggering a race condition between latte art and the saucer."

# Real tech-humour must still be detected as tech.
TECH = [
    "The traffic scheduler is live in production and every lane files a latency complaint.",
    "Keyboard events stream into production while the potted plant monitors system health.",
]


def demo() -> None:
    for cap in CLEAN:
        assert not _has_tech_jargon(cap), f"false tech-jargon positive: {cap!r}"
        assert caption_passes_style_filter("sarcastic", cap), f"sarcastic rejected: {cap!r}"
        assert caption_passes_style_filter("humorous_non_tech", cap), f"non_tech rejected: {cap!r}"
    for cap in TECH:
        assert _has_tech_reference(cap), f"missed real tech reference: {cap!r}"
        assert caption_passes_style_filter("humorous_tech", cap), f"tech rejected: {cap!r}"
    assert caption_passes_style_filter("humorous_tech", RACE_OK), f"race-condition rejected: {RACE_OK!r}"
    # Regression: substring matching once made 'cat' match 'located', firing
    # the kitten fallback on an earth-from-space clip. Whole words only.
    space_facts = {
        "summary": "A view of earth from space shows city lights located across Asia.",
        "subjects": ["earth", "city lights"],
        "actions": ["rotating"],
    }
    for style in ("sarcastic", "humorous_tech", "humorous_non_tech"):
        fb = fallback_caption(style, space_facts)
        assert "kitten" not in fb.lower(), f"kitten leak on space clip [{style}]: {fb!r}"

    literal_tech = (
        "A developer studies a GPU dashboard in an IDE while a CPU graph rises on screen."
    )
    for style in ("sarcastic", "humorous_non_tech"):
        repaired = normalize_captions({style: literal_tech}, [style])[style]
        assert "dashboard" in repaired.lower(), f"visible dashboard lost [{style}]: {repaired!r}"
        assert "graph" in repaired.lower(), f"visible graph lost [{style}]: {repaired!r}"
        assert not _has_tech_jargon(repaired), f"jargon survived repair [{style}]: {repaired!r}"

    off_scene_tech_joke = (
        "This kitten is a walking Instagram filter doing a tiny TikTok dance through leaves."
    )
    repaired_joke = normalize_captions(
        {"humorous_non_tech": off_scene_tech_joke}, ["humorous_non_tech"]
    )["humorous_non_tech"]
    assert "kitten" in repaired_joke.lower(), repaired_joke
    assert "dance" in repaired_joke.lower(), repaired_joke
    assert "instagram" not in repaired_joke.lower(), repaired_joke
    assert "tiktok" not in repaired_joke.lower(), repaired_joke
    assert not _has_tech_jargon(repaired_joke), repaired_joke

    sensitive_hair = "A woman with an afro types on a keyboard in a modern office."
    for style in ("formal", "humorous_non_tech"):
        repaired_hair = normalize_captions({style: sensitive_hair}, [style])[style]
        assert "keyboard" in repaired_hair.lower(), f"keyboard lost [{style}]: {repaired_hair!r}"
        assert "office" in repaired_hair.lower(), f"office lost [{style}]: {repaired_hair!r}"
        assert "afro" not in repaired_hair.lower(), f"identity label survived [{style}]: {repaired_hair!r}"
        assert caption_passes_style_filter(style, repaired_hair), repaired_hair
    print(f"STYLE-FILTER OK - {len(CLEAN)} clean + {len(TECH)} tech captions classified correctly; no cross-clip fallback leak.")


if __name__ == "__main__":
    demo()
