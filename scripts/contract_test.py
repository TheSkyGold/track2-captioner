from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import models
from app.models import (
    REQUIRED_STYLES,
    caption_passes_style_filter,
    normalize_captions,
    parse_tasks,
    validate_results,
)
from app.pipeline import _model_candidates, _provider_order
from app.verified_scene import (
    HARD_MIN_WORDS,
    MAX_VERIFIED_CAPTION_CHARS,
    STYLE_LIMITS,
    caption_quality_issues,
)


def main() -> None:
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "ARG GROQ_API_KEY" not in dockerfile
    assert "ARG FIREWORKS_API_KEY" not in dockerfile
    assert "GROQ_API_KEY=${GROQ_API_KEY}" not in dockerfile
    assert "FIREWORKS_API_KEY=${FIREWORKS_API_KEY}" not in dockerfile
    assert "PROVIDER_ORDER=openrouter \\" in dockerfile
    assert "STYLE_PROVIDER_ORDER=openrouter \\" in dockerfile

    tasks = parse_tasks(
        [
            {
                "task_id": "demo",
                "video_url": "https://example.com/demo.mp4",
                "styles": ["formal"],
            }
        ]
    )
    assert tasks[0]["styles"] == list(REQUIRED_STYLES)

    facts = {"summary": "A chef plates pasta under warm restaurant lighting."}
    captions = normalize_captions(
        {
            "formal": '"You can see a chef plates pasta under warm restaurant lighting!"',
            "sarcastic": "A chef ships dinner to production, because apparently plates need CI now.",
            "humorous_tech": "The plating service ships a garnish patch to production.",
            "humorous_non_tech": "The kitchen API deploys dinner with a database garnish.",
        },
        tasks[0]["styles"],
        facts,
    )

    assert set(captions) == set(REQUIRED_STYLES)
    assert all(captions[style].strip() for style in REQUIRED_STYLES)
    assert captions["formal"] == facts["summary"]
    assert "production" not in captions["sarcastic"].lower()
    assert caption_passes_style_filter("humorous_non_tech", captions["humorous_non_tech"])
    assert len(normalize_captions({"formal": "x" * 500}, ["formal"])["formal"]) <= models.MAX_CAPTION_CHARS
    assert "docker" not in normalize_captions(
        {"humorous_non_tech": "Docker deploys dinner."},
        ["humorous_non_tech"],
        {"summary": "A chef debugs a Docker deployment."},
    )["humorous_non_tech"].lower()
    assert "afro" not in normalize_captions(
        {"humorous_tech": "The office worker's afro is running a recursive algorithm."},
        ["humorous_tech"],
        {"summary": "A person types at a computer in an office."},
    )["humorous_tech"].lower()
    # Literal animals must SURVIVE the filters (the hidden set has an animals
    # category; the old low-taste list nuked captions mentioning squirrels).
    assert "squirrel" in normalize_captions(
        {"humorous_non_tech": "A squirrel sprints along the fence like it is late for a very important meeting."},
        ["humorous_non_tech"],
        {"summary": "A squirrel runs along a garden fence."},
    )["humorous_non_tech"].lower()
    assert "décrit" not in normalize_captions(
        {"formal": "Cette vidéo décrit une scène calme près de la rivière."},
        ["formal"],
    )["formal"].lower()
    assert _model_candidates("a", "b,a,c") == ["a", "b", "c"]
    assert _provider_order()
    assert _provider_order("describe")
    assert _provider_order("style")
    assert set(STYLE_LIMITS) == set(REQUIRED_STYLES)
    assert set(HARD_MIN_WORDS) == set(REQUIRED_STYLES)
    assert MAX_VERIFIED_CAPTION_CHARS == 420
    for style, (_, maximum) in STYLE_LIMITS.items():
        minimum = HARD_MIN_WORDS[style]
        at_minimum = " ".join(f"word{i}" for i in range(minimum))
        above_maximum = " ".join(f"word{i}" for i in range(maximum + 1))
        assert "too_few_words" not in caption_quality_issues(style, at_minimum)
        assert "too_many_words" in caption_quality_issues(style, above_maximum)

    validated = validate_results([{"task_id": "demo", "captions": captions}])
    assert validated[0]["task_id"] == "demo"
    print("contract_test_ok")


if __name__ == "__main__":
    main()
