"""Offline regression tests for the v37 verified-short engine."""

from __future__ import annotations

import asyncio
import re
import sys
import tempfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.verified_short import (
    _fallback_from_verified_facts,
    acquire_verified_facts,
    build_draft_payload,
    build_style_payload,
    build_verification_payload,
    caption_verified_frames,
    caption_violations,
    facts_from_verified_description,
    parse_json_object,
    sanitize_verified_facts,
    write_verified_caption,
)
from app import pipeline as P
from app import main as M


def test_unverified_and_high_risk_facts_are_excluded() -> None:
    raw = [
        {
            "id": "f1",
            "text": "An office worker types at a keyboard.",
            "category": "core",
            "frame_support": [1, 3],
            "verified": True,
            "conflict": False,
        },
        {
            "id": "f2",
            "text": "A Babe Ruth statue stands beside the road.",
            "category": "identity",
            "frame_support": [1, 2, 3],
            "verified": True,
            "conflict": False,
        },
        {
            "id": "f3",
            "text": "A sign reads POWELL HILL CHURCH.",
            "category": "ocr",
            "frame_support": [1, 2, 3],
            "verified": True,
            "conflict": False,
        },
        {
            "id": "f4",
            "text": "A coffee mug sits beside the keyboard.",
            "category": "object",
            "frame_support": [2],
            "verified": False,
            "conflict": False,
        },
        {
            "id": "f5",
            "text": "The setting is a private home office.",
            "category": "setting",
            "frame_support": [1, 2],
            "verified": True,
            "conflict": True,
        },
    ]

    clean = sanitize_verified_facts(raw)

    assert [fact["id"] for fact in clean] == ["f1"]
    assert clean[0]["text"] == "An office worker types at a keyboard."


def test_caption_rejects_literal_objects_absent_from_verified_facts() -> None:
    facts = [
        {"id": "f1", "text": "An office worker types at a keyboard."},
        {"id": "f2", "text": "The setting is an open-plan office."},
        {"id": "f3", "text": "Glass partitions and a potted plant are visible."},
    ]
    caption = (
        "The office worker types at the keyboard with a cat on her lap while "
        "a coffee mug waits beside a phone, because apparently deadlines need an audience."
    )

    violations = caption_violations("humorous_non_tech", caption, facts)

    assert "unsupported_object:cat" in violations
    assert "unsupported_object:coffee" in violations
    assert "unsupported_object:mug" in violations
    assert "unsupported_object:phone" in violations


def test_object_gate_accepts_singular_and_plural_forms_of_verified_objects() -> None:
    facts = [
        {"id": "f1", "text": "Cars and buses move along a city street."},
    ]
    caption = (
        "A car passes a bus along the city street while ordinary traffic keeps "
        "moving through the visible urban scene."
    )

    violations = caption_violations("formal", caption, facts)

    assert "unsupported_object:car" not in violations
    assert "unsupported_object:bus" not in violations


def test_color_count_and_precise_type_require_three_frame_supports() -> None:
    raw = [
        {
            "id": "f1",
            "text": "The kitten has orange fur.",
            "category": "color",
            "frame_support": [1, 2, 4],
            "verified": True,
            "conflict": False,
        },
        {
            "id": "f2",
            "text": "A red bus moves along the road.",
            "category": "color",
            "frame_support": [2],
            "verified": True,
            "conflict": False,
        },
        {
            "id": "f3",
            "text": "Exactly seven cars are visible.",
            "category": "count",
            "frame_support": [1, 3],
            "verified": True,
            "conflict": False,
        },
        {
            "id": "f4",
            "text": "The animal is a British Shorthair kitten.",
            "category": "type",
            "frame_support": [1, 2],
            "verified": True,
            "conflict": False,
        },
    ]

    clean = sanitize_verified_facts(raw)

    assert [fact["id"] for fact in clean] == ["f1"]


def test_style_and_length_gates_are_targeted() -> None:
    facts = [
        {"id": "f1", "text": "Cars move along a tree-lined city road."},
        {"id": "f2", "text": "Golden leaves fill the roadside trees."},
    ]
    short = "Cars move beneath golden trees, because commuting needed another grand ceremony."
    non_tech_with_api = (
        "Cars move beneath golden trees while the road API handles the morning crowd "
        "with the confidence of a parade nobody requested."
    )
    tech_without_tech = (
        "Cars move beneath golden trees while the crowded road puts on a surprisingly "
        "patient little show for everyone passing through."
    )
    formal_second_person = (
        "You can see several cars moving along a tree-lined city road as golden leaves "
        "frame the traffic and the surrounding buildings remain visible in the background."
    )

    assert "too_short" in caption_violations("sarcastic", short, facts)
    assert "tech_jargon_banned" in caption_violations(
        "humorous_non_tech", non_tech_with_api, facts
    )
    assert "missing_tech_term" in caption_violations(
        "humorous_tech", tech_without_tech, facts
    )
    assert "first_second_person" in caption_violations(
        "formal", formal_second_person, facts
    )


def test_verified_spine_is_capped_at_five_facts() -> None:
    raw = [
        {
            "id": f"f{index}",
            "text": f"Verified scene fact number {index}.",
            "category": "core",
            "frame_support": [index],
            "verified": True,
            "conflict": False,
        }
        for index in range(1, 8)
    ]

    clean = sanitize_verified_facts(raw)

    assert len(clean) == 5
    assert [fact["id"] for fact in clean] == ["f1", "f2", "f3", "f4", "f5"]


def test_verifier_reuses_the_exact_same_ordered_frames() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frames = []
        for index, payload in enumerate((b"first", b"second", b"third"), start=1):
            frame = Path(tmp) / f"frame-{index}.jpg"
            frame.write_bytes(payload)
            frames.append(frame)

        draft = "A subject is visible."
        first = build_draft_payload(frames, "vision-model")
        second = build_verification_payload(frames, draft, "vision-model")

    def image_urls(payload: dict) -> list[str]:
        content = payload["messages"][0]["content"]
        return [part["image_url"]["url"] for part in content if part["type"] == "image_url"]

    assert image_urls(first) == image_urls(second)
    assert len(image_urls(second)) == 3
    verifier_text = second["messages"][0]["content"][0]["text"]
    assert "A subject is visible." in verifier_text


def test_vision_payload_avoids_kimi_structured_output_mode() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frame = Path(tmp) / "frame.jpg"
        frame.write_bytes(b"frame")
        draft = build_draft_payload([frame], "accounts/fireworks/models/kimi-k2p6")
        verify = build_verification_payload(
            [frame], "A person works at a computer.", "accounts/fireworks/models/kimi-k2p6"
        )

    assert "response_format" not in draft
    assert "response_format" not in verify
    assert len(draft["messages"]) == 1
    assert len(verify["messages"]) == 1
    assert draft["messages"][0]["role"] == "user"
    assert verify["messages"][0]["role"] == "user"
    draft_instruction = draft["messages"][0]["content"][0]["text"]
    verify_instruction = verify["messages"][0]["content"][0]["text"]
    assert "2 to 4 short factual sentences" in draft_instruction
    assert "Return only the corrected description" in verify_instruction
    assert "A person works at a computer." in verify_instruction
    assert "JSON" not in draft_instruction
    assert "JSON" not in verify_instruction
    assert draft["max_tokens"] <= 220
    assert verify["max_tokens"] <= 220
    assert draft["reasoning_effort"] == "none"
    assert verify["reasoning_effort"] == "none"


def test_plain_verified_description_becomes_a_short_fact_spine() -> None:
    description = (
        "A worker types at a desktop computer. "
        "The person sits at a desk in a shared office. "
        "Other desks and glass partitions remain in the background. "
        "The worker continues using the keyboard throughout the clip. "
        "A potted plant is visible beside the workstation. "
        "The final frame still shows the same office setting."
    )

    facts = facts_from_verified_description(description)

    assert len(facts) == 5
    assert [fact["id"] for fact in facts] == ["f1", "f2", "f3", "f4", "f5"]
    assert facts[0]["text"] == "A worker types at a desktop computer."
    assert all(len(fact["text"].split()) <= 24 for fact in facts)


def test_style_payload_is_fact_locked_and_retry_is_targeted() -> None:
    facts = [
        {"id": "f1", "text": "An orange kitten walks toward the camera."},
        {"id": "f2", "text": "Green foliage surrounds a dirt path."},
    ]

    initial = build_style_payload("humorous_tech", facts, "writer-model")
    retry = build_style_payload(
        "humorous_tech",
        facts,
        "writer-model",
        previous_caption="A kitten walks down a path.",
        violations=["too_short", "missing_tech_term"],
    )

    initial_system = initial["messages"][0]["content"]
    initial_user = initial["messages"][1]["content"]
    retry_user = retry["messages"][1]["content"]
    assert "one clear technology or programming metaphor" in initial_system
    assert "API, code, cache, server, software, database, network, or algorithm" in initial_system
    assert "Never replace a general fact noun with a more specific noun" in initial_system
    assert "f1" in initial_user and "f2" in initial_user
    assert "18-30 words" in initial_system
    assert initial["reasoning_effort"] == "none"
    assert retry["reasoning_effort"] == "none"
    gpt_oss = build_style_payload(
        "humorous_tech",
        facts,
        "accounts/fireworks/models/gpt-oss-20b",
    )
    assert gpt_oss["reasoning_effort"] == "low"
    assert gpt_oss["max_tokens"] >= 384
    assert "A kitten walks down a path." in retry_user
    assert "too_short" in retry_user and "missing_tech_term" in retry_user
    assert "Do not add any new scene fact" in retry_user


def test_sarcastic_caption_requires_an_unmistakable_irony_signal() -> None:
    facts = [
        {"id": "f1", "text": "A kitten walks along a dirt path."},
        {"id": "f2", "text": "Green foliage surrounds the path."},
    ]
    merely_descriptive = (
        "A small kitten walks slowly down the dirt path while green foliage "
        "fills the quiet garden background."
    )
    clearly_ironic = (
        "A kitten walks down the dirt path, naturally treating this modest "
        "garden outing as a grand inspection of the foliage."
    )

    assert "missing_sarcastic_signal" in caption_violations(
        "sarcastic", merely_descriptive, facts
    )
    assert "missing_sarcastic_signal" not in caption_violations(
        "sarcastic", clearly_ironic, facts
    )


def test_caption_gate_rejects_corruption_repetition_and_inferred_emotion() -> None:
    facts = [
        {"id": "f1", "text": "A kitten walks along a dirt path."},
        {"id": "f2", "text": "Green foliage surrounds the path."},
    ]

    assert "corrupted_token" in caption_violations(
        "humorous_non_tech",
        "The kitten is marchingC marching along the dirt path like a tiny inspector.",
        facts,
    )
    assert "unsupported_inference" in caption_violations(
        "sarcastic",
        "A kitten walks along the path, clearly leaving the local wildlife terrified.",
        facts,
    )


def test_verifier_output_replaces_the_draft_before_styling() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frame = Path(tmp) / "frame.jpg"
        frame.write_bytes(b"frame")
        responses = [
            "A cat sits beside a coffee mug.",
            "An office worker types at a keyboard.",
        ]
        payloads = []

        async def fake_invoke(payload: dict) -> str:
            payloads.append(payload)
            return responses[len(payloads) - 1]

        facts = asyncio.run(
            acquire_verified_facts([frame], "vision-model", fake_invoke)
        )

    assert len(payloads) == 2
    assert [fact["id"] for fact in facts] == ["f1"]
    assert facts[0]["text"] == "An office worker types at a keyboard."
    assert all("cat" not in fact["text"].lower() for fact in facts)


def test_verifier_failure_falls_back_to_sanitized_draft() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frame = Path(tmp) / "frame.jpg"
        frame.write_bytes(b"frame")
        calls = 0

        async def fake_invoke(payload: dict) -> str:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise TimeoutError("verifier timed out")
            return "A kitten walks along a dirt path through surrounding foliage."

        facts = asyncio.run(
            acquire_verified_facts([frame], "vision-model", fake_invoke)
        )

    assert calls == 2
    assert [fact["id"] for fact in facts] == ["f1"]
    assert facts[0]["text"] == (
        "A kitten walks along a dirt path through surrounding foliage."
    )


def test_style_writer_retries_only_once_with_targeted_violations() -> None:
    facts = [
        {"id": "f1", "text": "An orange kitten walks toward the camera."},
        {"id": "f2", "text": "Green foliage surrounds a dirt path."},
    ]
    responses = [
        {"caption": "A kitten walks along a dirt path.", "fact_ids": ["f1"]},
        {
            "caption": (
                "The orange kitten runs a tiny navigation API through green foliage, "
                "keeping its paws in production while the dirt path handles every request."
            ),
            "fact_ids": ["f1", "f2"],
        },
    ]
    payloads = []

    async def fake_invoke(payload: dict) -> str:
        payloads.append(payload)
        return __import__("json").dumps(responses[len(payloads) - 1])

    caption = asyncio.run(
        write_verified_caption(
            "humorous_tech", facts, "writer-model", fake_invoke
        )
    )

    assert len(payloads) == 2
    assert caption == responses[1]["caption"]
    retry_user = payloads[1]["messages"][1]["content"]
    assert "too_short" in retry_user
    assert "missing_tech_term" in retry_user


def test_double_invalid_style_response_falls_back_without_new_objects() -> None:
    facts = [
        {"id": "f1", "text": "An office worker types at a keyboard."},
        {"id": "f2", "text": "A potted plant stands beside the desk."},
    ]
    bad = {
        "caption": "A cat guards the coffee mug while the worker types at a keyboard.",
        "fact_ids": ["f1", "f2"],
    }
    calls = 0

    async def fake_invoke(payload: dict) -> str:
        nonlocal calls
        calls += 1
        return __import__("json").dumps(bad)

    caption = asyncio.run(
        write_verified_caption("formal", facts, "writer-model", fake_invoke)
    )

    low = caption.lower()
    assert calls == 2
    assert "cat" not in low and "coffee" not in low and "mug" not in low
    assert "office worker" in low and "keyboard" in low


def test_grounded_fallbacks_stay_within_each_style_gate() -> None:
    facts = [
        {
            "id": "f1",
            "text": "Cars and buses move in both directions along a multi-lane city street.",
        },
        {
            "id": "f2",
            "text": "Trees with yellow foliage line the road beside tall background buildings.",
        },
        {
            "id": "f3",
            "text": "Traffic density changes while vehicles continue through the intersection.",
        },
    ]

    for style in ("formal", "sarcastic", "humorous_tech", "humorous_non_tech"):
        caption = _fallback_from_verified_facts(style, facts)
        assert caption_violations(style, caption, facts) == [], (style, caption)


def test_grounded_fallback_never_leaves_a_partial_fact_sentence() -> None:
    facts = [
        {
            "id": "f1",
            "text": "A small fluffy kitten with light orange fur walks forward along a dirt path between leafy bushes and tree trunks.",
        },
        {
            "id": "f2",
            "text": "The kitten moves toward the camera while lifting one front paw.",
        },
    ]

    caption = _fallback_from_verified_facts("humorous_non_tech", facts)

    assert "The kitten." not in caption
    assert caption_violations("humorous_non_tech", caption, facts) == []

    office_facts = [
        {
            "id": "f1",
            "text": "A person with hair styled in a bun sits at a white desk, wearing a light-colored jacket over an orange top.",
        },
        {
            "id": "f2",
            "text": "The person types on a keyboard in front of a large monitor.",
        },
    ]
    tech = _fallback_from_verified_facts("humorous_tech", office_facts)
    assert "jacket over." not in tech
    assert "sits at a white desk." in tech
    assert caption_violations("humorous_tech", tech, office_facts) == []


def test_formal_fallback_never_exceeds_its_48_word_gate() -> None:
    facts = [
        {
            "id": f"f{index}",
            "text": (
                f"Visible subject number {index} continues a clearly observable action "
                "within the same grounded setting across sampled frames."
            ),
        }
        for index in range(1, 4)
    ]

    caption = _fallback_from_verified_facts("formal", facts)

    assert "too_long" not in caption_violations("formal", caption, facts)

    short_facts = [{"id": "f1", "text": "A person types at a desk."}]
    short = _fallback_from_verified_facts("formal", short_facts)
    assert caption_violations("formal", short, short_facts) == []


def test_verified_engine_uses_two_vision_calls_and_four_independent_styles() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frame = Path(tmp) / "frame.jpg"
        frame.write_bytes(b"frame")
        calls = []
        facts_packet = (
            "An orange kitten walks toward the camera. "
            "Green foliage surrounds a dirt path."
        )
        captions = {
            "formal": (
                "An orange kitten walks steadily toward the camera along a dirt path. "
                "Dense green foliage surrounds the route as the animal continues forward "
                "throughout the short outdoor scene in a calm and clearly observable setting."
            ),
            "sarcastic": (
                "An orange kitten advances along the dirt path through green foliage, "
                "clearly treating this tiny outdoor expedition with the solemn importance "
                "of a royal inspection."
            ),
            "humorous_tech": (
                "The orange kitten runs a navigation API through green foliage, keeping "
                "tiny paws in production while the dirt path handles every incoming request."
            ),
            "humorous_non_tech": (
                "An orange kitten strolls along the dirt path through green foliage like "
                "a very small supervisor checking whether the outdoors is behaving properly today."
            ),
        }

        async def fake_invoke(payload: dict) -> str:
            calls.append(payload)
            first_content = payload["messages"][0]["content"]
            if isinstance(first_content, list):
                return facts_packet
            system = first_content
            style = next(name for name in captions if f"the {name} video caption" in system)
            return __import__("json").dumps(
                {"caption": captions[style], "fact_ids": ["f1", "f2"]}
            )

        result = asyncio.run(
            caption_verified_frames(
                [frame],
                ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"],
                "vision-model",
                "writer-model",
                fake_invoke,
            )
        )

    assert len(calls) == 6
    assert [payload["model"] for payload in calls[:2]] == [
        "vision-model",
        "vision-model",
    ]
    assert set(result) == set(captions)
    assert result == captions


def test_json_parser_recovers_one_object_from_model_wrapping() -> None:
    wrapped = (
        "Brief internal note.\n```json\n"
        '{"caption":"A grounded caption.","fact_ids":["f1"]}'
        "\n```\n"
    )

    parsed = parse_json_object(wrapped)

    assert parsed == {"caption": "A grounded caption.", "fact_ids": ["f1"]}


def test_one_style_provider_failure_does_not_zero_the_other_styles() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frame = Path(tmp) / "frame.jpg"
        frame.write_bytes(b"frame")
        packet = (
            "An orange kitten walks along a dirt path. "
            "Green foliage surrounds the kitten."
        )
        good = {
            "sarcastic": (
                "An orange kitten walks along the dirt path through green foliage, "
                "treating this modest outing with all the solemn importance of a grand inspection."
            ),
            "humorous_tech": (
                "The orange kitten runs a navigation API along the dirt path, keeping "
                "tiny paws in production while green foliage handles the scenery requests."
            ),
            "humorous_non_tech": (
                "An orange kitten strolls along the dirt path through green foliage like "
                "a tiny supervisor making sure the outdoors has completed its chores."
            ),
        }

        async def fake_invoke(payload: dict) -> str:
            first_content = payload["messages"][0]["content"]
            if isinstance(first_content, list):
                return packet
            system = first_content
            if "the formal video caption" in system:
                raise RuntimeError("formal writer unavailable")
            style = next(name for name in good if f"the {name} video caption" in system)
            return __import__("json").dumps(
                {"caption": good[style], "fact_ids": ["f1", "f2"]}
            )

        result = asyncio.run(
            caption_verified_frames(
                [frame],
                ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"],
                "vision-model",
                "writer-model",
                fake_invoke,
            )
        )

    assert set(result) == {"formal", "sarcastic", "humorous_tech", "humorous_non_tech"}
    assert "orange kitten" in result["formal"].lower()
    assert result["sarcastic"] == good["sarcastic"]
    assert result["humorous_tech"] == good["humorous_tech"]
    assert result["humorous_non_tech"] == good["humorous_non_tech"]


def test_style_calls_are_limited_to_two_at_a_time() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frame = Path(tmp) / "frame.jpg"
        frame.write_bytes(b"frame")
        description = (
            "A kitten walks along a dirt path. "
            "Green foliage surrounds the visible outdoor scene."
        )
        active = 0
        peak = 0

        async def fake_invoke(payload: dict) -> str:
            nonlocal active, peak
            first_content = payload["messages"][0]["content"]
            if isinstance(first_content, list):
                return description
            style = next(
                name
                for name in ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")
                if f"the {name} video caption" in first_content
            )
            captions = {
                "formal": (
                    "A kitten walks along a dirt path while green foliage surrounds "
                    "the visible outdoor setting throughout the sampled video sequence."
                ),
                "sarcastic": (
                    "A kitten walks along the dirt path, naturally treating this modest "
                    "outing as a grand inspection of the surrounding foliage."
                ),
                "humorous_tech": (
                    "The kitten runs a tiny navigation algorithm along the dirt path "
                    "while green foliage keeps the outdoor cache pleasantly full."
                ),
                "humorous_non_tech": (
                    "A kitten walks along the dirt path like a tiny supervisor making "
                    "sure the surrounding foliage finished its chores."
                ),
            }
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return __import__("json").dumps(
                {"caption": captions[style], "fact_ids": ["f1", "f2"]}
            )

        asyncio.run(
            caption_verified_frames(
                [frame],
                ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"],
                "vision-model",
                "writer-model",
                fake_invoke,
            )
        )

    assert peak == 2


def test_pipeline_flag_executes_verified_path_and_blocks_direct_video_bypass() -> None:
    styles = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
    expected = {
        "formal": (
            "An office worker types at a keyboard in a modern shared office. Glass "
            "partitions and a potted plant remain visible behind the desk throughout the scene."
        ),
        "sarcastic": (
            "An office worker types at a keyboard while glass partitions and a potted "
            "plant witness another thrilling chapter in the history of shared offices."
        ),
        "humorous_tech": (
            "The office worker streams keyboard input into production while glass "
            "partitions and a potted plant keep the shared-office runtime looking stable."
        ),
        "humorous_non_tech": (
            "An office worker types at the keyboard while the potted plant and glass "
            "partitions quietly compete for employee-of-the-month attention."
        ),
    }

    async def run_case() -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            frame = Path(tmp) / "frame.jpg"
            video.write_bytes(b"video")
            frame.write_bytes(b"frame")

            async def fake_download(url: str, dst: Path) -> Path:
                return video

            def fake_extract(*args, **kwargs) -> list[Path]:
                return [frame]

            async def forbidden_direct(*args, **kwargs):
                raise AssertionError("direct video facts bypassed v37 verification")

            async def fake_verified(*args, **kwargs) -> dict[str, str]:
                return dict(expected)

            originals = {
                "flag": getattr(P, "VERIFIED_SHORT_SPINE", None),
                "download": P._download,
                "extract": P._extract_keyframes,
                "direct": P._direct_video_facts,
                "verified": getattr(P, "caption_verified_frames", None),
            }
            had_flag = hasattr(P, "VERIFIED_SHORT_SPINE")
            had_verified = hasattr(P, "caption_verified_frames")
            try:
                P.VERIFIED_SHORT_SPINE = True
                P._download = fake_download
                P._extract_keyframes = fake_extract
                P._direct_video_facts = forbidden_direct
                P.caption_verified_frames = fake_verified
                return await P.caption_one_video("https://example.com/clip.mp4", styles)
            finally:
                if had_flag:
                    P.VERIFIED_SHORT_SPINE = originals["flag"]
                else:
                    delattr(P, "VERIFIED_SHORT_SPINE")
                P._download = originals["download"]
                P._extract_keyframes = originals["extract"]
                P._direct_video_facts = originals["direct"]
                if had_verified:
                    P.caption_verified_frames = originals["verified"]
                else:
                    delattr(P, "caption_verified_frames")

    result = asyncio.run(run_case())

    assert result == expected


def test_verified_failure_uses_the_proven_legacy_pipeline() -> None:
    styles = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
    expected = {
        "formal": (
            "A kitten walks steadily along a dirt path through green foliage, remaining "
            "clearly visible as the quiet outdoor scene continues in natural daylight."
        ),
        "sarcastic": "A kitten inspects the dirt path through green foliage with the gravity this tiny expedition obviously deserves.",
        "humorous_tech": "The kitten runs a navigation API along the dirt path while green foliage keeps the outdoor runtime in production.",
        "humorous_non_tech": "A kitten walks along the dirt path like a tiny supervisor checking whether the green foliage finished its chores.",
    }

    async def run_case() -> tuple[dict, int, int]:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            frame = Path(tmp) / "frame.jpg"
            video.write_bytes(b"video")
            frame.write_bytes(b"frame")
            verified_calls = 0
            direct_calls = 0

            async def fake_download(url: str, dst: Path) -> Path:
                return video

            def fake_extract(*args, **kwargs) -> list[Path]:
                return [frame]

            async def broken_verified(*args, **kwargs):
                nonlocal verified_calls
                verified_calls += 1
                raise ValueError("no verified facts available")

            async def fake_direct(*args, **kwargs):
                nonlocal direct_calls
                direct_calls += 1
                return {
                    "summary": "A kitten walks along a dirt path through green foliage.",
                    "subjects": ["kitten"],
                    "actions": ["walking"],
                    "setting": "foliage beside a dirt path",
                }

            async def fake_style_all(facts: dict, requested: list[str]) -> dict[str, str]:
                return dict(expected)

            originals = (
                P.VERIFIED_SHORT_SPINE,
                P._download,
                P._extract_keyframes,
                P._direct_video_facts,
                P.caption_verified_frames,
                P._style_all,
            )
            try:
                P.VERIFIED_SHORT_SPINE = True
                P._download = fake_download
                P._extract_keyframes = fake_extract
                P._direct_video_facts = fake_direct
                P.caption_verified_frames = broken_verified
                P._style_all = fake_style_all
                result = await P.caption_one_video(
                    "https://example.com/clip.mp4", styles
                )
            finally:
                (
                    P.VERIFIED_SHORT_SPINE,
                    P._download,
                    P._extract_keyframes,
                    P._direct_video_facts,
                    P.caption_verified_frames,
                    P._style_all,
                ) = originals
            return result, verified_calls, direct_calls

    result, verified_calls, direct_calls = asyncio.run(run_case())

    assert result == expected
    assert verified_calls == 1
    assert direct_calls == 1


def test_docker_profile_runs_the_verified_engine() -> None:
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(
        encoding="utf-8"
    )
    required = {
        "CAPTION_ENGINE": "pipeline",
        "VERIFIED_SHORT_SPINE": "1",
        "VERIFIED_WRITER_MODEL": "accounts/fireworks/models/gpt-oss-20b",
        "VERIFIED_WRITER_FALLBACKS": "accounts/fireworks/models/deepseek-v4-flash",
        "VERIFIED_OPENROUTER_VISION_FALLBACK": "qwen/qwen3-vl-235b-a22b-instruct",
        "VERIFIED_OPENROUTER_WRITER_FALLBACK": "openai/gpt-oss-120b",
        "VERIFIED_HTTP_TIMEOUT": "30",
        "VERIFIED_429_RETRIES": "1",
        "VERIFIED_429_MAX_WAIT_S": "3",
        "NUM_FRAMES": "6",
        "FRAME_MAX_EDGE": "768",
        "AUDIO_TRANSCRIBE_ENABLED": "0",
        "SCENE_DETECT_ENABLED": "0",
        "DETERMINISTIC_FORMAL": "0",
        "EVIDENCE_LOCK_ENABLED": "0",
        "MAX_CAPTION_CHARS": "300",
        "MAX_CONCURRENCY": "2",
        "PER_TASK_TIMEOUT_S": "70",
        "GLOBAL_BUDGET_S": "540",
    }

    missing = sorted(
        f"{key}={value}"
        for key, value in required.items()
        if not re.search(
            rf"(?m)^\s*{re.escape(key)}={re.escape(value)}(?:\s*\\)?\s*$",
            dockerfile,
        )
    )

    assert not missing, f"missing v37 Docker settings: {missing}"


def test_main_prefills_a_complete_results_file_before_network_work() -> None:
    async def run_case() -> tuple[int, list[dict]]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "tasks.json"
            output_path = root / "results.json"
            input_path.write_text(
                __import__("json").dumps(
                    [
                        {
                            "task_id": "one",
                            "video_url": "https://example.com/one.mp4",
                            "styles": ["formal"],
                        },
                        {
                            "task_id": "two",
                            "video_url": "https://example.com/two.mp4",
                            "styles": ["sarcastic"],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            saw_prefill = []

            async def fake_run_one(sem, task: dict) -> dict:
                prefilled = __import__("json").loads(output_path.read_text(encoding="utf-8"))
                saw_prefill.append(prefilled)
                return {
                    "task_id": task["task_id"],
                    "captions": {
                        "formal": "A complete formal result remains grounded in the visible scene.",
                        "sarcastic": "The visible scene proceeds with all the ceremony it clearly requested.",
                        "humorous_tech": "The visible scene enters production while the runtime keeps every grounded detail in its proper queue.",
                        "humorous_non_tech": "The visible scene carries on like a tiny everyday moment enjoying its turn in the spotlight.",
                    },
                }

            originals = (M.INPUT_PATH, M.OUTPUT_PATH, M._run_one)
            try:
                M.INPUT_PATH = input_path
                M.OUTPUT_PATH = output_path
                M._run_one = fake_run_one
                rc = await M._amain()
                final = __import__("json").loads(output_path.read_text(encoding="utf-8"))
            finally:
                M.INPUT_PATH, M.OUTPUT_PATH, M._run_one = originals

        assert len(saw_prefill) == 2
        for snapshot in saw_prefill:
            assert [row["task_id"] for row in snapshot] == ["one", "two"]
            assert all(set(row["captions"]) == {
                "formal", "sarcastic", "humorous_tech", "humorous_non_tech"
            } for row in snapshot)
            assert all(all(value.strip() for value in row["captions"].values()) for row in snapshot)
        return rc, final

    rc, final = asyncio.run(run_case())

    assert rc == 0
    assert [row["task_id"] for row in final] == ["one", "two"]


def test_main_checkpoints_each_completed_task_atomically() -> None:
    async def run_case() -> int:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "tasks.json"
            output_path = root / "results.json"
            input_path.write_text(
                __import__("json").dumps(
                    [
                        {"task_id": "one", "video_url": "https://example.com/one.mp4"},
                        {"task_id": "two", "video_url": "https://example.com/two.mp4"},
                    ]
                ),
                encoding="utf-8",
            )
            first_done = asyncio.Event()
            marker = "Task one completed with a grounded visible result."

            def result(task_id: str, formal: str) -> dict:
                return {
                    "task_id": task_id,
                    "captions": {
                        "formal": formal,
                        "sarcastic": "The scene completes its modest assignment with impressive ceremony.",
                        "humorous_tech": "The scene reaches production while the runtime keeps its grounded details in the queue.",
                        "humorous_non_tech": "The scene finishes its little task like it has been waiting all day for applause.",
                    },
                }

            async def fake_run_one(sem, task: dict) -> dict:
                if task["task_id"] == "one":
                    first_done.set()
                    return result("one", marker)
                await first_done.wait()
                await asyncio.sleep(0.03)
                checkpoint = __import__("json").loads(
                    output_path.read_text(encoding="utf-8")
                )
                assert checkpoint[0]["captions"]["formal"] == marker
                return result("two", "Task two completed with another grounded visible result.")

            originals = (M.INPUT_PATH, M.OUTPUT_PATH, M._run_one)
            try:
                M.INPUT_PATH = input_path
                M.OUTPUT_PATH = output_path
                M._run_one = fake_run_one
                return await M._amain()
            finally:
                M.INPUT_PATH, M.OUTPUT_PATH, M._run_one = originals

    assert asyncio.run(run_case()) == 0


def test_verified_provider_falls_back_to_the_next_model() -> None:
    attempted = []

    async def fake_chat(base_url: str, api_key: str, payload: dict) -> str:
        attempted.append(payload["model"])
        if len(attempted) == 1:
            request = httpx.Request("POST", "https://example.com/chat/completions")
            response = httpx.Response(503, request=request)
            raise httpx.HTTPStatusError("primary unavailable", request=request, response=response)
        return '{"facts":[]}'

    originals = (
        P.FIREWORKS_API_KEY,
        P.OPENROUTER_API_KEY,
        getattr(P, "VERIFIED_WRITER_FALLBACKS", None),
        getattr(P, "_verified_chat_content_at", None),
    )
    had_fallbacks = hasattr(P, "VERIFIED_WRITER_FALLBACKS")
    had_helper = hasattr(P, "_verified_chat_content_at")
    try:
        P.FIREWORKS_API_KEY = "test-key"
        P.OPENROUTER_API_KEY = ""
        P.VERIFIED_WRITER_FALLBACKS = "backup-model"
        P._verified_chat_content_at = fake_chat
        result = asyncio.run(
            P._invoke_verified_payload({"model": "primary-model", "messages": []})
        )
    finally:
        P.FIREWORKS_API_KEY = originals[0]
        P.OPENROUTER_API_KEY = originals[1]
        if had_fallbacks:
            P.VERIFIED_WRITER_FALLBACKS = originals[2]
        else:
            delattr(P, "VERIFIED_WRITER_FALLBACKS")
        if had_helper:
            P._verified_chat_content_at = originals[3]
        else:
            delattr(P, "_verified_chat_content_at")

    assert attempted == ["primary-model", "backup-model"]
    assert result == '{"facts":[]}'


def test_verified_timeout_does_not_repeat_the_same_slow_request() -> None:
    attempted = []

    async def fake_chat(base_url: str, api_key: str, payload: dict) -> str:
        attempted.append(payload["model"])
        request = httpx.Request("POST", "https://example.com/chat/completions")
        raise httpx.ReadTimeout("provider stalled", request=request)

    originals = (
        P.FIREWORKS_API_KEY,
        P.OPENROUTER_API_KEY,
        getattr(P, "VERIFIED_WRITER_FALLBACKS", None),
        getattr(P, "_verified_chat_content_at", None),
    )
    had_fallbacks = hasattr(P, "VERIFIED_WRITER_FALLBACKS")
    had_helper = hasattr(P, "_verified_chat_content_at")
    try:
        P.FIREWORKS_API_KEY = "test-key"
        P.OPENROUTER_API_KEY = ""
        P.VERIFIED_WRITER_FALLBACKS = "backup-model"
        P._verified_chat_content_at = fake_chat
        try:
            asyncio.run(
                P._invoke_verified_payload(
                    {"model": "primary-model", "messages": []}
                )
            )
        except httpx.ReadTimeout:
            pass
        else:
            raise AssertionError("verified timeout was not surfaced")
    finally:
        P.FIREWORKS_API_KEY = originals[0]
        P.OPENROUTER_API_KEY = originals[1]
        if had_fallbacks:
            P.VERIFIED_WRITER_FALLBACKS = originals[2]
        else:
            delattr(P, "VERIFIED_WRITER_FALLBACKS")
        if had_helper:
            P._verified_chat_content_at = originals[3]
        else:
            delattr(P, "_verified_chat_content_at")

    assert attempted == ["primary-model"]


def test_verified_connect_error_gets_one_fast_same_model_retry() -> None:
    attempted = []

    async def fake_chat(base_url: str, api_key: str, payload: dict) -> str:
        attempted.append(payload["model"])
        if len(attempted) == 1:
            request = httpx.Request("POST", "https://example.com/chat/completions")
            raise httpx.ConnectError("temporary DNS failure", request=request)
        return '{"caption":"Recovered response.","fact_ids":["f1"]}'

    originals = (
        P.FIREWORKS_API_KEY,
        P.OPENROUTER_API_KEY,
        getattr(P, "VERIFIED_WRITER_FALLBACKS", None),
        getattr(P, "_verified_chat_content_at", None),
    )
    try:
        P.FIREWORKS_API_KEY = "test-key"
        P.OPENROUTER_API_KEY = ""
        P.VERIFIED_WRITER_FALLBACKS = ""
        P._verified_chat_content_at = fake_chat
        result = asyncio.run(
            P._invoke_verified_payload({"model": "primary-model", "messages": []})
        )
    finally:
        (
            P.FIREWORKS_API_KEY,
            P.OPENROUTER_API_KEY,
            P.VERIFIED_WRITER_FALLBACKS,
            P._verified_chat_content_at,
        ) = originals

    assert attempted == ["primary-model", "primary-model"]
    assert "Recovered response" in result


def test_verified_uses_openrouter_after_fireworks_status_failures() -> None:
    attempted = []

    async def fake_chat(base_url: str, api_key: str, payload: dict) -> str:
        attempted.append((base_url, payload["model"]))
        if base_url == P.FIREWORKS_BASE_URL:
            request = httpx.Request("POST", "https://example.com/chat/completions")
            response = httpx.Response(503, request=request)
            raise httpx.HTTPStatusError(
                "fireworks unavailable", request=request, response=response
            )
        return '{"caption":"OpenRouter recovered the style.","fact_ids":["f1"]}'

    originals = (
        P.FIREWORKS_API_KEY,
        P.OPENROUTER_API_KEY,
        getattr(P, "VERIFIED_WRITER_FALLBACKS", None),
        getattr(P, "VERIFIED_OPENROUTER_WRITER_FALLBACK", None),
        getattr(P, "_verified_chat_content_at", None),
    )
    try:
        P.FIREWORKS_API_KEY = "test-key"
        P.OPENROUTER_API_KEY = "openrouter-key"
        P.VERIFIED_WRITER_FALLBACKS = ""
        P.VERIFIED_OPENROUTER_WRITER_FALLBACK = "openrouter-writer"
        P._verified_chat_content_at = fake_chat
        result = asyncio.run(
            P._invoke_verified_payload({"model": "primary-model", "messages": []})
        )
    finally:
        (
            P.FIREWORKS_API_KEY,
            P.OPENROUTER_API_KEY,
            P.VERIFIED_WRITER_FALLBACKS,
            P.VERIFIED_OPENROUTER_WRITER_FALLBACK,
            P._verified_chat_content_at,
        ) = originals

    assert attempted == [
        (P.FIREWORKS_BASE_URL, "primary-model"),
        (P.OPENROUTER_BASE_URL, "openrouter-writer"),
    ]
    assert "OpenRouter recovered" in result


def test_verified_can_run_with_only_an_openrouter_key() -> None:
    attempted = []

    async def fake_chat(base_url: str, api_key: str, payload: dict) -> str:
        attempted.append((base_url, payload["model"]))
        return '{"caption":"OpenRouter-only response.","fact_ids":["f1"]}'

    originals = (
        P.FIREWORKS_API_KEY,
        P.OPENROUTER_API_KEY,
        P.VERIFIED_OPENROUTER_WRITER_FALLBACK,
        P._verified_chat_content_at,
    )
    try:
        P.FIREWORKS_API_KEY = ""
        P.OPENROUTER_API_KEY = "openrouter-key"
        P.VERIFIED_OPENROUTER_WRITER_FALLBACK = "openrouter-writer"
        P._verified_chat_content_at = fake_chat
        result = asyncio.run(
            P._invoke_verified_payload({"model": "primary-model", "messages": []})
        )
    finally:
        (
            P.FIREWORKS_API_KEY,
            P.OPENROUTER_API_KEY,
            P.VERIFIED_OPENROUTER_WRITER_FALLBACK,
            P._verified_chat_content_at,
        ) = originals

    assert attempted == [(P.OPENROUTER_BASE_URL, "openrouter-writer")]
    assert "OpenRouter-only" in result


def test_invalid_complete_response_moves_to_the_next_model() -> None:
    attempted = []

    async def fake_chat(base_url: str, api_key: str, payload: dict) -> str:
        attempted.append(payload["model"])
        if len(attempted) == 1:
            raise ValueError("verified response rejected: finish_reason=length")
        return '{"caption":"Backup model response.","fact_ids":["f1"]}'

    originals = (
        P.FIREWORKS_API_KEY,
        P.OPENROUTER_API_KEY,
        P.VERIFIED_WRITER_FALLBACKS,
        P._verified_chat_content_at,
    )
    try:
        P.FIREWORKS_API_KEY = "test-key"
        P.OPENROUTER_API_KEY = ""
        P.VERIFIED_WRITER_FALLBACKS = "backup-model"
        P._verified_chat_content_at = fake_chat
        result = asyncio.run(
            P._invoke_verified_payload({"model": "primary-model", "messages": []})
        )
    finally:
        (
            P.FIREWORKS_API_KEY,
            P.OPENROUTER_API_KEY,
            P.VERIFIED_WRITER_FALLBACKS,
            P._verified_chat_content_at,
        ) = originals

    assert attempted == ["primary-model", "backup-model"]
    assert "Backup model" in result


def test_provider_payload_is_normalized_for_deepseek_and_openrouter() -> None:
    base = {
        "model": "accounts/fireworks/models/gpt-oss-20b",
        "messages": [],
        "max_tokens": 384,
        "reasoning_effort": "low",
    }

    deepseek = P._verified_candidate_payload(
        base, "accounts/fireworks/models/deepseek-v4-flash"
    )
    openrouter = P._verified_candidate_payload(
        base, "openai/gpt-oss-120b", openrouter=True
    )

    assert deepseek["reasoning_effort"] == "none"
    assert deepseek["max_tokens"] <= 256
    assert "reasoning_effort" not in openrouter
    assert openrouter["reasoning"] == {"effort": "low", "exclude": True}
    assert openrouter["max_tokens"] >= 512


def test_verified_429_wait_never_consumes_the_task_budget() -> None:
    long_request = httpx.Request("POST", "https://example.com/chat/completions")
    long_response = httpx.Response(
        429, request=long_request, headers={"retry-after": "12"}
    )
    short_response = httpx.Response(
        429, request=long_request, headers={"retry-after": "1"}
    )

    assert P._verified_429_wait(long_response, 0) < 0
    assert 0 <= P._verified_429_wait(short_response, 0) <= 3


def test_truncated_verified_response_is_never_treated_as_facts() -> None:
    body = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": "I am still analyzing the video frames"},
            }
        ]
    }

    try:
        P._verified_response_content(body)
    except ValueError as error:
        assert "finish_reason=length" in str(error)
    else:
        raise AssertionError("truncated response was accepted")


def test_empty_verified_spine_escalates_to_the_legacy_pipeline() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frame = Path(tmp) / "frame.jpg"
        frame.write_bytes(b"frame")

        async def fake_invoke(payload: dict) -> str:
            return ""

        try:
            asyncio.run(
                caption_verified_frames(
                    [frame],
                    ["formal"],
                    "vision-model",
                    "writer-model",
                    fake_invoke,
                )
            )
        except ValueError as error:
            assert "no verified facts" in str(error)
        else:
            raise AssertionError("empty verified spine did not escalate")


def main() -> None:
    test_unverified_and_high_risk_facts_are_excluded()
    test_caption_rejects_literal_objects_absent_from_verified_facts()
    test_object_gate_accepts_singular_and_plural_forms_of_verified_objects()
    test_color_count_and_precise_type_require_three_frame_supports()
    test_style_and_length_gates_are_targeted()
    test_verified_spine_is_capped_at_five_facts()
    test_verifier_reuses_the_exact_same_ordered_frames()
    test_vision_payload_avoids_kimi_structured_output_mode()
    test_plain_verified_description_becomes_a_short_fact_spine()
    test_style_payload_is_fact_locked_and_retry_is_targeted()
    test_sarcastic_caption_requires_an_unmistakable_irony_signal()
    test_caption_gate_rejects_corruption_repetition_and_inferred_emotion()
    test_verifier_output_replaces_the_draft_before_styling()
    test_verifier_failure_falls_back_to_sanitized_draft()
    test_style_writer_retries_only_once_with_targeted_violations()
    test_double_invalid_style_response_falls_back_without_new_objects()
    test_grounded_fallbacks_stay_within_each_style_gate()
    test_grounded_fallback_never_leaves_a_partial_fact_sentence()
    test_formal_fallback_never_exceeds_its_48_word_gate()
    test_verified_engine_uses_two_vision_calls_and_four_independent_styles()
    test_json_parser_recovers_one_object_from_model_wrapping()
    test_one_style_provider_failure_does_not_zero_the_other_styles()
    test_style_calls_are_limited_to_two_at_a_time()
    test_pipeline_flag_executes_verified_path_and_blocks_direct_video_bypass()
    test_verified_failure_uses_the_proven_legacy_pipeline()
    test_docker_profile_runs_the_verified_engine()
    test_main_prefills_a_complete_results_file_before_network_work()
    test_main_checkpoints_each_completed_task_atomically()
    test_verified_provider_falls_back_to_the_next_model()
    test_verified_timeout_does_not_repeat_the_same_slow_request()
    test_verified_connect_error_gets_one_fast_same_model_retry()
    test_verified_uses_openrouter_after_fireworks_status_failures()
    test_verified_can_run_with_only_an_openrouter_key()
    test_invalid_complete_response_moves_to_the_next_model()
    test_provider_payload_is_normalized_for_deepseek_and_openrouter()
    test_verified_429_wait_never_consumes_the_task_budget()
    test_truncated_verified_response_is_never_treated_as_facts()
    test_empty_verified_spine_escalates_to_the_legacy_pipeline()
    print("verified_short_tests_ok")


if __name__ == "__main__":
    main()
