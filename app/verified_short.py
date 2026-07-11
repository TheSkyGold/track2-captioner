"""Conservative fact and style gates for the v37 verified-short engine.

The external model is allowed to propose details, but only a small verified
subset can reach the caption writer.  This module intentionally keeps those
gates deterministic and testable.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from app.models import style_filter_reason


log = logging.getLogger("track2.verified_short")

_ALWAYS_DROP_CATEGORIES = {
    "brand",
    "dialogue",
    "identity",
    "location",
    "motive",
    "ocr",
}
_MULTIFRAME_CATEGORIES = {"color", "count", "spatial", "type"}
MAX_VERIFIED_FACTS = 5
STYLE_WORD_LIMITS = {
    "formal": (32, 48),
    "sarcastic": (18, 30),
    "humorous_tech": (18, 30),
    "humorous_non_tech": (18, 32),
}

_DRAFT_INSTRUCTION = """Analyze these sampled frames from one video in chronological order.
Write 2 to 4 short factual sentences describing only the visible subjects, main
action or state, setting, and any important change across the frames. Do not
identify a person, place, brand, or exact text. Do not infer intent, dialogue,
emotion, or off-screen events. Avoid exact counts and uncertain colors or fine
object types. Use cautious generic wording when unsure. Return only the factual
description: no analysis, preamble, bullets, labels, or markdown."""

_VERIFY_INSTRUCTION = """Recheck the draft description against these same ordered video frames.
Correct or remove every unsupported detail. Do not add new details. Keep only
the visible subjects, main action or state, setting, and an important temporal
change if clearly supported. Remove identities, place names, brands, transcribed
text, dialogue, intent, exact counts, and uncertain colors or fine object types.
Return only the corrected description as 2 to 4 short factual sentences. Do not
include analysis, a preamble, bullets, labels, or markdown."""

_STYLE_PERSONAS = {
    "formal": (
        "Write in a professional, objective documentary voice. State only visible facts; "
        "use no joke, opinion, exclamation, or first/second person."
    ),
    "sarcastic": (
        "Use dry, lightly mocking irony through understatement or contrast. Keep the main "
        "visible subject and action clear; use no technical jargon and no exclamation. "
        "Make the irony unmistakable with one dry signal such as naturally, apparently, "
        "clearly, obviously, of course, or a pointed grand-versus-ordinary contrast."
    ),
    "humorous_tech": (
        "Use one clear technology or programming metaphor tied directly to a verified "
        "subject, object, or action. Keep the literal scene accurate and use one crisp joke. "
        "Include at least one literal tech term from API, code, cache, server, software, "
        "database, network, or algorithm."
    ),
    "humorous_non_tech": (
        "Use warm everyday observational humor with one relatable comparison or punchline. "
        "Use no technology or programming jargon."
    ),
}

# Concrete entities that models commonly introduce while trying to make a
# caption vivid.  Tech-metaphor words (server, cache, pipeline...) are omitted
# deliberately because they are figurative vocabulary in humorous_tech.
_GUARDED_OBJECTS = {
    "airplane", "airplanes", "audience", "bicycle", "bicycles", "bird", "birds",
    "boat", "boats", "bus", "buses", "car", "cars", "cat", "cats", "child",
    "children", "coffee", "computer", "computers", "cup", "cups", "dog", "dogs",
    "driver", "drivers", "food", "kitten", "kittens", "laptop", "laptops", "man",
    "men", "monitor", "monitors", "motorcycle", "motorcycles", "mouse", "mug",
    "mugs", "person", "people", "phone", "phones", "puppy", "puppies", "screen",
    "screens", "sign", "signs", "statue", "statues", "tablet", "tablets", "train",
    "trains", "truck", "trucks", "van", "vans", "woman", "women",
}

_SARCASM_MARKERS = (
    "apparently",
    "because",
    "clearly",
    "grand",
    "heroic",
    "naturally",
    "obviously",
    "of course",
    "remarkably",
    "solemn",
    "thrilling",
)

_SPECULATIVE_TERMS = re.compile(
    r"\b(?:afraid|decides?|feels?|hopes?|intends?|terrified|thinks?|wants?)\b",
    re.IGNORECASE,
)
_CORRUPTED_CAMEL_TOKEN = re.compile(r"\b[a-z]{3,}[A-Z][A-Za-z]*\b")
_REPEATED_WORD = re.compile(r"\b([A-Za-z]{3,})\s+\1\b", re.IGNORECASE)


def sanitize_verified_facts(raw_facts: Any) -> list[dict[str, Any]]:
    """Return at most five short, independently verified scene facts.

    Exact identities, locations, brands, dialogue and OCR are deliberately
    discarded even when a model marks them verified: their judge reward is
    small and the observed contradiction rate is high.  Other high-variance
    details such as colors and counts need support from at least three sampled
    frames.  Core subjects/actions/settings need one explicit frame reference.
    """

    if not isinstance(raw_facts, list):
        return []

    clean: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, candidate in enumerate(raw_facts, start=1):
        if not isinstance(candidate, dict):
            continue
        if candidate.get("verified") is not True or candidate.get("conflict") is True:
            continue

        category = str(candidate.get("category", "core")).strip().lower()
        if category in _ALWAYS_DROP_CATEGORIES:
            continue

        support_raw = candidate.get("frame_support", [])
        support = sorted(
            {
                value
                for value in support_raw
                if isinstance(value, int) and not isinstance(value, bool) and value > 0
            }
        ) if isinstance(support_raw, list) else []
        required_support = 3 if category in _MULTIFRAME_CATEGORIES else 1
        if len(support) < required_support:
            continue

        text = " ".join(str(candidate.get("text", "")).split()).strip()
        if not text or len(text.split()) > 18:
            continue
        key = text.casefold().rstrip(".")
        if key in seen:
            continue
        seen.add(key)

        fact_id = str(candidate.get("id", "")).strip() or f"f{index}"
        clean.append(
            {
                "id": fact_id,
                "text": text,
                "category": category,
                "frame_support": support,
            }
        )
        if len(clean) >= MAX_VERIFIED_FACTS:
            break
    return clean


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9'-]*", text.casefold()))


_IRREGULAR_OBJECT_SINGULARS = {
    "children": "child",
    "men": "man",
    "people": "person",
    "women": "woman",
}


def _singular_object(word: str) -> str:
    if word in _IRREGULAR_OBJECT_SINGULARS:
        return _IRREGULAR_OBJECT_SINGULARS[word]
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "ses", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _object_tokens(text: str) -> set[str]:
    return {
        _singular_object(word)
        for word in _tokens(text)
        if word in _GUARDED_OBJECTS
    }


def parse_json_object(text: str) -> dict[str, Any]:
    """Recover the first JSON object from a model response."""

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("model response contains no JSON object")


def facts_from_verified_description(text: str) -> list[dict[str, Any]]:
    """Turn a short verified prose description into fact-locked writer input.

    Kimi's vision endpoint is markedly more reliable when asked for a concise
    description than for a nested JSON schema.  This local conversion restores
    stable fact IDs without another model call or another opportunity to invent
    scene content.
    """

    cleaned = str(text or "").strip()
    if not cleaned:
        return []

    # Accept a JSON wrapper defensively, while keeping prose as the primary and
    # explicitly requested provider contract.
    try:
        wrapped = parse_json_object(cleaned)
    except ValueError:
        wrapped = {}
    if wrapped:
        cleaned = str(
            wrapped.get("description")
            or wrapped.get("caption")
            or wrapped.get("summary")
            or ""
        ).strip()
        if not cleaned:
            return []

    cleaned = re.sub(r"```(?:text|markdown)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    cleaned = re.sub(
        r"^\s*(?:verified|final|factual)?\s*description\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"(?m)^\s*(?:[-*]|\d+[.)])\s*", "", cleaned)

    candidates = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    facts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        sentence = " ".join(candidate.split()).strip(" -\t")
        if not sentence:
            continue
        lowered = sentence.casefold()
        if lowered.startswith(
            (
                "analysis",
                "based on",
                "here is",
                "i can",
                "i will",
                "looking at",
                "the user",
            )
        ):
            continue
        # These constructions are almost always identity, OCR, or branding
        # claims.  Omitting the whole small sentence is safer than attempting a
        # potentially misleading rewrite.
        if re.search(
            r"\b(?:named|called|identified as|logo|brand|reads|says|labelled|labeled)\b",
            sentence,
            flags=re.IGNORECASE,
        ):
            continue

        words = sentence.split()
        if len(words) > 24:
            sentence = " ".join(words[:24]).rstrip(",;:") + "."
        elif sentence[-1:] not in ".!?":
            sentence += "."
        key = sentence.casefold().rstrip(".!?")
        if key in seen:
            continue
        seen.add(key)
        facts.append(
            {
                "id": f"f{len(facts) + 1}",
                "text": sentence,
                "category": "core",
                "frame_support": [],
            }
        )
        if len(facts) >= MAX_VERIFIED_FACTS:
            break
    return facts


def _image_parts(frames: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,"
                + base64.b64encode(frame.read_bytes()).decode("ascii")
            },
        }
        for frame in frames
    ]


def build_draft_payload(frames: list[Path], model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _DRAFT_INSTRUCTION},
                    *_image_parts(frames),
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 220,
        "reasoning_effort": "none",
    }


def build_verification_payload(
    frames: list[Path],
    draft: str,
    model: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{_VERIFY_INSTRUCTION}\n\nDraft description:\n{draft}",
                    },
                    *_image_parts(frames),
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": 220,
        "reasoning_effort": "none",
    }


def build_style_payload(
    style: str,
    facts: list[dict[str, Any]],
    model: str,
    *,
    previous_caption: str = "",
    violations: list[str] | None = None,
) -> dict[str, Any]:
    min_words, max_words = STYLE_WORD_LIMITS.get(style, (20, 60))
    persona = _STYLE_PERSONAS.get(style, _STYLE_PERSONAS["formal"])
    system = (
        f"You write the {style} video caption. {persona} "
        f"Write {min_words}-{max_words} words in one or two sentences. The numbered facts "
        "are the complete literal world of the video: never add a person, animal, object, "
        "place, speech, intent, brand, count, color, or event absent from them. A figurative "
        "comparison must not claim that a new object is actually visible. Never replace a "
        "general fact noun with a more specific noun: keep vehicles as vehicles, buildings "
        "as buildings, and an animal as an animal. Return JSON only: "
        '{"caption":"...","fact_ids":["f1","f2"]}. Include the IDs of every fact used.'
    )
    fact_json = json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
    user = f"Verified facts:\n{fact_json}\n\nWrite the caption now."
    if violations:
        user += (
            f"\n\nPrevious caption:\n{previous_caption}\n"
            f"Fix only these violations: {', '.join(violations)}. "
            "Do not add any new scene fact and return the same JSON schema."
        )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2 if style == "formal" else 0.65,
        "max_tokens": 384 if "gpt-oss" in model.casefold() else 220,
        "reasoning_effort": "low" if "gpt-oss" in model.casefold() else "none",
        "response_format": {"type": "json_object"},
    }


async def acquire_verified_facts(
    frames: list[Path],
    model: str,
    invoke: Callable[[dict[str, Any]], Awaitable[str]],
) -> list[dict[str, Any]]:
    """Draft facts, recheck them against the same frames, then gate them."""

    draft = await invoke(build_draft_payload(frames, model))
    try:
        verified_text = await invoke(build_verification_payload(frames, draft, model))
    except Exception:
        # A short, visually grounded draft retains more video signal than a static
        # generic caption when the second provider call times out.
        verified_text = draft
    return facts_from_verified_description(verified_text)


async def write_verified_caption(
    style: str,
    facts: list[dict[str, Any]],
    model: str,
    invoke: Callable[[dict[str, Any]], Awaitable[str]],
) -> str:
    """Write one style and perform at most one targeted repair call."""

    raw = parse_json_object(await invoke(build_style_payload(style, facts, model)))
    caption = " ".join(str(raw.get("caption", "")).split())
    violations = caption_violations(style, caption, facts)
    known_ids = {str(fact.get("id", "")) for fact in facts}
    used_ids = raw.get("fact_ids", [])
    if not isinstance(used_ids, list) or not used_ids:
        violations.append("missing_fact_ids")
    elif any(str(fact_id) not in known_ids for fact_id in used_ids):
        violations.append("unknown_fact_id")
    violations = list(dict.fromkeys(violations))
    if not violations:
        return caption
    log.info("%s caption needs one repair: %s", style, ",".join(violations))

    repaired = parse_json_object(
        await invoke(
            build_style_payload(
                style,
                facts,
                model,
                previous_caption=caption,
                violations=violations,
            )
        )
    )
    repaired_caption = " ".join(str(repaired.get("caption", "")).split())
    repaired_violations = caption_violations(style, repaired_caption, facts)
    repaired_ids = repaired.get("fact_ids", [])
    if not isinstance(repaired_ids, list) or not repaired_ids:
        repaired_violations.append("missing_fact_ids")
    elif any(str(fact_id) not in known_ids for fact_id in repaired_ids):
        repaired_violations.append("unknown_fact_id")
    if repaired_violations:
        log.warning(
            "%s caption repair rejected: %s; using grounded fallback",
            style,
            ",".join(dict.fromkeys(repaired_violations)),
        )
        return _fallback_from_verified_facts(style, facts)
    return repaired_caption


def _fallback_from_verified_facts(style: str, facts: list[dict[str, Any]]) -> str:
    def shorten_sentence(sentence: str, word_budget: int) -> str:
        words = sentence.split()
        if len(words) <= word_budget:
            return sentence
        # Prefer a complete leading clause over a mid-clause token cut.  Kimi
        # commonly emits `subject/action, wearing ...` or `subject/action; ...`.
        for clause in re.split(r"\s*(?:[,;]|\bwhile\b|\bwhereas\b)\s*", sentence):
            clause = clause.strip().rstrip(".!?")
            clause_words = clause.split()
            if 5 <= len(clause_words) <= word_budget:
                return clause + "."
        # A long intact sentence is preferable to a broken factual fragment;
        # the final 300-character normalizer can still trim at a sentence edge.
        return sentence

    def clipped_fact_base(count: int, word_budget: int) -> str:
        parts: list[str] = []
        used = 0
        for fact in facts[:count]:
            sentence = str(fact.get("text", "")).strip()
            words = sentence.split()
            if not words:
                continue
            if used + len(words) <= word_budget:
                parts.append(sentence)
                used += len(words)
                continue
            if not parts:
                parts.append(shorten_sentence(sentence, word_budget))
            break
        return " ".join(parts).strip()

    max_words = STYLE_WORD_LIMITS.get(style, (20, 50))[1]
    base = clipped_fact_base(3 if style == "formal" else 2, max_words)
    if not base:
        base = "Visible subjects and movement appear in a clearly observable setting."
    if style == "formal":
        min_words = STYLE_WORD_LIMITS["formal"][0]
        if len(base.split()) < min_words:
            padding = (
                "Across the clip, the camera maintains a direct view of the visible "
                "scene as the documented subject and activity remain observable within "
                "their surrounding setting overall."
            )
            room = max_words - len(base.split())
            if len(padding.split()) <= room:
                base = f"{base} {padding}"
        return base
    suffixes = {
        "sarcastic": "Naturally, the whole routine carries itself with monumental importance.",
        "humorous_tech": "The scene runs like a server processing its cache without a crash.",
        "humorous_non_tech": "It is quite a performance for such an ordinary moment.",
    }
    suffix = suffixes.get(style, suffixes["humorous_non_tech"])
    suffix_words = len(suffix.split())
    base = clipped_fact_base(2, max(1, max_words - suffix_words)) or base
    return f"{base} {suffix}"


async def caption_verified_frames(
    frames: list[Path],
    styles: list[str],
    vision_model: str,
    writer_model: str,
    invoke: Callable[[dict[str, Any]], Awaitable[str]],
) -> dict[str, str]:
    """Run draft -> visual verification -> independent style generation."""

    facts = await acquire_verified_facts(frames, vision_model, invoke)
    if not facts:
        raise ValueError("no verified facts available")

    # Two tasks may already run concurrently at the container level.  Capping
    # each task to two style requests avoids eight simultaneous Fireworks
    # connections, which caused intermittent DNS/connect failures in live A/Bs.
    style_semaphore = asyncio.Semaphore(2)

    async def write_one(style: str) -> str:
        async with style_semaphore:
            try:
                return await write_verified_caption(style, facts, writer_model, invoke)
            except Exception as error:
                log.warning(
                    "%s caption writer failed (%s); using grounded fallback",
                    style,
                    type(error).__name__,
                )
                return _fallback_from_verified_facts(style, facts)

    captions = await asyncio.gather(
        *(write_one(style) for style in styles)
    )
    return dict(zip(styles, captions))


def caption_violations(
    style: str,
    caption: str,
    facts: list[dict[str, Any]],
) -> list[str]:
    """Return deterministic violations before a caption can be emitted."""

    violations: list[str] = []
    word_count = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", caption))
    min_words, max_words = STYLE_WORD_LIMITS.get(style, (20, 60))
    if word_count < min_words:
        violations.append("too_short")
    if word_count > max_words:
        violations.append("too_long")

    style_reason = style_filter_reason(style, caption)
    if style_reason != "ok":
        violations.append(style_reason)
    if style == "sarcastic" and not any(
        marker in caption.casefold() for marker in _SARCASM_MARKERS
    ):
        violations.append("missing_sarcastic_signal")
    if _CORRUPTED_CAMEL_TOKEN.search(caption) or _REPEATED_WORD.search(caption):
        violations.append("corrupted_token")
    if _SPECULATIVE_TERMS.search(caption):
        violations.append("unsupported_inference")

    caption_words = _object_tokens(caption)
    fact_words = _object_tokens(
        " ".join(str(fact.get("text", "")) for fact in facts)
    )
    unsupported = sorted(caption_words - fact_words)
    violations.extend(f"unsupported_object:{word}" for word in unsupported)
    return list(dict.fromkeys(violations))
