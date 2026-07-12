"""Deterministic blind assignment primitives for paired caption evaluation.

This module intentionally stops before judge scoring. It builds the concealed
baseline/candidate mapping that a later evaluator can persist separately from
the anonymous Caption A / Caption B prompt.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Mapping, Sequence


Arm = Literal["baseline", "candidate"]
Assignment = tuple[Arm, Arm]
AssignmentMap = dict[tuple[str, str], Assignment]

REQUIRED_STYLES = (
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
)
CACHE_SCHEMA = "blind-assignment-cache/v1"


def blind_assignment(
    seed: str,
    task_id: str,
    style: str,
    judge_model: str,
) -> Assignment:
    """Return a stable single-pair fallback assignment derived from SHA-256."""
    material = "|".join((seed, task_id, style, judge_model)).encode("utf-8")
    bit = hashlib.sha256(material).digest()[0] & 1
    return (
        ("baseline", "candidate")
        if bit == 0
        else ("candidate", "baseline")
    )


def build_blind_assignments(
    seed: str,
    judge_model: str,
    pairs: Sequence[tuple[str, str]],
) -> AssignmentMap:
    """Build balanced assignments for a complete four-style corpus run."""
    pair_list = list(pairs)
    if not pair_list:
        raise ValueError("complete blind assignment requires at least one pair")
    if len(set(pair_list)) != len(pair_list):
        raise ValueError("blind assignment pairs must be unique")

    expected_styles = set(REQUIRED_STYLES)
    styles_by_clip: dict[str, set[str]] = {}
    for task_id, style in pair_list:
        if style not in expected_styles:
            raise ValueError(f"unknown requested style: {style}")
        styles_by_clip.setdefault(task_id, set()).add(style)
    for task_id, styles in styles_by_clip.items():
        if styles != expected_styles:
            missing = sorted(expected_styles - styles)
            raise ValueError(f"incomplete styles for {task_id}: {missing}")

    clips = sorted(
        styles_by_clip,
        key=lambda task_id: hashlib.sha256(
            f"{seed}|{judge_model}|{task_id}".encode("utf-8")
        ).digest(),
    )
    assignments: AssignmentMap = {}
    for clip_index, task_id in enumerate(clips):
        pattern = (0, 1, 0, 1) if clip_index % 2 == 0 else (1, 0, 1, 0)
        for style_index, style in enumerate(REQUIRED_STYLES):
            assignments[(task_id, style)] = (
                ("baseline", "candidate")
                if pattern[style_index] == 0
                else ("candidate", "baseline")
            )
    return assignments


def judge_caption_prompt(caption_a: str, caption_b: str) -> str:
    """Render only anonymous caption labels and their text for the judge."""
    return f"Caption A:\n{caption_a}\n\nCaption B:\n{caption_b}"


def write_assignment_cache(
    path: Path,
    seed: str,
    judge_model: str,
    assignments: Mapping[tuple[str, str], Assignment],
) -> None:
    """Persist the concealed remapping separately from judge-facing prompts."""
    style_rank = {style: index for index, style in enumerate(REQUIRED_STYLES)}
    records = []
    for (task_id, style), order in sorted(
        assignments.items(),
        key=lambda item: (item[0][0], style_rank.get(item[0][1], len(style_rank))),
    ):
        if order not in {
            ("baseline", "candidate"),
            ("candidate", "baseline"),
        }:
            raise ValueError(f"invalid blind assignment for {task_id}/{style}")
        records.append(
            {
                "caption_a_source": order[0],
                "caption_b_source": order[1],
                "style": style,
                "task_id": task_id,
            }
        )
    payload = {
        "assignments": records,
        "judge_model": judge_model,
        "schema": CACHE_SCHEMA,
        "seed": seed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_assignment_cache(path: Path) -> AssignmentMap:
    """Load a persisted concealed mapping and reject malformed cache records."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != CACHE_SCHEMA:
        raise ValueError("unsupported blind assignment cache schema")
    records = payload.get("assignments")
    if not isinstance(records, list):
        raise ValueError("blind assignment cache must contain an assignments list")

    assignments: AssignmentMap = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("blind assignment cache record must be an object")
        task_id = record.get("task_id")
        style = record.get("style")
        order = (record.get("caption_a_source"), record.get("caption_b_source"))
        if not isinstance(task_id, str) or style not in REQUIRED_STYLES:
            raise ValueError("invalid blind assignment cache key")
        if order not in {
            ("baseline", "candidate"),
            ("candidate", "baseline"),
        }:
            raise ValueError(f"invalid blind assignment cache order for {task_id}/{style}")
        key = (task_id, style)
        if key in assignments:
            raise ValueError(f"duplicate blind assignment cache key: {task_id}/{style}")
        assignments[key] = order  # type: ignore[assignment]
    return assignments
