"""Paired local A/B for the v19 writer prompt versus the fact-parity variant.

The expensive visual observations are produced once per clip and reused by both
writers.  This isolates FACT_PARITY from frame sampling and observer randomness.
The script is development-only; the Docker image copies only ``app/``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import urlparse

import httpx

from app import ensemble as E
from app import models as M
from app import pipeline as P
from app.models import FALLBACK_CAPTIONS, REQUIRED_STYLES, normalize_captions, parse_tasks


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    captions = [
        caption
        for row in results
        for caption in row.get("captions", {}).values()
        if isinstance(caption, str)
    ]
    words = [len(caption.split()) for caption in captions]
    chars = [len(caption) for caption in captions]
    static_fallbacks = set(FALLBACK_CAPTIONS.values())
    return {
        "tasks": len(results),
        "captions": len(captions),
        "mean_words": round(mean(words), 2) if words else 0.0,
        "min_words": min(words, default=0),
        "max_words": max(words, default=0),
        "mean_chars": round(mean(chars), 2) if chars else 0.0,
        "static_fallbacks": sum(caption in static_fallbacks for caption in captions),
    }


def _writer_prompt(fact_parity: bool) -> str:
    old = E.FACT_PARITY
    try:
        E.FACT_PARITY = fact_parity
        return E._writer_system_prompt()
    finally:
        E.FACT_PARITY = old


async def _run_task(
    client: httpx.AsyncClient,
    task: dict[str, Any],
    *,
    observer_models: list[str],
    writer_model: str,
    observer_max_tokens: int,
    writer_max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    task_id = str(task["task_id"])
    styles = list(task.get("styles") or REQUIRED_STYLES)
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix=f"ab-{task_id}-") as tmp:
        workdir = Path(tmp)
        video = await P._download(str(task["video_url"]), workdir / "clip.mp4")
        frames = P._extract_keyframes(video, workdir, P.NUM_FRAMES, P.FRAME_MAX_EDGE)
        content = E._frames_content(frames)

        async def observe(model: str) -> tuple[str, list[str]]:
            raw = await E._call(
                client,
                model,
                E.OBSERVE_SYSTEM,
                content,
                observer_max_tokens,
                temperature=0.5,
            )
            return model, E._parse_list(raw)

        observations = await asyncio.gather(*(observe(model) for model in observer_models))
        blocks = [
            f"### observer-{index + 1} ({len(details)} details):\n"
            + "\n".join(f"- {detail}" for detail in details)
            for index, (_, details) in enumerate(observations)
            if details
        ]
        if not blocks:
            raise RuntimeError(f"{task_id}: every observer returned an empty list")

        writer_content = (
            "Independent observation lists from several vision models for ONE clip. "
            "Cross-reference and write the four captions.\n\n" + "\n\n".join(blocks)
        )

        raw_outputs: dict[str, dict[str, str]] = {}
        normalized_outputs: dict[str, dict[str, str]] = {}
        for arm, parity in (("control", False), ("candidate", True)):
            raw = await E._call(
                client,
                writer_model,
                _writer_prompt(parity),
                writer_content,
                writer_max_tokens,
                temperature=0.0,
            )
            parsed = {key: str(value) for key, value in E._parse_obj(raw).items()}
            raw_outputs[arm] = parsed
            normalized_outputs[arm] = normalize_captions(parsed, styles)

    control = {"task_id": task_id, "captions": normalized_outputs["control"]}
    candidate = {"task_id": task_id, "captions": normalized_outputs["candidate"]}
    audit = {
        "task_id": task_id,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "frame_count": len(frames),
        "observations": [details for _, details in observations],
        "raw_outputs": raw_outputs,
        "normalized_outputs": normalized_outputs,
    }
    return control, candidate, audit


async def _amain(args: argparse.Namespace) -> int:
    raw_tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    tasks = parse_tasks(raw_tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    observer_models = [f"local/qwen-observer-{index + 1}" for index in range(args.observers)]

    endpoint_host = urlparse(E.OR_URL).hostname
    if not args.allow_remote and endpoint_host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError(
            f"Refusing non-local endpoint {E.OR_URL!r}; pass --allow-remote explicitly"
        )

    profile = {
        "MAX_CAPTION_CHARS": M.MAX_CAPTION_CHARS,
        "NUM_FRAMES": P.NUM_FRAMES,
        "FRAME_MAX_EDGE": P.FRAME_MAX_EDGE,
        "SCENE_DETECT_ENABLED": P.SCENE_DETECT_ENABLED,
        "TIMESTAMP_FRAMES": os.environ.get("TIMESTAMP_FRAMES", "0") != "0",
        "STYLE_EXEMPLARS": E.EXEMPLARS,
        "STRICT_GROUNDING": E.STRICT_GROUNDING,
        "ENSEMBLE_CONCISE": E.CONCISE,
    }
    expected_profile = {
        "MAX_CAPTION_CHARS": 1600,
        "NUM_FRAMES": 10,
        "FRAME_MAX_EDGE": 896,
        "SCENE_DETECT_ENABLED": False,
        "TIMESTAMP_FRAMES": False,
        "STYLE_EXEMPLARS": True,
        "STRICT_GROUNDING": False,
        "ENSEMBLE_CONCISE": False,
    }
    drift = {
        key: {"expected": expected_profile[key], "actual": value}
        for key, value in profile.items()
        if value != expected_profile[key]
    }
    if drift and not args.allow_profile_drift:
        raise RuntimeError(
            f"Refusing non-v19 benchmark profile: {json.dumps(drift, sort_keys=True)}"
        )

    control: list[dict[str, Any]] = []
    candidate: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> dict[str, Any]:
        (args.output_dir / "control_results.json").write_text(
            json.dumps(control, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.output_dir / "candidate_results.json").write_text(
            json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report = {
            "tasks": str(args.tasks),
            "api_base": E.OR_URL,
            "num_frames": P.NUM_FRAMES,
            "frame_max_edge": P.FRAME_MAX_EDGE,
            "observers": args.observers,
            "profile": profile,
            "control": _summary(control),
            "candidate": _summary(candidate),
            "audits": audits,
        }
        (args.output_dir / "ab_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report

    async with httpx.AsyncClient(timeout=httpx.Timeout(args.http_timeout)) as client:
        for task in tasks:
            left, right, audit = await _run_task(
                client,
                task,
                observer_models=observer_models,
                writer_model="local/qwen-writer",
                observer_max_tokens=args.observer_max_tokens,
                writer_max_tokens=args.writer_max_tokens,
            )
            control.append(left)
            candidate.append(right)
            audits.append(audit)
            checkpoint()
            print(
                f"{task['task_id']}: {audit['elapsed_s']:.1f}s; "
                f"control={_summary([left])['mean_words']} words; "
                f"candidate={_summary([right])['mean_words']} words",
                flush=True,
            )

    report = checkpoint()
    print(json.dumps({"control": report["control"], "candidate": report["candidate"]}))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=Path("data/sample_tasks.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/fact_parity_ab"))
    parser.add_argument("--observers", type=int, default=2)
    parser.add_argument("--observer-max-tokens", type=int, default=1400)
    parser.add_argument("--writer-max-tokens", type=int, default=2200)
    parser.add_argument("--http-timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N tasks")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Permit a paid/non-local API endpoint (disabled by default)",
    )
    parser.add_argument(
        "--allow-profile-drift",
        action="store_true",
        help="Permit settings that differ from the official v19 baseline",
    )
    args = parser.parse_args()
    if args.observers < 1:
        parser.error("--observers must be >= 1")
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
