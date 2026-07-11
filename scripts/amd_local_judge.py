"""Local multimodal judge helpers for the AMD/Qwen calibration bench."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
import sys
from typing import Any
from urllib.request import Request, urlopen


DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")
DEFAULT_SET_SPECS: list[tuple[str, float | None, str]] = [
    ("v5_base", 0.8908, "out/ab_baseline.json"),
    ("v10_arm", 0.8900, "out/ab_exemplars.json"),
    ("v10_regen", 0.8900, "out/ab_v10.json"),
    ("v7_concise", 0.8400, "out/calib_concise.json"),
    ("v13_gemma", 0.7717, "out/calib_gemma.json"),
    ("v16_grounded", 0.8475, "out/judge_sim/output/results.json"),
    ("v19_gate", 0.8942, "out/v19_gate/output/results.json"),
]


def _clamp(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_score_json(text: str) -> dict[str, Any]:
    """Extract and normalize one strict score object from model output."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("judge response contains no JSON object")
    obj = json.loads(text[start : end + 1])
    for key in ("accuracy", "style_match", "coverage"):
        obj[key] = _clamp(obj[key])
    obj.setdefault("unsupported_claims", [])
    obj.setdefault("missing", [])
    return obj


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[index][1]:
            stop += 1
        average_rank = (index + 1 + stop) / 2
        for offset in range(index, stop):
            ranks[ordered[offset][0]] = average_rank
        index = stop
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    """Return Spearman rank correlation, including average ranks for ties."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have the same length >= 2")
    x = _ranks(left)
    y = _ranks(right)
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        raise ValueError("Spearman correlation is undefined for constant input")
    result = numerator / (dx * dy)
    if abs(1.0 - abs(result)) < 1e-12:
        return math.copysign(1.0, result)
    return result


def build_messages(
    task_id: str,
    style: str,
    caption: str,
    frame_data_urls: list[str],
) -> list[dict[str, Any]]:
    """Build a frame-grounded judge request without exposing known scores."""
    system = (
        "You are a strict video-caption evaluator. Inspect all chronological frames. "
        "Score the caption independently on accuracy, requested style_match, and coverage "
        "of salient subjects, actions, setting, background, and temporal change. Unsupported "
        "specifics are worse than omissions, but missing important visible elements lowers "
        "coverage. Output strict JSON only with keys accuracy, style_match, coverage, "
        "unsupported_claims, missing, and reason. Each numeric score is from 0 to 1."
    )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"Task {task_id}; requested style: {style}; caption:\n{caption}",
        }
    ]
    content.extend(
        {"type": "image_url", "image_url": {"url": data_url}}
        for data_url in frame_data_urls
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def build_batch_messages(
    task_id: str,
    captions: dict[str, str],
    frame_data_urls: list[str],
) -> list[dict[str, Any]]:
    """Build one image-backed request that scores the four styles independently."""
    system = (
        "You are a strict video-caption evaluator. Inspect all chronological frames, then "
        "score each of four captions independently. For every requested style, assess "
        "accuracy, style_match, and coverage of salient subjects, actions, setting, "
        "background, and temporal change. Unsupported specifics are worse than omissions, "
        "but missing important visible elements lowers coverage. Humor and metaphors that "
        "make no literal visual claim do not count as factual errors. Output strict JSON "
        "whose four top-level keys are formal, sarcastic, humorous_tech, and "
        "humorous_non_tech. Each value must contain numeric accuracy, style_match, coverage "
        "from 0 to 1, plus unsupported_claims, missing, and reason."
    )
    ordered = {style: str(captions[style]) for style in STYLES}
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"Task {task_id}. Captions by requested style:\n{json.dumps(ordered)}",
        }
    ]
    content.extend(
        {"type": "image_url", "image_url": {"url": data_url}}
        for data_url in frame_data_urls
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def parse_batch_score_json(text: str) -> dict[str, dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("judge response contains no JSON object")
    obj = json.loads(text[start : end + 1])
    parsed: dict[str, dict[str, Any]] = {}
    for style in STYLES:
        if style not in obj:
            raise ValueError(f"judge response is missing style: {style}")
        parsed[style] = parse_score_json(json.dumps(obj[style]))
    return parsed


def load_caption_sets(
    root: Path,
    specs: list[tuple[str, float | None, str]],
) -> list[dict[str, Any]]:
    """Load named caption result files and retain their optional official score."""
    loaded: list[dict[str, Any]] = []
    for name, official_score, relative_path in specs:
        path = (root / relative_path).resolve()
        rows = json.loads(path.read_text(encoding="utf-8"))
        by_task = {str(row["task_id"]): row["captions"] for row in rows}
        loaded.append(
            {
                "name": name,
                "official_score": official_score,
                "path": path,
                "captions": by_task,
            }
        )
    return loaded


def common_task_ids(
    caption_sets: list[dict[str, Any]], task_urls: dict[str, str]
) -> list[str]:
    """Return tasks present in every result set and in the video URL map."""
    common = set(task_urls)
    for caption_set in caption_sets:
        common &= set(caption_set["captions"])
    return sorted(common)


def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate the official two-axis score and keep coverage as a diagnostic."""
    if not rows:
        raise ValueError("cannot aggregate an empty score list")
    accuracy = mean(float(row["accuracy"]) for row in rows)
    style_match = mean(float(row["style_match"]) for row in rows)
    coverage = mean(float(row["coverage"]) for row in rows)
    return {
        "accuracy": accuracy,
        "style_match": style_match,
        "coverage": coverage,
        "final": (accuracy + style_match) / 2,
    }


def parse_candidate_spec(spec: str) -> tuple[str, Path]:
    """Parse NAME=PATH and reject ambiguous or missing candidate files."""
    name, separator, raw_path = spec.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise ValueError("candidate must use NAME=PATH")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"candidate file does not exist: {path}")
    return name.strip(), path


def frame_timestamps(duration: float, count: int) -> list[float]:
    """Sample uniformly from 2% through 98%, retaining clip-edge evidence."""
    if duration <= 0 or count < 1:
        raise ValueError("duration and frame count must be positive")
    if count == 1:
        return [duration / 2]
    start = duration * 0.02
    stop = duration * 0.98
    step = (stop - start) / (count - 1)
    return [round(start + index * step, 3) for index in range(count)]


def calibration_report(
    summaries: list[dict[str, Any]], minimum_correlation: float
) -> dict[str, Any]:
    """Gate local judging on rank correlation with known official scores."""
    known = [row for row in summaries if row.get("official_score") is not None]
    if len(known) < 3:
        raise ValueError("at least three officially scored sets are required")
    correlation = spearman(
        [float(row["official_score"]) for row in known],
        [float(row["local_final"]) for row in known],
    )
    return {
        "spearman": correlation,
        "minimum_correlation": minimum_correlation,
        "accepted": correlation >= minimum_correlation,
        "known_sets": len(known),
    }


def _video_duration(video_path: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(proc.stdout.strip())


def extract_frames(
    video_path: Path,
    output_dir: Path,
    count: int = 12,
    max_edge: int = 896,
) -> list[Path]:
    """Extract uniformly spaced JPEG evidence frames with FFmpeg."""
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = _video_duration(video_path)
    timestamps = frame_timestamps(duration, count)
    frames: list[Path] = []
    for index, timestamp in enumerate(timestamps):
        seek_timestamp = min(timestamp, max(0.0, duration - 0.5))
        frame = output_dir / f"frame_{index:02d}.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{seek_timestamp:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                f"scale={max_edge}:{max_edge}:force_original_aspect_ratio=decrease",
                "-q:v",
                "3",
                str(frame),
            ],
            check=True,
        )
        frames.append(frame)
    return frames


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def call_local_judge(
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Call a local OpenAI-compatible multimodal endpoint and parse its score."""
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 700,
        }
    ).encode("utf-8")
    request = Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - explicit local endpoint
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    return parse_score_json(content)


def call_local_batch_judge(
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: float = 240.0,
) -> dict[str, dict[str, Any]]:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 2400,
        }
    ).encode("utf-8")
    request = Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - explicit local endpoint
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    return parse_batch_score_json(content)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _download(url: str, destination: Path) -> Path:
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "AMD-Track2-local-bench/1.0"})
    with urlopen(request, timeout=120) as response, temporary.open("wb") as output:  # noqa: S310
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(destination)
    return destination


def _task_urls(root: Path, relative_path: str) -> dict[str, str]:
    rows = json.loads((root / relative_path).read_text(encoding="utf-8"))
    return {str(row["task_id"]): row["video_url"] for row in rows}


def _candidate_set(name: str, path: Path) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        "name": name,
        "official_score": None,
        "path": path,
        "captions": {str(row["task_id"]): row["captions"] for row in rows},
    }


def _score_cache_key(
    model: str,
    task_id: str,
    style: str,
    caption: str,
    frames: int,
    max_edge: int,
) -> str:
    raw = json.dumps(
        [model, task_id, style, caption, frames, max_edge],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    work_dir = Path(args.work_dir).resolve()
    task_urls = _task_urls(root, args.tasks)
    caption_sets = load_caption_sets(root, DEFAULT_SET_SPECS)
    for raw_spec in args.candidate:
        name, path = parse_candidate_spec(raw_spec)
        caption_sets.append(_candidate_set(name, path))

    task_ids = common_task_ids(caption_sets, task_urls)
    if args.limit_tasks:
        task_ids = task_ids[: args.limit_tasks]
    if not task_ids:
        raise ValueError("no video task is shared by every caption set")

    evidence: dict[str, list[str]] = {}
    for task_id in task_ids:
        video = _download(task_urls[task_id], work_dir / "videos" / f"{task_id}.mp4")
        frame_dir = work_dir / "frames" / f"{task_id}_{args.frames}_{args.max_edge}"
        existing = sorted(frame_dir.glob("frame_*.jpg"))
        frame_paths = (
            existing
            if len(existing) == args.frames
            else extract_frames(video, frame_dir, count=args.frames, max_edge=args.max_edge)
        )
        evidence[task_id] = [image_data_url(path) for path in frame_paths]

    summaries: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    cache_dir = work_dir / "cache"
    for caption_set in caption_sets:
        set_rows: list[dict[str, Any]] = []
        for task_id in task_ids:
            captions = caption_set["captions"][task_id]
            ordered_captions = {style: str(captions[style]) for style in STYLES}
            key = _score_cache_key(
                args.model,
                task_id,
                "all_styles",
                json.dumps(ordered_captions, sort_keys=True),
                args.frames,
                args.max_edge,
            )
            cache_path = cache_dir / f"{key}.json"
            if cache_path.is_file():
                batch_scores = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                messages = build_batch_messages(task_id, ordered_captions, evidence[task_id])
                batch_scores = call_local_batch_judge(
                    args.endpoint, args.model, messages, timeout=args.timeout
                )
                _atomic_json(cache_path, batch_scores)
            for style in STYLES:
                caption = str(captions[style])
                score = batch_scores[style]
                row = {
                    "set": caption_set["name"],
                    "task_id": task_id,
                    "style": style,
                    "caption": caption,
                    **score,
                }
                set_rows.append(row)
                detail_rows.append(row)
                print(
                    f"{caption_set['name']} {task_id}/{style}: "
                    f"acc={score['accuracy']:.2f} style={score['style_match']:.2f} "
                    f"coverage={score['coverage']:.2f}",
                    flush=True,
                )
        aggregate = aggregate_scores(set_rows)
        summaries.append(
            {
                "name": caption_set["name"],
                "official_score": caption_set["official_score"],
                "local_final": aggregate["final"],
                **aggregate,
                "captions_scored": len(set_rows),
            }
        )

    calibration = calibration_report(summaries, args.min_correlation)
    v19 = next(row for row in summaries if row["name"] == "v19_gate")
    for summary in summaries:
        if summary["official_score"] is None:
            summary["delta_vs_v19"] = summary["local_final"] - v19["local_final"]
            summary["eligible"] = bool(calibration["accepted"]) and (
                summary["local_final"] > v19["local_final"]
            )

    report = {
        "model": args.model,
        "endpoint": args.endpoint,
        "task_ids": task_ids,
        "frames": args.frames,
        "max_edge": args.max_edge,
        "calibration": calibration,
        "summaries": summaries,
        "details": detail_rows,
    }
    _atomic_json(Path(args.report).resolve(), report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate a zero-credit Qwen3-VL judge on known AMD Track 2 scores."
    )
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--tasks", default="data/official_new12.json")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--max-edge", type=int, default=896)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--min-correlation", type=float, default=0.60)
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--work-dir", default="out/amd_local_judge")
    parser.add_argument("--report", default="out/amd_local_judge/report.json")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run_benchmark(args)
    print(json.dumps({
        "calibration": report["calibration"],
        "summaries": report["summaries"],
    }, indent=2))
    return 0 if report["calibration"]["accepted"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:  # noqa: BLE001 - CLI prints one actionable failure
        print(f"AMD local judge failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
