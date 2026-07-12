"""Offline contract tests for the opt-in direct Qwen caption engine.

Run:
    PYTHONPATH=. python scripts/test_qwen_direct.py
"""

from __future__ import annotations

import asyncio
import base64
import sys
import tempfile
import types
from pathlib import Path

import httpx

sys.path.insert(0, ".")

from app import main as M  # noqa: E402
from app import pipeline as P  # noqa: E402
from app import qwen_direct as Q  # noqa: E402


STYLES = [
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
]


def _frames(workdir: Path) -> list[Path]:
    frames: list[Path] = []
    for index in range(1, 5):
        frame = workdir / f"frame_{index}.jpg"
        frame.write_bytes(f"jpeg-{index}".encode("ascii"))
        frames.append(frame)
    return frames


def test_profile_matches_the_predeclared_direct_candidate() -> None:
    assert Q.FRAME_COUNT == 4
    assert Q.FRAME_MAX_EDGE == 1024
    assert Q.MODEL == "accounts/fireworks/models/qwen3p7-plus"
    assert Q.TEMPERATURE == 0.7
    assert Q.MAX_TOKENS == 400
    assert Q.REASONING_EFFORT == "none"
    assert Q.MAX_ATTEMPTS == 2


def test_build_request_contains_four_images_and_one_style_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frames = _frames(Path(tmp))
        payload = Q.build_request("formal", frames)

    assert payload["model"] == Q.MODEL
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 400
    assert payload["reasoning_effort"] == "none"
    assert len(payload["messages"]) == 2
    system = payload["messages"][0]["content"]
    assert "professional" in system.lower()
    assert "objective" in system.lower()
    assert "sarcastic" not in system.lower()
    assert "cross-reference" not in system.lower()
    assert "observation list" not in system.lower()

    content = payload["messages"][1]["content"]
    image_blocks = [part for part in content if part["type"] == "image_url"]
    text_blocks = [part for part in content if part["type"] == "text"]
    assert len(image_blocks) == 4
    assert len(text_blocks) == 1
    assert "one caption" in text_blocks[0]["text"].lower()
    for index, part in enumerate(image_blocks, start=1):
        url = part["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        encoded = url.split(",", 1)[1]
        assert base64.b64decode(encoded) == f"jpeg-{index}".encode("ascii")


def test_four_independent_style_calls_preserve_requested_keys() -> None:
    requests: list[dict] = []

    async def requester(payload: dict) -> str:
        requests.append(payload)
        return f"Caption response {len(requests)}."

    with tempfile.TemporaryDirectory() as tmp:
        captions = asyncio.run(
            Q.caption_styles_from_frames(
                _frames(Path(tmp)),
                STYLES,
                requester=requester,
                retry_delay_s=0,
            )
        )

    assert list(captions) == STYLES
    assert len(requests) == 4
    assert all(captions[style] for style in STYLES)
    assert len({request["messages"][0]["content"] for request in requests}) == 4
    assert all(
        len(
            [
                part
                for part in request["messages"][1]["content"]
                if part["type"] == "image_url"
            ]
        )
        == 4
        for request in requests
    )


def test_transient_failure_retries_only_the_failed_style() -> None:
    attempts: dict[str, int] = {}

    async def requester(payload: dict) -> str:
        system = payload["messages"][0]["content"]
        attempts[system] = attempts.get(system, 0) + 1
        if "dry" in system.lower() and attempts[system] == 1:
            raise httpx.TransportError("temporary transport failure")
        return "A grounded caption."

    with tempfile.TemporaryDirectory() as tmp:
        captions = asyncio.run(
            Q.caption_styles_from_frames(
                _frames(Path(tmp)),
                STYLES,
                requester=requester,
                retry_delay_s=0,
            )
        )

    assert captions["sarcastic"] == "A grounded caption."
    assert sorted(attempts.values()) == [1, 1, 1, 2]


def test_exhausted_style_returns_empty_for_pipeline_fallback() -> None:
    calls = 0

    async def requester(_payload: dict) -> str:
        nonlocal calls
        calls += 1
        raise httpx.TransportError("still unavailable")

    with tempfile.TemporaryDirectory() as tmp:
        captions = asyncio.run(
            Q.caption_styles_from_frames(
                _frames(Path(tmp)),
                ["formal"],
                requester=requester,
                retry_delay_s=0,
            )
        )

    assert captions == {"formal": ""}
    assert calls == 2


def test_extract_profile_uses_the_confirmed_fps_geometry() -> None:
    captured: dict = {}
    original_probe = P._ffprobe_duration
    original_extract = getattr(Q, "_extract_fps_frames", None)
    try:
        P._ffprobe_duration = lambda _video: 80.0

        def fake_extract(**kwargs):
            captured.update(kwargs)
            return _frames(kwargs["workdir"])

        Q._extract_fps_frames = fake_extract
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            video = workdir / "clip.mp4"
            video.write_bytes(b"video")
            frames = Q.extract_frames(video, workdir)
    finally:
        P._ffprobe_duration = original_probe
        if original_extract is None:
            delattr(Q, "_extract_fps_frames")
        else:
            Q._extract_fps_frames = original_extract

    assert len(frames) == 4
    assert captured == {
        "video": video,
        "workdir": workdir,
        "duration": 80.0,
    }


def test_fps_extractor_runs_one_four_frame_1024px_ffmpeg_filter() -> None:
    calls: list[tuple[list[str], dict]] = []
    original_run = Q.subprocess.run
    try:
        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            pattern = Path(command[-1])
            for index in range(1, 5):
                Path(str(pattern).replace("%02d", f"{index:02d}")).write_bytes(
                    f"jpeg-{index}".encode("ascii")
                )

        Q.subprocess.run = fake_run
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            video = workdir / "clip.mp4"
            video.write_bytes(b"video")
            frames = Q._extract_fps_frames(video, workdir, duration=80.0)
    finally:
        Q.subprocess.run = original_run

    assert len(frames) == 4
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:6] == [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i"
    ]
    assert command[6] == str(video)
    vf = command[command.index("-vf") + 1]
    assert vf.startswith("fps=0.05000000,")
    assert "min(1024,iw)" in vf
    assert "min(1024,ih)" in vf
    assert "force_original_aspect_ratio=decrease" in vf
    assert command[command.index("-vframes") + 1] == "4"
    assert command.index("-vf") < command.index("-vframes")
    assert kwargs == {"check": True, "timeout": 12.0}


def test_missing_direct_caption_uses_existing_pipeline_once() -> None:
    original_key = Q.FIREWORKS_API_KEY
    original_download = P._download
    original_extract = Q.extract_frames
    original_caption_styles = Q.caption_styles_from_frames
    original_pipeline = P.caption_one_video
    fallback_calls: list[list[str]] = []
    try:
        Q.FIREWORKS_API_KEY = "test-key"

        async def fake_download(_url: str, dst: Path) -> Path:
            dst.write_bytes(b"video")
            return dst

        def fake_extract(_video: Path, workdir: Path) -> list[Path]:
            return _frames(workdir)

        async def fake_direct(_frames_arg, styles, **_kwargs):
            return {
                "formal": "A person walks through a quiet park.",
                "sarcastic": "",
                "humorous_tech": "The walking pipeline completes its tiny runtime.",
                "humorous_non_tech": "The walking API deploys a cheerful stroll.",
            }

        async def fake_pipeline(video_url: str, styles: list[str]):
            assert video_url == "https://example.test/v.mp4"
            fallback_calls.append(styles)
            return {
                "sarcastic": "Clearly, this walk has reached historic importance.",
                "humorous_tech": "The walking pipeline completes its tiny runtime.",
                "humorous_non_tech": "A stroll enters like it booked the whole path.",
            }

        P._download = fake_download
        Q.extract_frames = fake_extract
        Q.caption_styles_from_frames = fake_direct
        P.caption_one_video = fake_pipeline
        captions = asyncio.run(Q.caption_qwen_direct("https://example.test/v.mp4", STYLES))
    finally:
        Q.FIREWORKS_API_KEY = original_key
        P._download = original_download
        Q.extract_frames = original_extract
        Q.caption_styles_from_frames = original_caption_styles
        P.caption_one_video = original_pipeline

    assert fallback_calls == [["sarcastic", "humorous_non_tech"]]
    assert list(captions) == STYLES
    assert all(captions.values())


def test_main_dispatches_only_when_explicitly_selected() -> None:
    original_engine = M.CAPTION_ENGINE
    original_pipeline = M.caption_one_video
    original_module = sys.modules.get("app.qwen_direct")
    called: list[tuple[str, list[str]]] = []
    fake_module = types.ModuleType("app.qwen_direct")

    async def fake_direct(video_url: str, styles: list[str]) -> dict[str, str]:
        called.append((video_url, styles))
        return {
            "formal": "A person walks through a quiet park.",
            "sarcastic": "Clearly, this walk has reached historic importance.",
            "humorous_tech": "The walking pipeline completes its tiny runtime.",
            "humorous_non_tech": "A stroll enters like it booked the whole path.",
        }

    async def forbidden_pipeline(**_kwargs):
        raise AssertionError("legacy pipeline selected for qwen_direct")

    fake_module.caption_qwen_direct = fake_direct
    try:
        sys.modules["app.qwen_direct"] = fake_module
        M.CAPTION_ENGINE = "qwen_direct"
        M.caption_one_video = forbidden_pipeline
        result = asyncio.run(
            M._run_one(
                asyncio.Semaphore(1),
                {
                    "task_id": "v1",
                    "video_url": "https://example.test/v.mp4",
                    "styles": STYLES,
                },
            )
        )
    finally:
        M.CAPTION_ENGINE = original_engine
        M.caption_one_video = original_pipeline
        if original_module is None:
            sys.modules.pop("app.qwen_direct", None)
        else:
            sys.modules["app.qwen_direct"] = original_module

    assert called == [("https://example.test/v.mp4", STYLES)]
    assert result["task_id"] == "v1"
    assert list(result["captions"]) == STYLES


def test_docker_default_is_untouched() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "CAPTION_ENGINE=ensemble" in dockerfile
    assert "CAPTION_ENGINE=qwen_direct" not in dockerfile


def main() -> None:
    test_profile_matches_the_predeclared_direct_candidate()
    test_build_request_contains_four_images_and_one_style_only()
    test_four_independent_style_calls_preserve_requested_keys()
    test_transient_failure_retries_only_the_failed_style()
    test_exhausted_style_returns_empty_for_pipeline_fallback()
    test_extract_profile_uses_the_confirmed_fps_geometry()
    test_fps_extractor_runs_one_four_frame_1024px_ffmpeg_filter()
    test_missing_direct_caption_uses_existing_pipeline_once()
    test_main_dispatches_only_when_explicitly_selected()
    test_docker_default_is_untouched()
    print("qwen_direct_ok")


if __name__ == "__main__":
    main()
