"""Serve Qwen3-VL from the AMD notebook behind a tiny OpenAI-compatible API.

This development-only adapter intentionally depends on Transformers instead of
vLLM so it also works on Radeon/ROCm instances where a matching vLLM wheel is
not available. It accepts the image payloads emitted by ``amd_local_judge.py``.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


DEFAULT_MODEL = os.environ.get(
    "LOCAL_VLM_MODEL", "/workspace/models/Qwen3-VL-8B-Instruct"
)


def normalize_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate OpenAI image/video parts into Qwen's chat-template schema."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            normalized.append(dict(message))
            continue
        parts: list[dict[str, Any]] = []
        for part in content:
            kind = part.get("type")
            if kind == "image_url":
                image_url = part.get("image_url", {})
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if not isinstance(url, str) or not url:
                    raise ValueError("image_url part has no URL")
                parts.append({"type": "image", "image": url})
            elif kind == "video_url":
                video_url = part.get("video_url", {})
                url = video_url.get("url") if isinstance(video_url, dict) else video_url
                if not isinstance(url, str) or not url:
                    raise ValueError("video_url part has no URL")
                parts.append({"type": "video", "video": url})
            else:
                parts.append(dict(part))
        normalized.append({**message, "content": parts})
    return normalized


def generation_options(payload: dict[str, Any]) -> dict[str, Any]:
    """Map bounded OpenAI generation controls to Transformers options."""
    max_new_tokens = max(1, min(4096, int(payload.get("max_tokens", 1024))))
    temperature = float(payload.get("temperature", 0.0) or 0.0)
    if temperature <= 0:
        return {"max_new_tokens": max_new_tokens, "do_sample": False}
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": max(0.01, min(1.0, float(payload.get("top_p", 0.9)))),
    }


def openai_response(content: str, model: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-local-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class QwenRuntime:
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.model_name = model_path
        max_pixels = int(os.environ.get("LOCAL_MAX_PIXELS", str(512 * 28 * 28)))
        min_pixels = int(os.environ.get("LOCAL_MIN_PIXELS", str(128 * 28 * 28)))
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).eval()
        self.lock = threading.Lock()

    def complete(self, payload: dict[str, Any]) -> str:
        messages = normalize_openai_messages(payload.get("messages", []))
        if not messages:
            raise ValueError("messages must be a non-empty list")
        with self.lock, self.torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            generated = self.model.generate(**inputs, **generation_options(payload))
            trimmed = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated)
            ]
            return self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]


def make_handler(runtime: QwenRuntime):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # noqa: N802 - stdlib callback name
            if self.path.rstrip("/") == "/v1/models":
                self._json(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": runtime.model_name, "object": "model"}],
                    },
                )
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802 - stdlib callback name
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                content = runtime.complete(payload)
                self._json(200, openai_response(content, runtime.model_name))
            except Exception as exc:  # noqa: BLE001 - surface notebook diagnostics
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}", flush=True)

    return Handler


def main() -> None:
    host = os.environ.get("LOCAL_VLM_HOST", "127.0.0.1")
    port = int(os.environ.get("LOCAL_VLM_PORT", "8000"))
    runtime = QwenRuntime(DEFAULT_MODEL)
    server = ThreadingHTTPServer((host, port), make_handler(runtime))
    server.daemon_threads = True
    print(f"READY {runtime.model_name} http://{host}:{port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
