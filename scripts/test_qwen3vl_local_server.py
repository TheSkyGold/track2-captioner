"""Unit tests for the notebook's OpenAI-compatible Qwen3-VL adapter."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("qwen3vl_local_server.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("qwen3vl_local_server", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load qwen3vl_local_server")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = _load_module()
    messages = [
        {"role": "system", "content": "Judge the captions."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect the evidence."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,AA=="},
                },
            ],
        },
    ]
    normalized = module.normalize_openai_messages(messages)
    assert normalized[0] == messages[0]
    assert normalized[1]["content"][0] == messages[1]["content"][0]
    assert normalized[1]["content"][1] == {
        "type": "image",
        "image": "data:image/jpeg;base64,AA==",
    }

    payload = {"model": "local", "max_tokens": 321, "temperature": 0.0}
    assert module.generation_options(payload) == {
        "max_new_tokens": 321,
        "do_sample": False,
    }
    sampled = module.generation_options(
        {"max_tokens": 99, "temperature": 0.6, "top_p": 0.8}
    )
    assert sampled["max_new_tokens"] == 99
    assert sampled["do_sample"] is True
    assert sampled["temperature"] == 0.6
    assert sampled["top_p"] == 0.8

    old_thinking = os.environ.get("LOCAL_ENABLE_THINKING")
    try:
        os.environ.pop("LOCAL_ENABLE_THINKING", None)
        assert module.chat_template_options() == {"enable_thinking": False}
        os.environ["LOCAL_ENABLE_THINKING"] = "1"
        assert module.chat_template_options() == {"enable_thinking": True}
    finally:
        if old_thinking is None:
            os.environ.pop("LOCAL_ENABLE_THINKING", None)
        else:
            os.environ["LOCAL_ENABLE_THINKING"] = old_thinking

    response = module.openai_response("{\"ok\":true}", "local-model")
    assert response["model"] == "local-model"
    assert response["choices"][0]["message"]["content"] == '{"ok":true}'
    assert response["choices"][0]["finish_reason"] == "stop"

    print("QWEN3VL SERVER TEST OK")


if __name__ == "__main__":
    main()
