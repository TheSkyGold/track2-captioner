"""Self-check: a transient Groq 429 must be absorbed, not collapse to a stub.

Reproduces the v3 failure mode (429 under concurrency) with a MockTransport that
returns 429 twice then 200, and asserts _chat_content_at recovers the content.
Run: python scripts/test_429_retry.py
"""
from __future__ import annotations

import asyncio
import itertools

import httpx

import app.pipeline as P


def _run() -> None:
    calls = itertools.count()

    def handler(request: httpx.Request) -> httpx.Response:
        n = next(calls)
        if n < 2:  # first two attempts rate-limited
            return httpx.Response(429, headers={"retry-after": "0"}, json={})
        return httpx.Response(200, json={"choices": [{"message": {"content": "  ok  "}}]})

    transport = httpx.MockTransport(handler)
    orig_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):  # inject the mock transport
        kwargs.pop("timeout", None)
        return orig_client(transport=transport)

    httpx.AsyncClient = patched_client
    slept: list[float] = []
    orig_sleep = asyncio.sleep

    async def fake_sleep(s: float) -> None:  # keep the test instant
        slept.append(s)

    asyncio.sleep = fake_sleep
    try:
        out = asyncio.run(P._chat_content_at("http://x", "k", {"model": "m", "messages": []}))
    finally:
        httpx.AsyncClient = orig_client
        asyncio.sleep = orig_sleep

    assert out == "ok", f"expected recovered content, got {out!r}"
    assert len(slept) == 2, f"expected 2 backoff sleeps for 2x429, got {slept}"
    print("OK: 429x2 -> 200 recovered; content =", repr(out), "; sleeps =", slept)


if __name__ == "__main__":
    _run()
