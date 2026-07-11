# Track 2 Submission Notes

## Project title

**Verified Scene Gate - grounded video captions in four distinct voices**

## Short description

> Two vision observers build facts, a pixel-level gate verifies them, and four fact-locked writers produce formal, sarcastic, tech-humorous, and everyday-humorous captions.

## Architecture

The container reads `/input/tasks.json`, downloads each clip, samples eight
chronological frames, and always writes a complete `/output/results.json`.

For every clip:

1. GPT-5.5 and Gemini 3.1 Pro independently list high-confidence atomic facts
   about the central subject, action/state, setting, and temporal change.
2. Local code assigns immutable fact IDs. GPT-5.5 re-reads the pixels and can
   keep IDs only. Free-form rewrites are impossible. Risky colors, counts,
   directions, OCR, brands, and breeds require independent corroboration.
3. Four Opus 4.8 calls run in parallel, each with one exact style rubric and the
   same closed fact ledger. Humor can be figurative but cannot introduce an
   unseen narrator, relationship, device, place, time, dialogue, or event.
4. GPT-5.5 audits each caption for literal accuracy, salient subject/action
   coverage, temporal consistency, and style match. Failed styles alone are
   repaired and re-audited.
5. Local validation enforces all four keys, English output, style separation,
   safe appearance handling, bounded length, prompt-injection isolation, and
   atomic JSON output.

The legacy writer is available only if fewer than two facts can be verified.
After the gate succeeds, late failures use a caption derived exclusively from
the verified ledger rather than reintroducing unsupported scene details.

## Generalization and runtime

Prompts contain no facts from the three public examples. The pipeline was
stress-tested on twelve additional clips spanning animals, cities, water,
mountains, weather, food, people, transport, and sports. It completed in 245.6
seconds and produced 12 valid rows and 48/48 captions. This is local evidence;
the official hidden score can only come from the competition judge.

## Reliability

- Linux/amd64 image, far below the 10 GB limit.
- OpenRouter-only submission profile; no Groq or Fireworks credential.
- Three clips concurrently; six API calls maximum in flight.
- Wall-clock stage deadlines include queue wait.
- One bounded retry only for 429/5xx, never auth/payment failures.
- Per-task deadline 125 seconds; global deadline 535 seconds.
- Pre-seeded results, schema validation, and atomic output replacement.

## Reproduction

```bash
export OPENROUTER_API_KEY=...
python scripts/preflight.py --strict --docker-build --docker-run
PYTHONPATH=. python scripts/test_verified_scene_gate.py
```

The repository contains no real keys. Because the judging harness injects no
credentials, use a dedicated OpenRouter competition credential with a total
spend cap and short expiry when publishing the image. Revoke it after judging.

## Technology

Python 3.11, asyncio, HTTPX, FFmpeg, Docker, OpenRouter, GPT-5.5,
Gemini 3.1 Pro, Claude Opus 4.8, Pydantic.
