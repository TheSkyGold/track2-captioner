# AMD Track 2 - Verified Scene Gate

Dockerized video-captioning agent. It reads `/input/tasks.json`, generates all
four requested English styles, writes `/output/results.json`, and exits.

## Run the public image

```bash
docker pull ghcr.io/theskygold/track2-captioner:latest
docker run --rm \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output \
  ghcr.io/theskygold/track2-captioner:latest
```

Required styles: `formal`, `sarcastic`, `humorous_tech`, and
`humorous_non_tech`.

## v30 architecture

The submission profile is `CAPTION_ENGINE=ensemble` with
`VERIFIED_SCENE_GATE=1`.

1. FFmpeg samples eight chronological frames and provides timestamp labels
   beside, never over, the pixels.
2. GPT-5.5 and Gemini 3.1 Pro independently list 2-12 high-confidence atomic
   observations.
3. Local code assigns immutable fact IDs. GPT-5.5 re-reads the frames and may
   keep existing IDs only; it cannot rewrite or create a fact.
4. Risky colors, counts, directions, OCR, brands, and breeds require
   independent corroboration on the same subject.
5. Four style-specific Opus 4.8 writers run in parallel from verified facts.
6. GPT-5.5 audits literal accuracy, central subject/action coverage, temporal
   consistency, and the official style rubric. Only failed styles are repaired
   and re-audited.
7. Local guards enforce all four keys, English output, style separation, safe
   appearance handling, prompt-text isolation, and a 420-character ceiling.

Once at least two facts are verified, a late provider failure cannot return to
the older combined writer. A per-style deterministic fallback uses the closed
fact ledger only.

## Evidence

- Targeted v30 suite: 40 unit/integration tests.
- Public examples: 3/3 completed through the verified path in 63.2 seconds in
  the final Docker run; all three local output audits passed.
- Broader local set: 12/12 rows, 48/48 captions, 245.6 seconds, structural
  self-check passed.

These are runtime and generalization checks, not an official score. The guide
contains three public development clips; the announced evaluation uses a
hidden set of about twelve clips.

## I/O contract

Input, `/input/tasks.json`:

```json
[
  {
    "task_id": "v1",
    "video_url": "https://example.com/clip.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

Output, `/output/results.json`:

```json
[
  {
    "task_id": "v1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```

## Validate locally

```bash
python scripts/preflight.py
PYTHONPATH=. python scripts/test_verified_scene_gate.py
python scripts/contract_test.py
```

Build and run the mounted-I/O contract test:

```bash
export OPENROUTER_API_KEY=...
python scripts/preflight.py --strict --docker-build --docker-run
```

## Submission profile

| Variable | Docker value |
|---|---|
| `CAPTION_ENGINE` | `ensemble` |
| `VERIFIED_SCENE_GATE` | `1` |
| `ENSEMBLE_OBSERVERS` | `openai/gpt-5.5,google/gemini-3.1-pro-preview` |
| `VERIFIED_SCENE_MODEL` | `openai/gpt-5.5` |
| `VERIFIED_WRITER_MODEL` | `anthropic/claude-opus-4.8` |
| `VERIFIED_REPAIR_MODEL` | `anthropic/claude-opus-4.8` |
| `VERIFIED_AUDITOR_MODEL` | `openai/gpt-5.5` |
| Last-resort observer / writer | `openai/gpt-5.5` / `anthropic/claude-opus-4.8` |
| Provider order | `openrouter` only |
| `NUM_FRAMES` / `FRAME_MAX_EDGE` | `8` / `768` |
| `MAX_CONCURRENCY` / API in-flight | `3` / `6` |
| Per-task / global deadline | `125s` / `535s` |
| Caption ceiling | `420` characters |

The repository is key-free. The competition harness injects no environment
variables, so a judging image must use a dedicated, capped, expiring and
revocable OpenRouter key at build time. Groq and Fireworks credentials are not
included in the submission image. Never publish a personal unrestricted
credential; rotate the competition credential after judging.

See [SUBMISSION.md](SUBMISSION.md), [RUNBOOK.md](RUNBOOK.md), and
[SUBMIT_STATUS.md](SUBMIT_STATUS.md).
