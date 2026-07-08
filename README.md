# Track 2 Captioner - AMD Developer Hackathon ACT II

Dockerized video-captioning agent for Track 2. It reads `/input/tasks.json`,
generates four styled English captions per clip, and writes
`/output/results.json`.

## Architecture

Pattern: understand once, style four times in parallel.

1. Download each MP4 URL from `tasks.json`.
2. Extract keyframes with FFmpeg using scene-change sampling plus uniform fallback.
3. Optionally transcribe audio with Groq Whisper when `GROQ_API_KEY` is set.
4. Ask a VLM for compact factual scene JSON.
5. Generate the four required styles independently:
   `formal`, `sarcastic`, `humorous_tech`, `humorous_non_tech`.
6. Normalize and validate the final JSON with Pydantic before writing output.

The runtime is built to degrade gracefully: provider failures, missing API keys,
and timeouts produce non-empty styled fallbacks rather than malformed output.

## Quickstart

Offline preflight, no API keys required:

```bash
python scripts/preflight.py
```

Build and run the Docker contract check:

```bash
python scripts/preflight.py --docker-build --docker-run
```

Real inference run:

```bash
export FIREWORKS_API_KEY=fw_xxx
python scripts/preflight.py --strict --docker-build --docker-run
make submit-check
```

Low-rate live run with Groq/OpenRouter first and Fireworks as fallback:

```bash
PROVIDER_ORDER=groq,fireworks,openrouter \
DESCRIBE_PROVIDER_ORDER=groq,openrouter,fireworks \
STYLE_PROVIDER_ORDER=openrouter,groq,fireworks \
MAX_CONCURRENCY=1 \
NUM_FRAMES=5 \
FRAME_MAX_EDGE=512 \
INPUT_PATH=data/sample_tasks.json \
OUTPUT_PATH=out/groq_results_final.json \
python -m app.main

python eval/self_check.py --results out/groq_results_final.json
python eval/quality_audit.py --results out/groq_results_final.json
python scripts/quality_gate.py --results out/demo_quality_results.json --scores eval/scores_quality_openrouter.json
```

Best measured quality profile on the public sample:

```bash
PROVIDER_ORDER=openrouter,groq,fireworks \
DESCRIBE_PROVIDER_ORDER=openrouter,groq,fireworks \
STYLE_PROVIDER_ORDER=openrouter,groq,fireworks \
MAX_CONCURRENCY=1 \
NUM_FRAMES=8 \
FRAME_MAX_EDGE=640 \
DESCRIBE_MAX_TOKENS=900 \
STYLE_MAX_TOKENS=180 \
INPUT_PATH=data/sample_tasks.json \
OUTPUT_PATH=out/demo_quality_results.json \
python -m app.main
```

The describe prompt asks for conservative fine-grained observations: approximate
traffic quantities, animal type without breed guesses, leaf/sunlight detail,
jewelry, hand position, nail color, cables, peripherals, and nearby objects.
Uncertain details are kept out of final captions unless they are visually clear.

Publish and verify a public image:

```bash
export PUBLIC_IMAGE=ghcr.io/<user>/track2-captioner:final
make publish
make verify-public
```

## I/O Contract

Input: `/input/tasks.json`

```json
[
  {
    "task_id": "v1",
    "video_url": "https://example.com/clip.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

Output: `/output/results.json`

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

## Runtime Variables

Copy `.env.example` to `.env` for local development. Do not commit real keys.

| Variable | Default | Purpose |
|---|---|---|
| `FIREWORKS_API_KEY` | empty | Required for real VLM/style inference. |
| `FIREWORKS_BASE_URL` | `https://api.fireworks.ai/inference/v1` | OpenAI-compatible Fireworks endpoint. |
| `PROVIDER_ORDER` | `groq,fireworks,openrouter` | Default provider priority. |
| `DESCRIBE_PROVIDER_ORDER` | `PROVIDER_ORDER` | Provider priority for video understanding. |
| `STYLE_PROVIDER_ORDER` | `PROVIDER_ORDER` | Provider priority for style caption writing. |
| `VLM_MODEL` | `accounts/fireworks/models/qwen2p5-vl-7b-instruct` | Describe-stage VLM. |
| `VLM_FALLBACK_MODELS` | empty | Comma-separated describe-stage fallback models. |
| `DIRECT_VIDEO_MODEL` | empty | Optional dedicated Fireworks video/audio model deployment. |
| `DIRECT_VIDEO_MAX_SECONDS` | `60` | Max seconds sent to the direct video/audio path. |
| `GROQ_VISION_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq vision fallback/priority model. |
| `OPENROUTER_API_KEY` | empty | Enables OpenRouter fallback. |
| `OPENROUTER_VLM_MODEL` | `qwen/qwen3-vl-8b-instruct` | OpenRouter describe-stage model. |
| `DESCRIBE_MAX_TOKENS` | `700` | Token budget for rich scene-facts JSON. |
| `STYLE_MODEL` | `accounts/fireworks/models/gemma-3-27b-it` | Style rewriter, Gemma bonus path. |
| `STYLE_LORA` | empty | Optional deployed LoRA model id. |
| `STYLE_FALLBACK_MODELS` | empty | Comma-separated style fallback models. |
| `GROQ_STYLE_MODEL` | `llama-3.3-70b-versatile` | Groq style model. |
| `OPENROUTER_STYLE_MODEL` | `qwen/qwen3-vl-8b-instruct` | OpenRouter style fallback. |
| `STYLE_MAX_TOKENS` | `140` | Token budget for one styled caption. |
| `EVIDENCE_LOCK_ENABLED` | `0` | Enables experimental candidate/repair pass against visual evidence. |
| `STYLE_CANDIDATES` | `2` | Candidate count when evidence-lock mode is enabled. |
| `STYLE_REPAIR_ENABLED` | `1` | Enables model/deterministic repair for thin evidence-lock captions. |
| `JUDGE_PROVIDER_ORDER` | `fireworks,openrouter,groq` | Provider priority for local LLM-judge proxy. |
| `FIREWORKS_JUDGE_MODEL` | `accounts/fireworks/models/qwen3p7-plus` | Fireworks judge model when available. |
| `GROQ_JUDGE_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq judge fallback. |
| `OPENROUTER_JUDGE_MODEL` | `qwen/qwen3-vl-8b-instruct` | OpenRouter judge fallback. |
| `GROQ_API_KEY` | empty | Enables optional Whisper transcription. |
| `NUM_FRAMES` | `8` | Target number of keyframes. |
| `FRAME_MAX_EDGE` | `720` | Max frame edge before upload. |
| `SCENE_DETECT_ENABLED` | `1` | Enable scene-change sampling before uniform fill. |
| `MAX_CONCURRENCY` | `3` | Parallel clips. |
| `PER_TASK_TIMEOUT_S` | `25` | Per-clip hard timeout. |

## Validation

Local gates:

```bash
python scripts/contract_test.py
python scripts/mock_run.py --tasks data/sample_tasks.json --out out/mock_results.json
python eval/self_check.py --results out/mock_results.json
python eval/quality_audit.py --results out/mock_results.json
python eval/detail_audit.py --results out/demo_quality_results.json
python finetune/train_gemma_lora.py --dataset finetune/dataset_v2.jsonl --dry-run
```

Current status and remaining submission gates are tracked in `SUBMIT_STATUS.md`.
Prompt/skill/plugin research is tracked in `docs/quality-research.md`.

Interactive project briefing:

```text
docs/mission-control.html
```

## Repository Layout

- `app/` - container runtime.
- `data/` - sample tasks.
- `docs/` - interactive briefing and project poster.
- `eval/` - zero-cost audits and provider-agnostic local judge.
- `finetune/` - synthetic dataset, scene generator, LoRA training/deploy notes.
- `scripts/` - preflight, Docker build/run/publish helpers.
- `SUBMISSION.md` - lablab.ai submission copy.
- `RUNBOOK.md` - operational commands.

## Notes

- Build target is `linux/amd64`.
- The image does not bake credentials.
- Generated `out/`, `in/`, caches, zips, and flattened duplicate exports are
  ignored by git.
