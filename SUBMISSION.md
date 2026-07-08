# Track 2 Submission Notes

Everything you need to fill in on the lablab.ai submission form.

## Project Title
**Track2 Captioner — a two-stage video-caption agent tuned for style-match**

## Short Description (≤ 200 chars)
> Docker agent for Track 2. Two-stage pipeline: Qwen2.5-VL DESCRIBE → Gemma-3-27B STYLE ×4 in parallel, with an optional LoRA rewriter and Whisper transcript hint. Optimised for the LLM-Judge accuracy × style-match rubric.

## Long Description

Track 2 asks for *four captions in four distinct tones* per clip, judged on **accuracy** (does the caption match the video?) and **style match** (does it hit the target tone?). Our system optimises those two axes independently.

**Stage 1 — DESCRIBE.** We sample N frames from the clip (uniform, with optional shot-detection top-up via FFmpeg's `select='gt(scene,T)'` filter) and, when enabled, extract a Whisper transcript. A single VLM call (Qwen2.5-VL 7B on Fireworks) returns a compact JSON of scene facts: setting, subjects, actions, mood, audio hint. This anchors the *accuracy* score.

**Stage 2 — STYLE ×4.** Four LLM calls run in parallel (`asyncio.gather`), each with a style-specific system prompt containing 4 few-shot examples drawn from different domains (garden / boulevard / office / kitchen — so the model doesn't overfit the guide's three example clips). Each prompt explicitly **bans the traits of the other three styles**: the sarcastic prompt bans exclamations and tech jargon, the humorous_non_tech prompt bans all technical vocabulary, and so on. This is what gives a clean *style match* score.

**Optional LoRA layer.** We fine-tune Gemma 3 (4B for dev, 27B for submission) via Unsloth on ~800 synthesised examples spanning the 8 evaluation categories (nature, urban, animals, people, sports, food, weather, technology). When a deployed LoRA id is passed via `STYLE_LORA`, all four style calls route to it — the base prompts stay compatible.

**Safety net.** The container never crashes: per-task timeout at 25 s, retries with exponential backoff, non-empty styled fallbacks, and Pydantic validation before `results.json` is written. No requested style is ever missing or blank.

## Technology Tags
Docker, Python, Fireworks AI, Gemma 3, Qwen2.5-VL, FFmpeg, faster-whisper, LoRA, Unsloth, asyncio

## Category Tags
AI Agent, Video, Multimodal, Developer Tools

## What we use from AMD platforms
- **AMD Developer Cloud** — MI300X for training the LoRA (Unsloth, ROCm 6).
- **ROCm** — PyTorch runtime for the training script.
- **Fireworks AI** — inference for the VLM (Qwen2.5-VL 7B) and the style rewriter (Gemma 3 27B). This makes us eligible for the **Best Use of Gemma via Fireworks** and **Best Use of Gemma in Video Captioning** prize slots.

## Repo layout
- `app/` — container code (main / pipeline / prompts / frames / audio)
- `finetune/` — dataset builder + LoRA trainer + Fireworks deploy guide
- `eval/` — local LLM-Judge proxy + zero-cost self-check
- `scripts/` — build.sh, run_local.sh, smoke_test.sh, mock_run.py
- `Makefile` — `make smoke`, `make dataset`, `make train`, `make submit-check`

## What we chose NOT to do (and why)
- **We do not overfit the three example clips** — the guide explicitly warns that "agents that only work on the three example clips will score poorly". Our few-shots and dataset span 8 categories × 200 scenes.
- **We do not bundle a `.env`** — credentials come from env at runtime.
- **We do not stack multiple providers** (like Groq + OpenRouter + Fireworks) — one provider reduces variance in the 30-second-per-request budget.

## Reproduction (5 minutes)
```bash
export FIREWORKS_API_KEY=fw_xxx
make build          # docker buildx amd64
make run            # runs the container on the 3 example clips
make self-check     # zero-cost structural + ban validation
make judge          # LLM-Judge proxy for a real accuracy × style-match score
```

## Reference bonuses eligible
| Prize | Amount | Reason we qualify |
|---|---|---|
| Best Use of Gemma in Video Captioning | **3 000 $** | Gemma 3 27B is our default style-layer model, on Fireworks |
| Best AMD-Hosted Gemma Project | **2 000 $** | LoRA training happens on AMD Developer Cloud (MI300X, ROCm) |
| Track 2 leaderboard prize | share of 20 000 $+ | Optimised specifically for LLM-Judge accuracy × style-match |
