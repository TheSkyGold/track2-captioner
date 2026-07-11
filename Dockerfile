# =============================================================================
# AMD Developer Hackathon: ACT II — Track 2 Video Captioning Agent
# Base slim + FFmpeg. Build for linux/amd64 explicitly (judging VM runs amd64).
#
#   docker buildx build --platform linux/amd64 \
#     --tag ghcr.io/<you>/track2-captioner:latest --push .
# =============================================================================
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# FFmpeg is used for keyframe extraction + audio track split.
# ca-certificates lets requests/openai talk HTTPS.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# The harness mounts /input and /output at runtime and injects NO env vars,
# so the submission profile is pinned here. Non-secret config is plain ENV.
# Submission engine = PIPELINE. The 12-clip stress benchmark measured the
# Qwen3-VL-8B + Gemma style profile as the strongest reliable path
# (scores_stress_gemma_v6.json: final 0.969). The ensemble path depends on
# frontier models and is kept as an opt-in experiment, not the default image.
ENV PYTHONPATH=/app \
    CAPTION_ENGINE=pipeline \
    ENSEMBLE_OBSERVERS=openai/gpt-5.5,google/gemini-3.1-pro-preview,anthropic/claude-opus-4.5 \
    ENSEMBLE_WRITER=anthropic/claude-opus-4.5 \
    CANDIDATE_SETS=1 \
    FACT_CONSENSUS=0 \
    STYLE_EXEMPLARS=1 \
    STRICT_GROUNDING=0 \
    WRITER_TEMP=0.5 \
    TIMESTAMP_FRAMES=0 \
    TIMESTAMP_TEXT=0 \
    FRAME_ANCHOR=1 \
    MAX_CAPTION_CHARS=1600 \
    OPENROUTER_VLM_MODEL=qwen/qwen3-vl-8b-instruct \
    OPENROUTER_STYLE_MODEL=google/gemma-3-27b-it \
    PROVIDER_ORDER=openrouter,groq,fireworks \
    DESCRIBE_PROVIDER_ORDER=openrouter,groq,fireworks \
    STYLE_PROVIDER_ORDER=openrouter,groq,fireworks \
    STYLE_MODEL=accounts/fireworks/models/gpt-oss-120b \
    STYLE_REASONING_EFFORT=low \
    STYLE_MAX_TOKENS=1400 \
    DETERMINISTIC_FORMAL=1 \
    NUM_FRAMES=8 \
    FRAME_MAX_EDGE=720 \
    GROQ_MAX_IMAGES=4 \
    GROQ_FRAME_MAX_EDGE=448 \
    HTTP_429_RETRIES=5 \
    HTTP_429_MAX_WAIT_S=45 \
    RETRY_AFTER_GIVEUP_S=60 \
    DESCRIBE_MAX_TOKENS=900 \
    SCENE_DETECT_ENABLED=1 \
    MAX_CONCURRENCY=2 \
    PER_TASK_TIMEOUT_S=150 \
    GLOBAL_BUDGET_S=540
# Judging VM = 2 vCPU / 4 GB RAM (Participant Guide p.5). Scene-detection
# decodes the ENTIRE UHD clip per video and three parallel ffmpeg passes
# thrash 2 cores into the 10-minute wall - uniform -ss seeks are near-free.

# API keys arrive as build args at publish time only (CI secrets) — the repo
# and default builds stay key-free; without keys the image degrades safely.
ARG OPENROUTER_API_KEY=""
ARG GROQ_API_KEY=""
ARG FIREWORKS_API_KEY=""
ENV OPENROUTER_API_KEY=${OPENROUTER_API_KEY} \
    GROQ_API_KEY=${GROQ_API_KEY} \
    FIREWORKS_API_KEY=${FIREWORKS_API_KEY}

# Sanity: must start and be ready in < 60s. Keep the container thin.
CMD ["python", "-u", "-m", "app.main"]
