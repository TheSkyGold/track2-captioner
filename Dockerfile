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
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# The harness mounts /input and /output at runtime and injects NO env vars,
# so the submission profile is pinned here. Non-secret config is plain ENV.
ENV PYTHONPATH=/app \
    PROVIDER_ORDER=openrouter,groq,fireworks \
    DESCRIBE_PROVIDER_ORDER=openrouter,groq,fireworks \
    STYLE_PROVIDER_ORDER=openrouter,groq,fireworks \
    OPENROUTER_VLM_MODEL=qwen/qwen3-vl-235b-a22b-instruct \
    OPENROUTER_STYLE_MODEL=anthropic/claude-sonnet-4 \
    DETERMINISTIC_FORMAL=0 \
    NUM_FRAMES=10 \
    FRAME_MAX_EDGE=896 \
    DESCRIBE_MAX_TOKENS=1300 \
    STYLE_MAX_TOKENS=220 \
    MAX_CONCURRENCY=3 \
    PER_TASK_TIMEOUT_S=90

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
