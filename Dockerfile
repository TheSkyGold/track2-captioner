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
# Submission engine = v30 Verified Scene Gate. Three independent vision observers
# build candidate facts; GPT-5.5 verifies a closed registry against the pixels;
# four style-specific Opus writers run in parallel; GPT-5.5 audits and triggers
# selective repair. The legacy ensemble remains only an early-failure fallback.
ENV PYTHONPATH=/app \
    CAPTION_ENGINE=ensemble \
    VERIFIED_SCENE_GATE=1 \
    ENSEMBLE_OBSERVERS=openai/gpt-5.5,google/gemini-3.1-pro-preview,anthropic/claude-opus-4.8 \
    VERIFIED_SCENE_MODEL=openai/gpt-5.5 \
    VERIFIED_WRITER_MODEL=anthropic/claude-opus-4.8 \
    VERIFIED_REPAIR_MODEL=anthropic/claude-opus-4.8 \
    VERIFIED_AUDIT=1 \
    VERIFIED_AUDITOR_MODEL=openai/gpt-5.5 \
    ENSEMBLE_WRITER=anthropic/claude-opus-4.8 \
    ENSEMBLE_OBSERVER_TIMEOUT_S=28 \
    ENSEMBLE_WRITER_TIMEOUT_S=28 \
    OPENROUTER_MAX_INFLIGHT=6 \
    OPENROUTER_HTTP_RETRIES=1 \
    OPENROUTER_RETRY_MAX_WAIT_S=2 \
    CANDIDATE_SETS=1 \
    FACT_CONSENSUS=1 \
    STYLE_EXEMPLARS=0 \
    STRICT_GROUNDING=1 \
    WRITER_TEMP=0.5 \
    TIMESTAMP_FRAMES=0 \
    TIMESTAMP_TEXT=1 \
    FRAME_ANCHOR=1 \
    MAX_CAPTION_CHARS=300 \
    OPENROUTER_VLM_MODEL=openai/gpt-5.5 \
    OPENROUTER_STYLE_MODEL=anthropic/claude-opus-4.8 \
    PROVIDER_ORDER=openrouter \
    STYLE_PROVIDER_ORDER=openrouter \
    STYLE_MODEL=accounts/fireworks/models/gpt-oss-120b \
    STYLE_REASONING_EFFORT=low \
    STYLE_MAX_TOKENS=1400 \
    DETERMINISTIC_FORMAL=1 \
    NUM_FRAMES=8 \
    FRAME_MAX_EDGE=768 \
    GROQ_MAX_IMAGES=4 \
    GROQ_FRAME_MAX_EDGE=448 \
    HTTP_429_RETRIES=5 \
    HTTP_429_MAX_WAIT_S=45 \
    RETRY_AFTER_GIVEUP_S=60 \
    DESCRIBE_MAX_TOKENS=1300 \
    SCENE_DETECT_ENABLED=0 \
    MAX_CONCURRENCY=3 \
    PER_TASK_TIMEOUT_S=125 \
    MIN_TASK_START_S=8 \
    GLOBAL_BUDGET_S=535
# Judging VM = 2 vCPU / 4 GB RAM (Participant Guide p.5). Scene-detection
# decodes the ENTIRE UHD clip per video and three parallel ffmpeg passes
# thrash 2 cores into the 10-minute wall - uniform -ss seeks are near-free.

# The judge injects no runtime secret, so the public submission carries one
# dedicated, capped, short-lived OpenRouter key. Never use a personal or
# unlimited key here. Groq and Fireworks stay absent from the final image.
ARG OPENROUTER_API_KEY=""
ENV OPENROUTER_API_KEY=${OPENROUTER_API_KEY}

# Sanity: must start and be ready in < 60s. Keep the container thin.
CMD ["python", "-u", "-m", "app.main"]
