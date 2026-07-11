#!/usr/bin/env bash
# Build and push the public linux/amd64 image for submission.
set -euo pipefail

if [[ -z "${PUBLIC_IMAGE:-}" ]]; then
    echo "PUBLIC_IMAGE is required, e.g. ghcr.io/<user>/track2-captioner:final" >&2
    exit 2
fi
if [[ -z "${FIREWORKS_API_KEY:-}" || -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "FIREWORKS_API_KEY and OPENROUTER_API_KEY are required for a jury image" >&2
    exit 2
fi

echo ">>> Building and pushing ${PUBLIC_IMAGE} for linux/amd64"
docker buildx build \
    --platform linux/amd64 \
    --tag "${PUBLIC_IMAGE}" \
    --build-arg FIREWORKS_API_KEY \
    --build-arg OPENROUTER_API_KEY \
    --build-arg GROQ_API_KEY \
    --push \
    .

echo ">>> Pushed ${PUBLIC_IMAGE}"
echo ">>> Verify anonymous pull from a clean environment:"
echo "    PUBLIC_IMAGE=${PUBLIC_IMAGE} bash scripts/verify_public_image.sh"
