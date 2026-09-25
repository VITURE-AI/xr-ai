# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The app-owned DashScope STT/TTS shim. Build context: the repository root.
FROM python:3.12-slim-bookworm

ARG APT_MIRROR_HOST=deb.debian.org
ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_DEFAULT_INDEX=${PIP_INDEX_URL} \
    UV_PYTHON_DOWNLOADS=never

RUN find /etc/apt -type f \( -name '*.sources' -o -name '*.list' \) \
        -exec sed -i "s|deb.debian.org|${APT_MIRROR_HOST}|g" {} + \
    && apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -i "${PIP_INDEX_URL}" uv

WORKDIR /workspace
COPY utils/xr-ai-logging /workspace/utils/xr-ai-logging
COPY agent-sdk/xr-ai-models /workspace/agent-sdk/xr-ai-models
COPY apps/sop-guidance/services/dashscope-speech /workspace/apps/sop-guidance/services/dashscope-speech
RUN --mount=type=cache,target=/tmp/uv-cache \
    uv sync --project apps/sop-guidance/services/dashscope-speech --no-dev

RUN useradd --uid 1000 --create-home appuser \
    && chown -R appuser:appuser /workspace
USER 1000:1000
WORKDIR /workspace/apps/sop-guidance/services/dashscope-speech
EXPOSE 8106
CMD ["uv", "run", "--no-sync", "dashscope_speech", "--config", "dashscope_speech.yaml"]
