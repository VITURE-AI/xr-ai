# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DeviceIOHub for the SOP guidance stack. Build context: the repository root.
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
    && apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -i "${PIP_INDEX_URL}" uv

WORKDIR /workspace
COPY agent-sdk/xr-ai-hub /workspace/agent-sdk/xr-ai-hub
COPY utils/xr-ai-logging /workspace/utils/xr-ai-logging
COPY services/device-io-hub /workspace/services/device-io-hub
COPY client-samples/web /workspace/client-samples/web
RUN --mount=type=cache,target=/tmp/uv-cache \
    uv sync --project services/device-io-hub --no-dev

RUN useradd --uid 1000 --create-home appuser \
    && chown -R appuser:appuser /workspace
USER 1000:1000
WORKDIR /workspace/services/device-io-hub
CMD ["uv", "run", "--no-sync", "device_io_hub", "--config", "/etc/xr-ai/device_io_hub.yaml"]
