# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The SOP guidance worker. Build context: the repository root.
#
# The detectors run as OpenVINO int8 exports on the CPU; torch (which
# ultralytics imports) comes from the CPU wheel index pinned in the worker's
# pyproject, so the image carries no CUDA stack.
FROM python:3.12-slim-bookworm

ARG APT_MIRROR_HOST=deb.debian.org
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_DEFAULT_INDEX=${PIP_INDEX_URL} \
    UV_INDEX=pytorch-cpu=${TORCH_INDEX_URL} \
    UV_PYTHON_DOWNLOADS=never \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN find /etc/apt -type f \( -name '*.sources' -o -name '*.list' \) \
        -exec sed -i "s|deb.debian.org|${APT_MIRROR_HOST}|g" {} + \
    && apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -i "${PIP_INDEX_URL}" uv

WORKDIR /workspace
COPY agent-sdk /workspace/agent-sdk
COPY utils /workspace/utils
# Dependencies first so code edits do not reinstall torch.
COPY apps/sop-guidance/guidance/pyproject.toml /workspace/apps/sop-guidance/guidance/
COPY apps/sop-guidance/worker/pyproject.toml /workspace/apps/sop-guidance/worker/
RUN mkdir -p apps/sop-guidance/guidance/sop_guidance apps/sop-guidance/worker/sop_guidance_worker \
    && touch apps/sop-guidance/guidance/sop_guidance/__init__.py \
             apps/sop-guidance/worker/sop_guidance_worker/__init__.py
RUN --mount=type=cache,target=/tmp/uv-cache \
    uv sync --project apps/sop-guidance/worker --no-dev
COPY apps/sop-guidance /workspace/apps/sop-guidance
RUN --mount=type=cache,target=/tmp/uv-cache \
    uv sync --project apps/sop-guidance/worker --no-dev

RUN useradd --uid 1000 --create-home appuser \
    && mkdir -p /data/run \
    && chown -R appuser:appuser /workspace /data/run
USER 1000:1000
ENV XR_RUN_DIR=/data/run
WORKDIR /workspace/apps/sop-guidance/worker
CMD ["uv", "run", "--no-sync", "sop_guidance_worker", "--config", "../yaml/sop_guidance_worker.yaml"]
