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
    UV_PYTHON_DOWNLOADS=never

RUN find /etc/apt -type f \( -name '*.sources' -o -name '*.list' \) \
        -exec sed -i "s|deb.debian.org|${APT_MIRROR_HOST}|g" {} + \
    && apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -i "${PIP_INDEX_URL}" uv

# The user first, so every later layer is written as it: a closing
# `chown -R` over the environment copied all of it into a new layer on every
# code change.
RUN useradd --uid 1000 --create-home appuser \
    && mkdir -p /workspace /data/run /home/appuser/.config/ultralytics \
                /usr/local/share/nltk_data /tmp/uv-cache \
    && chown appuser:appuser /workspace /data/run /home/appuser/.config \
                             /home/appuser/.config/ultralytics \
                             /usr/local/share/nltk_data /tmp/uv-cache
USER 1000:1000
WORKDIR /workspace
COPY --chown=appuser:appuser agent-sdk /workspace/agent-sdk
COPY --chown=appuser:appuser utils /workspace/utils
# Dependencies first so code edits do not reinstall torch.
COPY --chown=appuser:appuser apps/sop-guidance/guidance/pyproject.toml /workspace/apps/sop-guidance/guidance/
COPY --chown=appuser:appuser apps/sop-guidance/worker/pyproject.toml /workspace/apps/sop-guidance/worker/
COPY --chown=appuser:appuser apps/sop-guidance/backends/rpi-hat-judge/pyproject.toml \
     /workspace/apps/sop-guidance/backends/rpi-hat-judge/
RUN mkdir -p apps/sop-guidance/guidance/sop_guidance apps/sop-guidance/worker/sop_guidance_worker \
             apps/sop-guidance/backends/rpi-hat-judge/rpi_hat_judge \
    && touch apps/sop-guidance/guidance/sop_guidance/__init__.py \
             apps/sop-guidance/worker/sop_guidance_worker/__init__.py \
             apps/sop-guidance/backends/rpi-hat-judge/rpi_hat_judge/__init__.py
RUN --mount=type=cache,target=/tmp/uv-cache,uid=1000,gid=1000 \
    uv sync --project apps/sop-guidance/worker --no-dev

# The voice pipeline's sentence splitter needs NLTK's punkt_tab data and
# downloads it from GitHub on first start when it is missing. Baked in here,
# before the code is copied so code edits keep this layer: where GitHub is
# slow, that download held pipeline setup past its timeout on every start of
# a fresh container.
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN apps/sop-guidance/worker/.venv/bin/python -c \
        "import nltk, sys; sys.exit(0 if nltk.download('punkt_tab', download_dir='${NLTK_DATA}', quiet=True) else 1)"

COPY --chown=appuser:appuser apps/sop-guidance /workspace/apps/sop-guidance
RUN --mount=type=cache,target=/tmp/uv-cache,uid=1000,gid=1000 \
    uv sync --project apps/sop-guidance/worker --no-dev

# /tmp is the volume shared with the hub, so Ultralytics keeps its settings in
# the user's home instead.
ENV XR_RUN_DIR=/data/run \
    YOLO_CONFIG_DIR=/home/appuser/.config/ultralytics
WORKDIR /workspace/apps/sop-guidance/worker
CMD ["uv", "run", "--no-sync", "sop_guidance_worker", "--config", "../yaml/sop_guidance_worker.yaml"]
