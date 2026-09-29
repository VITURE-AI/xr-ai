# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
sop-guidance orchestrator: spoken, camera-checked guidance through procedures.

Pipeline
--------
Mic → STT → wake word / gates → foreground (idle or active tools) → TTS
Camera → detector overlay → return video + grounded step checks (backend)

Model deployment
----------------
The default models file uses hosted DashScope models: the LLM and VLM are
called directly, and speech goes through the app-owned DashScope shim this
orchestrator starts on port 8106 (set DASHSCOPE_API_KEY). With
``yaml/models.local.json`` the sample reuses self-hosted services instead and
``--no-speech`` skips the shim.

How to run (from apps/sop-guidance/):
    uv sync && uv run sop_guidance
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from xr_ai_launcher import Process, run_stack
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

_CAPTURE_PROCESS = Process(
    "capture",
    "../../services/device-io-hub",
    "device_io_capture",
    config="yaml/media_capture.yaml",
)

_SPEECH_PROCESS = Process(
    "speech",
    "services/dashscope-speech",
    "dashscope_speech",
    config="services/dashscope-speech/dashscope_speech.yaml",
    port=8106,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spoken, camera-checked guidance through procedures.",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="record participant video, bidirectional audio, and data-channel traffic",
    )
    parser.add_argument(
        "--no-speech",
        action="store_true",
        help="do not start the DashScope speech shim (self-hosted STT/TTS)",
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _parser().parse_args(sys.argv[1:] if argv is None else argv)


def _build_processes(*, capture: bool = False, speech: bool = True) -> list[Process]:
    processes = [
        Process(
            "hub",
            "../../services/device-io-hub",
            "device_io_hub",
            config="yaml/device_io_hub.yaml",
        ),
    ]
    if speech:
        processes.append(_SPEECH_PROCESS)
    if capture:
        processes.append(_CAPTURE_PROCESS)
    processes.append(
        Process(
            "worker",
            "worker",
            "sop_guidance_worker",
            config="yaml/sop_guidance_worker.yaml",
        )
    )
    return processes


PROCESSES = _build_processes()


def run(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    setup_logging("orchestrator", namespace="sop-guidance")
    run_stack(_build_processes(capture=args.capture, speech=not args.no_speech), _BASE)


if __name__ == "__main__":
    run()
