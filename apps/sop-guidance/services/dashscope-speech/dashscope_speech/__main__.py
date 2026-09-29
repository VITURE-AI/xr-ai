# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point: ``dashscope_speech --config <yaml> [--ready-file <path>]``.

Launcher convention: the ready file is created only once uvicorn is bound
and accepting connections, so the stack never advances to a worker that
would race a not-yet-listening port.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import uvicorn
from loguru import logger
from xr_ai_logging import setup_logging

from .app import create_app
from .config import ConfigError, Settings, load_settings


async def serve(settings: Settings, ready_file: Path | None = None) -> None:
    app = create_app(settings)
    server = uvicorn.Server(uvicorn.Config(app, host=settings.host, port=settings.port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        if task.done():
            await task
            raise RuntimeError("HTTP server exited before becoming ready")
        await asyncio.sleep(0.05)
    logger.info("Ready  ->  http://{}:{}/v1", settings.host, settings.port)
    if ready_file is not None:
        ready_file.touch()
    await task
    logger.info("Stopped.")


def run() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    setup_logging("dashscope-speech")

    p = argparse.ArgumentParser(prog="dashscope_speech", description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=None, help="YAML config (see dashscope_speech.yaml)")
    p.add_argument("--ready-file", type=Path, default=None, help="touched once the HTTP port is listening")
    ns, _ = p.parse_known_args()

    try:
        settings = load_settings(ns.config)
    except ConfigError as exc:
        raise SystemExit(f"[dashscope_speech] config error: {exc}") from None
    missing = settings.missing_keys()
    if missing:
        # Fail the stack now: a shim without a key would pass /health and then
        # answer every utterance with an error.
        raise SystemExit("[dashscope_speech] " + "; ".join(missing))
    if not (settings.stt.enabled or settings.tts.enabled):
        raise SystemExit("[dashscope_speech] both stt and tts are disabled; nothing to serve")

    for role, section in (("stt", settings.stt), ("tts", settings.tts)):
        if section.enabled:
            logger.info("{}: model={} endpoint={} key from {}", role, section.model, section.endpoint.base_url,
                        "/".join(section.endpoint.api_key_envs))
    if settings.stt.enabled:
        logger.info("stt: context={}", f"{len(settings.stt.context)} chars" if settings.stt.context else "none")
    if settings.tts.enabled:
        logger.info("tts: voice={} sample_rate={} instruction={} streaming={}", settings.tts.voice,
                    settings.tts.sample_rate, "set" if settings.tts.instruction else "none",
                    settings.tts.streaming_enabled)

    asyncio.run(serve(settings, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
