# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for the SOP guidance worker."""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Sequence
from pathlib import Path

from loguru import logger

from .app import run_app
from .config import load_config


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ready-file", type=Path, default=None)
    args, _ = parser.parse_known_args(argv)

    asyncio.run(
        run_app(
            load_config(args.config),
            ready_file=args.ready_file,
        )
    )
    # Exit now rather than wait on worker threads at interpreter shutdown: a
    # library thread stuck in a network call (NLTK's data download, observed)
    # kept the process alive after the voice session ended, so a supervisor
    # never restarted a worker that had stopped serving.
    logger.complete()
    os._exit(0)


if __name__ == "__main__":
    run()
