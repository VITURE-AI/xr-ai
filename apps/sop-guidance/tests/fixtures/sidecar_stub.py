# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A sidecar serving the scripted backend, for the ``remote`` backend tests.

The shape a real sidecar takes: build an ordinary in-process backend and hand
it to :func:`sop_guidance.backends.remote.serve`. Run it by hand with::

    python tests/fixtures/sidecar_stub.py --endpoint ipc:///tmp/scripted.sock \\
        --capabilities '{"voice_advance": false, "provides_overlay": true}'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# tests/ on the path, as pytest has it, so the scripted backend is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures.scripted_backend import ScriptedBackend  # noqa: E402
from sop_guidance.backends.base import Capabilities  # noqa: E402
from sop_guidance.backends.remote import serve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--capabilities", default="{}", help="Capabilities as JSON")
    args = parser.parse_args()
    backend = ScriptedBackend(Capabilities.model_validate(json.loads(args.capabilities)))
    asyncio.run(serve(backend, args.endpoint))


if __name__ == "__main__":
    main()
