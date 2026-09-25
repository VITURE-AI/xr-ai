# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A deterministic fake procedure backend that proves the host/backend seam.

It is deliberately not the ``vlm`` one. It decides progress from frame labels
and commands, the way a custom detector-and-state-machine backend would, which
is what shows the host, tools and interaction layer work for a backend they
know nothing about. The same class runs in-process and, through
``sidecar_stub.py``, behind :mod:`sop_guidance.backends.remote`.

A frame's label is its first pixel value, so it survives the shared-memory
ring unchanged: :data:`DONE` confirms the current step, :data:`HAND` is a hand
hiding the work and draws a correction cue. Every frame yields one overlay box.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np
from sop_guidance.backends.base import (
    Capabilities,
    CommandResult,
    Cue,
    OverlayUpdate,
    RunCommand,
    RunContext,
    RunFinished,
    RunSnapshot,
    StepChanged,
    StepInfo,
    TimedFrame,
    TurnContext,
)
from sop_guidance.vision.overlay import Detection

STEPS = ("Open the lid.", "Lift the tray.", "Close the lid.")

DONE = 200
"""Frame label: the current step's target state is in view."""

HAND = 100
"""Frame label: a hand hides the work."""

HAND_CUE = "Move your hand off the lid so I can see it."
LID_BOX = Detection("lid", 10.0, 12.0, 50.0, 40.0, 0.9)


def labelled_frame(label: int, participant_id: str = "alice", *, size: int = 8) -> TimedFrame:
    """A uniform BGR frame whose pixels carry *label*."""

    image = np.full((size, size, 3), label, dtype=np.uint8)
    return TimedFrame(participant_id=participant_id, timestamp_us=time.time_ns() // 1_000,
                      width=size, height=size, image=image)


class ScriptedRun:
    def __init__(self, backend: ScriptedBackend, ctx: RunContext, start_step: int) -> None:
        self._backend = backend
        self._ctx = ctx
        self.step = start_step
        self.cues = 0
        self.closed = ""
        backend.runs.append(self)

    async def start(self) -> None:
        await self._ctx.emit(StepChanged(self.step, reason="start"))

    async def on_frame(self, frame: TimedFrame) -> None:
        label = int(frame.image.flat[0]) if frame.image.size else 0
        await self._ctx.emit(OverlayUpdate(frame.timestamp_us, (LID_BOX,),
                                           extra={"step": self.step}))
        if label == HAND:
            self.cues += 1
            await self._ctx.emit(Cue(HAND_CUE, kind="correction"))
        elif label == DONE:
            await self._advance()

    async def command(self, command: RunCommand) -> CommandResult:
        if command.kind in ("next", "advance"):
            await self._advance()
            return CommandResult(True)
        if command.kind == "reset":
            self.step = 0
            self.cues = 0
            await self._ctx.emit(StepChanged(0, reason="reset"))
            return CommandResult(True)
        if command.kind == "check":
            return CommandResult(True, reason="lid is open")
        return CommandResult(False, speech="not supported")

    async def _advance(self) -> None:
        if self.step + 1 >= len(STEPS):
            await self._ctx.emit(RunFinished("completed"))
            return
        self.step += 1
        await self._ctx.emit(StepChanged(self.step, reason="advance"))

    def snapshot(self) -> RunSnapshot:
        return RunSnapshot(self.step, len(STEPS), STEPS[self.step],
                           extra={"holes": [self.step], "cues": self.cues},
                           state={"step": self.step})

    def turn_context(self) -> TurnContext:
        return TurnContext(prompt_block="\n\nScripted context.")

    async def input_changed(self) -> None:
        return None

    async def close(self, reason: str) -> None:
        self.closed = reason


class ScriptedBackend:
    name = "scripted"

    def __init__(self, capabilities: Capabilities | None = None) -> None:
        self.capabilities = capabilities or Capabilities()
        self.runs: list[ScriptedRun] = []

    @property
    def title(self) -> str:
        return "scripted"

    def steps(self) -> list[StepInfo]:
        return [StepInfo(number=i + 1, instruction=s) for i, s in enumerate(STEPS)]

    def parts(self) -> tuple[str, ...]:
        return ()

    def validate(self) -> list[str]:
        return []

    def instructions_digest(self) -> list[str]:
        return list(STEPS)

    def preview_annotator(self) -> Any:
        return None

    async def open_run(self, ctx: RunContext, *, start_step: int,
                       checkpoint: Mapping[str, Any] | None) -> ScriptedRun:
        return ScriptedRun(self, ctx, start_step)
