# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures: a scripted procedure backend and in-memory host ports.

The scripted backend is deliberately not the ``vlm`` one. It decides progress
from commands alone, the way a custom detector-and-state-machine backend
would, which is what proves the host, tools and interaction layer work for a
backend they know nothing about.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest_asyncio
from sop_guidance.backends.base import (
    Capabilities,
    CommandResult,
    ModelHandles,
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
from sop_guidance.host import GuidanceHost, HostSettings, LoadedProcedure
from sop_guidance.procedures import GuidanceDefaults, load_procedure
from sop_guidance.recorder import SessionStore

STEPS = ("Open the lid.", "Lift the tray.", "Close the lid.")


class ScriptedRun:
    def __init__(self, backend: ScriptedBackend, ctx: RunContext, start_step: int) -> None:
        self._backend = backend
        self._ctx = ctx
        self.step = start_step
        self.closed = ""
        backend.runs.append(self)

    async def start(self) -> None:
        await self._ctx.emit(StepChanged(self.step, reason="start"))

    async def on_frame(self, frame: TimedFrame) -> None:
        return None

    async def command(self, command: RunCommand) -> CommandResult:
        if command.kind in ("next", "advance"):
            if self.step + 1 >= len(STEPS):
                await self._ctx.emit(RunFinished("completed"))
                return CommandResult(True)
            self.step += 1
            await self._ctx.emit(StepChanged(self.step, reason="advance"))
            return CommandResult(True)
        if command.kind == "check":
            return CommandResult(True, reason="lid is open")
        return CommandResult(False, speech="not supported")

    def snapshot(self) -> RunSnapshot:
        return RunSnapshot(self.step, len(STEPS), STEPS[self.step],
                           extra={"holes": [self.step]}, state={"step": self.step})

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


@dataclass
class FakePorts:
    said: list[tuple[str, str, str]] = field(default_factory=list)
    states: list[dict[str, Any]] = field(default_factory=list)
    previews: list[tuple[str, str, str]] = field(default_factory=list)
    overlays: list[OverlayUpdate] = field(default_factory=list)
    connected: set[str] = field(default_factory=lambda: {"alice", "bob", "glasses"})

    async def say(self, owner: str, text: str, *, kind: str) -> None:
        self.said.append((owner, text, kind))

    def speech_remaining_s(self, owner: str) -> float:
        return 0.0

    def last_heard_us(self, owner: str) -> int:
        return 0

    async def publish_state(self, state: Mapping[str, Any]) -> None:
        self.states.append(dict(state))

    def latest_frame(self, participant_id: str) -> TimedFrame | None:
        return None

    async def fetch_frame(self, participant_id: str) -> TimedFrame | None:
        return None

    async def start_preview(self, owner: str, input_pid: str, backend: Any) -> None:
        self.previews.append(("start", owner, input_pid))

    async def stop_preview(self, owner: str) -> None:
        self.previews.append(("stop", owner, ""))

    async def overlay_update(self, owner: str, update: OverlayUpdate) -> None:
        self.overlays.append(update)

    def resolve_input(self, owner: str, saved: str) -> str:
        return saved if saved in self.connected else owner

    def texts(self, owner: str | None = None) -> list[str]:
        return [t for o, t, _ in self.said if owner is None or o == owner]


def write_procedure(root: Path, procedure_id: str = "lid-demo", **extra: str) -> Path:
    folder = root / procedure_id
    folder.mkdir(parents=True)
    lines = [f"id: {procedure_id}", "title: lid demo", "aliases: [the lid]", "backend: scripted"]
    lines += [f"{k}: {v}" for k, v in extra.items()]
    (folder / "procedure.yaml").write_text("\n".join(lines) + "\n")
    return folder


@dataclass
class HostHarness:
    host: GuidanceHost
    ports: FakePorts
    backend: ScriptedBackend
    store: SessionStore
    llm_replies: list[str]


async def make_harness(
    tmp_path: Path,
    *,
    capabilities: Capabilities | None = None,
    settings: HostSettings | None = None,
) -> HostHarness:
    folder = write_procedure(tmp_path / "procedures")
    entry = load_procedure(folder, GuidanceDefaults())
    backend = ScriptedBackend(capabilities)
    store = SessionStore(tmp_path / "run", level="off")
    await store.start()
    ports = FakePorts()
    replies: list[str] = []

    async def llm_text(system: str, user: str, max_tokens: int, temperature: float) -> str:
        return replies.pop(0) if replies else "NONE"

    host = GuidanceHost(
        procedures=[LoadedProcedure(entry=entry, backend=backend, models=ModelHandles())],
        store=store, ports=ports, settings=settings or HostSettings(step_ack_timeout_s=0),
        llm_text=llm_text,
    )
    return HostHarness(host, ports, backend, store, replies)


@pytest_asyncio.fixture
async def harness(tmp_path: Path) -> AsyncIterator[HostHarness]:
    h = await make_harness(tmp_path)
    try:
        yield h
    finally:
        await h.host.shutdown()
        await h.store.aclose()


async def settle() -> None:
    """Let host tasks scheduled from run events finish."""

    for _ in range(5):
        await asyncio.sleep(0)
