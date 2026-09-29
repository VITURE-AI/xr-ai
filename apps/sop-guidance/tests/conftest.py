# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures: in-memory host ports and a host around a procedure backend.

The default backend is the scripted one from ``fixtures/scripted_backend.py``,
deliberately not the ``vlm`` one: it proves the host, tools and interaction
layer work for a backend they know nothing about.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest_asyncio
from fixtures.scripted_backend import ScriptedBackend
from sop_guidance.backends.base import (
    Capabilities,
    ModelHandles,
    OverlayUpdate,
    ProcedureBackend,
    TimedFrame,
)
from sop_guidance.host import GuidanceHost, HostSettings, LoadedProcedure
from sop_guidance.procedures import GuidanceDefaults, load_procedure
from sop_guidance.recorder import SessionStore


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
    backend: Any
    """A :class:`ScriptedBackend` unless the test passed its own."""

    store: SessionStore
    llm_replies: list[str]


async def make_harness(
    tmp_path: Path,
    *,
    capabilities: Capabilities | None = None,
    settings: HostSettings | None = None,
    backend: ProcedureBackend | None = None,
    level: str = "off",
) -> HostHarness:
    folder = write_procedure(tmp_path / "procedures")
    entry = load_procedure(folder, GuidanceDefaults())
    backend = backend if backend is not None else ScriptedBackend(capabilities)
    store = SessionStore(tmp_path / "run", level=level)
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


async def eventually(condition: Callable[[], bool], *, timeout_s: float = 5.0) -> None:
    """Wait for *condition*, for events that cross a task or process boundary."""

    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)

