# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A second SOP is data only: a folder the built-in ``vlm`` backend can guide.

``fixtures/procedures/filter-swap`` has a ``procedure.yaml``, a schema v1
``sop.json`` and one reference frame, and no detector profile, geometry
module or Python. These tests load it next to the shipped nose pad procedure
through the same catalog, tools and backend factory the worker uses, then
guide a run on it against a stubbed vision model.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from conftest import FakePorts, eventually
from sop_guidance.backends.base import BackendServices, ModelHandles, RunCommand, TimedFrame
from sop_guidance.backends.registry import resolve_backend
from sop_guidance.backends.vlm import VlmBackend
from sop_guidance.host import GuidanceHost, HostSettings, LoadedProcedure
from sop_guidance.procedures import (
    GuidanceDefaults,
    ProcedureEntry,
    discover_procedures,
    load_procedure,
)
from sop_guidance.recorder import SessionStore
from sop_guidance.tools import build_guidance_tools
from sop_guidance_worker.config import load_config
from xr_ai_tools.tool_calling import tool_definitions

APP = Path(__file__).resolve().parents[1]
FIXTURE = APP / "tests" / "fixtures" / "procedures" / "filter-swap"
SHIPPED = APP / "procedures" / "nosepad-replacement"


def _catalog(tmp_path: Path) -> list[ProcedureEntry]:
    """Both folders under one procedures dir, loaded with the shipped defaults."""

    root = tmp_path / "procedures"
    root.mkdir()
    (root / SHIPPED.name).symlink_to(SHIPPED, target_is_directory=True)
    (root / FIXTURE.name).symlink_to(FIXTURE, target_is_directory=True)
    config = load_config(APP / "yaml" / "sop_guidance_worker.yaml")
    return discover_procedures(root, config.guidance_defaults)


def _build(entry: ProcedureEntry, tmp_path: Path) -> Any:
    """The worker's build path, without vision support installed."""

    factory = resolve_backend(entry.spec.backend)
    return factory(BackendServices(entry=entry, config=entry.spec.backend_config,
                                   artifacts_dir=tmp_path / "artifacts"))


def test_catalog_loads_the_shipped_and_the_new_procedure(tmp_path: Path) -> None:
    entries = {e.id: e for e in _catalog(tmp_path)}

    assert sorted(entries) == ["filter-swap", "nosepad-replacement"]
    spec = entries["filter-swap"].spec
    assert spec.backend == "vlm"
    assert "detector" not in spec.backend_config and "geometry" not in spec.backend_config
    # The worker's defaults still reach a procedure that set none of them.
    assert spec.backend_config["monitor"]["check_interval_s"] == 1.0
    assert spec.foreground.reminder_interval_s == 60


def test_tool_enum_offers_both_procedures(tmp_path: Path) -> None:
    procedures = [LoadedProcedure(entry=e, backend=_build(e, tmp_path), models=ModelHandles())
                  for e in _catalog(tmp_path)]
    host = GuidanceHost(procedures=procedures, store=SessionStore(tmp_path / "run", level="off"),
                        ports=FakePorts())
    tools = build_guidance_tools(host, active=False)
    schema = next(d.parameters for d in tool_definitions(tools) if d.name == "guidance__start")
    procedure_id = schema["properties"]["procedure_id"]

    assert procedure_id["enum"] == ["filter-swap", "nosepad-replacement"]
    assert 'filter-swap: "water filter swap" (also: water filter' in procedure_id["description"]
    assert 'nosepad-replacement: "nosepad replacement"' in procedure_id["description"]


def test_new_procedure_builds_a_vlm_backend_without_vision(tmp_path: Path) -> None:
    entries = {e.id: e for e in _catalog(tmp_path)}
    fixture = _build(entries["filter-swap"], tmp_path)
    shipped = _build(entries["nosepad-replacement"], tmp_path)

    assert isinstance(fixture, VlmBackend)
    assert fixture.preview_annotator() is None
    assert fixture.validate() == []
    assert [s.gradeable for s in fixture.steps()] == [False, True, False]
    assert fixture.steps()[1].reference_images == (str(FIXTURE / "frames" / "step_02.jpg"),)
    # The contrast: the nose pad needs its detector, this one needs nothing.
    assert any("vision support is unavailable" in p for p in shipped.validate())


@dataclass
class _Reply:
    content: str


class StubVlm:
    """Answers every grading question with a grounded pass."""

    PASS = json.dumps({
        "observation": "a white cartridge sits in the jug funnel",
        "requirements": {"New cartridge seated in the funnel": {
            "visible": True, "evidence": "white cartridge flush in the funnel"}},
        "issue": "",
    })

    def __init__(self) -> None:
        self.asked: list[list[str]] = []

    async def ask_image(self, path: Path, question: str) -> _Reply:
        return await self.ask_images([path], question)

    async def ask_images(self, paths: list[Path], question: str) -> _Reply:
        self.asked.append([str(p) for p in paths])
        return _Reply(self.PASS)


class CameraPorts(FakePorts):
    """Host ports with a live camera that always has a fresh frame."""

    async def fetch_frame(self, participant_id: str) -> TimedFrame | None:
        # A textured frame at a gradeable size: blank or tiny frames are skipped.
        image = np.tile(np.arange(320, dtype=np.uint8), (240, 1))[:, :, None].repeat(3, axis=2)
        return TimedFrame(participant_id=participant_id, timestamp_us=time.time_ns() // 1_000,
                          width=320, height=240, image=image)


async def test_guidance_runs_on_the_new_procedure(tmp_path: Path) -> None:
    # Checks as fast as the loop allows; every other setting is the default.
    fast = GuidanceDefaults(backend_config={"vlm": {"monitor": {
        "tick_s": 0.01, "check_interval_s": 0.01, "speech_lead_s": 0.0}}})
    entry = load_procedure(FIXTURE, fast)
    vlm = StubVlm()
    ports = CameraPorts()
    store = SessionStore(tmp_path / "run", level="off")
    await store.start()
    host = GuidanceHost(
        procedures=[LoadedProcedure(entry=entry, backend=_build(entry, tmp_path),
                                    models=ModelHandles(vlm=vlm))],
        store=store, ports=ports, settings=HostSettings(step_ack_timeout_s=0),
    )

    def announced() -> list[str]:
        return [t for _, t, kind in ports.said if kind == "announcement"]

    try:
        assert (await host.begin("alice", "filter-swap")).status == "started"
        assert announced() == ["Step 1 of 3: Lift the old filter cartridge out of the jug."]

        # Step 1 has no reference frame, so the wearer's word advances it.
        assert (await host.command("alice", RunCommand("advance"))).accepted
        # Step 2 has one: the monitor grades it and advances on its own.
        await eventually(lambda: len(announced()) == 3)

        assert announced()[1:] == [
            "Step 2 of 3: Press the new filter cartridge into the jug until it clicks.",
            "Step 3 of 3: Put the lid back on the jug.",
        ]
        teacher = str(FIXTURE / "frames" / "step_02.jpg")
        assert sum(paths[0] == teacher for paths in vlm.asked) >= 2  # the pass streak
        state = ports.states[-1]
        assert (state["procedure_id"], state["backend"], state["step"]) == ("filter-swap", "vlm", 3)
    finally:
        await host.shutdown()
        await store.aclose()


def test_blank_and_tiny_frames_are_not_graded() -> None:
    from sop_guidance.backends.vlm.config import EvaluatorSettings
    from sop_guidance.backends.vlm.grading import is_parser_issue
    from sop_guidance.backends.vlm.run import _unusable_frame

    evaluator = EvaluatorSettings()
    black = np.zeros((180, 320, 3), dtype=np.uint8)
    tiny = np.tile(np.arange(64, dtype=np.uint8), (48, 1))[:, :, None].repeat(3, axis=2)
    textured = np.tile(np.arange(320, dtype=np.uint8), (240, 1))[:, :, None].repeat(3, axis=2)

    # A camera that is off sends black frames; grading one described the
    # teacher's reference and advanced the step.
    assert "blank" in _unusable_frame(black, evaluator)
    assert "too small" in _unusable_frame(tiny, evaluator)
    assert _unusable_frame(textured, evaluator) == ""
    # Never spoken as a correction: the monitor treats it as a missing frame.
    assert is_parser_issue(_unusable_frame(black, evaluator))
