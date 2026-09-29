# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The judge behind the real guidance host, with a scripted detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import make_harness, settle
from rpi_hat_judge.backend import RpiHatJudgeBackend
from rpi_hat_judge.board import HOLE_ANCHORS
from rpi_hat_judge.config import RpiHatJudgeConfig
from rpi_hat_judge.spec import SpecError, load_spec
from sop_guidance.backends.base import RunCommand, TimedFrame
from sop_guidance.vision import Detection, DetectorProfile

SPEC = Path(__file__).resolve().parents[3] / "procedures" / "rpi-hat-assembly" / "sop_rpi_hat.json"
BOARD = (400.0, 300.0, 1400.0, 1000.0)
TICK_US = 333_333


class FakeAnnotator:
    """Hands the run one scripted frame of boxes per call."""

    def __init__(self) -> None:
        labels = {c: c for c in ("board", "hole", "screw", "hole_filled", "fpc")}
        self.profile = DetectorProfile.model_validate({
            "enabled": True, "overlay": {"class_labels": labels},
            "hands": {"enabled": True, "model": "hand.pt"},
        })
        self.next: list[Detection] = []

    async def detect_array(self, image):
        return list(self.next)

    def draw(self, image, detections):
        return image


def boxes(installed=(), cable_seated=False, hand=False) -> list[Detection]:
    x1, y1, x2, y2 = BOARD
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    out = [Detection("board", *BOARD, 0.97)]
    for k, (rx, ry) in HOLE_ANCHORS.items():
        ax, ay = cx + rx * bw, cy + ry * bh
        label = "hole_filled" if k in installed else "hole"
        out.append(Detection(label, ax - 15, ay - 15, ax + 15, ay + 15, 0.9))
    if cable_seated:
        out.append(Detection("fpc", x1 + 20, cy - 80, x1 + 200, cy + 80, 0.9))
    if hand:
        out.append(Detection("hand", x1, y1, x2, y2, 0.9))
    return out


class Feeder:
    def __init__(self, harness, annotator: FakeAnnotator) -> None:
        self.harness = harness
        self.annotator = annotator
        self.t_us = 1_000_000_000

    async def feed(self, frame_boxes: list[Detection], ticks: int) -> None:
        session = self.harness.host.session_of("alice")
        for _ in range(ticks):
            if session.ended:
                return
            self.annotator.next = frame_boxes
            self.t_us += TICK_US
            await session.run.on_frame(TimedFrame(
                participant_id="alice", timestamp_us=self.t_us, width=1920, height=1080,
                image=np.zeros((4, 4, 3), dtype=np.uint8),
            ))
        await settle()


async def judge(tmp_path: Path):
    annotator = FakeAnnotator()
    backend = RpiHatJudgeBackend(procedure_id="lid-demo", spec=load_spec(SPEC),
                                 config=RpiHatJudgeConfig(), annotator=annotator)
    harness = await make_harness(tmp_path, backend=backend)
    await harness.host.begin("alice", "lid-demo")
    return harness, Feeder(harness, annotator)


async def close(harness) -> None:
    await harness.host.shutdown()
    await harness.store.aclose()


async def test_the_camera_walks_the_whole_assembly(tmp_path: Path) -> None:
    harness, feeder = await judge(tmp_path)
    try:
        await feeder.feed(boxes(), 8)
        for up_to in (1, 2, 3, 4):
            await feeder.feed(boxes(installed=range(1, up_to + 1)), 12)
        await feeder.feed(boxes(installed=(1, 2, 3, 4), cable_seated=True), 14)
        assert harness.ports.texts("alice") == [
            "Step 1 of 5: Lay the board flat on the desk and drive screw 1 into the top-left hole.",
            "Step 1, install the top-left screw, done. Step 2 of 5: Tighten diagonally: "
            "drive screw 2 into the bottom-right hole.",
            "Step 2, install the bottom-right screw, done. Step 3 of 5: Drive screw 3 into "
            "the top-right hole.",
            "Step 3, install the top-right screw, done. Step 4 of 5: Drive screw 4 into the "
            "bottom-left hole. That completes all four corners.",
            "Step 4, install the bottom-left screw, done. Step 5 of 5: Insert the FPC ribbon "
            "cable into the connector on the board's left edge and press the latch down.",
            "Step 5, seat the ribbon cable, done.",
            "You've completed all steps in 'lid demo'. Well done!",
        ]
        assert harness.host.session_of("alice") is None
        assert harness.ports.states[-1]["outcome"] == "completed"
        # The run drew its own boxes and published the hole map.
        labels = {d.label for d in harness.ports.overlays[-1].detections}
        assert {"board", "hole", "hole_filled", "fpc"} <= labels
        running = [s for s in harness.ports.states if s["status"] == "running"]
        assert running[-1]["extra"]["holes"]["4"] == "installed"
    finally:
        await close(harness)


async def test_an_out_of_order_screw_is_a_standing_correction(tmp_path: Path) -> None:
    harness, feeder = await judge(tmp_path)
    try:
        await feeder.feed(boxes(), 8)
        await feeder.feed(boxes(installed=[1]), 12)
        await feeder.feed(boxes(installed=[1, 3]), 12)
        said = harness.ports.texts("alice")[-1]
        assert said == ("Out of order. screw 3 went in while hole 2, bottom-right is still "
                        "empty. Next, install the bottom-right screw")
        state = harness.ports.states[-1]
        assert state["step"] == 2
        assert state["correction"]["text"] == said
        assert state["extra"]["steps_owed"] == [2]
        # Putting in the missing screw moves on and clears the correction.
        await feeder.feed(boxes(installed=[1, 2, 3]), 12)
        assert harness.ports.states[-1]["step"] == 4
        assert harness.ports.states[-1]["correction"] == {}
    finally:
        await close(harness)


async def test_a_hand_over_the_board_holds_progress(tmp_path: Path) -> None:
    harness, feeder = await judge(tmp_path)
    try:
        await feeder.feed(boxes(installed=[1], hand=True), 20)
        assert harness.host.session_of("alice").run.snapshot().step_index == 0
        extra = harness.ports.states[-1]["extra"]
        assert not extra["trusted"] and "hand covers" in extra["reject_reason"]
    finally:
        await close(harness)


async def test_voice_cannot_advance_and_repeat_reads_the_step(tmp_path: Path) -> None:
    harness, feeder = await judge(tmp_path)
    try:
        refused = await harness.host.command("alice", RunCommand("next"))
        assert not refused.accepted
        assert refused.speech.startswith("The camera confirms each step")
        repeat = await harness.host.command("alice", RunCommand("repeat"))
        assert repeat.speech == ("Step 1 of 5: Install the top-left screw. Lay the board flat "
                                 "on the desk and drive screw 1 into the top-left hole.")
        context = harness.host.turn_context("alice")
        assert "never say a step is done" in context.prompt_block
        assert "Camera view: no frame judged yet." in context.prompt_block
    finally:
        await close(harness)


async def test_reset_starts_the_judge_over(tmp_path: Path) -> None:
    harness, feeder = await judge(tmp_path)
    try:
        await feeder.feed(boxes(), 8)
        await feeder.feed(boxes(installed=[1]), 12)
        assert harness.host.session_of("alice").run.snapshot().step_index == 1
        await harness.host.command("alice", RunCommand("reset"))
        await settle()
        assert harness.host.session_of("alice").run.snapshot().step_index == 0
        assert harness.ports.texts("alice")[-1].startswith("Step 1 of 5:")
    finally:
        await close(harness)


def test_validate_names_a_profile_that_does_not_label_the_judge_classes() -> None:
    annotator = FakeAnnotator()
    annotator.profile = annotator.profile.model_copy(update={
        "overlay": annotator.profile.overlay.model_copy(update={"class_labels": {"board": "b"}}),
    })
    backend = RpiHatJudgeBackend(procedure_id="x", spec=load_spec(SPEC),
                                 config=RpiHatJudgeConfig(), annotator=annotator)
    problems = backend.validate()
    assert any("does not label" in p for p in problems)
    assert any("hand.pt do not exist" in p for p in problems)


def test_spec_must_follow_the_board_screw_order(tmp_path: Path) -> None:
    import json

    raw = json.loads(SPEC.read_text())
    raw["steps"][0]["hole"] = 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw))
    with pytest.raises(SpecError, match="names hole 2"):
        load_spec(bad)
