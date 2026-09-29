# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The judge's own SOP file, ``sop_rpi_hat.json``, kept in its native shape.

It is not forced into the schema v1 SOP the ``vlm`` backend reads: there are
no reference frames to grade against, and each step names the hole it fills
instead. Screen text and speech are written separately in it, because a
bracket or a circled digit reads fine but speaks badly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .board import SOP_SEQUENCE
from .texts import tts


class SpecError(ValueError):
    """The SOP file is missing a field or disagrees with the board layout."""


@dataclass(frozen=True, slots=True)
class JudgeStep:
    id: int
    title: str
    instruction: str
    """Screen text, as written (circled digits and all)."""

    tip: str
    speech: str
    """The short name spoken in cues: "install the top-left screw"."""

    hole: int | None
    nominal_sec: float


@dataclass(frozen=True, slots=True)
class JudgeSpec:
    name: str
    steps: tuple[JudgeStep, ...]

    @property
    def total(self) -> int:
        return len(self.steps)

    def index_of_hole(self, hole: int) -> int:
        return next(i for i, s in enumerate(self.steps) if s.hole == hole)

    @property
    def fpc_index(self) -> int:
        """The cable step: the first one after the four screws."""
        return len(SOP_SEQUENCE)

    def spoken_instruction(self, index: int) -> str:
        """The instruction the host reads out, with circled digits made speakable."""
        text = tts(self.steps[index].instruction)
        return text if text[-1:] in ".!?" else f"{text}."


def load_spec(path: Path) -> JudgeSpec:
    """Load and check the SOP file; the English fields are required."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})
    steps: list[JudgeStep] = []
    for entry in raw.get("steps", []):
        if not entry.get("required", True):
            continue
        missing = [k for k in ("title_en", "instruction_en", "tip_en") if not entry.get(k)]
        if missing:
            raise SpecError(f"{path}: step {entry.get('id')} lacks {', '.join(missing)}")
        steps.append(JudgeStep(
            id=int(entry["id"]),
            title=entry["title_en"],
            instruction=entry["instruction_en"],
            tip=entry["tip_en"],
            speech=entry.get("speech_en") or entry["title_en"],
            hole=entry.get("hole"),
            nominal_sec=float(entry.get("nominal_sec", defaults.get("nominal_sec", 10.0))),
        ))
    # The screw order is a fact of the board (SOP_SEQUENCE); a step's `hole`
    # only makes the file self-explaining, and a disagreement is a config error.
    if len(steps) != len(SOP_SEQUENCE) + 1:
        raise SpecError(f"{path}: expected {len(SOP_SEQUENCE)} screw steps and the cable step, "
                        f"found {len(steps)} steps")
    for step, hole in zip(steps, SOP_SEQUENCE, strict=False):
        if step.hole is not None and step.hole != hole:
            raise SpecError(f"{path}: step {step.id} names hole {step.hole}, "
                            f"but the screw order puts hole {hole} there")
    steps[:len(SOP_SEQUENCE)] = [
        JudgeStep(s.id, s.title, s.instruction, s.tip, s.speech, hole, s.nominal_sec)
        for s, hole in zip(steps, SOP_SEQUENCE, strict=False)
    ]
    return JudgeSpec(name=raw.get("name_en") or raw.get("name", ""), steps=tuple(steps))


__all__ = ["JudgeSpec", "JudgeStep", "SpecError", "load_spec"]
