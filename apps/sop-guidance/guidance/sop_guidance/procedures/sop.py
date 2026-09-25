# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema v1 SOP files: the step data a VLM-graded procedure is guided from.

The format is the one the old glasses worker loaded from ``sop_dir``::

    {
      "schema_version": 1,
      "name": "nosepad replacement",
      "summary": "...",
      "parts": ["Size 0 nose pad (detector label nosepad_0) -- ..."],
      "steps": [
        {
          "description": "Grasp the installed nose pad ...",
          "image": "frames/step_02_b.jpg",
          "before_image": "frames/step_02_a.jpg",
          "teacher_caption": "...",
          "expected_requirements": ["bridge of the glasses is now empty"],
          "key_info": {"objects": [...], "action": "...", "position": "...",
                       "target_state": "...", "ignore": [...]},
          "reference_images": ["frames/step_02_b.jpg"],
          "geometry_gate": "no_pad_on_glasses",
          "hold_seconds": 5.0
        }
      ]
    }

``image`` is the teacher's AFTER state, the frame the student is compared
against. ``before_image`` names the state before the step; supply it for any
step defined by a change rather than an appearance, so the grader is asked
which of two states the student resembles instead of how closely they resemble
one. ``geometry_gate`` names a deterministic detector check that can only veto
a positive verdict; which names are valid depends on the procedure's geometry
profile, so it is validated by the backend rather than here.

Loading is lenient where guidance can still run (a missing image falls back to
text-only checks) and strict where it cannot (no name, no usable step).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

SOP_SCHEMA_VERSION = 1

_DEFAULT_IGNORE = ("background", "lighting", "camera angle")

# A hold delays an advance, so a typo such as 600 would strand the wearer on a
# step with no indication why. Clamp instead of trusting it.
_HOLD_SECONDS_MAX = 60.0


class SopFormatError(ValueError):
    """An SOP file exists but cannot be turned into a guidable procedure."""


@dataclass(frozen=True, slots=True)
class KeyInfo:
    """The few facts that define a step, so the grader checks only these."""

    objects: tuple[str, ...] = ()
    action: str = ""
    position: str = ""
    target_state: str = ""
    ignore: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        return not (self.objects or self.action or self.position or self.target_state)

    def as_prompt_block(self) -> str:
        """Render the key info as a compact block for model prompts."""

        lines: list[str] = []
        if self.objects:
            lines.append(f"  Key objects: {', '.join(self.objects)}")
        if self.action:
            lines.append(f"  Action: {self.action}")
        if self.position:
            lines.append(f"  Position/placement: {self.position}")
        if self.target_state:
            lines.append(f"  Target end-state: {self.target_state}")
        ignore = list(self.ignore) or list(_DEFAULT_IGNORE)
        lines.append(f"  IGNORE (must not affect the verdict): {', '.join(ignore)}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SopStep:
    """One authored step."""

    number: int
    description: str
    image_path: str = ""
    before_image_path: str = ""
    teacher_caption: str = ""
    expected_requirements: tuple[str, ...] = ()
    key_info: KeyInfo | None = None
    reference_image_paths: tuple[str, ...] = ()
    geometry_gate: str = ""
    hold_seconds: float = 0.0

    @property
    def reference_reliable(self) -> bool:
        """Whether the automatic monitor can ever grade this step by looking.

        One definition because the monitor's short-circuit, the guidance
        prompt's "take their word" note, and the advance gate's ungradeable
        escape hatch all read it and must agree.
        """

        return bool(self.image_path)


@dataclass(frozen=True, slots=True)
class Sop:
    """A parsed schema v1 SOP."""

    name: str
    steps: tuple[SopStep, ...]
    summary: str = ""
    parts: tuple[str, ...] = ()
    source_path: str = ""
    instructions: tuple[str, ...] = field(default=())

    def instruction(self, index: int) -> str:
        """The spoken instruction for the 0-based step *index*."""

        if self.instructions and index < len(self.instructions):
            return self.instructions[index]
        return self.steps[index].description


def _strlist(value: Any, *, limit: int | None = None) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out = [str(v).strip() for v in value if str(v).strip()]
    return out[:limit] if limit else out


def _key_info(value: Any) -> KeyInfo | None:
    if not isinstance(value, dict):
        return None
    ignore = _strlist(value.get("ignore"), limit=8)
    lowered = {i.lower() for i in ignore}
    ignore.extend(d for d in _DEFAULT_IGNORE if d not in lowered)
    info = KeyInfo(
        objects=tuple(_strlist(value.get("objects"), limit=3)),
        action=str(value.get("action", "")).strip(),
        position=str(value.get("position", "")).strip(),
        target_state=str(value.get("target_state", "")).strip(),
        ignore=tuple(ignore),
    )
    return None if info.is_empty() else info


def _hold_seconds(raw: dict[str, Any], name: str, number: int) -> float:
    if "hold_seconds" not in raw:
        return 0.0
    try:
        held = float(raw.get("hold_seconds") or 0.0)
    except (TypeError, ValueError):
        logger.warning("sop {!r}: step {} hold_seconds={!r} is not a number; ignored",
                       name, number, raw.get("hold_seconds"))
        return 0.0
    if held < 0.0:
        logger.warning("sop {!r}: step {} hold_seconds={} is negative; ignored",
                       name, number, held)
        return 0.0
    if held > _HOLD_SECONDS_MAX:
        logger.warning("sop {!r}: step {} hold_seconds={} exceeds {:.0f}s; clamped",
                       name, number, held, _HOLD_SECONDS_MAX)
        return _HOLD_SECONDS_MAX
    return held


def _resolve_image(raw: Any, base_dir: Path) -> str:
    """Resolve one image reference, or "" when it is not a readable file.

    A missing image is not fatal: the step falls back to text-only checks.
    """

    text = str(raw or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        logger.warning("sop: image {} not found; step falls back to text-only checks", path)
        return ""
    return str(path.resolve())


def _parts(data: dict[str, Any], name: str) -> tuple[str, ...]:
    raw = data.get("parts")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        logger.warning("sop {!r}: 'parts' must be a list of strings; ignored", name)
        return ()
    out: list[str] = []
    for entry in raw:
        if isinstance(entry, str) and entry.strip():
            out.append(entry.strip())
        else:
            logger.warning("sop {!r}: ignoring non-string 'parts' entry {!r}", name, entry)
    return tuple(out)


def sop_from_dict(
    data: dict[str, Any],
    *,
    base_dir: Path,
    default_name: str = "",
    source_path: str = "",
) -> Sop:
    """Build an :class:`Sop` from a parsed mapping.

    Raises :class:`SopFormatError` for an unreadable version, a missing name,
    or no step with a description.
    """

    version = data.get("schema_version", SOP_SCHEMA_VERSION)
    if not isinstance(version, int) or version < 1:
        raise SopFormatError(f"schema_version must be a positive integer, got {version!r}")
    if version > SOP_SCHEMA_VERSION:
        raise SopFormatError(
            f"schema_version {version} is newer than this worker understands "
            f"({SOP_SCHEMA_VERSION})"
        )
    name = str(data.get("name", "") or default_name).strip()
    if not name:
        raise SopFormatError("missing 'name' and no filename to fall back to")
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise SopFormatError(f"{name!r}: 'steps' must be a non-empty list")

    steps: list[SopStep] = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            logger.warning("sop {!r}: step {} is not an object; skipped", name, index + 1)
            continue
        description = str(raw.get("description", "")).strip()
        if not description:
            logger.warning("sop {!r}: step {} has no description; skipped", name, index + 1)
            continue
        number = len(steps) + 1
        image = _resolve_image(raw.get("image"), base_dir)
        references = [
            resolved
            for candidate in _strlist(raw.get("reference_images"))
            if (resolved := _resolve_image(candidate, base_dir))
        ]
        if image and image not in references:
            references.insert(0, image)
        before = _resolve_image(raw.get("before_image"), base_dir)
        steps.append(SopStep(
            number=number,
            description=description,
            image_path=image,
            before_image_path=before,
            teacher_caption=str(raw.get("teacher_caption", "")).strip() if image else "",
            expected_requirements=tuple(_strlist(raw.get("expected_requirements"), limit=4)),
            key_info=_key_info(raw.get("key_info")),
            reference_image_paths=tuple(references),
            geometry_gate=str(raw.get("geometry_gate", "")).strip().lower(),
            hold_seconds=_hold_seconds(raw, name, number),
        ))
    if not steps:
        raise SopFormatError(f"{name!r}: no step had a description")
    return Sop(
        name=name,
        steps=tuple(steps),
        summary=str(data.get("summary", "")).strip(),
        parts=_parts(data, name),
        source_path=source_path,
        instructions=tuple(s.description for s in steps),
    )


def load_sop_file(path: Path) -> Sop:
    """Parse one ``.json``, ``.yaml`` or ``.yml`` SOP file."""

    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise SopFormatError(f"{path}: not parseable ({exc})") from exc
    if not isinstance(data, dict):
        raise SopFormatError(f"{path}: top level must be an object")
    return sop_from_dict(
        data,
        base_dir=path.parent,
        default_name=path.stem.replace("_", " ").replace("-", " ").strip(),
        source_path=str(path),
    )


__all__ = [
    "SOP_SCHEMA_VERSION",
    "KeyInfo",
    "Sop",
    "SopFormatError",
    "SopStep",
    "load_sop_file",
    "sop_from_dict",
]
