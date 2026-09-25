# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Procedure discovery and per-task configuration layering.

Each procedure is one self-contained folder with a ``procedure.yaml``::

    procedures/
      nosepad-replacement/
        procedure.yaml    # id, title, aliases, backend, task settings
        sop.json          # read by the vlm backend
        frames/  prompts/

Settings merge in this order, later layers winning: backend code defaults,
the worker's ``guidance_defaults`` block, ``procedure.yaml``, then per-step
fields inside the backend's own step data. Mappings merge key by key; lists
and scalars replace. Every layer is validated at startup, so a typo in a task
file stops the worker with the file and field named rather than surfacing in
the middle of a session.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PROCEDURE_FILE = "procedure.yaml"

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class ProcedureConfigError(ValueError):
    """A procedure folder or the defaults block is invalid."""


class ForegroundSettings(BaseModel):
    """How the conversational foreground behaves while this task runs."""

    model_config = ConfigDict(extra="forbid")

    active_prompt_file: str = ""
    """Task prompt appended to the shared active-mode prompt, relative to the folder."""

    reminder_interval_s: float = Field(default=60.0, ge=0.0)
    """Minimum seconds between "we are still on step N" reminders after Q&A; 0 disables."""


class ModelSettings(BaseModel):
    """Which model roles from the shared models file this task uses."""

    model_config = ConfigDict(extra="forbid")

    llm_role: str = "llm"
    """Role for text calls such as acknowledgements and request extraction."""

    vlm_role: str = "vlm"
    """Role for visual grading and camera questions."""


class ProcedureSpec(BaseModel):
    """The validated content of one ``procedure.yaml`` after merging defaults."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    enabled: bool = True
    backend: str = "vlm"
    foreground: ForegroundSettings = Field(default_factory=ForegroundSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    backend_config: dict[str, Any] = Field(default_factory=dict)
    """Backend-owned settings, validated by that backend's own schema."""

    request_qualifiers: list[str] = Field(default_factory=list)
    """Words that tell this task's interchangeable parts apart ("solid", "wire").

    Two spoken requests that differ only in these words are one choice revised,
    so the newer replaces the older instead of standing beside it.
    """

    ui: dict[str, Any] = Field(default_factory=dict)
    """Client-facing presentation data served by the app API as-is."""

    @field_validator("id")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not _ID_PATTERN.fullmatch(value):
            raise ValueError(
                "must be lowercase letters, digits and hyphens, starting with a "
                "letter or digit, at most 64 characters"
            )
        return value

    @field_validator("aliases")
    @classmethod
    def _aliases(cls, value: list[str]) -> list[str]:
        return [a.strip() for a in value if a and a.strip()]


class GuidanceDefaults(BaseModel):
    """The worker-wide ``guidance_defaults`` block every task inherits."""

    model_config = ConfigDict(extra="forbid")

    foreground: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, Any] = Field(default_factory=dict)
    backend_config: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Per-backend defaults keyed by backend name, e.g. ``{"vlm": {...}}``."""


@dataclass(frozen=True, slots=True)
class ProcedureEntry:
    """One discovered procedure folder and its merged specification."""

    spec: ProcedureSpec
    directory: Path

    @property
    def id(self) -> str:
        return self.spec.id

    def resolve(self, relative: str) -> Path:
        """Resolve a path written in ``procedure.yaml`` against the folder."""

        path = Path(relative)
        return path if path.is_absolute() else (self.directory / path).resolve()

    def active_prompt(self) -> str:
        name = self.spec.foreground.active_prompt_file
        if not name:
            return ""
        return self.resolve(name).read_text(encoding="utf-8").strip()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge *override* onto *base*: mappings recurse, everything else replaces."""

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _format_validation(error: ValidationError) -> str:
    parts = []
    for issue in error.errors():
        where = ".".join(str(p) for p in issue["loc"]) or "<root>"
        parts.append(f"{where}: {issue['msg']}")
    return "; ".join(parts)


def load_procedure(folder: Path, defaults: GuidanceDefaults) -> ProcedureEntry:
    """Load and validate one procedure folder against the worker defaults."""

    path = folder / PROCEDURE_FILE
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ProcedureConfigError(f"{path}: not valid YAML ({exc})") from exc
    if not isinstance(raw, dict):
        raise ProcedureConfigError(f"{path}: top level must be a mapping")
    backend = str(raw.get("backend", "vlm"))
    layered = dict(raw)
    layered["foreground"] = deep_merge(defaults.foreground, raw.get("foreground") or {})
    layered["models"] = deep_merge(defaults.models, raw.get("models") or {})
    layered["backend_config"] = deep_merge(
        defaults.backend_config.get(backend, {}), raw.get("backend_config") or {},
    )
    try:
        spec = ProcedureSpec.model_validate(layered)
    except ValidationError as exc:
        raise ProcedureConfigError(f"{path}: {_format_validation(exc)}") from exc
    if spec.id != folder.name:
        raise ProcedureConfigError(
            f"{path}: id {spec.id!r} must match its folder name {folder.name!r}"
        )
    entry = ProcedureEntry(spec=spec, directory=folder.resolve())
    if spec.foreground.active_prompt_file and not entry.resolve(
        spec.foreground.active_prompt_file
    ).is_file():
        raise ProcedureConfigError(
            f"{path}: foreground.active_prompt_file "
            f"{spec.foreground.active_prompt_file!r} does not exist"
        )
    return entry


def discover_procedures(
    root: Path, defaults: GuidanceDefaults,
) -> list[ProcedureEntry]:
    """Load every ``*/procedure.yaml`` under *root*, sorted by id.

    Disabled procedures are loaded and validated too, so a broken file is
    reported even while it is switched off, but they are left out of the
    returned list.
    """

    if not root.is_dir():
        raise ProcedureConfigError(f"procedures directory {root} does not exist")
    entries: list[ProcedureEntry] = []
    seen: dict[str, Path] = {}
    for folder in sorted(p for p in root.iterdir() if (p / PROCEDURE_FILE).is_file()):
        entry = load_procedure(folder, defaults)
        if entry.id in seen:
            raise ProcedureConfigError(
                f"duplicate procedure id {entry.id!r} in {folder} and {seen[entry.id]}"
            )
        seen[entry.id] = folder
        if entry.spec.enabled:
            entries.append(entry)
    return entries


__all__ = [
    "PROCEDURE_FILE",
    "ForegroundSettings",
    "GuidanceDefaults",
    "ModelSettings",
    "ProcedureConfigError",
    "ProcedureEntry",
    "ProcedureSpec",
    "deep_merge",
    "discover_procedures",
    "load_procedure",
]
