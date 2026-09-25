# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for the SOP guidance worker.

The worker YAML holds process-wide settings and normal (idle) mode: one voice
pipeline, one wake word, one hub, one session policy. Everything that decides
how one task is guided lives in that task's ``procedure.yaml`` and inherits the
``guidance_defaults`` block here. Paths are resolved relative to the YAML file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sop_guidance.host import HostSettings
from sop_guidance.procedures import GuidanceDefaults

_PROMPTS = Path(__file__).parent / "prompts"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WakeSettings(_Section):
    """How speech is addressed to the assistant."""

    phrase: str = "hey helix"
    aliases: list[str] = Field(default_factory=lambda: [
        "helix", "helics", "healix", "heelix", "helyx", "hylix", "elix", "ilix",
        "alex", "aleks", "alix", "hex",
    ])
    """Names STT turns the wake word into; short ones only match exactly."""

    max_distance: int = Field(default=2, ge=0)
    max_probe: int = Field(default=3, ge=1)
    """Only this many leading words are searched for the name."""

    required_in_guidance: bool = True
    """While guiding, only addressed speech (and bare exits) reaches the assistant."""

    required_in_live: bool = True
    """Default for idle mode; each client may change it with ``xr.wake_mode``."""

    def names(self) -> list[str]:
        name = self.phrase.split()[-1] if self.phrase.split() else ""
        return [n for n in [name, *self.aliases] if n]


class ForegroundConfig(_Section):
    """The conversational foreground."""

    idle_prompt_file: str = str(_PROMPTS / "idle.txt")
    active_prompt_file: str = str(_PROMPTS / "active.txt")
    current_view_prompt_file: str = str(_PROMPTS / "current_view.txt")
    llm_role: str = "llm"
    vlm_role: str = "vlm"
    max_tool_rounds: int = Field(default=4, ge=1)
    max_tokens: int = Field(default=512, ge=16)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    turn_timeout_s: float = Field(default=10.0, gt=0.0)
    send_frame: bool = True
    """Show the active-mode turn the wearer's current annotated view."""

    noise_classifier: bool = True
    """In idle mode, ask the LLM whether an unaddressed-looking transcript is a request."""

    quick_ack: bool = True
    """Send a short text-only progress line while an idle answer is prepared."""


class PreviewConfig(_Section):
    """The annotated return-video track."""

    fps: float = Field(default=30.0, ge=0.0)
    """Target rate of the annotated preview; 0 disables it."""

    live_profile: str = "live-coco"
    """Detector profile drawn outside guidance for clients in ``default`` overlay mode."""

    jpeg_debug_fps: float = Field(default=1.0, ge=0.0)


class VoiceConfig(_Section):
    silence_duration: float = 0.8
    min_speech: float = 0.15
    silero_threshold: float = 0.5
    idle_timeout_secs: float = 0.0


class ApiConfig(_Section):
    """The app HTTP API the UI's server reads procedures and sessions from."""

    host: str = "127.0.0.1"
    port: int = Field(default=8093, ge=0, le=65535)
    token_env: str = "SOP_GUIDANCE_API_TOKEN"
    """Environment variable with the shared bearer secret; unset means no auth."""


class DebugConfig(_Section):
    level: Literal["off", "frames", "full"] = "frames"
    max_bytes: int = Field(default=2_000_000_000, ge=0)
    clip_seconds: float = Field(default=10.0, gt=0.0)
    clip_fps: float = Field(default=1.0, gt=0.0)


class WorkerConfig(_Section):
    """Resolved worker settings."""

    models_config: Path
    voice_gate_yaml: Path
    detectors_yaml: Path
    procedures_dir: Path
    run_dir: Path
    frame_max_age_s: float = 3.0
    frame_timeout_s: float = 5.0
    wake: WakeSettings = Field(default_factory=WakeSettings)
    foreground: ForegroundConfig = Field(default_factory=ForegroundConfig)
    preview: PreviewConfig = Field(default_factory=PreviewConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    guidance: HostSettings = Field(default_factory=HostSettings)
    guidance_defaults: GuidanceDefaults = Field(default_factory=GuidanceDefaults)

    def prompt(self, name: Literal["idle", "active", "current_view"]) -> str:
        path = Path(getattr(self.foreground, f"{name}_prompt_file"))
        return path.read_text(encoding="utf-8").strip()


def _resolve(base: Path | None, raw: str) -> Path:
    path = Path(os.path.expandvars(raw)).expanduser()
    if base is not None and not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(path: Path | None) -> WorkerConfig:
    """Load and validate the worker YAML; paths are relative to it."""

    data: dict[str, Any] = {}
    if path is not None and path.exists():
        with path.open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"worker config must be a YAML mapping: {path}")
        data = loaded
    base = path.parent if path is not None else None
    for key, default in (
        ("models_config", "models.dashscope.json"),
        ("voice_gate_yaml", "voice_gate.yaml"),
        ("detectors_yaml", "detectors.yaml"),
        ("procedures_dir", "../procedures"),
        ("run_dir", "../run"),
    ):
        data[key] = _resolve(base, str(data.get(key, default)))
    run_dir_env = os.environ.get("XR_RUN_DIR", "").strip()
    if run_dir_env:
        data["run_dir"] = Path(run_dir_env).resolve()
    foreground = data.get("foreground") or {}
    for key in ("idle_prompt_file", "active_prompt_file", "current_view_prompt_file"):
        if key in foreground:
            foreground[key] = str(_resolve(base, str(foreground[key])))
    data["foreground"] = foreground
    try:
        return WorkerConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"{path}: {exc}") from exc


__all__ = [
    "ApiConfig",
    "DebugConfig",
    "ForegroundConfig",
    "PreviewConfig",
    "VoiceConfig",
    "WakeSettings",
    "WorkerConfig",
    "load_config",
]
