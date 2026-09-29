# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Settings for the ``rpi_hat_judge`` backend, read from a procedure's ``backend_config``.

The defaults are the values the old judge ran live with (``sop_judge.yaml``
plus ``glasses_demo.py`` defaults). A worker's
``guidance_defaults.backend_config.rpi_hat_judge`` block and the procedure's
own ``backend_config`` override them key by key.

Durations are seconds. The vote counts the state layer takes are ticks, and
are derived as ``round(seconds * tick_hz)`` once per run, as the old judge did.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DetectorSettings(_Section):
    profile: str = "rpi-hat-v7"
    """Name of a profile in the worker's ``detectors.yaml``."""


class TrackerSettings(_Section):
    stable_sec: float = Field(default=0.5, gt=0.0)
    """A new screw count must hold this long to move forward one step."""

    back_sec: float = Field(default=1.5, gt=0.0)
    """...and this long to jump ahead or go back."""

    tol_sec: float = Field(default=1.0, gt=0.0)
    """Untrusted frames tolerated before a pending step change is dropped."""

    hold_display_sec: float = Field(default=1.5, ge=0.0)
    """Minimum time between two changes of the step shown."""

    slot_window: int = Field(default=7, ge=1)
    slot_need: int = Field(default=5, ge=1)
    """A hole flips on slot_need of its last slot_window observations."""

    slot_back_sec: float = Field(default=3.0, ge=0.0)
    """A screw counts as removed only after this long of contrary evidence."""

    hf_conf_min: float = Field(default=0.65, ge=0.0, lt=1.0)
    """Confidence floor for hole_filled boxes. The judge default is 0.80; on the
    glasses a seated screw passed 0.80 in only 14-33% of frames, and 0.65 cut
    false removal alerts 6 -> 2 on replay."""

    installed_tol: float = Field(default=0.12, gt=0.0)
    screw_as_installed: bool = False


class OcclusionSettings(_Section):
    th_hand: float = Field(default=0.40, gt=0.0, le=1.0)
    th_conf: float = Field(default=0.75, ge=0.0, le=1.0)
    min_holes: int = Field(default=2, ge=0, le=6)
    use_hand: bool = True


class SmoothingSettings(_Section):
    hold_sec: float = Field(default=0.20, ge=0.0)
    window: int = Field(default=5, ge=1)


class FpcSettings(_Section):
    iob_min: float = Field(default=0.90, gt=0.0, le=1.0)
    need_sec: float = Field(default=2.5, gt=0.0)
    window_sec: float = Field(default=3.0, gt=0.0)
    unseat_sec: float = Field(default=3.0, gt=0.0)


class RpiHatJudgeConfig(_Section):
    """Everything the ``rpi_hat_judge`` backend reads for one procedure."""

    spec: str = "sop_rpi_hat.json"
    """The judge's SOP file, relative to the procedure folder."""

    tick_hz: float = Field(default=3.0, gt=0.0, le=30.0)
    """Detection rate. The v7 detector at 1280 takes ~300 ms a frame on CPU, so
    3 Hz is its real throughput; vote counts assume this rate is met."""

    record_interval_s: float = Field(default=1.0, gt=0.0)
    """With debug capture on, the least time between two recorded judge frames
    while the reading changes. Step changes, alerts and the finish are always
    recorded."""

    detector: DetectorSettings = Field(default_factory=DetectorSettings)
    tracker: TrackerSettings = Field(default_factory=TrackerSettings)
    occlusion: OcclusionSettings = Field(default_factory=OcclusionSettings)
    smoothing: SmoothingSettings = Field(default_factory=SmoothingSettings)
    fpc: FpcSettings = Field(default_factory=FpcSettings)


__all__ = ["RpiHatJudgeConfig"]
