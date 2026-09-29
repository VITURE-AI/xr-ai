# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Settings for the ``vlm`` backend, read from a procedure's ``backend_config``.

The defaults are the values the tuned glasses deployment ran with. A worker's
``guidance_defaults.backend_config.vlm`` block and each ``procedure.yaml``
override them key by key.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MonitorSettings(_Section):
    """Cadence of the automatic completion checks."""

    check_interval_s: float = Field(default=1.0, gt=0.0)
    """Minimum spacing between check launches. A slower VLM sets the real cadence."""

    tick_s: float = Field(default=0.25, gt=0.0)
    """How often the loop looks for a finished check or a due launch."""

    speech_lead_s: float = Field(default=1.5, ge=0.0)
    """How early a check may start against the tail of the step's own speech."""

    settle_s: float = Field(default=0.0, ge=0.0)
    """Extra pause after speech ends before grading resumes."""

    check_timeout_s: float = Field(default=15.0, gt=0.0)
    """Deadline for one grounded check; an overrun counts as no evidence."""

    skip_static_frames: bool = True
    """Skip a check when no newer frame arrived since the last one."""


class ProgressionSettings(_Section):
    """When a positive verdict advances the step."""

    pass_streak: int = Field(default=2, ge=1)
    """Consecutive grounded passes required before advancing."""

    default_hold_s: float = Field(default=0.0, ge=0.0, le=60.0)
    """Seconds a pass must hold when a step sets no ``hold_seconds`` of its own."""

    grounded_verdict_max_age_s: float = Field(default=20.0, gt=0.0)
    """Oldest verdict a spoken advance request may be granted on."""

    verdict_fresh_s: float = Field(default=5.0, gt=0.0)
    """Past this age the foreground is told the observation may be out of date."""


class CorrectionSettings(_Section):
    """How often the same problem is spoken on one step."""

    first_gap_s: float = Field(default=5.0, ge=0.0)
    """Quiet time before the first correction, measured from the last thing heard."""

    backoff_s: float = Field(default=10.0, ge=0.0)
    """Each repeat waits this many seconds times the number of times already spoken."""

    max_gap_s: float = Field(default=60.0, ge=0.0)
    """Longest wait between repeats, so a long step still gets an occasional nudge."""


class EvaluatorSettings(_Section):
    """How one grounded check is run."""

    require_reliable_reference: bool = True
    """Automatic grading only for steps with a teacher reference frame."""

    tier2_parallel: bool = False
    """Run the live-image tier alongside the comparison to cut latency."""

    frame_reuse_max_age_ms: float = Field(default=500.0, ge=0.0)
    """Grade the preview's newest annotated frame when it is at most this old."""

    jpeg_quality: int = Field(default=85, ge=10, le=100)
    """Encoding quality of the graded student frame."""

    min_frame_size: tuple[int, int] = (160, 120)
    """Smallest (width, height) graded; smaller frames wait for the camera."""

    blank_max_std: float = Field(default=4.0, ge=0.0)
    """A frame whose pixel spread is at most this is blank and never graded.

    A camera that is off, covered or not yet streaming sends uniform frames,
    and the comparison then describes the teacher's reference instead.
    """


class DetectorSettings(_Section):
    """The detector profile whose boxes are drawn and fed to geometry."""

    profile: str = ""
    """Name of a profile in the worker's ``detectors.yaml``; empty disables the overlay."""

    conf: float | None = Field(default=None, gt=0.0, lt=1.0)
    """Per-procedure confidence override."""

    iou: float | None = Field(default=None, gt=0.0, lt=1.0)
    """Per-procedure NMS IoU override."""


class VlmBackendConfig(_Section):
    """Everything the ``vlm`` backend reads for one procedure."""

    sop: str = "sop.json"
    """Schema v1 SOP file, relative to the procedure folder."""

    detector: DetectorSettings = Field(default_factory=DetectorSettings)
    geometry: str = ""
    """Geometry profile module, relative to the procedure folder; empty for none."""

    spatial_context: bool = False
    """Add the geometry profile's prose about the boxes to grading prompts."""

    monitor: MonitorSettings = Field(default_factory=MonitorSettings)
    progression: ProgressionSettings = Field(default_factory=ProgressionSettings)
    corrections: CorrectionSettings = Field(default_factory=CorrectionSettings)
    evaluator: EvaluatorSettings = Field(default_factory=EvaluatorSettings)


__all__ = [
    "CorrectionSettings",
    "DetectorSettings",
    "EvaluatorSettings",
    "MonitorSettings",
    "ProgressionSettings",
    "VlmBackendConfig",
]
