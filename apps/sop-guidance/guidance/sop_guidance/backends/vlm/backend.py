# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``vlm`` backend: a schema v1 SOP graded by a vision-language model.

Optional detector and geometry profiles add boxes to the graded frame, prose
about them to the prompt, and deterministic gates that can veto a pass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import ValidationError

from ...procedures import Sop, SopFormatError, load_sop_file
from ...vision.overlay import weight_problems
from ..base import BackendServices, Capabilities, RunContext, StepInfo
from .config import VlmBackendConfig
from .grading import OverlayPrompting
from .run import VlmRun


class VlmBackend:
    """Grades a procedure's steps against teacher reference frames."""

    name = "vlm"
    capabilities = Capabilities()

    def __init__(
        self,
        *,
        procedure_id: str,
        sop: Sop,
        config: VlmBackendConfig,
        artifacts_dir: Path,
        annotator: Any = None,
        problems: Sequence[str] = (),
    ) -> None:
        self.procedure_id = procedure_id
        self.sop = sop
        self.config = config
        self.artifacts_dir = artifacts_dir
        self._annotator = annotator
        self._problems = list(problems)

    # ── description ──────────────────────────────────────────────────────────

    @property
    def title(self) -> str:
        return self.sop.name

    def steps(self) -> list[StepInfo]:
        return [
            StepInfo(
                number=step.number,
                instruction=self.sop.instruction(index),
                reference_images=step.reference_image_paths,
                before_image=step.before_image_path,
                gradeable=step.reference_reliable,
                requirements=step.expected_requirements,
                done_when=step.key_info.target_state if step.key_info else "",
            )
            for index, step in enumerate(self.sop.steps)
        ]

    def parts(self) -> tuple[str, ...]:
        return self.sop.parts

    def instructions_digest(self) -> list[str]:
        return [self.sop.instruction(i) for i in range(len(self.sop.steps))]

    def preview_annotator(self) -> Any:
        return self._annotator

    # ── validation ───────────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        problems = list(self._problems)
        annotator = self._annotator
        gates = tuple(getattr(getattr(annotator, "geometry", None), "GATES", ()) or ())
        for step in self.sop.steps:
            if step.geometry_gate and step.geometry_gate not in gates:
                problems.append(
                    f"step {step.number} geometry_gate {step.geometry_gate!r} is not "
                    f"provided by the geometry profile (have: {', '.join(gates) or 'none'})"
                )
        if annotator is not None:
            problems.extend(weight_problems(annotator.profile))
        if not any(step.reference_reliable for step in self.sop.steps):
            logger.warning(
                "procedure {!r}: no step has a reference image, so no step can "
                "advance automatically", self.procedure_id,
            )
        return problems

    # ── grading support for runs ─────────────────────────────────────────────

    def overlay_prompting(self) -> OverlayPrompting:
        annotator = self._annotator
        if annotator is None:
            return OverlayPrompting()
        geometry = annotator.geometry
        return OverlayPrompting(
            enabled=bool(annotator.profile.enabled),
            live_guide=annotator.prompt_block(["the live student image"]),
            comparison_guide=annotator.prompt_block,
            spoken_example=geometry.spoken_example(),
            contradiction_example=geometry.contradiction_example(),
            veto=geometry.veto,
        )

    async def annotate_teacher(self, path: str) -> tuple[str, bool]:
        annotator = self._annotator
        if annotator is None:
            return path, False
        artifact = await annotator.annotate_file(path, role="teacher")
        if artifact.applied:
            return artifact.path, True
        return path, False

    async def open_run(
        self,
        ctx: RunContext,
        *,
        start_step: int,
        checkpoint: Mapping[str, Any] | None,
    ) -> VlmRun:
        if not 0 <= start_step < len(self.sop.steps):
            raise ValueError(f"step {start_step + 1} is out of range")
        return VlmRun(self, ctx, start_step=start_step)


def create_backend(services: BackendServices) -> VlmBackend:
    """Build the ``vlm`` backend for one procedure folder."""

    entry = services.entry
    try:
        config = VlmBackendConfig.model_validate(dict(services.config))
    except ValidationError as exc:
        raise ValueError(
            f"{entry.directory / 'procedure.yaml'}: backend_config: {exc}"
        ) from exc
    try:
        sop = load_sop_file(entry.resolve(config.sop))
    except (OSError, SopFormatError) as exc:
        raise ValueError(f"procedure {entry.id!r}: cannot load {config.sop}: {exc}") from exc

    problems: list[str] = []
    annotator = None
    if config.detector.profile:
        if services.frame_annotator is None:
            problems.append("detector profiles are configured but vision support is unavailable")
        else:
            geometry_path = entry.resolve(config.geometry) if config.geometry else None
            annotator = services.frame_annotator(
                config.detector.profile,
                overrides={
                    k: v for k, v in (("conf", config.detector.conf),
                                      ("iou", config.detector.iou))
                    if v is not None
                },
                geometry_path=geometry_path,
                spatial_context=config.spatial_context,
            )
    elif config.geometry:
        problems.append("geometry needs a detector profile to read boxes from")

    return VlmBackend(
        procedure_id=entry.id,
        sop=sop,
        config=config,
        artifacts_dir=services.artifacts_dir / entry.id,
        annotator=annotator,
        problems=problems,
    )


__all__ = ["VlmBackend", "create_backend"]
