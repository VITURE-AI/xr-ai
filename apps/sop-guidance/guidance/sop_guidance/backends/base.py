# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The seam between the guidance host and the thing that decides progress.

The host owns sessions, ownership, speech, client state, the overlay track and
persistence. It never decides whether a step is done. A *procedure backend*
does: the built-in ``vlm`` backend grades camera frames with a vision model,
and a future backend may run its own detector and state machine instead.

A backend reports everything through :meth:`RunContext.emit`. The host turns
a :class:`StepChanged` into the verbatim "Step n of N: ..." announcement and a
checkpoint, a :class:`Cue` into speech, an :class:`OverlayUpdate` into the
return-video track, and a :class:`RunFinished` into the end of the session.
Keeping that one output path is what lets a new backend arrive without the
host, tools, speech or UI learning anything about it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import numpy as np

    from ..procedures import ProcedureEntry
    from ..vision.overlay import AnnotatedFrame, Detection


class Capabilities(BaseModel):
    """What a backend lets the host and the conversational foreground do."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    voice_advance: bool = True
    """Whether "next" and grounded advance requests may move the step."""

    jump_to_step: bool = True
    """Whether a run can start at, or navigate to, an arbitrary step."""

    resume: bool = True
    """Whether a stopped run can be restored from its checkpoint."""

    wearer_requests: bool = True
    """Whether spoken choices ("the size one pad") constrain grading."""

    check_on_demand: bool = True
    """Whether the foreground may ask for an immediate grounded check."""

    frame_hz: float = Field(default=0.0, ge=0.0)
    """Rate the host pushes frames to :meth:`ProcedureRun.on_frame`; 0 means the run pulls."""

    min_frame_size: tuple[int, int] | None = None
    """Smallest (width, height) the backend can grade, for operator warnings."""

    provides_overlay: bool = False
    """The backend draws its own boxes; the host preview skips its detector."""

    max_concurrent_runs: int = Field(default=1, ge=1)
    """How many sessions may run this backend at once."""


@dataclass(frozen=True, slots=True)
class StepInfo:
    """What the host, tools, API and UI need to know about one step."""

    number: int
    instruction: str
    title: str = ""
    reference_images: tuple[str, ...] = ()
    before_image: str = ""
    gradeable: bool = True
    """Whether the backend can confirm this step on its own."""

    requirements: tuple[str, ...] = ()
    """Short conditions that must all be visible for the step to count as done."""

    done_when: str = ""
    """The finished state in one description, for tutorials and operators."""


@dataclass(slots=True)
class TimedFrame:
    """One camera frame of the session's input participant."""

    participant_id: str
    timestamp_us: int
    width: int
    height: int
    image: np.ndarray
    """Unannotated BGR pixels."""

    annotated: AnnotatedFrame | None = None
    """The host preview's annotation of the same pixels, when one exists."""


# ── events: the only way a run speaks to the host ────────────────────────────


@dataclass(frozen=True, slots=True)
class StepChanged:
    """The run is now on step *index* (0-based) and the wearer must hear it."""

    index: int
    reason: Literal["start", "resume", "advance", "navigate", "restep", "reset"] = "advance"
    acknowledge: bool = False
    """Lead the announcement with a short acknowledgement of the finished step."""

    lead: str = ""
    """The backend's own words before the announcement, such as confirming the
    finished step; used instead of the host's acknowledgement when set."""


@dataclass(frozen=True, slots=True)
class Cue:
    """Speech the run wants the wearer to hear, such as a correction."""

    text: str
    kind: Literal["correction", "hint", "alert", "status"] = "correction"
    priority: int = 0
    """Higher speaks sooner; announcements and answers outrank every cue."""


@dataclass(frozen=True, slots=True)
class Verdict:
    """One grading result, recorded and summarised for the foreground."""

    result: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class OverlayUpdate:
    """Boxes a backend wants drawn on the wearer's return-video track."""

    timestamp_us: int
    detections: tuple[Detection, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunFinished:
    """The run ended on its own, normally because the last step was confirmed."""

    outcome: Literal["completed", "stopped", "interrupted"] = "completed"
    reason: str = ""


RunEvent = StepChanged | Cue | Verdict | OverlayUpdate | RunFinished


# ── commands: what the host and foreground may ask of a run ──────────────────


CommandKind = Literal["repeat", "advance", "next", "go_to", "check", "reset"]


@dataclass(frozen=True, slots=True)
class RunCommand:
    """A request from the wearer, forwarded by the host."""

    kind: CommandKind
    step_index: int | None = None
    """Target for ``go_to``, 0-based."""

    transcript: str = ""


@dataclass(frozen=True, slots=True)
class CommandResult:
    """The run's answer to a command.

    ``speech`` is what the wearer should hear when the command itself produced
    no :class:`StepChanged`; empty means the host says nothing further.
    """

    accepted: bool
    speech: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """The run's current position, for client state and checkpoints."""

    step_index: int
    total_steps: int
    instruction: str
    extra: Mapping[str, Any] = field(default_factory=dict)
    """Backend-owned JSON for clients, such as a hole map."""

    state: Mapping[str, Any] = field(default_factory=dict)
    """Backend-owned JSON stored in the checkpoint and handed back on resume."""


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Backend knowledge the foreground adds to its active-mode prompt."""

    prompt_block: str = ""
    frame_jpeg: bytes = b""
    """The wearer's current view, annotated when possible; empty for text only."""


@dataclass(frozen=True, slots=True)
class ModelHandles:
    """Model services a backend may use, resolved from the task's roles."""

    llm: Any = None
    vlm: Any = None


class RunContext(Protocol):
    """Host services handed to one run."""

    session_id: str
    owner: str

    @property
    def input_participant(self) -> str:
        """Whose camera the run is graded against."""

    @property
    def models(self) -> ModelHandles: ...

    @property
    def recorder(self) -> Any:
        """The session's :class:`~sop_guidance.recorder.SessionHandle`."""

    async def emit(self, event: RunEvent) -> None: ...

    def speech_remaining_s(self) -> float:
        """Seconds of speech still playing to the owner."""

    def last_heard_us(self) -> int:
        """When the wearer last spoke to us or was spoken to, in Unix microseconds."""

    def wearer_requests(self) -> tuple[str, ...]: ...

    def latest_frame(self) -> TimedFrame | None:
        """Newest preview frame of the input participant, of any age."""

    async def fetch_frame(self) -> TimedFrame | None:
        """Pull a fresh frame from the hub when the preview has none."""


@runtime_checkable
class ProcedureRun(Protocol):
    """One session's progress through a procedure. Fresh state per session."""

    async def start(self) -> None:
        """Begin; emits the first :class:`StepChanged`."""

    async def on_frame(self, frame: TimedFrame) -> None:
        """Receive a pushed frame when ``capabilities.frame_hz`` is non-zero."""

    async def command(self, command: RunCommand) -> CommandResult: ...

    def snapshot(self) -> RunSnapshot: ...

    def turn_context(self) -> TurnContext: ...

    async def input_changed(self) -> None:
        """The session's camera moved to another participant; re-arm the step."""

    async def close(self, reason: str) -> None: ...


@runtime_checkable
class ProcedureBackend(Protocol):
    """A procedure's grader, built once per procedure folder at startup."""

    name: str
    capabilities: Capabilities

    @property
    def title(self) -> str: ...

    def steps(self) -> Sequence[StepInfo]: ...

    def parts(self) -> tuple[str, ...]:
        """Descriptions for telling interchangeable parts apart, by shape."""

    def validate(self) -> list[str]:
        """Problems that must stop startup, as human-readable strings."""

    def instructions_digest(self) -> list[str]:
        """The step instructions a checkpoint is validated against on resume."""

    def preview_annotator(self) -> Any:
        """The frame annotator the host preview draws with, or None.

        When ``capabilities.provides_overlay`` is set the backend sends its own
        boxes as :class:`OverlayUpdate` events, and this annotator only paints
        them in its profile's colours; its detector is never run.
        """

    async def open_run(
        self,
        ctx: RunContext,
        *,
        start_step: int,
        checkpoint: Mapping[str, Any] | None,
    ) -> ProcedureRun: ...


@dataclass(frozen=True, slots=True)
class BackendServices:
    """What a backend factory may use to build itself."""

    entry: ProcedureEntry
    config: Mapping[str, Any]
    """The merged ``backend_config`` block for this procedure."""

    artifacts_dir: Path
    detector_profiles: Mapping[str, Any] = field(default_factory=dict)
    frame_annotator: Callable[..., Any] | None = None
    """Builds a frame annotator: ``(profile_name, *, overrides, geometry_path,
    spatial_context) -> FrameAnnotator``; None when vision support is absent."""

    extra: Mapping[str, Any] = field(default_factory=dict)


BackendFactory = Callable[[BackendServices], ProcedureBackend]
Emit = Callable[[RunEvent], Awaitable[None]]


__all__ = [
    "BackendFactory",
    "BackendServices",
    "Capabilities",
    "CommandKind",
    "CommandResult",
    "Cue",
    "Emit",
    "ModelHandles",
    "OverlayUpdate",
    "ProcedureBackend",
    "ProcedureRun",
    "RunCommand",
    "RunContext",
    "RunEvent",
    "RunFinished",
    "RunSnapshot",
    "StepChanged",
    "StepInfo",
    "TimedFrame",
    "TurnContext",
    "Verdict",
]
