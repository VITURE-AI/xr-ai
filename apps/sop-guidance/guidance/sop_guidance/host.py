# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The guidance host: sessions, ownership, speech and persistence.

A session belongs to the participant who started it (the *owner*) and is
graded against one participant's camera (the *input*, often a wearer's
glasses driven from an operator browser). The host enforces how many sessions
may run at once; with the default of one it reproduces the old room-wide
single run, including the takeover confirmation another participant must give
before replacing it.

The host never decides whether a step is done. It forwards commands to the
procedure's backend run and turns the run's events into speech, client state
and checkpoints, which is what keeps every backend interchangeable.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from .backends.base import (
    CommandResult,
    Cue,
    ModelHandles,
    OverlayUpdate,
    ProcedureBackend,
    ProcedureRun,
    RunCommand,
    RunEvent,
    RunFinished,
    StepChanged,
    TimedFrame,
    TurnContext,
    Verdict,
)
from .procedures import ProcedureEntry
from .recorder import SessionHandle, SessionStore
from .text import DEFAULT_REQUEST_QUALIFIERS, guidance_request, request_subject, step_announcement

SpeechKind = Literal["announcement", "answer", "correction", "status"]


def _now_us() -> int:
    return time.time_ns() // 1_000


class HostSettings(BaseModel):
    """Worker-wide guidance session policy."""

    model_config = ConfigDict(extra="forbid")

    max_concurrent_sessions: int = Field(default=1, ge=1)
    """Sessions that may run at once; 1 keeps one run per room with takeover."""

    takeover_confirm_s: float = Field(default=60.0, gt=0.0)
    """How long a takeover offer stays valid."""

    restart_confirm_s: float = Field(default=120.0, ge=0.0)
    """After a run ends, ask before starting the same procedure again; 0 disables."""

    step_ack_timeout_s: float = Field(default=2.0, ge=0.0)
    """Budget for the few words that lead into the next step; 0 disables."""

    turn_history_max: int = Field(default=3, ge=0)
    """Exchanges on the current step kept for the foreground's context."""

    wearer_requests_max: int = Field(default=5, ge=0)
    """Spoken requirements kept per session; 0 disables extraction."""

    wearer_request_timeout_s: float = Field(default=10.0, gt=0.0)
    heartbeat_s: float = Field(default=5.0, gt=0.0)
    """How often a running session refreshes its liveness for clients."""


class HostPorts(Protocol):
    """What the host needs from the worker around it."""

    async def say(self, owner: str, text: str, *, kind: SpeechKind) -> None: ...

    def speech_remaining_s(self, owner: str) -> float: ...

    def last_heard_us(self, owner: str) -> int: ...

    async def publish_state(self, state: Mapping[str, Any]) -> None: ...

    def latest_frame(self, participant_id: str) -> TimedFrame | None: ...

    async def fetch_frame(self, participant_id: str) -> TimedFrame | None: ...

    async def start_preview(self, owner: str, input_pid: str, backend: ProcedureBackend) -> None: ...

    async def stop_preview(self, owner: str) -> None: ...

    async def overlay_update(self, owner: str, update: OverlayUpdate) -> None: ...

    def resolve_input(self, owner: str, saved: str) -> str: ...


@dataclass(slots=True)
class LoadedProcedure:
    """A discovered procedure with its built backend."""

    entry: ProcedureEntry
    backend: ProcedureBackend
    models: ModelHandles

    @property
    def id(self) -> str:
        return self.entry.id

    @property
    def title(self) -> str:
        return self.entry.spec.title

    @property
    def total_steps(self) -> int:
        return len(self.backend.steps())

    def instruction(self, index: int) -> str:
        return self.backend.steps()[index].instruction


@dataclass(slots=True)
class HostReply:
    """The outcome of a session command, for voice and for client controls."""

    status: Literal["ok", "error", "confirmation_required", "started"] = "ok"
    message: str = ""
    session_id: str = ""
    owner: str = ""
    input_participant: str = ""
    token: str = ""

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status if self.status != "started" else "ok",
                               "message": self.message}
        for key in ("session_id", "owner", "input_participant", "token"):
            value = getattr(self, key)
            if value:
                out[key] = value
        return out


@dataclass(slots=True)
class GuidanceSession:
    """One running guidance session."""

    session_id: str
    owner: str
    input_pid: str
    procedure: LoadedProcedure
    recorder: SessionHandle
    started_at_us: int
    generation: int
    run: ProcedureRun | None = None
    wearer_requests: list[str] = field(default_factory=list)
    turn_history: list[tuple[str, str]] = field(default_factory=list)
    history: list[tuple[str, str]] = field(default_factory=list)
    last_verdict: Mapping[str, Any] = field(default_factory=dict)
    last_reminder_us: int = 0
    heartbeat: asyncio.Task[None] | None = None
    frame_pump: asyncio.Task[None] | None = None
    ended: bool = False

    @property
    def step_index(self) -> int:
        return self.run.snapshot().step_index if self.run is not None else 0


@dataclass(slots=True)
class _Finished:
    procedure_id: str
    at_us: int
    step_index: int
    completed: bool


@dataclass(slots=True)
class _Takeover:
    token: str
    active_session_id: str
    owner: str
    procedure_id: str
    step_index: int
    resume_id: str
    expires: float
    generation: int


LlmText = Callable[[str, str, int, float], Awaitable[str]]
"""``(system, user, max_tokens, temperature) -> text`` for host LLM features."""


class _SessionContext:
    """The :class:`~sop_guidance.backends.base.RunContext` for one session."""

    def __init__(self, host: GuidanceHost, session: GuidanceSession) -> None:
        self._host = host
        self._session = session
        self.session_id = session.session_id
        self.owner = session.owner

    @property
    def input_participant(self) -> str:
        return self._session.input_pid

    @property
    def models(self) -> ModelHandles:
        return self._session.procedure.models

    @property
    def recorder(self) -> SessionHandle:
        return self._session.recorder

    async def emit(self, event: RunEvent) -> None:
        await self._host._on_event(self._session, event)

    def speech_remaining_s(self) -> float:
        return self._host._ports.speech_remaining_s(self._session.owner)

    def last_heard_us(self) -> int:
        return self._host._ports.last_heard_us(self._session.owner)

    def wearer_requests(self) -> tuple[str, ...]:
        return tuple(self._session.wearer_requests)

    def latest_frame(self) -> TimedFrame | None:
        return self._host._ports.latest_frame(self._session.input_pid)

    async def fetch_frame(self) -> TimedFrame | None:
        return await self._host._ports.fetch_frame(self._session.input_pid)


class GuidanceHost:
    """Owns guidance sessions for one worker."""

    def __init__(
        self,
        *,
        procedures: Sequence[LoadedProcedure],
        store: SessionStore,
        ports: HostPorts,
        settings: HostSettings | None = None,
        llm_text: LlmText | None = None,
    ) -> None:
        self._procedures = {p.id: p for p in procedures}
        self._store = store
        self._ports = ports
        self._settings = settings or HostSettings()
        self._llm_text = llm_text
        self._lock = asyncio.Lock()
        self._sessions: dict[str, GuidanceSession] = {}
        self._takeovers: dict[str, _Takeover] = {}
        self._finished: dict[str, _Finished] = {}
        self._restart_pending: dict[str, str] = {}
        self._generation = 0
        self._background: set[asyncio.Task[Any]] = set()

    # ── catalog ──────────────────────────────────────────────────────────────

    @property
    def settings(self) -> HostSettings:
        return self._settings

    def procedures(self) -> list[LoadedProcedure]:
        return list(self._procedures.values())

    def procedure(self, procedure_id: str) -> LoadedProcedure | None:
        return self._procedures.get(procedure_id)

    def match_procedure(self, spoken: str) -> LoadedProcedure | None:
        """A procedure named exactly by *spoken*: its id, title or an alias."""

        wanted = " ".join(spoken.casefold().replace("-", " ").split())
        candidates = {wanted, wanted.removeprefix("the ")}
        for procedure in self._procedures.values():
            names = [procedure.id.replace("-", " "), procedure.title, *procedure.entry.spec.aliases]
            if any(" ".join(n.casefold().replace("-", " ").split()) in candidates for n in names):
                return procedure
        return None

    # ── session lookup ───────────────────────────────────────────────────────

    def sessions(self) -> list[GuidanceSession]:
        return list(self._sessions.values())

    def session_of(self, owner: str) -> GuidanceSession | None:
        return self._sessions.get(owner)

    def session_for_input(self, participant_id: str) -> GuidanceSession | None:
        """The session whose camera is *participant_id*, if any."""

        for session in self._sessions.values():
            if session.input_pid == participant_id:
                return session
        return None

    def session_by_id(self, session_id: str) -> GuidanceSession | None:
        for session in self._sessions.values():
            if session.session_id == session_id:
                return session
        return None

    def is_guiding(self, participant_id: str | None = None) -> bool:
        if participant_id is None:
            return bool(self._sessions)
        return participant_id in self._sessions

    def audience(self, session: GuidanceSession) -> set[str]:
        return {session.owner, session.input_pid}

    # ── starting ─────────────────────────────────────────────────────────────

    async def begin(
        self,
        pid: str,
        procedure_id: str,
        *,
        at_step: int = 0,
        entry_mode: str = "start",
        intent_quote: str = "",
        request: str | None = None,
        explicit: bool = False,
    ) -> HostReply:
        """Start, resume or jump into a procedure on *pid*'s behalf.

        *request* is the utterance that asked for it. When present, the entry
        policy only honours a resume or a step the request itself asked for:
        model arguments cannot turn an ordinary request into a resume, and a
        quoted intent must actually appear in what was said. *explicit* marks
        a client control or a deterministic spoken entry, which is consent in
        itself and skips the restart confirmation.
        """

        async with self._lock:
            return await self._begin(
                pid, procedure_id, at_step=at_step, entry_mode=entry_mode,
                intent_quote=intent_quote, request=request, explicit=explicit,
            )

    async def _begin(
        self,
        pid: str,
        procedure_id: str,
        *,
        at_step: int,
        entry_mode: str,
        intent_quote: str,
        request: str | None,
        explicit: bool = False,
    ) -> HostReply:
        logger.info("START_GUIDANCE procedure={!r} at_step={} pid={}", procedure_id, at_step, pid)
        procedure = self._procedures.get(procedure_id)
        if procedure is None:
            names = ", ".join(f"'{p.title}'" for p in self._procedures.values())
            return HostReply("error", f"I do not have that procedure. I have {names}.")
        caps = procedure.backend.capabilities
        resume_id = ""
        entry = guidance_request(request) if request else None
        if request is not None and (intent_quote or entry_mode != "start"):
            quote = " ".join(intent_quote.casefold().split())
            said = " ".join(request.casefold().split())
            if entry_mode not in {"start", "resume", "step"} or not quote or quote not in said:
                return HostReply("ok", "Should I start from the beginning, resume where we "
                                       "stopped, or start at a particular step?")
            entry = (entry_mode, at_step if entry_mode == "step" else 0, procedure_id)
        if request is not None:
            at_step = entry[1] if entry and entry[0] == "step" else 0
            if entry and entry[0] == "resume":
                current = self._sessions.get(pid)
                saved_id = (
                    self._store.latest_resumable(pid, procedure.id) if caps.resume else ""
                )
                if saved_id and current is None:
                    try:
                        _, saved_step = self._validate_resume(saved_id)
                    except ValueError as exc:
                        return HostReply("error", str(exc))
                    conflict = self._conflict_for(pid, procedure)
                    if conflict is not None:
                        return self._offer_takeover(pid, conflict, procedure.id, saved_step,
                                                    saved_id)
                    resume_id = saved_id
                    at_step = saved_step + 1
                elif current is not None and current.procedure.id == procedure.id:
                    at_step = current.step_index + 1
                else:
                    finished = self._finished.get(pid)
                    if (finished is None or finished.procedure_id != procedure.id
                            or finished.completed):
                        return HostReply("ok", "There is no stopped guidance to resume. Ask me "
                                               "to guide you through it to start at the beginning.")
                    at_step = finished.step_index + 1
            logger.info("GUIDANCE_ENTRY_POLICY request={!r} at_step={}", request, at_step)
        step_index = 0
        if at_step or (entry and entry[0] == "step"):
            total = procedure.total_steps
            if not 1 <= at_step <= total:
                # Said rather than clamped: silently starting somewhere else is
                # how the wearer follows the wrong instruction believing they
                # asked for this one.
                return HostReply("ok", f"'{procedure.title}' has {total} steps, so there is "
                                       f"no step {at_step}. Which one did you mean?")
            if at_step > 1 and not caps.jump_to_step and not resume_id:
                return HostReply("ok", f"'{procedure.title}' always starts from its first step.")
            step_index = at_step - 1
        # An explicit entry is already consent. An ambiguous call right after a
        # run of the same procedure asks once: after a run the conversation is
        # saturated with it and the model starts it again on "yes".
        if (entry is None and not explicit and step_index == 0
                and self._conflict_for(pid, procedure) is None
                and self._needs_restart_confirmation(pid, procedure.id)):
            self._restart_pending[pid] = procedure.id
            finished = self._finished[pid]
            if finished.step_index and not finished.completed:
                return HostReply("ok", f"We stopped '{procedure.title}' at step "
                                       f"{finished.step_index + 1} of {procedure.total_steps}. "
                                       "Shall I pick up from there, or start again from the "
                                       "beginning?")
            return HostReply("ok", f"We just finished '{procedure.title}'. Do you want to go "
                                   "through it again from the beginning?")
        self._restart_pending.pop(pid, None)
        return await self._enter(procedure, pid, step_index=step_index, resume_id=resume_id)

    def _needs_restart_confirmation(self, pid: str, procedure_id: str) -> bool:
        window_s = self._settings.restart_confirm_s
        finished = self._finished.get(pid)
        if window_s <= 0 or finished is None:
            return False
        if self._restart_pending.get(pid) == procedure_id:
            return False
        if finished.procedure_id != procedure_id:
            return False
        age_s = (_now_us() - finished.at_us) / 1_000_000
        return 0 <= age_s <= window_s

    def _conflict_for(self, pid: str, procedure: LoadedProcedure) -> GuidanceSession | None:
        """A session another participant owns that blocks *pid* from starting."""

        others = [s for s in self._sessions.values() if s.owner != pid]
        if len(self._sessions) - (pid in self._sessions) >= self._settings.max_concurrent_sessions:
            return others[0] if others else None
        same_backend = [s for s in others if s.procedure.backend is procedure.backend]
        if len(same_backend) >= procedure.backend.capabilities.max_concurrent_runs:
            return same_backend[0]
        return None

    async def _enter(
        self,
        procedure: LoadedProcedure,
        pid: str,
        *,
        step_index: int,
        resume_id: str = "",
        inherit_input: str = "",
    ) -> HostReply:
        conflict = self._conflict_for(pid, procedure)
        if conflict is not None:
            return self._offer_takeover(pid, conflict, procedure.id, step_index, resume_id)
        current = self._sessions.get(pid)
        if current is not None:
            await self._exit(current, outcome="superseded", reason="new_run", speak=False)

        checkpoint: dict[str, Any] = {}
        if resume_id:
            handle, checkpoint = self._store.resume_session(resume_id, pid)
        else:
            handle = self._store.open_session(
                procedure_id=procedure.id,
                procedure_name=procedure.title,
                total_steps=procedure.total_steps,
                owner=pid,
                config=procedure.entry.spec.model_dump(mode="json"),
            )
        self._generation += 1
        session = GuidanceSession(
            session_id=handle.session_id,
            owner=pid,
            input_pid=self._ports.resolve_input(pid, self._saved_input(pid, checkpoint,
                                                                       inherit_input)),
            procedure=procedure,
            recorder=handle,
            started_at_us=_now_us(),
            generation=self._generation,
        )
        if checkpoint:
            session.history = [tuple(p) for p in checkpoint.get("history", [])]
            session.turn_history = [tuple(p) for p in checkpoint.get("turn_history", [])]
            session.wearer_requests = list(checkpoint.get("wearer_requests", []))
        self._sessions[pid] = session
        logger.info("GUIDANCE_START procedure={} steps={} owner={} input={}",
                    procedure.id, procedure.total_steps, pid, session.input_pid)
        # Always a preview: a backend that draws its own overlay gets its boxes
        # drawn on it instead of the host detector's.
        await self._ports.start_preview(pid, session.input_pid, procedure.backend)
        try:
            session.run = await procedure.backend.open_run(
                _SessionContext(self, session),
                start_step=step_index,
                checkpoint=checkpoint.get("backend_state") if checkpoint else None,
            )
            await session.run.start()
        except Exception:
            logger.exception("guidance run failed to start")
            await self._exit(session, outcome="interrupted", reason="backend_error", speak=False)
            return HostReply("error", "I could not start that procedure. Please try again.")
        self._checkpoint(session)
        session.heartbeat = asyncio.create_task(self._heartbeat(session),
                                                name=f"guidance-heartbeat-{pid}")
        if procedure.backend.capabilities.frame_hz > 0:
            session.frame_pump = asyncio.create_task(self._pump_frames(session),
                                                     name=f"guidance-frames-{pid}")
        return HostReply("started", "", session_id=session.session_id, owner=pid,
                         input_participant=session.input_pid)

    # ── takeover ─────────────────────────────────────────────────────────────

    def _saved_input(self, pid: str, checkpoint: Mapping[str, Any], inherit: str) -> str:
        """The camera a new session should prefer before *pid*'s own default.

        A resumed session keeps its saved camera. A takeover inherits the
        superseded session's camera, unless *pid* picked one explicitly.
        """

        saved = str(checkpoint.get("input_participant", "")) if checkpoint else ""
        if saved or not inherit:
            return saved
        return inherit if self._ports.resolve_input(pid, "") == pid else ""

    def _offer_takeover(
        self, pid: str, active: GuidanceSession, procedure_id: str, step_index: int,
        resume_id: str,
    ) -> HostReply:
        pending = _Takeover(
            token=secrets.token_urlsafe(16),
            active_session_id=active.session_id,
            owner=active.owner,
            procedure_id=procedure_id,
            step_index=step_index,
            resume_id=resume_id,
            expires=time.monotonic() + self._settings.takeover_confirm_s,
            generation=active.generation,
        )
        self._takeovers[pid] = pending
        return HostReply(
            "confirmation_required",
            f'There is an active guidance session run by participant "{active.owner}". '
            "Are you sure you want to stop that session and run your own?",
            owner=active.owner,
            token=pending.token,
        )

    def pending_takeover(self, pid: str) -> bool:
        return pid in self._takeovers

    async def confirm_takeover(self, pid: str, token: str | None = None) -> HostReply:
        async with self._lock:
            pending = self._takeovers.pop(pid, None)
            active = self.session_by_id(pending.active_session_id) if pending else None
            if (pending is None or (token is not None and pending.token != token)
                    or pending.expires < time.monotonic()
                    or active is None or active.owner != pending.owner
                    or active.generation != pending.generation):
                return HostReply("error", "The active session changed or the confirmation "
                                          "expired. Please try again.")
            procedure = self._procedures.get(pending.procedure_id)
            if procedure is None:
                return HostReply("error", "This procedure is not ready.")
            if pending.resume_id:
                try:
                    self._validate_resume(pending.resume_id)
                except ValueError as exc:
                    return HostReply("error", str(exc))
            old_owner = active.owner
            # Taking over a session keeps its camera: the operator takes over
            # what the wearer's glasses are showing, not their own webcam.
            old_input = active.input_pid
            await self._exit(active, outcome="superseded", reason="participant_takeover",
                             speak=False)
            notice = f'Your guidance was stopped because participant "{pid}" took over.'
            await self._ports.say(old_owner, notice, kind="status")
            return await self._enter(procedure, pid, step_index=pending.step_index,
                                     resume_id=pending.resume_id, inherit_input=old_input)

    def cancel_takeover(self, pid: str, token: str | None = None) -> HostReply:
        pending = self._takeovers.get(pid)
        if pending is not None and (token is None or pending.token == token):
            self._takeovers.pop(pid, None)
        return HostReply("ok", "Okay, the active session will continue.")

    # ── stopping, resuming, leaving ──────────────────────────────────────────

    async def stop(self, requester: str, *, session_id: str = "",
                   reason: str = "") -> HostReply:
        """Stop a session: the requester's own, or *session_id* from a client control."""

        async with self._lock:
            session = (
                self.session_by_id(session_id) if session_id else self._sessions.get(requester)
            )
            if session is None:
                if session_id:
                    return HostReply("error", "That session is no longer running. Refresh "
                                              "the session list.")
                if self._sessions:
                    owner = next(iter(self._sessions))
                    return HostReply("ok", f'Guidance belongs to participant "{owner}". Use '
                                           "the session page to stop it.")
                return HostReply("ok", "Guidance is not active.")
            message = await self._exit(session, outcome="stopped",
                                       reason=reason or "session_control", speak=False)
            return HostReply("ok", message, session_id=session.session_id,
                             owner=session.owner)

    def latest_resumable(self, pid: str) -> str:
        """*pid*'s most recently stopped session that can still resume, or "".

        What a bare "resume" means: the stop message tells the wearer to say
        just that, without naming the procedure again.
        """

        return self._latest_resumable(pid)[0]

    def resumable_procedure(self, pid: str) -> LoadedProcedure | None:
        """The procedure of :meth:`latest_resumable`'s session, if any."""

        return self._latest_resumable(pid)[1]

    def _latest_resumable(self, pid: str) -> tuple[str, LoadedProcedure | None]:
        session_id = self._store.latest_resumable(pid)
        if not session_id:
            return "", None
        try:
            procedure, _ = self._validate_resume(session_id)
        except ValueError:
            return "", None
        return session_id, procedure

    async def resume(self, pid: str, session_id: str) -> HostReply:
        """Resume a saved session from a client control."""

        if not session_id:
            session_id = self.latest_resumable(pid)
            if not session_id:
                return HostReply("error", "There is no stopped guidance to resume.")
        async with self._lock:
            try:
                procedure, step = self._validate_resume(session_id)
            except ValueError as exc:
                return HostReply("error", str(exc))
            return await self._enter(procedure, pid, step_index=step, resume_id=session_id)

    def _validate_resume(self, session_id: str) -> tuple[LoadedProcedure, int]:
        meta, checkpoint = self._store.load_session(session_id)
        procedure = self._procedures.get(str(meta.get("procedure_id", "")))
        if procedure is None:
            for candidate in self._procedures.values():
                if candidate.title == meta.get("demo"):
                    procedure = candidate
                    break
        if procedure is None:
            raise ValueError("This procedure is not ready.")
        if not procedure.backend.capabilities.resume:
            raise ValueError(f"'{procedure.title}' cannot be resumed; start it again instead.")
        if checkpoint.get("instructions") != procedure.backend.instructions_digest():
            raise ValueError("The procedure changed since this session. Start a new run instead.")
        step = checkpoint.get("step")
        if type(step) is not int or not 0 <= step < procedure.total_steps:
            raise ValueError("The saved step is invalid.")
        return procedure, step

    async def participant_left(self, pid: str) -> None:
        async with self._lock:
            self._takeovers.pop(pid, None)
            session = self._sessions.get(pid)
            if session is not None:
                await self._exit(session, outcome="interrupted", reason="owner_disconnected",
                                 speak=False)

    async def shutdown(self) -> None:
        async with self._lock:
            for session in list(self._sessions.values()):
                await self._exit(session, outcome="interrupted", reason="worker_shutdown",
                                 speak=False)
        tasks = list(self._background)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def change_input(self, owner: str, source: str) -> None:
        """Re-point a running session at another participant's camera."""

        async with self._lock:
            session = self._sessions.get(owner)
            if session is None or source == session.input_pid or session.run is None:
                return
            await self._ports.stop_preview(owner)
            session.input_pid = source
            await session.run.input_changed()
            await self._ports.start_preview(owner, source, session.procedure.backend)
            self._checkpoint(session)
            await self._publish(session)

    async def _exit(self, session: GuidanceSession, *, outcome: str, reason: str,
                    speak: bool) -> str:
        """End *session* and return its closing sentence."""

        if session.ended:
            return ""
        session.ended = True
        snapshot = session.run.snapshot() if session.run is not None else None
        stopped_at = snapshot.step_index if snapshot is not None else 0
        for task in (session.heartbeat, session.frame_pump):
            if task is not None and task is not asyncio.current_task():
                task.cancel()
        self._checkpoint(session)
        if session.run is not None:
            try:
                await session.run.close(reason or outcome)
            except Exception:
                logger.exception("guidance run close failed")
        await self._ports.stop_preview(session.owner)
        procedure = session.procedure
        completed = outcome == "completed"
        if completed:
            message = f"You've completed all steps in '{procedure.title}'. Well done!"
        else:
            # Naming the step makes the resume offer actionable: congratulating
            # someone who stopped at step 2 of 5 tells them we lost track.
            message = (
                f"Stopped guidance for '{procedure.title}' at step {stopped_at + 1} "
                f"of {procedure.total_steps}. Say resume to pick up there, or guide "
                "me through it to start again."
            )
        self._finished[session.owner] = _Finished(
            procedure_id=procedure.id, at_us=_now_us(),
            step_index=0 if completed else stopped_at, completed=completed,
        )
        self._restart_pending.pop(session.owner, None)
        logger.info("GUIDANCE_DONE outcome={} reason={} stopped_at={}",
                    outcome, reason, stopped_at + 1)
        session.recorder.chat("agent", message, session.owner)
        session.recorder.end(outcome, reason)
        self._sessions.pop(session.owner, None)
        await self._publish(session, ended=True, outcome=outcome)
        if speak:
            await self._ports.say(session.owner, message, kind="status")
        return message

    # ── commands from the foreground ─────────────────────────────────────────

    async def command(self, owner: str, command: RunCommand) -> CommandResult:
        session = self._sessions.get(owner)
        if session is None or session.run is None:
            return CommandResult(False, speech="Guidance is not active.", reason="no-session")
        caps = session.procedure.backend.capabilities
        if command.kind in ("advance", "next") and not caps.voice_advance:
            # Voice never moves a camera-confirmed step; "next" re-reads it.
            return CommandResult(False, speech=(
                "The camera confirms each step, so I can't skip ahead. "
                + self.step_line(owner)
            ), reason="voice-advance-disabled")
        if command.kind == "go_to" and not caps.jump_to_step:
            return CommandResult(False, speech=(
                f"'{session.procedure.title}' has to be done in order."
            ), reason="jump-disabled")
        if command.kind == "check" and not caps.check_on_demand:
            return CommandResult(False, reason="check-disabled")
        result = await session.run.command(command)
        self._checkpoint(session)
        return result

    def step_line(self, owner: str) -> str:
        session = self._sessions.get(owner)
        if session is None or session.run is None:
            return ""
        snapshot = session.run.snapshot()
        instruction = snapshot.instruction.rstrip()
        tail = "" if instruction[-1:] in ".!?" else "."
        return f"Step {snapshot.step_index + 1} of {snapshot.total_steps} is: {instruction}{tail}"

    def turn_context(self, owner: str) -> TurnContext:
        session = self._sessions.get(owner)
        if session is None or session.run is None:
            return TurnContext()
        return session.run.turn_context()

    def record_user(self, owner: str, text: str) -> None:
        session = self._sessions.get(owner)
        if session is not None:
            session.recorder.chat("user", text, owner)

    def record_turn(self, owner: str, user: str, reply: str) -> None:
        """Keep one exchange for the step's context and the checkpoint."""

        session = self._sessions.get(owner)
        if session is None:
            return
        session.turn_history.append((user, reply))
        limit = self._settings.turn_history_max
        if limit:
            session.turn_history = session.turn_history[-limit:]
        session.history = (session.history + [(user, reply)])[-4:]
        self._checkpoint(session)

    async def say(self, owner: str, text: str, *, kind: SpeechKind = "answer") -> None:
        """Speak to a session owner and record it in the session conversation."""

        session = self._sessions.get(owner)
        if session is not None:
            session.recorder.chat("agent", text, owner)
        await self._ports.say(owner, text, kind=kind)

    def reminder_due(self, owner: str) -> bool:
        """Whether an off-topic answer should close with a step reminder."""

        session = self._sessions.get(owner)
        if session is None:
            return False
        interval = session.procedure.entry.spec.foreground.reminder_interval_s
        if interval <= 0:
            return False
        return (_now_us() - session.last_reminder_us) / 1_000_000 >= interval

    def mark_reminded(self, owner: str) -> None:
        session = self._sessions.get(owner)
        if session is not None:
            session.last_reminder_us = _now_us()

    # ── wearer requests ──────────────────────────────────────────────────────

    async def extract_wearer_request(self, owner: str, transcript: str) -> None:
        """Record a lasting requirement stated in *transcript*, via its own call.

        Its own call, not a field on the conversational turn, because a side
        field on a busy reply is exactly what the model drops. Writes nothing if
        the session changed while it was in flight.
        """

        session = self._sessions.get(owner)
        said = (transcript or "").strip()
        if (session is None or not said or self._llm_text is None
                or self._settings.wearer_requests_max <= 0
                or not session.procedure.backend.capabilities.wearer_requests):
            return
        parts = session.procedure.backend.parts()
        parts_text = "\n".join(f"- {p}" for p in parts)
        system = (
            "You extract standing requirements from one thing a person said "
            "while being guided through a physical task. Answer with ONLY the "
            "requirement, in five words or fewer, or the single word NONE.\n\n"
            "A requirement is a CHOICE they are stating about which part, size, "
            "side or option to use, and it is meant to hold for the rest of the "
            "task.\n"
            "Answer NONE for: questions of any kind; anything describing what "
            "they currently have, hold or see; progress reports; "
            "acknowledgements; and chatter. Saying which part they are HOLDING "
            "is not asking for it -- that is the case this exists to get right.\n"
            "Their words come through speech recognition and arrive garbled, so "
            "map a mangled part name onto the nearest real one.\n\n"
            + (f"The parts this task involves:\n{parts_text}\n\n" if parts else "")
            + 'Examples: "I wanna change into the size one nose pad" -> size one '
            'nose pad. "but I\'m holding a size zero" -> NONE. "is it this one?" '
            '-> NONE. "actually give me the solid one" -> solid saddle pad.'
        )
        try:
            raw = await asyncio.wait_for(
                self._llm_text(system, said, 24, 0.0),
                timeout=self._settings.wearer_request_timeout_s,
            )
        except Exception:
            # Silent: the wearer is owed an answer, not a report that a
            # background extraction failed.
            logger.exception("wearer-request extraction failed")
            return
        cleaned = raw.strip().strip("\"'").rstrip(".")
        if not cleaned or cleaned.upper().startswith("NONE"):
            return
        if self._sessions.get(owner) is not session or session.ended:
            return
        self._record_wearer_request(session, cleaned)

    def _record_wearer_request(self, session: GuidanceSession, text: str) -> None:
        said = text.strip()
        limit = self._settings.wearer_requests_max
        # A whole sentence is the model echoing the utterance, not a request.
        if not said or limit <= 0 or len(said) > 80:
            return
        if session.wearer_requests and session.wearer_requests[-1] == said:
            return
        qualifiers = DEFAULT_REQUEST_QUALIFIERS | frozenset(
            q.lower() for q in session.procedure.entry.spec.request_qualifiers
        )
        subject = request_subject(said, qualifiers)
        # A revision replaces what it revises: a header clause saying "the last
        # one wins" lost live to the grader demanding the first request.
        if len(subject) >= 2:
            for old in list(session.wearer_requests):
                if len(subject & request_subject(old, qualifiers)) >= 2:
                    session.wearer_requests.remove(old)
        session.wearer_requests.append(said)
        if len(session.wearer_requests) > limit:
            del session.wearer_requests[:-limit]
        logger.info("GUIDANCE_WEARER_REQUEST kept={}/{} text={!r}",
                    len(session.wearer_requests), limit, said[:60])
        self._checkpoint(session)

    # ── run events ───────────────────────────────────────────────────────────

    async def _on_event(self, session: GuidanceSession, event: RunEvent) -> None:
        if session.ended:
            return
        if isinstance(event, StepChanged):
            await self._announce(session, event)
        elif isinstance(event, Cue):
            if event.text.strip():
                await self.say(session.owner, event.text, kind="correction")
        elif isinstance(event, Verdict):
            session.last_verdict = dict(event.result)
            await self._publish(session)
        elif isinstance(event, OverlayUpdate):
            await self._ports.overlay_update(session.owner, event)
        elif isinstance(event, RunFinished):
            # Scheduled, not awaited: the run may be emitting from inside a
            # command the host is holding its lock for.
            task = asyncio.create_task(self._finish(session, event), name="guidance-finish")
            self._background.add(task)
            task.add_done_callback(self._background.discard)

    async def _finish(self, session: GuidanceSession, event: RunFinished) -> None:
        async with self._lock:
            if session.ended:
                return
            await self._exit(session, outcome=event.outcome, reason=event.reason, speak=True)

    async def _announce(self, session: GuidanceSession, event: StepChanged) -> None:
        procedure = session.procedure
        total = procedure.total_steps
        instruction = procedure.instruction(event.index)
        lead = ""
        if event.acknowledge and event.index > 0:
            lead = await self._step_ack(session, procedure.instruction(event.index - 1))
        # The turn history is the conversation about one step; a new step
        # starts a new one.
        if event.reason in ("advance", "navigate", "reset"):
            session.turn_history.clear()
        text = step_announcement(event.index, total, instruction)
        if lead:
            text = f"{lead} {text}"
        self._checkpoint(session)
        await self._publish(session)
        await self.say(session.owner, text, kind="announcement")

    async def _step_ack(self, session: GuidanceSession, done: str) -> str:
        """A few words confirming the finished step, before the next is read.

        The announcement itself stays the authored instruction verbatim. On the
        critical path, so it is capped hard and a late pleasantry is dropped.
        """

        history = session.turn_history
        if (not history or self._llm_text is None
                or self._settings.step_ack_timeout_s <= 0):
            return ""
        recent = "\n".join(
            f"Wearer: {u}\nYou: {a}"
            for u, a in history[-max(1, self._settings.turn_history_max):]
        )
        system = (
            "You are guiding someone through a physical task, hands-on, out "
            "loud. They have just finished the step you were on and you are "
            "about to read them the next one.\n\n"
            "Write ONLY the handful of words that come immediately before it: "
            "confirm they got it right, and connect it to what they were just "
            "asking about if that reads naturally. At most ten words, spoken "
            "plainly, no emoji.\n"
            "Do NOT state, summarise or hint at the next step -- it is read out "
            "straight after you and must not be said twice.\n"
            "Answer with the single word NONE if the exchange gives you nothing "
            "worth acknowledging.\n\n"
            'Examples: "Yes, exactly." / "That\'s it -- straight off, like you '
            'asked." / "Good, that\'s the one."'
        )
        user = f"The step they just finished: {done}\n\nWhat was said on it:\n{recent}"
        try:
            raw = await asyncio.wait_for(
                self._llm_text(system, user, 32, 0.3),
                timeout=self._settings.step_ack_timeout_s,
            )
        except Exception:
            logger.info("step acknowledgement unavailable; announcing the step bare")
            return ""
        ack = " ".join(raw.strip().strip("\"'").split())
        if not ack or ack.upper().startswith("NONE"):
            return ""
        # A model that ignores the word cap has almost certainly gone on to
        # paraphrase the next step, the one thing this must not do.
        if len(ack.split()) > 14:
            return ""
        return ack if ack[-1:] in ".!?," else ack + "."

    # ── persistence and client state ─────────────────────────────────────────

    def _checkpoint(self, session: GuidanceSession) -> None:
        if session.run is None:
            return
        snapshot = session.run.snapshot()
        session.recorder.save_checkpoint(
            step=snapshot.step_index,
            procedure=session.procedure.title,
            procedure_id=session.procedure.id,
            input_participant=session.input_pid,
            instructions=session.procedure.backend.instructions_digest(),
            history=session.history,
            turn_history=session.turn_history,
            wearer_requests=session.wearer_requests,
            backend_state=dict(snapshot.state),
        )

    def state_of(self, session: GuidanceSession, *, ended: bool = False,
                 outcome: str = "") -> dict[str, Any]:
        snapshot = session.run.snapshot() if session.run is not None else None
        procedure = session.procedure
        return {
            "session_id": session.session_id,
            "status": "ended" if ended else "running",
            "outcome": outcome,
            "procedure_id": procedure.id,
            "procedure": procedure.title,
            "backend": procedure.backend.name,
            "owner": session.owner,
            "input_participant": session.input_pid,
            "step": (snapshot.step_index + 1) if snapshot else 0,
            "total_steps": procedure.total_steps,
            "instruction": snapshot.instruction if snapshot else "",
            "extra": dict(snapshot.extra) if snapshot else {},
            "capabilities": procedure.backend.capabilities.model_dump(mode="json"),
            "wearer_requests": list(session.wearer_requests),
            "updated_us": _now_us(),
        }

    async def republish(self) -> None:
        """Send every running session's state again, for a client that just connected."""

        for session in self.sessions():
            if not session.ended:
                await self._publish(session)

    async def _publish(self, session: GuidanceSession, *, ended: bool = False,
                       outcome: str = "") -> None:
        try:
            await self._ports.publish_state(self.state_of(session, ended=ended, outcome=outcome))
        except Exception:
            logger.exception("guidance state publish failed")

    async def _pump_frames(self, session: GuidanceSession) -> None:
        """Push the input camera to a run that asked for frames (``frame_hz``).

        The preview's newest frame is reused when it exists, so a pushed frame
        and the wearer's overlay are the same pixels; otherwise one is fetched.
        Always the session's CURRENT input, so a camera change follows at once.
        """

        period = 1.0 / session.procedure.backend.capabilities.frame_hz
        last_ts = 0
        try:
            while not session.ended:
                await asyncio.sleep(period)
                run = session.run
                if run is None or session.ended:
                    continue
                frame = self._ports.latest_frame(session.input_pid)
                if frame is None or frame.timestamp_us == last_ts:
                    frame = await self._ports.fetch_frame(session.input_pid)
                if frame is None or frame.timestamp_us == last_ts or session.ended:
                    continue
                last_ts = frame.timestamp_us
                try:
                    await run.on_frame(frame)
                except Exception:
                    logger.exception("guidance run rejected a frame")
        except asyncio.CancelledError:
            pass

    async def _heartbeat(self, session: GuidanceSession) -> None:
        try:
            while not session.ended:
                await asyncio.sleep(self._settings.heartbeat_s)
                session.recorder.heartbeat()
        except asyncio.CancelledError:
            pass


__all__ = [
    "GuidanceHost",
    "GuidanceSession",
    "HostPorts",
    "HostReply",
    "HostSettings",
    "LlmText",
    "LoadedProcedure",
    "SpeechKind",
]
