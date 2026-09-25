# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native ``guidance__*`` tools for the conversational foreground.

``procedure_id`` is typed as a ``Literal`` of the enabled procedure ids, so
the JSON schema the model sees is an enum and an unknown id fails validation
before the tool runs; the model then retries with a valid one. The tool never
picks a "closest" match. Titles and aliases go in the description so the
model can map the wearer's words onto an id.

Tools whose result is what the wearer should hear return directly. When a
tool started or advanced a run, the host has already spoken the step, so the
direct result is empty and nothing more is said.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model
from xr_ai_tools import Tool, ToolSet

from .backends.base import RunCommand
from .host import GuidanceHost, HostReply


@dataclass(frozen=True, slots=True)
class TurnScope:
    """Who the current model turn acts for and what they said."""

    participant_id: str
    request: str | None


current_turn: ContextVar[TurnScope | None] = ContextVar("sop_guidance_turn", default=None)
"""Set by the foreground around each model turn."""


def _scope() -> TurnScope:
    scope = current_turn.get()
    if scope is None:
        raise RuntimeError("guidance tools require an active foreground turn")
    return scope


class _Empty(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SpokenResult(BaseModel):
    """What the wearer should hear; empty when the host already spoke."""

    say: str = ""
    status: str = "ok"


class ProcedureSummary(BaseModel):
    id: str
    title: str
    aliases: list[str]
    steps: int


class ProcedureList(BaseModel):
    procedures: list[ProcedureSummary]


class GuidanceStatus(BaseModel):
    guiding: bool
    procedure_id: str = ""
    procedure: str = ""
    step: int = 0
    total_steps: int = 0
    instruction: str = ""


class CheckSummary(BaseModel):
    summary: str


def _catalog_description(host: GuidanceHost) -> str:
    lines = []
    for procedure in host.procedures():
        also = procedure.entry.spec.aliases
        suffix = f" (also: {', '.join(also)})" if also else ""
        lines.append(f'{procedure.id}: "{procedure.title}"{suffix}')
    return "; ".join(lines)


def _spoken(reply: HostReply) -> SpokenResult:
    return SpokenResult(say=reply.message, status=reply.status)


def build_guidance_tools(host: GuidanceHost, *, active: bool) -> ToolSet:
    """Build the tool set for one foreground mode.

    Idle offers listing and starting. Active adds stopping, advancing,
    re-reading, checking and switching; which of those the running backend
    allows is decided again inside each call, from its capabilities.
    """

    ids = tuple(p.id for p in host.procedures())
    catalog = _catalog_description(host)
    procedure_type: Any = Literal[ids] if ids else str  # type: ignore[valid-type]
    # Always an enum: pydantic renders a one-value Literal as ``const``, which
    # not every tool-calling endpoint honours.
    id_schema: dict[str, Any] = {"enum": list(ids)} if ids else {}

    start_request = create_model(
        "StartGuidanceRequest",
        __config__=ConfigDict(extra="forbid"),
        procedure_id=(procedure_type, Field(description=f"Exact procedure id. {catalog}",
                                            json_schema_extra=id_schema)),
        entry_mode=(Literal["start", "resume", "step"], Field(
            default="start",
            description="start from the beginning, resume where the wearer stopped, "
                        "or step to begin at at_step",
        )),
        at_step=(int, Field(default=0, ge=0, description="1-based step for entry_mode=step")),
        intent_quote=(str, Field(
            default="",
            description="The wearer's words asking to resume or pick a step, copied "
                        "exactly from this request, including any negation",
        )),
    )
    switch_request = create_model(
        "SwitchProcedureRequest",
        __config__=ConfigDict(extra="forbid"),
        procedure_id=(procedure_type, Field(description=f"Exact procedure id. {catalog}",
                                            json_schema_extra=id_schema)),
    )

    async def list_procedures(_request: _Empty) -> ProcedureList:
        return ProcedureList(procedures=[
            ProcedureSummary(id=p.id, title=p.title, aliases=list(p.entry.spec.aliases),
                             steps=p.total_steps)
            for p in host.procedures()
        ])

    async def start(request: Any) -> SpokenResult:
        scope = _scope()
        reply = await host.begin(
            scope.participant_id,
            request.procedure_id,
            at_step=request.at_step,
            entry_mode=request.entry_mode,
            intent_quote=request.intent_quote,
            request=scope.request,
        )
        return _spoken(reply)

    async def status(_request: _Empty) -> GuidanceStatus:
        scope = _scope()
        session = host.session_of(scope.participant_id)
        if session is None or session.run is None:
            return GuidanceStatus(guiding=False)
        snapshot = session.run.snapshot()
        return GuidanceStatus(
            guiding=True,
            procedure_id=session.procedure.id,
            procedure=session.procedure.title,
            step=snapshot.step_index + 1,
            total_steps=snapshot.total_steps,
            instruction=snapshot.instruction,
        )

    tools: list[Tool[Any, Any]] = [
        Tool(
            "list_procedures",
            "List the procedures you can guide, with their ids and step counts.",
            _Empty, ProcedureList, list_procedures,
        ),
        Tool(
            "start",
            "Start guiding the wearer through a procedure. Only the host announces "
            "steps: never announce one yourself. The allowed ids are listed on "
            "procedure_id; if the request does not clearly name one, ask instead.",
            start_request, SpokenResult, start,
            return_direct=True, render_result=lambda r: r.say,
        ),
        Tool(
            "status",
            "Report whether guidance is running and which step the wearer is on.",
            _Empty, GuidanceStatus, status,
        ),
    ]
    if not active:
        return ToolSet.namespaced({"guidance": tools})

    async def stop(_request: _Empty) -> SpokenResult:
        scope = _scope()
        return _spoken(await host.stop(scope.participant_id, reason="wearer_request"))

    async def advance(_request: _Empty) -> SpokenResult:
        scope = _scope()
        result = await host.command(scope.participant_id, RunCommand("advance"))
        if result.accepted:
            return SpokenResult(say="")
        return SpokenResult(
            say=result.speech or host.step_line(scope.participant_id), status="refused",
        )

    async def repeat_step(_request: _Empty) -> SpokenResult:
        scope = _scope()
        return SpokenResult(say=host.step_line(scope.participant_id))

    async def check_now(_request: _Empty) -> CheckSummary:
        scope = _scope()
        result = await host.command(scope.participant_id, RunCommand("check"))
        return CheckSummary(summary=result.reason or "no fresh view of the work")

    async def switch_procedure(request: Any) -> SpokenResult:
        scope = _scope()
        reply = await host.begin(scope.participant_id, request.procedure_id)
        return _spoken(reply)

    tools += [
        Tool(
            "stop",
            "Stop the running guidance when the wearer wants to stop or exit it.",
            _Empty, SpokenResult, stop,
            return_direct=True, render_result=lambda r: r.say,
        ),
        Tool(
            "advance",
            "Ask to move to the next step because the wearer says they finished this "
            "one. It is granted only when a recent look at their work confirms it; "
            "otherwise the wearer hears what is still missing.",
            _Empty, SpokenResult, advance,
            return_direct=True, render_result=lambda r: r.say,
        ),
        Tool(
            "repeat_step",
            "Read the current step out again, verbatim. Only when they ask to hear the "
            "instruction again or say they missed it; it answers nothing else.",
            _Empty, SpokenResult, repeat_step,
            return_direct=True, render_result=lambda r: r.say,
        ),
        Tool(
            "check_now",
            "Take a fresh look at the wearer's work on this step when the result you "
            "have is missing or out of date. Returns what was seen.",
            _Empty, CheckSummary, check_now,
        ),
        Tool(
            "switch_procedure",
            "Switch to a different procedure the wearer names.",
            switch_request, SpokenResult, switch_procedure,
            return_direct=True, render_result=lambda r: r.say,
        ),
    ]
    return ToolSet.namespaced({"guidance": tools})


__all__ = [
    "CheckSummary",
    "GuidanceStatus",
    "ProcedureList",
    "SpokenResult",
    "TurnScope",
    "build_guidance_tools",
    "current_turn",
]
