# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conversational foreground: one bounded tool-calling turn per request.

Idle mode answers questions, looks through the camera and starts procedures.
Active mode answers the wearer about the step they are on: the shared active
prompt plus the procedure's own prompt, the backend's knowledge of the step,
and (when the model accepts images) what the wearer is looking at right now.
The foreground never moves a step itself; it asks the host through tools.
"""

from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import Callable
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from sop_guidance.host import GuidanceHost, GuidanceSession
from sop_guidance.text import STEP_ANNOUNCEMENT
from sop_guidance.tools import TurnScope, build_guidance_tools, current_turn
from xr_ai_models import ChatMessage, ChatResponse, ImagePart, LLMService, TextPart, ToolDef
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.tool_calling import ToolLoopError, run_tool_loop

from .config import WorkerConfig
from .preview import FrameCache
from .speech import SpeechRouter

_FALLBACK = "Something went wrong. Please try again."


class CurrentViewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="What to find out from the wearer's current view")


class CurrentViewResult(BaseModel):
    text: str
    available: bool = True


def _merge(*catalogs: ToolSet) -> ToolSet:
    tools: dict[str, Tool[Any, Any]] = {}
    for catalog in catalogs:
        for name, tool in catalog.items():
            if name in tools:
                raise ValueError(f"duplicate tool {name}")
            tools[name] = tool
    return ToolSet(tools)


def _jpeg_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


class Foreground:
    """Run the model turn for one accepted request."""

    def __init__(
        self,
        *,
        host: GuidanceHost,
        llm: LLMService,
        vlm: Any,
        config: WorkerConfig,
        speech: SpeechRouter,
        frames: FrameCache,
        input_of: Callable[[str], str],
    ) -> None:
        self._host = host
        self._llm = llm
        self._vlm = vlm
        self._cfg = config.foreground
        self._speech = speech
        self._frames = frames
        self._input_of = input_of
        self._idle_prompt = config.prompt("idle")
        self._active_prompt = config.prompt("active")
        self._view_prompt = config.prompt("current_view")
        capabilities = getattr(llm, "capabilities", None)
        self._llm_sees = bool(getattr(capabilities, "vision", False))
        self._last_ack: dict[str, str] = {}

    async def answer(self, pid: str, request: str, *, timestamp_us: int) -> None:
        """Answer *request* (the wake word removed) for *pid* and speak the reply.

        The same text is the turn's scope: the entry policy checks the model's
        quoted intent against what was actually said.
        """

        session = self._host.session_of(pid)
        token = current_turn.set(TurnScope(participant_id=pid, request=request))
        try:
            if session is not None and session.run is not None:
                await self._active_turn(pid, request, session, timestamp_us)
            else:
                await self._idle_turn(pid, request, timestamp_us)
        finally:
            current_turn.reset(token)

    # ── idle ─────────────────────────────────────────────────────────────────

    async def _idle_turn(self, pid: str, request: str, timestamp_us: int) -> None:
        ack_task = None
        if self._cfg.quick_ack:
            ack_task = asyncio.create_task(self._quick_ack(pid, request), name=f"ack:{pid}")
        try:
            tools = _merge(
                build_guidance_tools(self._host, active=False),
                ToolSet({"current_view": self._current_view_tool(pid)}),
            )
            reply, direct = await self._run(
                pid, self._idle_prompt, [TextPart(request)], tools,
            )
        finally:
            if ack_task is not None:
                ack_task.cancel()
        if not direct and STEP_ANNOUNCEMENT.search(reply) and not self._host.is_guiding(pid):
            # Only the host announces steps, from authored text. A reply in
            # that shape with no run behind it is invented, and it reads
            # exactly like an instruction the wearer should follow.
            logger.info("AGENT_INVENTED_STEP suppressed={!r}", reply[:120])
            names = [p.title for p in self._host.procedures()]
            how = f"guide me through {names[0]}" if len(names) == 1 else "guide me through it"
            reply = (
                "I am not guiding you through anything at the moment, so that step "
                f'was not from the procedure -- ignore it. Say "{how}" and I will '
                "take you through it properly, watching as you go."
            )
        await self._speech.say(pid, reply, timestamp_us=timestamp_us)

    async def _quick_ack(self, pid: str, request: str) -> None:
        """A few text-only words while the answer is prepared; never spoken."""

        try:
            await asyncio.sleep(0.6)
            response = await asyncio.wait_for(self._llm.chat(
                (
                    ChatMessage(role="system", content=(
                        "Write a two-to-five word acknowledgement that you are working "
                        "on the request, such as 'Let me check.' or 'One moment.'. "
                        "Answer with ONLY those words."
                    )),
                    ChatMessage(role="user", content=request),
                ),
                max_tokens=12, temperature=0.3,
            ), timeout=2.0)
        except (asyncio.CancelledError, Exception):
            return
        ack = " ".join(response.content.strip().strip("\"'").split())
        if not ack or len(ack.split()) > 6 or self._last_ack.get(pid) == ack:
            return
        self._last_ack[pid] = ack
        await self._speech.progress(ack)

    def _current_view_tool(self, pid: str) -> Tool[CurrentViewRequest, CurrentViewResult]:
        async def inspect(request: CurrentViewRequest) -> CurrentViewResult:
            source = self._input_of(pid)
            jpeg = await self._frames.fetch_jpeg(source)
            if jpeg is None:
                return CurrentViewResult(text="I cannot see a camera frame right now.",
                                         available=False)
            try:
                response = await asyncio.wait_for(
                    self._vlm.ask_image(jpeg, request.question, system_prompt=self._view_prompt,
                                        max_tokens=self._cfg.max_tokens),
                    timeout=self._cfg.turn_timeout_s,
                )
            except TimeoutError:
                return CurrentViewResult(text="I could not look in time. Please ask again.",
                                         available=False)
            return CurrentViewResult(text=response.content.strip() or "I could not tell.")

        return Tool(
            "current_view",
            "Look through the participant's camera, only when answering needs "
            "evidence from what is visible right now.",
            CurrentViewRequest, CurrentViewResult, inspect,
            return_direct=True, render_result=lambda r: r.text,
        )

    # ── active ───────────────────────────────────────────────────────────────

    async def _active_turn(self, pid: str, request: str, session: GuidanceSession,
                           timestamp_us: int) -> None:
        run = session.run
        assert run is not None
        step_at_start = run.snapshot().step_index
        context = self._host.turn_context(pid)
        frame = context.frame_jpeg if (self._cfg.send_frame and self._llm_sees) else b""
        system = self._active_system(session, context.prompt_block, bool(frame))
        content: list[Any] = [TextPart(request)]
        if frame:
            content.append(ImagePart(_jpeg_url(frame)))
        extract = asyncio.create_task(
            self._host.extract_wearer_request(pid, request), name=f"wearer-request:{pid}",
        )
        try:
            reply, direct = await self._run(
                pid, system, content, build_guidance_tools(self._host, active=True),
            )
        finally:
            await asyncio.gather(extract, return_exceptions=True)
        current = self._host.session_of(pid)
        if current is not session or session.run is None:
            # The turn stopped, switched or replaced the run; whatever the
            # host said about that is already out.
            if reply:
                await self._speech.say(pid, reply, timestamp_us=timestamp_us)
            return
        if session.run.snapshot().step_index != step_at_start:
            if reply and not direct:
                logger.info("GUIDANCE_TURN_STALE was={} now={}", step_at_start,
                            session.run.snapshot().step_index)
            return
        if not reply:
            return
        if not direct:
            self._host.record_turn(pid, request, reply)
            if self._host.reminder_due(pid) and not STEP_ANNOUNCEMENT.search(reply):
                # After a detour, one line brings them back to the work.
                self._host.mark_reminded(pid)
                ended = reply.rstrip()
                if ended[-1:] not in ".!?":
                    ended += "."
                reply = f"{ended} When you're ready, we're still on step {step_at_start + 1}."
        await self._host.say(pid, reply, kind="answer")

    def _active_system(self, session: GuidanceSession, block: str, has_frame: bool) -> str:
        procedure = session.procedure
        snapshot = session.run.snapshot() if session.run is not None else None
        index = snapshot.step_index if snapshot else 0
        steps = procedure.backend.steps()
        lines = "\n".join(f"{i + 1}. {s.instruction}" for i, s in enumerate(steps))
        requests = ""
        if session.wearer_requests:
            requests = (
                "\n\nWhat they have asked for during this procedure (oldest first; "
                "where two conflict the LAST is what they want now):\n"
                + "\n".join(f'- "{r}"' for r in session.wearer_requests)
                + "\nThese are conditions on the work, and they outlive the step "
                "they were said on. Never confirm something that contradicts one: "
                "if what you can see is not what they asked for, say which they "
                "asked for and which they have. Them SAYING which part they hold "
                "does not make it the right one. Speech recognition garbles these, "
                "so match a request to the nearest thing it could mean."
            )
        history = ""
        limit = self._host.settings.turn_history_max
        if session.turn_history and limit:
            history = "\n\n[Earlier on this step]\n" + "\n".join(
                f"Wearer: {u}\nAssistant: {a}" for u, a in session.turn_history[-limit:]
            )
        frame_note = (
            "\n\nThe message includes what the wearer is looking at right now, with "
            "detected objects outlined. Judge from that first; the check result "
            "above is older supporting context. If the two disagree, trust what you "
            "can see now. The outlines are a detector's guess, so a label is not proof."
            if has_frame else ""
        )
        task_prompt = procedure.entry.active_prompt()
        header = (
            f"You are guiding someone through {procedure.title!r}. THEY ARE ON STEP "
            f"{index + 1} OF {len(steps)}: {steps[index].instruction!r}.\n\n"
            "All step instructions:\n" + lines
        )
        parts = [header + block + frame_note + requests + history, self._active_prompt]
        if task_prompt:
            parts.append(task_prompt)
        return "\n\n".join(p.strip() for p in parts if p.strip())

    # ── the model loop ───────────────────────────────────────────────────────

    async def _run(self, pid: str, system: str, content: list[Any],
                   tools: ToolSet) -> tuple[str, bool]:
        round_index = 0

        async def call_model(messages: tuple[ChatMessage, ...],
                             definitions: tuple[ToolDef, ...]) -> ChatResponse:
            nonlocal round_index
            round_index += 1
            started = time.monotonic()
            response = await self._llm.chat(
                messages, tools=definitions,
                max_tokens=self._cfg.max_tokens, temperature=self._cfg.temperature,
            )
            logger.info("FOREGROUND pid={} round={} tools={} ms={:.0f}", pid, round_index,
                        [c.name for c in response.tool_calls or ()],
                        (time.monotonic() - started) * 1000)
            return response

        user = content[0].text if len(content) == 1 else content
        try:
            result = await asyncio.wait_for(run_tool_loop(
                (ChatMessage(role="system", content=system),
                 ChatMessage(role="user", content=user)),
                tools, call_model, max_iterations=self._cfg.max_tool_rounds,
            ), timeout=self._cfg.turn_timeout_s * self._cfg.max_tool_rounds)
        except TimeoutError:
            logger.warning("foreground turn timed out pid={}", pid)
            return "Sorry, that took too long. Please ask again.", False
        except ToolLoopError:
            logger.exception("foreground tool loop failed pid={}", pid)
            return _FALLBACK, False
        return result.content.strip(), result.return_direct


__all__ = ["Foreground"]
