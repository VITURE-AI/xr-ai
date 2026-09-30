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
import json
import re
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from sop_guidance.backends.base import RunCommand
from sop_guidance.backends.vlm.grading import extract_json
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

# A reply saying it cannot see, which is only true once it has looked.
_CANT_SEE = re.compile(
    r"\b(?:i\s+)?(?:can(?:no|')t|cannot|can not|am unable to|'m unable to|don't have the ability to)"
    r"\s+(?:see|view|look at)\b",
    re.IGNORECASE,
)

# The old fork's quick-ack prompt, verbatim.
_QUICK_ACK_PROMPT = (
    'Output ONLY one JSON object: {"ack": "<spoken phrase>", "think": false}\n'
    "ack: a SHORT natural spoken acknowledgment (3-6 words, no period). "
    "Sound like a helpful smart-glasses assistant about to START working on it. "
    "ALWAYS use present or future tense — the task is NOT done yet. "
    "NEVER use past tense. "
    "Examples: 'On it.' / 'Let me look.' / 'Sure, checking now.' / "
    "'Let me think about that.' / 'Looking back at that.' / 'Got it.'\n"
    "think: true if ANY of:\n"
    "  (A) questions about past events, what happened earlier, or recall "
    "('what was that?', 'what did I see earlier?', 'did I do that?')\n"
    "  (B) spatial or visual analysis that requires examining an image "
    "('what is that?', 'describe what I see', 'is there a ...?')\n"
    "  (C) questions about demonstrations or procedures "
    "('how many steps?', 'what was step 2?')\n"
    "  (D) corrections, follow-ups, or ambiguous references "
    "('no, not that', 'the one I saw earlier')\n"
    "think: false for: greetings, simple yes/no, acknowledgements, "
    "immediate next-step advances in guidance."
)


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
        scene: Callable[[str], str] | None = None,
    ) -> None:
        self._host = host
        self._llm = llm
        self._vlm = vlm
        self._cfg = config.foreground
        self._speech = speech
        self._frames = frames
        self._input_of = input_of
        # The scene memory's context block for a camera, when the observer runs.
        self._scene = scene
        self._idle_prompt = config.prompt("idle")
        self._active_prompt = config.prompt("active")
        self._view_prompt = config.prompt("current_view")
        capabilities = getattr(llm, "capabilities", None)
        self._llm_sees = bool(getattr(capabilities, "vision", False))
        self._last_ack: dict[str, str] = {}
        # Idle exchanges per participant, oldest first. Without them a "yes"
        # to the assistant's own offer ("shall I start the nose pad
        # procedure?") reached the model with nothing to say what it accepted.
        self._history: dict[str, deque[tuple[str, str]]] = {}

    def forget(self, pid: str) -> None:
        self._history.pop(pid, None)
        self._last_ack.pop(pid, None)

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
        # The old fork's order: a short acknowledgement first, which also says
        # whether the request needs visual or temporal care, then the agent.
        think = False
        if self._cfg.quick_ack:
            ack, think = await self._quick_ack(pid, request)
            if ack and self._last_ack.get(pid) != ack:
                self._last_ack[pid] = ack
                await self._speech.progress(ack)
        tools = _merge(
            build_guidance_tools(self._host, active=False),
            ToolSet({"current_view": self._current_view_tool(pid)}),
        )
        history = self._history.setdefault(pid, deque(maxlen=self._cfg.idle_history_max))
        prefix = (
            "This request may require visual or temporal care. Use the "
            "provided context and tools as needed, then return only a "
            "concise answer for the wearer.\n\n"
            if think else ""
        )
        prompt = (
            f"{prefix}[Context - use this before calling tools]\n"
            f"{self._idle_context(pid, timestamp_us, tuple(history))}\n\n"
            f"[User request]\n{request}"
        )
        reply, direct = await self._run(pid, self._idle_prompt, [TextPart(prompt)], tools)
        if not direct and _CANT_SEE.search(reply):
            # It answered a camera question without looking. The camera is
            # right there, so look rather than tell the wearer it cannot.
            logger.info("CANT_SEE_WITHOUT_LOOKING pid={} reply={!r}", pid, reply[:80])
            tool = tools.get("current_view")
            if tool is not None:
                view = await tool.handler(CurrentViewRequest(question=request))
                reply, direct = view.text, True
        if self._host.is_guiding(pid):
            # The run has its own step-scoped history; the chat that led to it
            # would only compete with the procedure once it ends.
            history.clear()
        elif reply:
            history.append((request, reply))
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

    def _idle_context(self, pid: str, timestamp_us: int,
                      history: tuple[tuple[str, str], ...]) -> str:
        """The old fork's agent context, minus the scene memory it no longer keeps."""

        parts = []
        if self._scene is not None:
            parts.append(self._scene(self._input_of(pid)))
        procedures = self._host.procedures()
        if procedures:
            lines = ["[Available procedures]"]
            for procedure in procedures:
                lines.append(f"  {procedure.id!r} ({procedure.title!r}): "
                             f"{procedure.total_steps} steps")
                if procedure.entry.spec.description:
                    lines.append(f"    Summary: {procedure.entry.spec.description}")
            parts.append("\n".join(lines))
        else:
            parts.append("[Available procedures]\nNone.")
        offered = self._host.start_offer(pid)
        if offered is not None:
            parts.append(
                f"[Offered guided mode: {offered.title!r}] You have just asked whether "
                "to start it and are waiting for their answer. If they agree, in any "
                "words, call guidance__start for it again with the same arguments. If "
                "they decline, or talk about something else, answer normally and do "
                "not start it."
            )
        finished = self._host.last_finished(pid)
        if finished is not None and not self._host.is_guiding(pid):
            procedure, step_index, at_us, completed = finished
            if not completed:
                step_no = step_index + 1
                parts.append(
                    f"[Stopped part-way: {procedure.title!r}, at step {step_no}] If they "
                    "ask to resume, continue, carry on, or pick up where they left off, "
                    f"call guidance__start with entry_mode=resume. Do not start it at "
                    "step 1 instead -- resuming and starting over are different "
                    "requests, and they asked for this one. General guidance requests "
                    "start at step 1, even when a stopped step is shown here. Never "
                    "infer a resume from conversation history."
                )
            window_s = self._host.settings.restart_confirm_s
            age_s = (time.time_ns() // 1_000 - at_us) / 1_000_000
            if window_s > 0 and 0 <= age_s <= window_s:
                parts.append(
                    f"[Just finished: {procedure.title!r}] That procedure is not running "
                    "now. Do NOT call guidance__start for it again unless they clearly "
                    "ask to go through it AGAIN, or to resume it. Reporting what they "
                    "did (\"I have attached it\"), agreeing (\"yes\", \"ok\"), or "
                    "thanking you is NOT such a request -- answer those normally."
                )
        others = [s for s in self._host.sessions() if s.owner != pid]
        if others:
            parts.append(
                f'[Another participant owns guidance: "{others[0].owner}"] This '
                "requester is not being guided. For a request to start their own "
                "guidance, call guidance__start to obtain the takeover confirmation. "
                "Do not announce or advance the other participant's step."
            )
        parts.append(f"Participant: {pid}")
        if timestamp_us:
            parts.append(f"Reference time (when user spoke): {timestamp_us} µs")
        if history:
            lines = []
            for user, agent in history:
                lines.append(f"  User: {user}")
                lines.append(f"  Agent: {agent}")
            parts.append("[Recent conversation]\n" + "\n".join(lines))
        return "\n\n".join(parts)

    async def _quick_ack(self, pid: str, request: str) -> tuple[str, bool]:
        """The old fork's quick acknowledgement: ``(spoken ack, needs thinking)``."""

        history = self._history.get(pid)
        context = ""
        if history:
            last_user, last_agent = history[-1]
            context = f"[Previous turn] User: {last_user} / Agent: {last_agent}\n"
        try:
            response = await asyncio.wait_for(self._llm.chat(
                (
                    ChatMessage(role="system", content=_QUICK_ACK_PROMPT),
                    ChatMessage(role="user", content=context + request),
                ),
                max_tokens=40, temperature=0.0,
            ), timeout=8.0)
        except Exception:
            logger.debug("quick-ack call failed", exc_info=True)
            return "", False
        raw = response.content.strip()
        obj = extract_json(raw)
        if obj:
            try:
                parsed = json.loads(obj)
                return str(parsed.get("ack", "")).strip(), bool(parsed.get("think", False))
            except (json.JSONDecodeError, AttributeError):
                pass
        return "", False

    def _current_view_tool(self, pid: str) -> Tool[CurrentViewRequest, CurrentViewResult]:
        async def inspect(request: CurrentViewRequest) -> CurrentViewResult:
            source = self._input_of(pid)
            jpeg = await self._frames.fetch_jpeg(source)
            if jpeg is None:
                return CurrentViewResult(text="I cannot see a camera frame right now.",
                                         available=False)
            question = request.question
            if self._scene is not None:
                # A question about the past that reached the camera anyway is
                # still answered from what was recorded, not the live picture.
                question = (
                    f"{self._scene(source)}\n\n[Question]\n{request.question}\n\n"
                    "If the question is about what was seen or done earlier, answer "
                    "from the observations above, not from the picture. Otherwise "
                    "answer from the picture."
                )
            try:
                response = await asyncio.wait_for(
                    self._vlm.ask_image(jpeg, question, system_prompt=self._view_prompt,
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
            "evidence from what is visible right now. Never for questions about "
            "the past (earlier, before, just now): those are answered from the "
            "recorded observations in the context.",
            CurrentViewRequest, CurrentViewResult, inspect,
            return_direct=True, render_result=lambda r: r.text,
        )

    # ── active ───────────────────────────────────────────────────────────────

    async def _active_turn(self, pid: str, request: str, session: GuidanceSession,
                           timestamp_us: int) -> None:
        extract = asyncio.create_task(
            self._host.extract_wearer_request(pid, request), name=f"wearer-request:{pid}",
        )
        try:
            await self._guidance_turn(pid, request, session)
        finally:
            await asyncio.gather(extract, return_exceptions=True)

    async def _guidance_turn(self, pid: str, transcript: str, session: GuidanceSession,
                             *, allow_check: bool = True) -> None:
        """Answer one utterance on the current step: the old fork's guidance turn.

        One model call with a fixed JSON contract, not a tool loop: the reply
        is spoken as it stands and the action is applied by code. The prompt
        text is the old fork's, verbatim, so the wearer hears what they did
        there.
        """

        run = session.run
        if run is None:
            return
        procedure = session.procedure
        step_at_start = run.snapshot().step_index
        steps = procedure.backend.steps()
        total = len(steps)
        indices = range(total) if total <= 12 else range(
            max(0, step_at_start - 2), min(total, step_at_start + 5),
        )
        lines = []
        for index in indices:
            prefix = ">> " if index == step_at_start else "   "
            lines.append(f"{prefix}Step {index + 1}: {steps[index].instruction}")
        context = self._host.turn_context(pid)
        requests = ""
        if session.wearer_requests:
            requests = (
                "\n\nWhat they have asked for during this procedure (oldest "
                "first; where two conflict the LAST is what they want now):\n"
                + "\n".join(f'- "{r}"' for r in session.wearer_requests)
                + "\nThese are conditions on the work, and they outlive the step "
                "they were said on. Never confirm something that contradicts one: "
                "if what you can see is not what they asked for, say which they "
                "asked for and which they have. In particular, them SAYING which "
                "part they hold does not make it the right one -- answer against "
                "the list above, not against their description. Speech recognition "
                "garbles these, so match a request to the nearest thing it could "
                "mean rather than treating it as being about nothing."
            )
        history = ""
        limit = self._host.settings.turn_history_max
        if session.turn_history and limit:
            history = "\n[Earlier on this step]\n" + "\n".join(
                f"Wearer: {u}\nAssistant: {a}" for u, a in session.turn_history[-limit:]
            )
        frame = context.frame_jpeg if (self._cfg.send_frame and self._llm_sees) else b""
        # Only describe the picture to the model when there IS one. Telling a
        # text-only call that it can see the wearer invites an invented
        # description, which is the worst failure available here.
        frame_note = (
            "\nThe message includes what the wearer is looking at right now, "
            "with detected objects outlined. Judge from that first; the check "
            "result above is older supporting context. If the two disagree, "
            "trust what you can see now. The outlines are a detector's guess, "
            "so do not treat a label as proof."
            if frame else ""
        )
        system = (
            f"You are guiding someone through {procedure.title!r}. THEY ARE ON STEP "
            f"{step_at_start + 1} OF {total}: {steps[step_at_start].instruction!r}.\n\n"
            "All step instructions:\n" + "\n".join(lines) + context.prompt_block
            + frame_note + requests + history + "\n\n" + self._active_prompt
        )
        task_prompt = procedure.entry.active_prompt()
        if task_prompt:
            system += "\n\n" + task_prompt
        content: Any = transcript
        if frame:
            content = [TextPart(transcript), ImagePart(_jpeg_url(frame))]
        started = time.monotonic()
        raw = ""
        error = ""
        try:
            response = await asyncio.wait_for(self._llm.chat(
                (ChatMessage(role="system", content=system),
                 ChatMessage(role="user", content=content)),
                max_tokens=self._cfg.guidance_turn_max_tokens,
                temperature=self._cfg.temperature,
            ), timeout=self._cfg.turn_timeout_s)
            raw = response.content.strip()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("guidance turn failed")
        finally:
            session.recorder.record_call(
                kind="llm", name="guidance_turn",
                request={"system": system, "user": transcript, "frame": bool(frame)},
                response=raw, latency_ms=(time.monotonic() - started) * 1000.0, error=error,
            )
        if error:
            await self._fallback(pid, session, step_at_start, "GUIDANCE_TURN_FAIL")
            return
        parsed: dict[str, Any] = {}
        obj = extract_json(raw)
        if not obj:
            # Salvage only when the reply IS speech. The JSON keys are matched
            # in their quoted form so a legitimate sentence containing the word
            # "reply" or "action" is not thrown away.
            if raw and len(raw) <= 400 and not any(
                x in raw.lower() for x in ("{", "}", '"action"', '"reply"')
            ):
                reply, action, target, reason = raw, "none", "", "salvaged-prose"
            else:
                await self._fallback(pid, session, step_at_start, "GUIDANCE_TURN_NONJSON")
                return
        else:
            try:
                loaded = json.loads(obj)
            except json.JSONDecodeError:
                loaded = {}
            parsed = loaded if isinstance(loaded, dict) else {}
            reply = str(parsed.get("reply", "")).strip()
            action = str(parsed.get("action", "none")).lower().strip()
            target = str(parsed.get("procedure", "")).strip()
            reason = str(parsed.get("reason", "")).strip()
        if action not in {"none", "advance", "exit", "restep", "check", "switch", "navigate"}:
            logger.info("GUIDANCE_TURN_BAD_ACTION action={!r}", action)
            action = "none"
        if action == "check" and not allow_check:
            action = "none"
        if len(reply) > 600:
            boundary = max(reply.rfind(c, 0, 600) for c in ".!?")
            reply = reply[:boundary + 1] if boundary >= 0 else reply[:600].rstrip()
        if (self._host.session_of(pid) is not session or session.run is None
                or session.run.snapshot().step_index != step_at_start):
            logger.info("GUIDANCE_TURN_STALE was={}", step_at_start)
            return
        logger.info("GUIDANCE_TURN action={} step={} frame={} reason={!r} reply={!r}",
                    action, step_at_start, "YES" if frame else "NO", reason[:60], reply[:60])
        if action == "exit":
            stopped = await self._host.stop(pid, reason="wearer_request")
            await self._host.say(pid, stopped.message, kind="status")
        elif action == "restep":
            await self._respond(pid, self._host.step_line(pid), transcript)
        elif action == "check":
            await self._host.command(pid, RunCommand("check", transcript=transcript))
            current = self._host.session_of(pid)
            if (current is session and session.run is not None
                    and session.run.snapshot().step_index == step_at_start):
                await self._guidance_turn(pid, transcript, session, allow_check=False)
        elif action == "advance":
            result = await self._host.command(pid, RunCommand("advance", transcript=transcript))
            if not result.accepted:
                await self._respond(pid, result.speech or reply or self._host.step_line(pid),
                                    transcript)
        elif action == "navigate":
            mode = parsed.get("entry_mode")
            quote = parsed.get("intent_quote")
            at_step = parsed.get("at_step", 0)
            if (mode not in {"start", "resume", "step"} or not isinstance(quote, str)
                    or not quote or type(at_step) is not int):
                await self._respond(pid, "Should I start from the beginning, keep this step, "
                                         "or go to a particular step?", transcript)
            else:
                begun = await self._host.begin(pid, procedure.id, at_step=at_step,
                                               entry_mode=mode, intent_quote=quote,
                                               request=transcript, confirm=True)
                if begun.message:
                    await self._respond(pid, begun.message, transcript)
        elif action == "switch":
            other = self._host.match_procedure(target) if target else None
            if other is None:
                names = ", ".join(f"'{p.title}'" for p in self._host.procedures())
                await self._respond(pid, f"I do not have that procedure. I have {names}.",
                                    transcript)
            else:
                begun = await self._host.begin(pid, other.id, explicit=True, confirm=True)
                if begun.message:
                    await self._respond(pid, begun.message, transcript)
        else:
            await self._respond(pid, reply or self._host.step_line(pid), transcript)

    async def _respond(self, pid: str, text: str, transcript: str) -> None:
        """Speak a guidance answer and keep the exchange for this step."""

        if not text:
            return
        if transcript:
            self._host.record_turn(pid, transcript, text)
        await self._host.say(pid, text, kind="answer")

    async def _fallback(self, pid: str, session: GuidanceSession, step: int, event: str) -> None:
        if (self._host.session_of(pid) is not session or session.run is None
                or session.run.snapshot().step_index != step):
            return
        logger.info("{} step={}", event, step)
        await self._host.say(pid, self._host.step_line(pid), kind="answer")

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
