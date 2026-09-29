# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""From an accepted transcript or typed line to the right action.

Speech passes three gates before anything answers it: a shape gate (too
short, no letters, only filler), the wake word, and in idle mode an LLM
classifier asking whether it is a request at all. Typed text skips all
three; the client sent it on purpose. Then the deterministic paths run before
any model does, so they hold whatever a model decides: a bare "next"
advances, "stop guidance" ends a run, a yes or no answers a pending takeover,
and "guide me through <procedure>" starts it.

Each participant has at most one turn in flight. A new accepted request
cancels the old turn and drops its queued speech.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress

import nemo_relay
from loguru import logger
from sop_guidance.backends.base import RunCommand
from sop_guidance.host import GuidanceHost
from sop_guidance.text import (
    confirmation_answer,
    guidance_request,
    is_fast_advance,
    is_guidance_stop,
    is_guidance_voice_exit,
    is_shape_noise,
    is_stop_speaking,
    match_wake_word,
    strip_filler,
)
from xr_ai_hub import ProcessorEndpoint
from xr_ai_models import ChatMessage, LLMService

from .config import WakeSettings, WorkerConfig
from .foreground import Foreground
from .protocol import ClientRegistry
from .speech import SpeechRouter

_CLASSIFIER_PROMPT = (
    "A microphone in a workshop picked up one utterance. Decide whether it is "
    "addressed to the voice assistant as a request or question, or whether it "
    "is background talk, a fragment, or someone speaking to another person. "
    "Answer with ONLY yes (a request for the assistant) or no."
)


def _now_us() -> int:
    return time.time_ns() // 1_000


class Interaction:
    """Gate, route and dispatch requests."""

    def __init__(
        self,
        *,
        host: GuidanceHost,
        foreground: Foreground,
        speech: SpeechRouter,
        clients: ClientRegistry,
        endpoint: ProcessorEndpoint,
        llm: LLMService,
        config: WorkerConfig,
    ) -> None:
        self._host = host
        self._foreground = foreground
        self._speech = speech
        self._clients = clients
        self._ep = endpoint
        self._llm = llm
        self._wake: WakeSettings = config.wake
        self._classify = config.foreground.noise_classifier
        self._names = config.wake.names()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _addressed(self, text: str) -> tuple[bool, str]:
        return match_wake_word(
            text, names=self._names,
            max_probe=self._wake.max_probe, max_distance=self._wake.max_distance,
        )

    def _as_owner(self, pid: str) -> str:
        """The session owner *pid* speaks for, when *pid* is that session's camera.

        A wearer's glasses are the input of a session an operator started for
        them; what the wearer says drives that session.
        """

        if self._host.session_of(pid) is not None:
            return pid
        session = self._host.session_for_input(pid)
        return session.owner if session is not None else pid

    async def _status(self, pid: str, status: str) -> None:
        # The wearer speaking for an operator's session sees the same status
        # as the operator. Guidance itself is not a status: the hub folds any
        # value it does not know into "processing", so clients read it from
        # guidance.state instead.
        for target in dict.fromkeys((pid, self._as_owner(pid))):
            with suppress(Exception):
                await self._ep.set_status(status, target)

    # ── entry points ─────────────────────────────────────────────────────────

    async def on_speech(self, pid: str, text: str, timestamp_us: int) -> None:
        """A final transcript the voice gate accepted."""

        text = text.strip()
        self._speech.heard(pid, timestamp_us)
        if is_shape_noise(text):
            logger.info("NOISE_L1_DROP pid={} {!r}", pid, text)
            return
        addressed, _ = self._addressed(text)
        speaker = self._as_owner(pid)
        if (not addressed and is_fast_advance(text)
                and not is_guidance_voice_exit(text)):
            # A short ASR command must not skip a step or reopen guidance.
            logger.info("ADVANCE_UNADDRESSED_DROP pid={} {!r}", pid, text)
            return
        if self._host.is_guiding(speaker):
            if not (addressed or not self._wake.required_in_guidance
                    or is_guidance_voice_exit(text)):
                logger.info("GUIDANCE_UNADDRESSED_DROP pid={} {!r}", pid, text)
                return
        else:
            if self._clients.wake_required_in_live(pid) and not addressed:
                logger.info("LIVE_UNADDRESSED_DROP pid={} {!r}", pid, text)
                return
            if self._classify and not addressed and not await self._is_request(text):
                logger.info("NOISE_L2_DROP pid={} {!r}", pid, text)
                return
        await self._speech.echo_user(text)
        await self.dispatch(speaker, text, timestamp_us)

    def gate_eats_exit(self, pid: str, text: str) -> bool:
        """Whether *text* is a bare "stop" to a running session, which the gate swallows."""

        return strip_filler(text) == "stop" and self._host.is_guiding(self._as_owner(pid))

    async def on_typed(self, pid: str, text: str, timestamp_us: int) -> None:
        """Typed text: no gates, and it speaks for the sender's picked camera.

        Only an explicit pick counts; with none, a wearer's own typed command
        stays theirs.
        """

        speaker = self._clients.explicit_input(pid) or pid
        owner = self._as_owner(speaker)
        timestamp_us = timestamp_us or _now_us()
        # Typing counts as the wearer talking, like speech: a correction must
        # leave its gap after it too.
        self._speech.heard(owner, timestamp_us)
        await self.dispatch(owner, text, timestamp_us)

    async def _is_request(self, text: str) -> bool:
        try:
            response = await asyncio.wait_for(self._llm.chat(
                (ChatMessage(role="system", content=_CLASSIFIER_PROMPT),
                 ChatMessage(role="user", content=text)),
                max_tokens=3, temperature=0.0,
            ), timeout=3.0)
        except Exception:
            logger.exception("intent classifier failed; accepting")
            return True
        return not response.content.strip().lower().startswith("no")

    # ── dispatch ─────────────────────────────────────────────────────────────

    async def dispatch(self, pid: str, text: str, timestamp_us: int) -> None:
        lock = self._locks.setdefault(pid, asyncio.Lock())
        async with lock:
            await self.cancel(pid)
            await self._speech.interrupt(pid)
            task = asyncio.create_task(
                self._run(pid, text, timestamp_us), name=f"turn:{pid}",
                context=nemo_relay.fork_asyncio_context(),
            )
            self._tasks[pid] = task
            task.add_done_callback(lambda t, p=pid: self._tasks.pop(p, None)
                                   if self._tasks.get(p) is t else None)

    def busy(self) -> bool:
        """Whether any participant's turn is in flight."""

        return any(not task.done() for task in self._tasks.values())

    async def cancel(self, pid: str) -> None:
        task = self._tasks.pop(pid, None)
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def aclose(self) -> None:
        for pid in list(self._tasks):
            await self.cancel(pid)

    async def _run(self, pid: str, text: str, timestamp_us: int) -> None:
        await self._status(pid, "processing")
        try:
            await self._handle(pid, text, timestamp_us)
        except asyncio.CancelledError:
            logger.info("turn cancelled pid={}", pid)
            raise
        except Exception:
            logger.exception("turn failed pid={}", pid)
            await self._speech.say(pid, "Something went wrong. Please try again.")
        finally:
            await self._status(pid, "ready")

    async def _handle(self, pid: str, raw: str, timestamp_us: int) -> None:
        addressed, request = self._addressed(raw.strip())
        request = request.strip()
        logger.info("USER pid={} raw={!r} request={!r}", pid, raw, request)
        if not request.strip(" .,!?"):
            if addressed:
                await self._speech.say(pid, "Yes?")
            return

        host = self._host
        if host.pending_takeover(pid):
            answer = confirmation_answer(request)
            if answer is not None:
                reply = (await host.confirm_takeover(pid) if answer
                         else host.cancel_takeover(pid))
                await self._speech.say(pid, reply.message)
                return

        session = host.session_of(pid)
        entry = guidance_request(request)
        if entry is not None and not (session is not None and entry[0] == "resume"
                                      and not entry[2]):
            target = entry[2]
            if target:
                procedure = host.match_procedure(target)
            elif session is not None:
                procedure = session.procedure
            elif entry[0] == "resume":
                # A bare "resume", as the stop message invites: the speaker's
                # last stopped session, whichever procedure it was.
                procedure = host.resumable_procedure(pid)
            elif len(host.procedures()) == 1:
                # "Guide me through it" names nothing, but with one procedure
                # there is nothing else it can mean.
                procedure = host.procedures()[0]
            else:
                procedure = None
            if procedure is not None:
                reply = await host.begin(pid, procedure.id, request=request, explicit=True)
                await self._speech.say(pid, reply.message)
                return

        if session is None and host.is_guiding():
            if is_fast_advance(request) or is_guidance_stop(request):
                owner = host.sessions()[0].owner
                await self._speech.say(pid, (
                    f'Guidance is owned by participant "{owner}". Ask to start your '
                    "own guidance, or use Stop on the session page."
                ))
                return

        if session is not None:
            host.record_user(pid, request)
            if is_guidance_voice_exit(request):
                reply = await host.stop(pid, reason="wearer_request")
                await host.say(pid, reply.message, kind="status")
                return
            if is_fast_advance(request):
                result = await host.command(pid, RunCommand("next", transcript=request))
                if not result.accepted and result.speech:
                    await host.say(pid, result.speech, kind="answer")
                return
        elif is_stop_speaking(request):
            # The queued speech was already dropped when this was accepted.
            return

        await self._foreground.answer(pid, request, timestamp_us=timestamp_us)


__all__ = ["Interaction"]
