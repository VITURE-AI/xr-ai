# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the worker's speech goes and when it has finished playing.

The guidance monitor must not grade a step while its own instruction is still
being read out, and a correction must not talk over the wearer. Both need an
estimate of when queued speech ends, which the voice runtime does not expose,
so :class:`PlaybackTracker` watches the audio actually handed to the hub: each
chunk extends a per-recipient deadline by its duration. Between queuing text
and the first synthesized chunk there is no audio to watch, so a word-rate
estimate stands in until audio flows.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from loguru import logger
from xr_ai_hub import AudioChunk, DataMessage, ProcessorEndpoint
from xr_ai_runtime import AgentRuntime
from xr_ai_voice import VOICE_OUTPUT_TOPIC, VoiceOutput

AGENT_RESPONSE_TOPIC = "agent.response"
USER_ECHO_TOPIC = "chat.user"
PROGRESS_TOPIC = "agent.progress"

_WORDS_PER_S = 2.6
_SYNTH_LATENCY_S = 1.0


def _now_us() -> int:
    return time.time_ns() // 1_000


class PlaybackTracker:
    """Estimate how much queued speech each recipient still has to hear."""

    def __init__(
        self,
        *,
        slack_s: float = 0.3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._slack_s = slack_s
        self._clock = clock
        self._deadline: dict[str, float] = {}
        self._pending: dict[str, float] = {}

    def install(self, endpoint: ProcessorEndpoint) -> None:
        """Observe *endpoint*'s return audio and flushes.

        The voice output transport holds this same endpoint object and calls
        these methods on it, so instance attributes shadowing the bound
        methods see every chunk without changing the SDK.
        """

        send = endpoint.send_return_audio
        flush = endpoint.flush_return_audio

        async def send_return_audio(chunk: AudioChunk) -> None:
            self.audio_sent(chunk.participant_id, chunk.samples / chunk.sample_rate)
            await send(chunk)

        async def flush_return_audio(participant_id: str) -> None:
            self.flushed(participant_id)
            await flush(participant_id)

        endpoint.send_return_audio = send_return_audio  # type: ignore[method-assign]
        endpoint.flush_return_audio = flush_return_audio  # type: ignore[method-assign]

    def enqueued(self, participant_id: str, text: str) -> None:
        estimate = _SYNTH_LATENCY_S + len(text.split()) / _WORDS_PER_S
        now = self._clock()
        base = max(self._deadline.get(participant_id, now), self._pending.get(participant_id, now))
        self._pending[participant_id] = max(base, now) + estimate

    def audio_sent(self, participant_id: str, duration_s: float) -> None:
        now = self._clock()
        start = max(self._deadline.get(participant_id, now), now)
        self._deadline[participant_id] = start + duration_s
        # Real audio replaces the estimate; later text re-arms it.
        self._pending.pop(participant_id, None)

    def flushed(self, participant_id: str) -> None:
        self._deadline.pop(participant_id, None)
        self._pending.pop(participant_id, None)

    def remaining_s(self, participant_id: str) -> float:
        now = self._clock()
        remaining = 0.0
        deadline = self._deadline.get(participant_id)
        if deadline is not None and deadline > now:
            remaining = deadline - now + self._slack_s
        pending = self._pending.get(participant_id)
        if pending is not None and pending > now:
            remaining = max(remaining, pending - now)
        return remaining


VoiceOutputMode = str
VOICE_OUTPUT_MODES = frozenset({"default", "selected_input", "web_client"})


class SpeechRouter:
    """Speak to participants and mirror the text to every connected client.

    Text lines go to the whole room: guidance accepts commands from anyone,
    so a transcript split by speaker left each person seeing half of it.
    Audio is private and follows the voice output mode: the participant the
    reply is for, their selected input (the wearer's glasses), or the browser
    that chose ``web_client``.
    """

    def __init__(
        self,
        *,
        endpoint: ProcessorEndpoint,
        tracker: PlaybackTracker,
        publish: Callable[[VoiceOutput, str], Awaitable[None]],
        selected_input: Callable[[str], str],
    ) -> None:
        self._ep = endpoint
        self._tracker = tracker
        self._publish = publish
        self._selected_input = selected_input
        self._mode: VoiceOutputMode = "default"
        self._mode_client = ""
        self._interrupt_next: set[str] = set()
        self._last_heard: dict[str, int] = {}

    # ── routing ──────────────────────────────────────────────────────────────

    @property
    def mode(self) -> VoiceOutputMode:
        return self._mode

    def recipient(self, pid: str) -> str:
        """The participant whose speaker plays replies meant for *pid*."""

        if self._mode == "selected_input":
            return self._selected_input(pid) or pid
        if self._mode == "web_client":
            client = self._mode_client
            if client and client in self._ep.connected_participants:
                return client
        return pid

    async def set_mode(self, sender: str, mode: VoiceOutputMode, audience: Iterable[str]) -> None:
        if mode not in VOICE_OUTPUT_MODES:
            raise ValueError(f"unsupported voice output mode {mode!r}")
        before = {pid: self.recipient(pid) for pid in audience}
        self._mode = mode
        self._mode_client = sender
        for pid, old in before.items():
            if self.recipient(pid) != old:
                await self.interrupt(pid, recipient=old)
        logger.info("VOICE_OUTPUT mode={} sender={}", mode, sender)

    # ── speech ───────────────────────────────────────────────────────────────

    async def say(
        self,
        pid: str,
        text: str,
        *,
        interrupt: bool = False,
        speak: bool = True,
        timestamp_us: int | None = None,
    ) -> None:
        """Show *text* to the room and, when *speak*, voice it to *pid*'s recipient."""

        text = text.strip()
        if not text:
            return
        await self.broadcast(AGENT_RESPONSE_TOPIC, text)
        if not speak:
            return
        recipient = self.recipient(pid)
        interrupt = interrupt or recipient in self._interrupt_next
        self._interrupt_next.discard(recipient)
        self._tracker.enqueued(recipient, text)
        await self._publish(
            VoiceOutput(text=text, interrupt=interrupt, timestamp_us=timestamp_us),
            recipient,
        )

    async def interrupt(self, pid: str, *, recipient: str | None = None) -> None:
        """Drop queued speech for *pid*; the next line then supersedes what remains."""

        target = recipient or self.recipient(pid)
        self._interrupt_next.add(target)
        try:
            await self._ep.flush_return_audio(target)
        except Exception:
            logger.exception("flush_return_audio failed pid={!r}", target)

    def remaining_s(self, pid: str) -> float:
        return self._tracker.remaining_s(self.recipient(pid))

    # ── what was heard ───────────────────────────────────────────────────────

    def heard(self, pid: str, timestamp_us: int | None = None) -> None:
        self._last_heard[pid] = timestamp_us or _now_us()

    def last_heard_us(self, pid: str) -> int:
        return self._last_heard.get(pid, 0)

    # ── text lines ───────────────────────────────────────────────────────────

    async def echo_user(self, text: str) -> None:
        await self.broadcast(USER_ECHO_TOPIC, text)

    async def progress(self, text: str) -> None:
        await self.broadcast(PROGRESS_TOPIC, text)

    async def broadcast(self, topic: str, payload: str | dict[str, Any]) -> None:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        for pid in sorted(self._ep.connected_participants):
            await self.send(pid, topic, data)

    async def send(self, pid: str, topic: str, payload: str | dict[str, Any]) -> None:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        try:
            await self._ep.send_return_data(DataMessage(
                participant_id=pid, topic=topic, pts_us=_now_us(), data=data.encode(),
            ))
        except Exception:
            logger.exception("send_return_data failed pid={!r} topic={}", pid, topic)


def runtime_publisher(runtime: AgentRuntime) -> Callable[[VoiceOutput, str], Awaitable[None]]:
    """Publish voice output addressed to one participant."""

    async def publish(output: VoiceOutput, participant_id: str) -> None:
        await runtime.publish(
            VOICE_OUTPUT_TOPIC, output, participant_id=participant_id, source="guidance",
        )

    return publish


__all__ = [
    "AGENT_RESPONSE_TOPIC",
    "PROGRESS_TOPIC",
    "USER_ECHO_TOPIC",
    "VOICE_OUTPUT_MODES",
    "PlaybackTracker",
    "SpeechRouter",
    "runtime_publisher",
]
