# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The runtime agent bridging voice topics to the guidance worker."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from xr_ai_runtime import Agent, RuntimeContext, Topic, subscribe
from xr_ai_voice import (
    VOICE_TRANSCRIPT_TOPIC,
    UserQuery,
    VoiceParticipantJoined,
    VoiceParticipantLeft,
    VoiceTranscript,
)

from .interaction import Interaction
from .speech import SpeechRouter

USER_QUERY_TOPIC = Topic("sop-guidance.user-query", UserQuery)
PARTICIPANT_JOINED_TOPIC = Topic("sop-guidance.participant-joined", VoiceParticipantJoined)
PARTICIPANT_LEFT_TOPIC = Topic("sop-guidance.participant-left", VoiceParticipantLeft)


def _participant(ctx: RuntimeContext) -> str:
    pid = ctx.metadata.participant_id
    if pid is None:
        raise ValueError("guidance messages require a participant")
    return pid


class GuidanceAgent(Agent):
    """Hand accepted speech and participant lifecycle to the worker."""

    def __init__(
        self,
        *,
        interaction: Interaction,
        speech: SpeechRouter,
        on_joined: Callable[[str], Awaitable[None]],
        on_left: Callable[[str], Awaitable[None]],
    ) -> None:
        super().__init__()
        self._interaction = interaction
        self._speech = speech
        self._on_joined = on_joined
        self._on_left = on_left

    @subscribe(USER_QUERY_TOPIC)
    async def query(self, query: UserQuery, ctx: RuntimeContext) -> None:
        await self._interaction.on_speech(_participant(ctx), query.text, query.timestamp_us)

    @subscribe(VOICE_TRANSCRIPT_TOPIC)
    async def transcript(self, transcript: VoiceTranscript, ctx: RuntimeContext) -> None:
        # Every final transcript counts as the wearer talking, including the
        # ones the gate drops: a correction must not talk over them.
        pid = ctx.metadata.participant_id
        if pid is None:
            return
        self._speech.heard(pid, transcript.timestamp_us)
        if self._interaction.gate_eats_exit(pid, transcript.text):
            # The voice gate treats a bare "stop" as a global stop and never
            # forwards it, but during guidance it is the wake-free exit.
            await self._interaction.on_speech(pid, transcript.text, transcript.timestamp_us)

    @subscribe(PARTICIPANT_JOINED_TOPIC)
    async def joined(self, _event: VoiceParticipantJoined, ctx: RuntimeContext) -> None:
        await self._on_joined(_participant(ctx))

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def left(self, _event: VoiceParticipantLeft, ctx: RuntimeContext) -> None:
        await self._on_left(_participant(ctx))


__all__ = [
    "PARTICIPANT_JOINED_TOPIC",
    "PARTICIPANT_LEFT_TOPIC",
    "USER_QUERY_TOPIC",
    "GuidanceAgent",
]
