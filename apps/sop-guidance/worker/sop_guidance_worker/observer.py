# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Background scene memory for normal mode, ported from the old fork.

Every ``interval_s`` the observer takes the newest frame of each camera a
client is watching and asks the VLM what changed since its last observation;
"unchanged" answers are dropped. Every ``condense_interval_s`` the recent
observations are condensed into a short scene summary. Both go into the
normal-mode prompt, which is what lets the assistant know roughly what is in
view before any tool call and answer "what did I see earlier?".

It yields to guidance, whose step checks own the VLM for a guided camera, and
to a turn in flight, as the old fork did.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from loguru import logger
from sop_guidance.backends.vlm.grading import extract_json
from sop_guidance.vision import bgr_to_jpeg
from xr_ai_models import ChatMessage

from .preview import FrameCache

# The old fork's prompts, verbatim.
_FIRST_QUESTION = (
    "In one short phrase, describe the most notable action or movement "
    "happening in this frame. Include spatial position if relevant "
    "(left/right/center of frame, above/below/beside what object)."
)
_CONDENSE_PROMPT = (
    "You are a scene context summarizer for smart glasses. "
    "Given a timeline of camera observations (each tagged with "
    "[HH:MM:SS|timestamp_us]), output ONLY valid JSON in this shape:\n"
    '{"overview":"1-2 sentence scene summary","events":['
    '{"timestamp_us":<int>,"time":"HH:MM:SS","description":"brief event"}]}\n'
    "Include only the 3-6 most significant events. "
    "Use the exact timestamp_us values from the input."
)
_UNCHANGED = frozenset({
    "unchanged", "no change", "nothing new", "same", "no changes",
    "nothing has changed", "nothing changed", "no significant change",
})


def _change_question(previous: str) -> str:
    prev_clean = previous.split("compared to")[0].strip().rstrip(",")
    return (
        f"Previous observation: {prev_clean}\n\n"
        "Compare this new frame to the previous observation. "
        "Has something visibly changed — a new action, movement, or object state?\n"
        "If yes: one short phrase describing the change. "
        "Include WHERE in the frame (left/right/center) and position "
        "relative to nearby objects if relevant.\n"
        "If nothing changed: respond with exactly: unchanged"
    )


def _hms(timestamp_us: int) -> str:
    return time.strftime("%H:%M:%S", time.localtime(timestamp_us / 1_000_000))


@dataclass(frozen=True, slots=True)
class Observation:
    timestamp_us: int
    description: str


class SceneMemory:
    """Observations and the condensed summary of one camera."""

    def __init__(self, max_observations: int) -> None:
        self.observations: deque[Observation] = deque(maxlen=max_observations)
        self.summary = ""
        self.last_description = ""
        self.last_timestamp_us = 0

    def add(self, observation: Observation) -> None:
        self.observations.append(observation)
        self.last_description = observation.description
        self.last_timestamp_us = observation.timestamp_us

    def context_block(self, max_recent: int) -> str:
        """The old fork's ``[Scene summary]`` and ``[Recent observations]`` sections."""

        parts = [f"[Scene summary]\n{self.summary}" if self.summary
                 else "[Scene summary]\nNo scene summary available yet."]
        recent = list(self.observations)[-max_recent:] if max_recent else []
        if recent:
            lines = ["[Recent observations]"]
            lines += [f"  {_hms(o.timestamp_us)}  {o.description}" for o in recent]
            parts.append("\n".join(lines))
        else:
            parts.append("[Recent observations]\nNone yet.")
        return "\n\n".join(parts)


class SceneObserver:
    """Observe watched cameras in the background and condense what was seen."""

    def __init__(
        self,
        *,
        frames: FrameCache,
        vlm: Any,
        llm: Any,
        sources: Callable[[], Iterable[str]],
        busy: Callable[[str], bool],
        interval_s: float = 2.0,
        condense_interval_s: float = 60.0,
        max_observations: int = 240,
        vlm_timeout_s: float = 15.0,
    ) -> None:
        self._frames = frames
        self._vlm = vlm
        self._llm = llm
        self._sources = sources
        self._busy = busy
        self._interval_s = interval_s
        self._condense_interval_s = condense_interval_s
        self._max_observations = max_observations
        self._vlm_timeout_s = vlm_timeout_s
        self._memories: dict[str, SceneMemory] = {}
        self._tasks: list[asyncio.Task[None]] = []

    def memory(self, source: str) -> SceneMemory:
        memory = self._memories.get(source)
        if memory is None:
            memory = self._memories[source] = SceneMemory(self._max_observations)
        return memory

    def forget(self, source: str) -> None:
        self._memories.pop(source, None)

    def start(self) -> None:
        if self._tasks:
            return
        self._tasks = [
            asyncio.create_task(self._observe_loop(), name="scene-observer"),
            asyncio.create_task(self._condense_loop(), name="scene-condenser"),
        ]

    async def aclose(self) -> None:
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # ── observing ────────────────────────────────────────────────────────────

    async def _observe_loop(self) -> None:
        logger.info("scene observer started gap={:.1f}s", self._interval_s)
        while True:
            try:
                for source in sorted(set(self._sources())):
                    if self._busy(source):
                        continue
                    await self.observe(source)
                await asyncio.sleep(self._interval_s)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("scene observer error")
                await asyncio.sleep(self._interval_s)

    async def observe(self, source: str) -> Observation | None:
        """Caption what is new in *source*'s newest frame; None when nothing is."""

        memory = self.memory(source)
        frame = self._frames.latest(source)
        if frame is None:
            frame = await self._frames.fetch(source)
        if frame is None:
            return None
        # Same frame as last time: the hub has not delivered a new one yet.
        if memory.last_timestamp_us and frame.timestamp_us == memory.last_timestamp_us:
            return None
        jpeg = await asyncio.to_thread(bgr_to_jpeg, frame.image, max_width=1280, quality=85)
        if not jpeg:
            return None
        previous = memory.last_description
        question = _change_question(previous) if previous else _FIRST_QUESTION
        try:
            response = await asyncio.wait_for(
                self._vlm.ask_image(jpeg, question, max_tokens=96),
                timeout=self._vlm_timeout_s,
            )
        except Exception as exc:
            logger.warning("scene observation failed source={} error={}", source, exc)
            return None
        description = (response.content or "").strip()
        if not description or description.lower().rstrip(".!") in _UNCHANGED:
            return None
        # The model regenerated the same text as before.
        if previous and description == previous:
            return None
        observation = Observation(frame.timestamp_us, description)
        memory.add(observation)
        logger.info("OBS source={} {}", source, description[:120])
        return observation

    # ── condensing ───────────────────────────────────────────────────────────

    async def _condense_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._condense_interval_s)
                for source in list(self._memories):
                    await self.condense(source)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("scene condenser error")

    async def condense(self, source: str) -> str:
        """Summarise *source*'s last 20 observations; returns the new summary."""

        memory = self.memory(source)
        recent = list(memory.observations)[-20:]
        if not recent:
            return memory.summary
        obs_text = "\n".join(
            f"  [{_hms(o.timestamp_us)}|{o.timestamp_us}]  {o.description}" for o in recent
        )
        try:
            response = await asyncio.wait_for(self._llm.chat(
                (ChatMessage(role="system", content=_CONDENSE_PROMPT),
                 ChatMessage(role="user", content=f"Observations:\n{obs_text}")),
                max_tokens=256, temperature=0.1,
            ), timeout=20.0)
        except Exception:
            logger.exception("scene condenser call failed")
            return memory.summary
        raw = (response.content or "").strip()
        obj = extract_json(raw)
        try:
            structured = json.loads(obj) if obj else {"overview": raw, "events": []}
        except json.JSONDecodeError:
            structured = {"overview": raw, "events": []}
        if not isinstance(structured, dict):
            return memory.summary
        overview = str(structured.get("overview", "")).strip()
        events = structured.get("events", [])
        lines = [overview] if overview else []
        for event in events if isinstance(events, list) else []:
            if isinstance(event, dict):
                lines.append(f"  [{event.get('time', '')} | {event.get('timestamp_us', 0)} us] "
                             f"{event.get('description', '')}")
        summary = "\n".join(lines).strip()
        if summary:
            memory.summary = summary
            logger.info("SCENE_SUMMARY source={} {}", source, summary.replace("\n", " | ")[:200])
        return memory.summary


__all__ = ["Observation", "SceneMemory", "SceneObserver"]
