# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera frames and the annotated return-video preview.

During guidance one inference serves both ends: the frame drawn for the
owner's preview is the same annotated array a grounded check reuses, so what
the wearer sees and what the model is shown cannot disagree. Outside guidance
a client in ``default`` overlay mode gets a live preview drawn by the general
object detector; those frames are decoration and never reach the frame cache
or the session recorder, because a check reusing a COCO detector's boxes would
be handed evidence about a "cell phone" when it asked about a nose pad.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass

from loguru import logger
from sop_guidance.backends.base import TimedFrame
from sop_guidance.recorder import SessionHandle
from sop_guidance.vision import FrameAnnotator, bgr_to_jpeg, bgr_to_rgb24, frame_to_bgr
from xr_ai_hub import FrameSignal, ParticipantEvent, PixelFormat, ProcessorEndpoint, ReturnVideoFrame


def _now_us() -> int:
    return time.time_ns() // 1_000


class FrameCache:
    """Frame signals, fresh pixels on request, and the newest annotated frame."""

    def __init__(self, endpoint: ProcessorEndpoint, *, max_age_s: float = 3.0,
                 timeout_s: float = 5.0) -> None:
        self._ep = endpoint
        self._max_age_us = int(max_age_s * 1_000_000)
        self._timeout_s = timeout_s
        self._signals: dict[str, FrameSignal] = {}
        self._waiters: dict[str, set[asyncio.Event]] = {}
        self._latest: dict[str, TimedFrame] = {}
        endpoint.on_frame(self._on_frame)
        endpoint.on_participant(self._on_participant)

    async def _on_frame(self, signal: FrameSignal) -> None:
        current = self._signals.get(signal.participant_id)
        if current is None or signal.pts_us >= current.pts_us:
            self._signals[signal.participant_id] = signal
        for event in self._waiters.get(signal.participant_id, ()):
            event.set()

    async def _on_participant(self, event: ParticipantEvent) -> None:
        if not event.joined:
            self.release(event.participant_id)

    def release(self, participant_id: str) -> None:
        self._signals.pop(participant_id, None)
        self._latest.pop(participant_id, None)
        for event in self._waiters.get(participant_id, ()):
            event.set()

    def publishing(self) -> list[str]:
        """Participants with a fresh camera frame."""

        now = _now_us()
        return sorted(pid for pid, s in self._signals.items()
                      if now - s.pts_us < self._max_age_us)

    def signal(self, participant_id: str) -> FrameSignal | None:
        return self._signals.get(participant_id)

    def latest(self, participant_id: str) -> TimedFrame | None:
        """The newest annotated preview frame, when one is fresh."""

        frame = self._latest.get(participant_id)
        if frame is None or _now_us() - frame.timestamp_us > self._max_age_us:
            return None
        return frame

    def store(self, frame: TimedFrame) -> None:
        self._latest[frame.participant_id] = frame

    async def _wait_signal(self, participant_id: str) -> FrameSignal | None:
        signal = self._signals.get(participant_id)
        if signal is not None and _now_us() - signal.pts_us < self._max_age_us:
            return signal
        event = asyncio.Event()
        self._waiters.setdefault(participant_id, set()).add(event)
        try:
            with suppress(TimeoutError):
                async with asyncio.timeout(self._timeout_s):
                    while True:
                        event.clear()
                        signal = self._signals.get(participant_id)
                        if signal is not None and _now_us() - signal.pts_us < self._max_age_us:
                            return signal
                        await event.wait()
            return None
        finally:
            waiters = self._waiters.get(participant_id)
            if waiters is not None:
                waiters.discard(event)
                if not waiters:
                    self._waiters.pop(participant_id, None)

    async def fetch(self, participant_id: str) -> TimedFrame | None:
        """Fresh, unannotated pixels from *participant_id*'s camera, or None."""

        signal = await self._wait_signal(participant_id)
        if signal is None:
            return None
        data = await self._ep.request_frame(signal)
        if data is None:
            return None
        image = await asyncio.to_thread(frame_to_bgr, data.data, data.width, data.height, data.fmt)
        return TimedFrame(
            participant_id=participant_id, timestamp_us=data.pts_us,
            width=data.width, height=data.height, image=image,
        )

    async def fetch_jpeg(self, participant_id: str, *, max_width: int = 1280) -> bytes | None:
        frame = await self.fetch(participant_id)
        if frame is None:
            return None
        return await asyncio.to_thread(bgr_to_jpeg, frame.image, max_width=max_width, quality=85)


@dataclass(slots=True)
class _Loop:
    recipient: str
    source: str
    guidance: bool
    task: asyncio.Task[None]


class PreviewManager:
    """One annotated preview loop per recipient."""

    def __init__(
        self,
        *,
        endpoint: ProcessorEndpoint,
        frames: FrameCache,
        fps: float,
        live_annotator: Callable[[], FrameAnnotator | None],
        recorder: Callable[[str], SessionHandle | None],
    ) -> None:
        self._ep = endpoint
        self._frames = frames
        self._fps = fps
        self._live_annotator = live_annotator
        self._recorder = recorder
        self._loops: dict[str, _Loop] = {}

    def running(self, recipient: str) -> _Loop | None:
        return self._loops.get(recipient)

    async def start_guidance(self, recipient: str, source: str,
                             annotator: FrameAnnotator | None) -> None:
        await self.stop(recipient)
        if self._fps <= 0:
            return
        self._start(recipient, source, annotator, guidance=True)

    async def start_live(self, recipient: str, source: str) -> None:
        """Start or re-point the live preview; a no-op when already on *source*."""

        current = self._loops.get(recipient)
        if current is not None and (current.guidance or current.source == source):
            return
        await self.stop(recipient)
        annotator = self._live_annotator()
        if self._fps <= 0 or annotator is None:
            return
        self._start(recipient, source, annotator, guidance=False)

    async def stop_live(self, recipient: str) -> None:
        current = self._loops.get(recipient)
        if current is not None and not current.guidance:
            await self.stop(recipient)

    async def stop(self, recipient: str) -> None:
        loop = self._loops.pop(recipient, None)
        if loop is None:
            return
        loop.task.cancel()
        await asyncio.gather(loop.task, return_exceptions=True)
        with suppress(Exception):
            await self._ep.stop_return_video(recipient)

    async def aclose(self) -> None:
        for recipient in list(self._loops):
            await self.stop(recipient)

    def _start(self, recipient: str, source: str, annotator: FrameAnnotator | None,
               *, guidance: bool) -> None:
        task = asyncio.create_task(
            self._run(recipient, source, annotator, guidance=guidance),
            name=f"preview:{recipient}",
        )
        self._loops[recipient] = _Loop(recipient, source, guidance, task)

    async def _run(self, recipient: str, source: str, annotator: FrameAnnotator | None,
                   *, guidance: bool) -> None:
        period = 1.0 / self._fps
        last_pts = 0
        failures = 0
        logger.info("PREVIEW start recipient={} source={} guidance={} fps={:.1f}",
                    recipient, source, guidance, self._fps)
        # Paced against a moving deadline: sleeping a full period after the
        # work makes each cycle cost period + work and the loop never reaches
        # the configured rate.
        next_tick = time.monotonic()
        try:
            while True:
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    # Re-base instead of chasing missed slots, which would run
                    # flat out and starve the rest of the event loop.
                    next_tick = time.monotonic()
                    await asyncio.sleep(0)
                signal = self._frames.signal(source)
                if signal is None or signal.pts_us == last_pts:
                    continue
                try:
                    data = await self._ep.request_frame(signal)
                    if data is None or data.pts_us == last_pts:
                        continue
                    last_pts = data.pts_us
                    image = await asyncio.to_thread(
                        frame_to_bgr, data.data, data.width, data.height, data.fmt,
                    )
                    drawn = image
                    annotated = None
                    if annotator is not None:
                        # The annotator draws in place; the raw pixels stay
                        # untouched for anything that needs them unannotated.
                        annotated = await annotator.annotate_array(image.copy(), stream=source)
                        drawn = annotated.image
                    if guidance:
                        self._frames.store(TimedFrame(
                            participant_id=source, timestamp_us=data.pts_us,
                            width=data.width, height=data.height, image=image,
                            annotated=annotated,
                        ))
                        handle = self._recorder(recipient)
                        if handle is not None and handle.preview_due():
                            handle.push_preview_frame(
                                await asyncio.to_thread(bgr_to_jpeg, drawn), data.pts_us,
                            )
                    height, width = drawn.shape[:2]
                    await self._ep.send_return_video(ReturnVideoFrame(
                        pts_us=data.pts_us,
                        width=width,
                        height=height,
                        fmt=PixelFormat.RGB24,
                        data=await asyncio.to_thread(bgr_to_rgb24, drawn),
                        participant_id=recipient,
                    ))
                    failures = 0
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # One bad frame must not end the preview for the session.
                    failures += 1
                    if failures in (1, 10) or failures % 100 == 0:
                        logger.warning("PREVIEW frame failed recipient={} count={} error={}",
                                       recipient, failures, exc)
        finally:
            logger.info("PREVIEW stop recipient={}", recipient)


__all__ = ["FrameCache", "PreviewManager"]
