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
from sop_guidance.backends.base import OverlayUpdate, TimedFrame
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
    backend_boxes: bool = False


# Backend boxes older than this are stale and no longer drawn.
_OVERLAY_MAX_AGE_S = 2.0


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
        box_painter: FrameAnnotator | None = None,
        watchers: Callable[[str], set[str]] | None = None,
    ) -> None:
        self._ep = endpoint
        self._frames = frames
        self._fps = fps
        self._live_annotator = live_annotator
        self._recorder = recorder
        # Draws boxes a backend supplies itself (``provides_overlay``); its own
        # detector is never run.
        self._box_painter = box_painter
        self._loops: dict[str, _Loop] = {}
        # Clients watching a camera: they see the guidance preview of that
        # camera, not only the session owner. Speech from the wearer's own
        # device makes the wearer the owner, while the operator's browser
        # watching that camera is who needs the task detector's boxes.
        self._watchers = watchers or (lambda _source: set())
        self._overlays: dict[str, tuple[float, OverlayUpdate]] = {}

    def running(self, recipient: str) -> _Loop | None:
        return self._loops.get(recipient)

    def guidance_for(self, source: str) -> _Loop | None:
        """The guidance preview running on *source*'s camera, if any."""

        for loop in self._loops.values():
            if loop.guidance and loop.source == source:
                return loop
        return None

    def audience(self, loop: _Loop) -> set[str]:
        """Who a loop's frames go to: its recipient, plus watchers of a guided camera."""

        if not loop.guidance:
            return {loop.recipient}
        others = {w for w in self._watchers(loop.source) if w != loop.recipient
                  and not ((own := self._loops.get(w)) is not None and own.guidance)}
        return {loop.recipient} | others

    async def start_guidance(self, recipient: str, source: str,
                             annotator: FrameAnnotator | None,
                             *, backend_boxes: bool = False) -> None:
        """Preview *source* for *recipient* during guidance.

        With ``backend_boxes`` the frames carry the boxes the backend last sent
        through :meth:`set_overlay` instead of *annotator*'s detections, and
        *annotator*, when given, only paints them in its profile's colours.
        """

        await self.stop(recipient)
        if self._fps <= 0:
            return
        self._start(recipient, source, None if backend_boxes else annotator,
                    guidance=True, backend_boxes=backend_boxes,
                    painter=annotator if backend_boxes else None)
        # Watchers' own live loops would draw the general detector over the
        # same track; the guidance loop serves them now.
        for watcher in self._watchers(source):
            await self.stop_live(watcher)

    def set_overlay(self, recipient: str, update: OverlayUpdate) -> None:
        """The boxes a backend wants on *recipient*'s preview from now on."""

        self._overlays[recipient] = (time.monotonic(), update)

    async def start_live(self, recipient: str, source: str) -> None:
        """Start or re-point the live preview; a no-op when already on *source*."""

        current = self._loops.get(recipient)
        if current is not None and (current.guidance or current.source == source):
            return
        await self.stop(recipient)
        if self.guidance_for(source) is not None:
            # That camera is being guided; its guidance preview reaches this
            # client as a watcher.
            return
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
        audience = self.audience(loop)
        loop.task.cancel()
        await asyncio.gather(loop.task, return_exceptions=True)
        self._overlays.pop(recipient, None)
        for pid in audience:
            with suppress(Exception):
                await self._ep.stop_return_video(pid)

    async def aclose(self) -> None:
        for recipient in list(self._loops):
            await self.stop(recipient)

    def _start(self, recipient: str, source: str, annotator: FrameAnnotator | None,
               *, guidance: bool, backend_boxes: bool = False,
               painter: FrameAnnotator | None = None) -> None:
        task = asyncio.create_task(
            self._run(recipient, source, annotator, guidance=guidance,
                      backend_boxes=backend_boxes, painter=painter),
            name=f"preview:{recipient}",
        )
        self._loops[recipient] = _Loop(recipient, source, guidance, task, backend_boxes)

    def _backend_boxes(self, recipient: str) -> OverlayUpdate | None:
        entry = self._overlays.get(recipient)
        if entry is None or time.monotonic() - entry[0] > _OVERLAY_MAX_AGE_S:
            return None
        return entry[1]

    async def _run(self, recipient: str, source: str, annotator: FrameAnnotator | None,
                   *, guidance: bool, backend_boxes: bool = False,
                   painter: FrameAnnotator | None = None) -> None:
        painter = painter or self._box_painter
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
                    elif backend_boxes and painter is not None:
                        update = self._backend_boxes(recipient)
                        if update is not None and update.detections:
                            drawn = painter.draw(image.copy(), list(update.detections))
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
                    pixels = await asyncio.to_thread(bgr_to_rgb24, drawn)
                    loop = self._loops.get(recipient)
                    for pid in sorted(self.audience(loop) if loop else {recipient}):
                        await self._ep.send_return_video(ReturnVideoFrame(
                            pts_us=data.pts_us,
                            width=width,
                            height=height,
                            fmt=PixelFormat.RGB24,
                            data=pixels,
                            participant_id=pid,
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
