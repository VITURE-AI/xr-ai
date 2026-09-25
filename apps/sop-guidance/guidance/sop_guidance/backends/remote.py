# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A procedure backend that runs in its own process: the ``remote`` backend.

Some backends cannot share the worker's environment, such as a judge pinned
to a MediaPipe or protobuf the worker does not use. Such a backend runs as a
*sidecar*, an application-owned process with its own uv project, and the
procedure names it with ``backend: remote``::

    backend: remote
    backend_config:
      endpoint: ipc:///tmp/rpi-hat-judge.sock

The sidecar wraps an ordinary in-process :class:`ProcedureBackend` with
:func:`serve`, so a backend author writes no protocol code and the host cannot
tell the two placements apart.

Frames go one way over a shared-memory ring the host creates per run; the
host sends only the session input participant's frames, only while the run
is open. Control and events share one ZMQ connection of msgpack maps: the host
sends requests, the sidecar answers each one and pushes run events in between.
Events emitted while serving a request arrive before its reply, so an
announcement made by ``start`` or ``command`` has been spoken by the time the
call returns, exactly as in-process. Every reply and event carries the run's
snapshot, which is how the synchronous :meth:`ProcedureRun.snapshot` answers
without a round trip. A served run gets no model clients and records nothing:
it brings its own models, and the host's recorder keeps the session.
"""

from __future__ import annotations

import asyncio
import os
import secrets
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import asdict
from multiprocessing import resource_tracker
from typing import Any

import msgpack
import numpy as np
import zmq
import zmq.asyncio
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from xr_ai_hub import FrameSignal, PixelFormat, ShmRingBuffer

from ..vision.overlay import Detection
from .base import (
    BackendServices,
    Capabilities,
    CommandResult,
    Cue,
    ModelHandles,
    OverlayUpdate,
    ProcedureBackend,
    ProcedureRun,
    RunCommand,
    RunContext,
    RunEvent,
    RunFinished,
    RunSnapshot,
    StepChanged,
    StepInfo,
    TimedFrame,
    TurnContext,
    Verdict,
)

PROTOCOL_VERSION = 1

Message = dict[str, Any]


class RemoteBackendError(RuntimeError):
    """The sidecar is unreachable, timed out, or failed a request."""


class RemoteBackendConfig(BaseModel):
    """The ``backend_config`` block of a ``backend: remote`` procedure."""

    model_config = ConfigDict(extra="forbid")

    endpoint: str
    """ZMQ endpoint the sidecar binds: ``ipc:///tmp/judge.sock``, ``tcp://127.0.0.1:8110``."""

    connect_timeout_s: float = Field(default=10.0, gt=0.0)
    """How long startup waits for the sidecar to describe itself."""

    request_timeout_s: float = Field(default=10.0, gt=0.0)
    """Deadline for one request, including the frame the sidecar is grading."""

    ring_slots: int = Field(default=2, ge=1)
    """Frame slots per run; one frame is in flight at a time."""

    max_frame_bytes: int = Field(default=1920 * 1080 * 3, gt=0)
    """Largest RGB24 frame the ring carries; larger frames are dropped."""


# ── wire encoding ────────────────────────────────────────────────────────────


def _pack(message: Message) -> bytes:
    return msgpack.packb(message, use_bin_type=True)


def _unpack(data: bytes) -> Message:
    message = msgpack.unpackb(data, raw=False)
    if not isinstance(message, dict):
        raise RemoteBackendError("remote backend frame is not a map")
    return message


def _json(value: Mapping[str, Any]) -> dict[str, Any]:
    return dict(value)


def encode_event(event: RunEvent) -> Message:
    if isinstance(event, StepChanged):
        return {"type": "step", "index": event.index, "reason": event.reason,
                "acknowledge": event.acknowledge}
    if isinstance(event, Cue):
        return {"type": "cue", "text": event.text, "kind": event.kind, "priority": event.priority}
    if isinstance(event, Verdict):
        return {"type": "verdict", "result": _json(event.result)}
    if isinstance(event, OverlayUpdate):
        return {"type": "overlay", "timestamp_us": event.timestamp_us,
                "detections": [asdict(d) for d in event.detections],
                "extra": _json(event.extra)}
    if isinstance(event, RunFinished):
        return {"type": "finished", "outcome": event.outcome, "reason": event.reason}
    raise TypeError(f"not a run event: {event!r}")


def decode_event(message: Message) -> RunEvent:
    kind = message.get("type")
    if kind == "step":
        return StepChanged(int(message["index"]), reason=message["reason"],
                           acknowledge=bool(message["acknowledge"]))
    if kind == "cue":
        return Cue(str(message["text"]), kind=message["kind"], priority=int(message["priority"]))
    if kind == "verdict":
        return Verdict(dict(message["result"]))
    if kind == "overlay":
        return OverlayUpdate(int(message["timestamp_us"]),
                             tuple(Detection(**d) for d in message["detections"]),
                             extra=dict(message["extra"]))
    if kind == "finished":
        return RunFinished(message["outcome"], reason=str(message["reason"]))
    raise RemoteBackendError(f"unknown remote event {kind!r}")


def _encode_snapshot(snapshot: RunSnapshot) -> Message:
    return {"step_index": snapshot.step_index, "total_steps": snapshot.total_steps,
            "instruction": snapshot.instruction, "extra": _json(snapshot.extra),
            "state": _json(snapshot.state)}


def _decode_snapshot(message: Message) -> RunSnapshot:
    return RunSnapshot(int(message["step_index"]), int(message["total_steps"]),
                       str(message["instruction"]), extra=dict(message["extra"]),
                       state=dict(message["state"]))


def _encode_step(step: StepInfo) -> Message:
    return asdict(step)


def _decode_step(message: Message) -> StepInfo:
    return StepInfo(
        number=int(message["number"]), instruction=str(message["instruction"]),
        title=str(message.get("title", "")),
        reference_images=tuple(message.get("reference_images", ())),
        before_image=str(message.get("before_image", "")),
        gradeable=bool(message.get("gradeable", True)),
    )


# ── host side ────────────────────────────────────────────────────────────────


EventHandler = Callable[[Message], Awaitable[None]]


class _Connection:
    """One DEALER socket to the sidecar, shared by every run of a backend."""

    def __init__(self, endpoint: str, *, timeout_s: float) -> None:
        self._timeout_s = timeout_s
        self._socket = zmq.asyncio.Context.instance().socket(zmq.DEALER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)
        self._pending: dict[str, asyncio.Future[Message]] = {}
        self._handlers: dict[str, EventHandler] = {}
        self._receiver = asyncio.create_task(self._receive(), name="remote-backend-receiver")

    def attach(self, run_id: str, handler: EventHandler) -> None:
        self._handlers[run_id] = handler

    def detach(self, run_id: str) -> None:
        self._handlers.pop(run_id, None)

    async def request(self, op: str, *, run: str = "", args: Message | None = None) -> Message:
        request_id = secrets.token_hex(8)
        future: asyncio.Future[Message] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._socket.send(_pack({"v": PROTOCOL_VERSION, "id": request_id, "op": op,
                                           "run": run, "args": args or {}}))
            async with asyncio.timeout(self._timeout_s):
                reply = await future
        except TimeoutError as exc:
            raise RemoteBackendError(f"remote backend {op!r} timed out") from exc
        finally:
            self._pending.pop(request_id, None)
        if not reply.get("ok"):
            raise RemoteBackendError(f"remote backend {op!r} failed: {reply.get('error', '')}")
        return reply

    async def _receive(self) -> None:
        while True:
            try:
                message = _unpack(await self._socket.recv())
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("remote backend sent an unreadable frame")
                continue
            if message.get("type") == "event":
                handler = self._handlers.get(str(message.get("run", "")))
                if handler is None:
                    continue
                # Awaited inline: an event emitted while serving a request is
                # handled before that request's reply resolves. Host event
                # handling therefore must never wait on the same run.
                try:
                    await handler(message)
                except Exception:
                    logger.exception("remote backend event failed")
                continue
            future = self._pending.get(str(message.get("id", "")))
            if future is not None and not future.done():
                future.set_result(message)

    async def aclose(self) -> None:
        self._receiver.cancel()
        await asyncio.gather(self._receiver, return_exceptions=True)
        self._socket.close()


class RemoteRun:
    """The host's handle on one run living in the sidecar."""

    def __init__(self, *, ctx: RunContext, connection: _Connection, run_id: str,
                 ring: ShmRingBuffer, ring_name: str, max_frame_bytes: int) -> None:
        self._ctx = ctx
        self._conn = connection
        self.run_id = run_id
        self._ring = ring
        self.ring_name = ring_name
        self._max_frame_bytes = max_frame_bytes
        self._seq = 0
        self._last_frame: TimedFrame | None = None
        self._closed = False
        self._snapshot = RunSnapshot(0, 0, "")
        self._turn = ""
        connection.attach(run_id, self._on_event)

    async def open(self, *, start_step: int, checkpoint: Mapping[str, Any] | None) -> None:
        await self._call("open", {
            "session_id": self._ctx.session_id, "owner": self._ctx.owner,
            "start_step": start_step, "checkpoint": dict(checkpoint) if checkpoint else None,
            "ring": self.ring_name,
        })

    async def start(self) -> None:
        await self._call("start")

    async def on_frame(self, frame: TimedFrame) -> None:
        if self._closed:
            return
        rgb = np.ascontiguousarray(frame.image[..., ::-1])
        if rgb.nbytes > self._max_frame_bytes:
            logger.warning("remote backend frame {}x{} exceeds max_frame_bytes; dropped",
                           frame.width, frame.height)
            return
        self._seq += 1
        height, width = rgb.shape[:2]
        try:
            slot = self._ring.write_frame(memoryview(rgb), width, height, PixelFormat.RGB24,
                                          frame.timestamp_us, self._seq)
        except RuntimeError:
            # Every slot still held by the sidecar: drop, the next tick is newer.
            return
        self._last_frame = frame
        try:
            await self._call("frame", {
                "slot": slot, "seq": self._seq, "pts_us": frame.timestamp_us, "width": width,
                "height": height, "data_sz": rgb.nbytes, "participant_id": frame.participant_id,
            })
        except RemoteBackendError as exc:
            # One lost frame is not the pump's problem; the next tick retries.
            logger.warning("remote backend frame failed: {}", exc)

    async def command(self, command: RunCommand) -> CommandResult:
        try:
            result = await self._call("command", {"kind": command.kind,
                                                  "step_index": command.step_index,
                                                  "transcript": command.transcript})
        except RemoteBackendError as exc:
            logger.warning("remote backend command failed: {}", exc)
            return CommandResult(False, reason="backend-unavailable")
        return CommandResult(bool(result["accepted"]), speech=str(result["speech"]),
                             reason=str(result["reason"]))

    def snapshot(self) -> RunSnapshot:
        return self._snapshot

    def turn_context(self) -> TurnContext:
        # The picture is the host's own copy of what it last pushed, so the
        # foreground sees what the backend judged without a round trip.
        frame = self._last_frame or self._ctx.latest_frame()
        jpeg = b""
        if frame is not None:
            from ..vision.frames import bgr_to_jpeg

            with suppress(Exception):
                jpeg = bgr_to_jpeg(frame.image, max_width=1280, quality=80)
        return TurnContext(prompt_block=self._turn, frame_jpeg=jpeg)

    async def input_changed(self) -> None:
        await self._call("input_changed")

    async def close(self, reason: str) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._call("close", {"reason": reason})
        except RemoteBackendError as exc:
            logger.warning("remote backend close failed: {}", exc)
        finally:
            self._conn.detach(self.run_id)
            self._ring.close()
            self._ring.unlink()

    async def _call(self, op: str, args: Message | None = None) -> Message:
        payload = dict(args or {})
        payload["status"] = {
            "input_participant": self._ctx.input_participant,
            "speech_remaining_s": self._ctx.speech_remaining_s(),
            "last_heard_us": self._ctx.last_heard_us(),
            "wearer_requests": list(self._ctx.wearer_requests()),
        }
        reply = await self._conn.request(op, run=self.run_id, args=payload)
        self._absorb(reply)
        return reply.get("result") or {}

    def _absorb(self, message: Message) -> None:
        if message.get("snapshot"):
            self._snapshot = _decode_snapshot(message["snapshot"])
        if "turn" in message:
            self._turn = str(message["turn"])

    async def _on_event(self, message: Message) -> None:
        self._absorb(message)
        if not self._closed:
            await self._ctx.emit(decode_event(message["event"]))


class RemoteBackend:
    """A :class:`ProcedureBackend` whose runs live in a sidecar process."""

    def __init__(self, *, config: RemoteBackendConfig, description: Message) -> None:
        self.config = config
        self.name = str(description["name"])
        self.capabilities = Capabilities.model_validate(description["capabilities"])
        self._title = str(description["title"])
        self._steps = [_decode_step(s) for s in description["steps"]]
        self._parts = tuple(str(p) for p in description["parts"])
        self._digest = [str(i) for i in description["instructions_digest"]]
        self._problems = [str(p) for p in description["problems"]]
        self._conn: _Connection | None = None

    @property
    def title(self) -> str:
        return self._title

    def steps(self) -> list[StepInfo]:
        return list(self._steps)

    def parts(self) -> tuple[str, ...]:
        return self._parts

    def validate(self) -> list[str]:
        return list(self._problems)

    def instructions_digest(self) -> list[str]:
        return list(self._digest)

    def preview_annotator(self) -> Any:
        return None

    async def open_run(self, ctx: RunContext, *, start_step: int,
                       checkpoint: Mapping[str, Any] | None) -> RemoteRun:
        if self._conn is None:
            self._conn = _Connection(self.config.endpoint,
                                     timeout_s=self.config.request_timeout_s)
        ring_name = f"sopg-{secrets.token_hex(6)}"
        ring = ShmRingBuffer(ring_name, num_slots=self.config.ring_slots,
                             max_frame_bytes=self.config.max_frame_bytes, create=True)
        run = RemoteRun(ctx=ctx, connection=self._conn, run_id=secrets.token_hex(8),
                        ring=ring, ring_name=ring_name,
                        max_frame_bytes=self.config.max_frame_bytes)
        try:
            await run.open(start_step=start_step, checkpoint=checkpoint)
        except Exception:
            self._conn.detach(run.run_id)
            ring.close()
            ring.unlink()
            raise
        return run

    async def aclose(self) -> None:
        """Drop the connection; runs should be closed first."""

        if self._conn is not None:
            await self._conn.aclose()
            self._conn = None


def describe(config: RemoteBackendConfig) -> Message:
    """Ask the sidecar what it serves, blocking for up to ``connect_timeout_s``.

    Synchronous because backends are built during startup, before any
    session exists; a sidecar still starting simply answers late.
    """

    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(config.endpoint)
        socket.send(_pack({"v": PROTOCOL_VERSION, "id": "describe", "op": "describe",
                           "run": "", "args": {}}))
        if not socket.poll(int(config.connect_timeout_s * 1000)):
            raise RemoteBackendError(
                f"no remote backend answered at {config.endpoint} within "
                f"{config.connect_timeout_s:g}s"
            )
        reply = _unpack(socket.recv())
    finally:
        socket.close()
        context.term()
    if not reply.get("ok"):
        raise RemoteBackendError(f"remote backend describe failed: {reply.get('error', '')}")
    return reply["result"]


def create_backend(services: BackendServices) -> RemoteBackend:
    """Build the ``remote`` backend for one procedure folder."""

    entry = services.entry
    try:
        config = RemoteBackendConfig.model_validate(dict(services.config))
    except ValidationError as exc:
        raise ValueError(f"{entry.directory / 'procedure.yaml'}: backend_config: {exc}") from exc
    try:
        description = describe(config)
    except RemoteBackendError as exc:
        raise ValueError(f"procedure {entry.id!r}: {exc}") from exc
    return RemoteBackend(config=config, description=description)


# ── sidecar side ─────────────────────────────────────────────────────────────


class _NullRecorder:
    """The sidecar records nothing; the host's recorder owns the session."""

    recording = False

    def __getattr__(self, name: str) -> Callable[..., None]:
        return lambda *args, **kwargs: None


def _attach_ring(name: str) -> ShmRingBuffer:
    """Open the host's ring without adopting it.

    Attaching registers the segment with this process's resource tracker,
    which would unlink it, from under a host still using it, when the
    sidecar exits. The host created the ring and is the one to remove it.
    """

    ring = ShmRingBuffer(name, create=False)
    if os.name == "posix":
        with suppress(Exception):
            resource_tracker.unregister(f"/{name}", "shared_memory")
    return ring


class _SidecarContext:
    """The :class:`RunContext` a served backend's run sees in the sidecar."""

    def __init__(self, server: _Server, identity: bytes, run_id: str, args: Message) -> None:
        self._server = server
        self._identity = identity
        self._run_id = run_id
        self.session_id = str(args["session_id"])
        self.owner = str(args["owner"])
        self.run: ProcedureRun | None = None
        self.ring = _attach_ring(str(args["ring"]))
        self.frame: TimedFrame | None = None
        self.status: Message = dict(args.get("status") or {})
        self._recorder = _NullRecorder()

    @property
    def input_participant(self) -> str:
        return str(self.status.get("input_participant", self.owner))

    @property
    def models(self) -> ModelHandles:
        return ModelHandles()

    @property
    def recorder(self) -> Any:
        return self._recorder

    async def emit(self, event: RunEvent) -> None:
        message: Message = {"v": PROTOCOL_VERSION, "type": "event", "run": self._run_id,
                            "event": encode_event(event)}
        if self.run is not None:
            message.update(_run_state(self.run))
        await self._server.send(self._identity, message)

    def speech_remaining_s(self) -> float:
        return float(self.status.get("speech_remaining_s", 0.0))

    def last_heard_us(self) -> int:
        return int(self.status.get("last_heard_us", 0))

    def wearer_requests(self) -> tuple[str, ...]:
        return tuple(self.status.get("wearer_requests", ()))

    def latest_frame(self) -> TimedFrame | None:
        return self.frame

    async def fetch_frame(self) -> TimedFrame | None:
        # Frames are pushed by the host; the newest one is the freshest there is.
        return self.frame

    def read_frame(self, args: Message) -> TimedFrame:
        signal = FrameSignal(slot=int(args["slot"]), seq=int(args["seq"]),
                             pts_us=int(args["pts_us"]), width=int(args["width"]),
                             height=int(args["height"]), fmt=PixelFormat.RGB24,
                             data_sz=int(args["data_sz"]))
        try:
            view = self.ring.read_slot(signal)
            try:
                # Copied out before the slot is released; nothing may keep a
                # view into shared memory past that.
                image = np.frombuffer(view.data, dtype=np.uint8).reshape(
                    signal.height, signal.width, 3)[..., ::-1].copy()
            finally:
                view.data.release()
        finally:
            # Released even when the read failed, or the ring fills and the
            # host drops every later frame.
            with suppress(RuntimeError, ValueError):
                self.ring.release_slot(signal.slot)
        self.frame = TimedFrame(participant_id=str(args["participant_id"]),
                                timestamp_us=signal.pts_us, width=signal.width,
                                height=signal.height, image=image)
        return self.frame


def _run_state(run: ProcedureRun) -> Message:
    # Sent with every reply and event because the host reads both
    # synchronously; a backend should keep them cheap to produce.
    return {"snapshot": _encode_snapshot(run.snapshot()),
            "turn": run.turn_context().prompt_block}


class _Server:
    """The sidecar end: one ROUTER socket serving one backend's runs."""

    def __init__(self, backend: ProcedureBackend, endpoint: str) -> None:
        self._backend = backend
        self._socket = zmq.asyncio.Context.instance().socket(zmq.ROUTER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(endpoint)
        self._send_lock = asyncio.Lock()
        self._runs: dict[str, _SidecarContext] = {}

    async def send(self, identity: bytes, message: Message) -> None:
        async with self._send_lock:
            await self._socket.send_multipart([identity, _pack(message)])

    async def serve_forever(self) -> None:
        try:
            while True:
                identity, data = await self._socket.recv_multipart()
                try:
                    request = _unpack(data)
                except Exception:
                    logger.exception("remote backend request unreadable")
                    continue
                # One request at a time keeps each run's frames, commands and
                # close in the order the host sent them.
                reply: Message = {"v": PROTOCOL_VERSION, "type": "reply",
                                  "id": request.get("id", "")}
                try:
                    reply["result"] = await self._handle(identity, request)
                    reply["ok"] = True
                except Exception as exc:
                    logger.exception("remote backend {} failed", request.get("op"))
                    reply.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                ctx = self._runs.get(str(request.get("run", "")))
                if ctx is not None and ctx.run is not None:
                    reply.update(_run_state(ctx.run))
                await self.send(identity, reply)
        finally:
            for ctx in list(self._runs.values()):
                if ctx.run is not None:
                    with suppress(Exception):
                        await ctx.run.close("sidecar_shutdown")
                ctx.ring.close()
            self._socket.close()

    async def _handle(self, identity: bytes, request: Message) -> Message:
        if request.get("v") != PROTOCOL_VERSION:
            raise RemoteBackendError(f"protocol version {request.get('v')!r} is not "
                                     f"{PROTOCOL_VERSION}; update the sidecar or the worker")
        op = request.get("op")
        run_id = str(request.get("run", ""))
        args: Message = request.get("args") or {}
        if op == "describe":
            backend = self._backend
            return {
                "name": backend.name, "title": backend.title,
                "capabilities": backend.capabilities.model_dump(mode="json"),
                "steps": [_encode_step(s) for s in backend.steps()],
                "parts": list(backend.parts()),
                "instructions_digest": list(backend.instructions_digest()),
                "problems": backend.validate(),
            }
        if op == "open":
            ctx = _SidecarContext(self, identity, run_id, args)
            self._runs[run_id] = ctx
            try:
                ctx.run = await self._backend.open_run(
                    ctx, start_step=int(args["start_step"]), checkpoint=args.get("checkpoint"),
                )
            except Exception:
                self._runs.pop(run_id, None)
                ctx.ring.close()
                raise
            return {}
        ctx = self._runs.get(run_id)
        if ctx is None or ctx.run is None:
            raise RemoteBackendError(f"no open run {run_id!r}")
        ctx.status = dict(args.get("status") or ctx.status)
        run = ctx.run
        if op == "start":
            await run.start()
        elif op == "frame":
            await run.on_frame(ctx.read_frame(args))
        elif op == "command":
            result = await run.command(RunCommand(args["kind"], step_index=args.get("step_index"),
                                                  transcript=str(args.get("transcript", ""))))
            return {"accepted": result.accepted, "speech": result.speech,
                    "reason": result.reason}
        elif op == "input_changed":
            await run.input_changed()
        elif op == "close":
            try:
                await run.close(str(args.get("reason", "")))
            finally:
                self._runs.pop(run_id, None)
                ctx.ring.close()
        else:
            raise RemoteBackendError(f"unknown op {op!r}")
        return {}


async def serve(backend: ProcedureBackend, endpoint: str) -> None:
    """Serve *backend* to a host's ``remote`` backend at *endpoint* until cancelled."""

    logger.info("remote backend {} serving at {}", backend.name, endpoint)
    await _Server(backend, endpoint).serve_forever()


__all__ = [
    "PROTOCOL_VERSION",
    "RemoteBackend",
    "RemoteBackendConfig",
    "RemoteBackendError",
    "RemoteRun",
    "create_backend",
    "decode_event",
    "describe",
    "encode_event",
    "serve",
]
