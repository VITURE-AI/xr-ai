# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The client data protocol and the host's view of the worker.

Clients steer the worker with small JSON messages on named data topics:

``xr.main_input``     ``{"participant_id": "..."}`` whose camera this client drives
``xr.voice_output``   ``{"mode": "default|selected_input|web_client"}``
``xr.yolo_mode``      ``{"mode": "default|guidance_only"}`` live overlay outside guidance
``xr.wake_mode``      ``{"required_in_live": bool}`` wake word outside guidance
``guidance.control``  ``{"action": ..., "request_id": ...}`` answered on ``guidance.result``

Untopiced data is typed text and is handled as a request. Every client acts
on the value it sends, not on a change: clients re-announce their modes on
connect, and that announcement is what tells the worker someone is there.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from loguru import logger
from sop_guidance.backends.base import OverlayUpdate, ProcedureBackend, TimedFrame
from sop_guidance.host import GuidanceHost, HostReply, SpeechKind
from xr_ai_hub import DataMessage, ProcessorEndpoint

from .preview import FrameCache, PreviewManager
from .speech import SpeechRouter

MAIN_INPUT_TOPIC = "xr.main_input"
VOICE_OUTPUT_TOPIC = "xr.voice_output"
YOLO_MODE_TOPIC = "xr.yolo_mode"
WAKE_MODE_TOPIC = "xr.wake_mode"
CONTROL_TOPIC = "guidance.control"
RESULT_TOPIC = "guidance.result"
STATE_TOPIC = "guidance.state"
OVERLAY_TOPIC = "guidance.overlay"
MODES_REQUEST_TOPIC = "xr.modes_request"
"""Worker to client: re-send your ``xr.*`` modes (they live in the client)."""

YOLO_MODES = frozenset({"default", "guidance_only"})


@dataclass(slots=True)
class ClientState:
    """What one client asked for."""

    input: str | None = None
    yolo_mode: str | None = None
    wake_in_live: bool | None = None
    wake_seq: int = 0
    """Order of the last ``xr.wake_mode``, so the newest of several drivers wins."""


class ClientRegistry:
    """Per-client selections, checked against who is actually connected."""

    def __init__(self, endpoint: ProcessorEndpoint, *, wake_in_live: bool) -> None:
        self._ep = endpoint
        self._clients: dict[str, ClientState] = {}
        self._wake_default = wake_in_live
        self._wake_seq = 0

    def connected(self) -> frozenset[str]:
        return self._ep.connected_participants

    def state(self, pid: str) -> ClientState:
        return self._clients.setdefault(pid, ClientState())

    def forget(self, pid: str) -> None:
        self._clients.pop(pid, None)
        for state in self._clients.values():
            if state.input == pid:
                state.input = None

    def explicit_input(self, pid: str) -> str | None:
        """*pid*'s own pick of another camera, when that participant is here.

        A stale pick (a client that missed a leave event) is ignored rather
        than honoured: acting as an absent participant would move ownership
        somewhere nobody is watching.
        """

        state = self._clients.get(pid)
        if state is None or not state.input or state.input == pid:
            return None
        return state.input if state.input in self.connected() else None

    def input_of(self, pid: str) -> str:
        return self.explicit_input(pid) or pid

    def set_wake_in_live(self, pid: str, required: bool) -> None:
        self._wake_seq += 1
        state = self.state(pid)
        state.wake_in_live = required
        state.wake_seq = self._wake_seq

    def wake_required_in_live(self, pid: str) -> bool:
        """Whether *pid*'s speech outside guidance needs the wake word.

        A wearer's glasses never announce a wake mode; the operator who picked
        their camera does. So *pid*'s own choice wins, then the newest choice
        of a connected client driving *pid*, then the configured default.
        """

        state = self._clients.get(pid)
        if state is not None and state.wake_in_live is not None:
            return state.wake_in_live
        connected = self.connected()
        drivers = [s for other, s in self._clients.items()
                   if other != pid and other in connected
                   and s.input == pid and s.wake_in_live is not None]
        if drivers:
            return bool(max(drivers, key=lambda s: s.wake_seq).wake_in_live)
        return self._wake_default

    def watchers(self, source: str) -> set[str]:
        """Connected clients showing *source*'s camera that asked for an overlay.

        A client that never announced an overlay mode (a wearer's glasses) is
        not sent return video at all.
        """

        return {pid for pid in self.connected()
                if (state := self._clients.get(pid)) is not None
                and state.yolo_mode is not None and self.input_of(pid) == source}

    def wants_live_preview(self, pid: str) -> bool:
        state = self._clients.get(pid)
        return state is not None and state.yolo_mode == "default"


class WorkerPorts:
    """:class:`~sop_guidance.host.HostPorts` over the worker's services."""

    def __init__(
        self,
        *,
        speech: SpeechRouter,
        frames: FrameCache,
        preview: PreviewManager,
        clients: ClientRegistry,
    ) -> None:
        self._speech = speech
        self._frames = frames
        self._preview = preview
        self._clients = clients

    async def say(self, owner: str, text: str, *, kind: SpeechKind) -> None:
        await self._speech.say(owner, text)

    def speech_remaining_s(self, owner: str) -> float:
        return self._speech.remaining_s(owner)

    def last_heard_us(self, owner: str) -> int:
        return self._speech.last_heard_us(owner)

    async def publish_state(self, state: Mapping[str, Any]) -> None:
        await self._speech.broadcast(STATE_TOPIC, dict(state))

    def latest_frame(self, participant_id: str) -> TimedFrame | None:
        return self._frames.latest(participant_id)

    async def fetch_frame(self, participant_id: str) -> TimedFrame | None:
        return await self._frames.fetch(participant_id)

    async def start_preview(self, owner: str, input_pid: str, backend: ProcedureBackend) -> None:
        if backend.capabilities.provides_overlay:
            await self._preview.start_guidance(owner, input_pid, backend.preview_annotator(),
                                               backend_boxes=True)
        else:
            await self._preview.start_guidance(owner, input_pid, backend.preview_annotator())

    async def stop_preview(self, owner: str) -> None:
        loop = self._preview.running(owner)
        viewers = self._preview.audience(loop) if loop is not None else {owner}
        await self._preview.stop(owner)
        connected = self._clients.connected()
        for pid in sorted(viewers | {owner}):
            if self._clients.wants_live_preview(pid) and pid in connected:
                await self._preview.start_live(pid, self._clients.input_of(pid))

    async def overlay_update(self, owner: str, update: OverlayUpdate) -> None:
        self._preview.set_overlay(owner, update)
        await self._speech.send(owner, OVERLAY_TOPIC, {
            "timestamp_us": update.timestamp_us,
            "detections": [asdict(d) for d in update.detections],
            "extra": dict(update.extra),
        })

    def resolve_input(self, owner: str, saved: str) -> str:
        if saved and saved in self._clients.connected():
            return saved
        return self._clients.input_of(owner)


def _load(payload: str) -> dict[str, Any] | None:
    try:
        value = json.loads(payload)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


TypedHandler = Callable[[str, str, int], Awaitable[None]]


class ClientProtocol:
    """Dispatch client data messages."""

    def __init__(
        self,
        *,
        host: GuidanceHost,
        clients: ClientRegistry,
        speech: SpeechRouter,
        preview: PreviewManager,
        on_typed: TypedHandler,
        cancel_turn: Callable[[str], Awaitable[None]],
    ) -> None:
        self._host = host
        self._clients = clients
        self._speech = speech
        self._preview = preview
        self._on_typed = on_typed
        self._cancel_turn = cancel_turn

    async def on_data(self, msg: DataMessage) -> None:
        try:
            text = (msg.data or b"").decode("utf-8", errors="replace").strip()
        except Exception:
            return
        if not text:
            return
        pid = msg.participant_id
        try:
            if not msg.topic:
                await self._on_typed(pid, text, msg.pts_us)
            elif msg.topic == MAIN_INPUT_TOPIC:
                await self._main_input(pid, text)
            elif msg.topic == VOICE_OUTPUT_TOPIC:
                await self._voice_output(pid, text)
            elif msg.topic == YOLO_MODE_TOPIC:
                await self._yolo_mode(pid, text)
            elif msg.topic == WAKE_MODE_TOPIC:
                self._wake_mode(pid, text)
            elif msg.topic == CONTROL_TOPIC:
                await self._control(pid, text)
        except Exception:
            logger.exception("client message failed pid={} topic={}", pid, msg.topic)

    # ── modes ────────────────────────────────────────────────────────────────

    async def _main_input(self, sender: str, payload: str) -> None:
        obj = _load(payload)
        if obj is None:
            logger.warning("main-input from {} ignored, unreadable: {!r}", sender, payload[:120])
            return
        target = str(obj.get("participant_id") or "").strip() or None
        if target is not None and target not in self._clients.connected():
            # Almost always a stale picker; honouring it would starve the
            # pipeline of frames.
            logger.warning("main-input from {} ignored, {!r} is not connected", sender, target)
            return
        self._clients.state(sender).input = target
        source = target or sender
        session = self._host.session_of(sender)
        if session is not None:
            if source != session.input_pid:
                await self._cancel_turn(sender)
                await self._host.change_input(sender, source)
            return
        # Clients re-announce their pick on every roster change; only a real
        # move restarts the live preview, or the overlay blinks out.
        if self._clients.wants_live_preview(sender):
            await self._preview.start_live(sender, source)

    async def _voice_output(self, sender: str, payload: str) -> None:
        obj = _load(payload) or {}
        mode = obj.get("mode")
        try:
            await self._speech.set_mode(sender, str(mode), self._clients.connected())
        except ValueError:
            logger.warning("voice-output from {} ignored, invalid: {!r}", sender, payload[:120])

    async def _yolo_mode(self, sender: str, payload: str) -> None:
        obj = _load(payload) or {}
        mode = obj.get("mode")
        if mode not in YOLO_MODES:
            logger.warning("yolo-mode from {} ignored, invalid: {!r}", sender, payload[:120])
            return
        self._clients.state(sender).yolo_mode = mode
        if self._host.session_of(sender) is not None:
            return
        # Acted on even when unchanged: a reconnecting client announces the
        # value the worker already assumed, and that is the signal to draw.
        if mode == "default":
            await self._preview.start_live(sender, self._clients.input_of(sender))
        else:
            await self._preview.stop_live(sender)

    def _wake_mode(self, sender: str, payload: str) -> None:
        obj = _load(payload) or {}
        required = obj.get("required_in_live")
        if not isinstance(required, bool):
            logger.warning("wake-mode from {} ignored, invalid: {!r}", sender, payload[:120])
            return
        self._clients.set_wake_in_live(sender, required)

    # ── session controls ─────────────────────────────────────────────────────

    async def _control(self, pid: str, payload: str) -> None:
        command = _load(payload)
        if command is None or not isinstance(command.get("request_id"), str):
            return
        action = command.get("action")
        source = command.get("input_participant")
        if action in {"resume", "confirm", "start"} and isinstance(source, str):
            if not source or source in self._clients.connected():
                self._clients.state(pid).input = source or None
        reply = await self._run_control(pid, action, command)
        result = reply.as_dict()
        result["request_id"] = command["request_id"]
        await self._speech.send(pid, RESULT_TOPIC, result)
        if reply.status in ("ok", "started") and reply.message:
            await self._speech.say(pid, reply.message)

    async def _run_control(self, pid: str, action: Any, command: dict[str, Any]) -> HostReply:
        session_id = command.get("session_id", "")
        if not isinstance(session_id, str):
            return HostReply("error", "Invalid session id.")
        token = command.get("token")
        token = token if isinstance(token, str) else None
        if action == "ready":
            await self._host.republish()
            return HostReply("ok", "")
        if action == "cancel":
            self._host.cancel_takeover(pid, token)
            return HostReply("ok", "")
        if action == "confirm":
            return await self._host.confirm_takeover(pid, token)
        if action == "stop":
            session = self._host.session_by_id(session_id) if session_id else None
            if session is not None:
                await self._cancel_turn(session.owner)
            return await self._host.stop(pid, session_id=session_id, reason="session_control")
        if action == "resume":
            return await self._host.resume(pid, session_id)
        if action == "start":
            procedure_id = command.get("procedure_id")
            step = command.get("step", 0)
            if not isinstance(procedure_id, str) or self._host.procedure(procedure_id) is None:
                return HostReply("error", "Unknown procedure.")
            at_step = step if type(step) is int and step > 0 else 0
            return await self._host.begin(
                pid, procedure_id, at_step=at_step,
                entry_mode="step" if at_step else "start", explicit=True,
            )
        return HostReply("error", "Unknown session action.")


__all__ = [
    "CONTROL_TOPIC",
    "MAIN_INPUT_TOPIC",
    "MODES_REQUEST_TOPIC",
    "OVERLAY_TOPIC",
    "RESULT_TOPIC",
    "STATE_TOPIC",
    "ClientProtocol",
    "ClientRegistry",
    "ClientState",
    "WorkerPorts",
]
