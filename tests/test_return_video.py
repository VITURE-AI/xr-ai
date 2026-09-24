# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped processed-video publishing, over IPC and in the room client."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from device_io_hub.transport.livekit import _room_client as room_client_module
from device_io_hub.transport.livekit.config import LiveKitConnectorConfig
from xr_ai_hub import (
    MsgType,
    PixelFormat,
    ReturnVideoFrame,
    ReturnVideoStop,
    decode,
    encode,
)

pytestmark = pytest.mark.asyncio

_SECRET = "test-secret-at-least-32-bytes-long"


def _frame(pid: str = "alice", **changes) -> ReturnVideoFrame:
    frame = ReturnVideoFrame(
        pts_us=100,
        width=2,
        height=2,
        fmt=PixelFormat.RGB24,
        data=bytes(range(12)),
        participant_id=pid,
    )
    return replace(frame, **changes)


async def test_return_video_codec_round_trips() -> None:
    frame = _frame(track_id="preview")
    assert decode(encode(MsgType.RETURN_VIDEO, frame)) == (MsgType.RETURN_VIDEO, frame)
    stop = ReturnVideoStop("alice", "preview")
    assert decode(encode(MsgType.RETURN_VIDEO_STOP, stop)) == (
        MsgType.RETURN_VIDEO_STOP, stop,
    )


async def test_return_video_reaches_only_the_target_connector(
    hub, make_connector, make_processor, settle,
) -> None:
    alice_conn = make_connector(connector_id="alice_conn")
    bob_conn = make_connector(connector_id="bob_conn")
    await alice_conn.register()
    await bob_conn.register()
    await settle()
    await alice_conn.notify_participant_joined("alice", pts_us=1)
    await bob_conn.notify_participant_joined("bob", pts_us=2)
    await settle()

    alice_frames: list[ReturnVideoFrame] = []
    alice_stops: list[ReturnVideoStop] = []
    bob_frames: list[ReturnVideoFrame] = []

    async def on_alice(frame): alice_frames.append(frame)
    async def on_alice_stop(stop): alice_stops.append(stop)
    async def on_bob(frame): bob_frames.append(frame)

    alice_conn.on_return_video(on_alice)
    alice_conn.on_return_video_stop(on_alice_stop)
    bob_conn.on_return_video(on_bob)
    tasks = [asyncio.create_task(alice_conn.run()), asyncio.create_task(bob_conn.run())]
    try:
        proc = make_processor()
        await settle()
        await proc.send_return_video(_frame())
        await proc.stop_return_video("alice")
        await proc.send_return_video(_frame("ghost"))
        for _ in range(40):
            if alice_frames and alice_stops:
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1)

        assert alice_frames == [_frame()]
        assert alice_stops == [ReturnVideoStop("alice")]
        assert bob_frames == []
    finally:
        alice_conn.stop()
        bob_conn.stop()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class _FakeVideoSource:
    def __init__(self, width: int, height: int, *, is_screencast: bool = False) -> None:
        self.size = (width, height)
        self.is_screencast = is_screencast
        self.timestamps: list[int] = []
        self.closed = False

    def capture_frame(self, _frame: object, *, timestamp_us: int) -> None:
        self.timestamps.append(timestamp_us)

    async def aclose(self) -> None:
        self.closed = True


class _FakeLocalParticipant:
    def __init__(self) -> None:
        self.published: list[tuple[object, object]] = []
        self.unpublished: list[str] = []
        self.permissions: list[dict[str, list[str]]] = []

    async def publish_track(self, track: object, options: object) -> object:
        self.published.append((track, options))
        return SimpleNamespace(sid=f"video-{len(self.published)}")

    async def unpublish_track(self, sid: str) -> None:
        self.unpublished.append(sid)

    def set_track_subscription_permissions(
        self, *, allow_all_participants: bool, participant_permissions: list,
    ) -> None:
        assert allow_all_participants is False
        self.permissions.append({
            p.participant_identity: p.allowed_track_sids for p in participant_permissions
        })


@pytest.fixture
def room_client(monkeypatch):
    sources: list[_FakeVideoSource] = []

    def make_source(width: int, height: int, *, is_screencast: bool = False):
        source = _FakeVideoSource(width, height, is_screencast=is_screencast)
        sources.append(source)
        return source

    monkeypatch.setattr(room_client_module.rtc, "VideoSource", make_source)
    monkeypatch.setattr(
        room_client_module.rtc, "LocalVideoTrack",
        SimpleNamespace(create_video_track=lambda name, source: SimpleNamespace(name=name)),
    )
    for name in ("VideoFrame", "TrackPublishOptions", "VideoEncoding",
                 "ParticipantTrackPermission"):
        monkeypatch.setattr(
            room_client_module.rtc, name, lambda **kwargs: SimpleNamespace(**kwargs),
        )

    def build(audience: str = "participant"):
        client = room_client_module.RoomClient.__new__(room_client_module.RoomClient)
        client._cfg = LiveKitConnectorConfig(
            api_key="key", api_secret=_SECRET, return_video_audience=audience,
        )
        client._room = SimpleNamespace(local_participant=_FakeLocalParticipant())
        client._participant_sessions = {"alice": "s-alice", "bob": "s-bob"}
        client._return_audio = {}
        client._return_video = {}
        client._return_video_lock = asyncio.Lock()
        return client

    build.sources = sources
    return build


async def test_room_client_publishes_updates_and_stops(room_client) -> None:
    client = room_client()
    local = client._room.local_participant

    await client.send_return_video(_frame())
    await client.send_return_video(_frame(pts_us=200))

    assert len(room_client.sources) == 1
    source = room_client.sources[0]
    assert source.is_screencast is True
    assert source.timestamps == [100, 200]
    track, options = local.published[0]
    assert track.name == "xr-hub-overlay-alice"
    assert options.simulcast is False
    assert options.video_encoding.max_bitrate == 6_000_000
    assert local.permissions[-1] == {"alice": ["video-1"]}

    await client.stop_return_video(ReturnVideoStop("alice"))

    assert local.unpublished == ["video-1"]
    assert source.closed is True
    assert client._return_video == {}
    assert local.permissions[-1] == {}


async def test_room_audience_lets_every_participant_subscribe(room_client) -> None:
    client = room_client("room")
    await client.send_return_video(_frame())
    assert client._room.local_participant.permissions[-1] == {
        "alice": ["video-1"], "bob": ["video-1"],
    }


async def test_resize_republishes_the_track(room_client) -> None:
    client = room_client()
    local = client._room.local_participant
    await client.send_return_video(_frame())
    await client.send_return_video(
        _frame(width=4, height=2, data=bytes(24), pts_us=300),
    )
    assert local.unpublished == ["video-1"]
    assert [s.size for s in room_client.sources] == [(2, 2), (4, 2)]
    assert room_client.sources[0].closed is True
    assert local.permissions[-1] == {"alice": ["video-2"]}


async def test_malformed_frame_and_departed_participant_are_dropped(room_client) -> None:
    client = room_client()
    await client.send_return_video(_frame(data=b"too short"))
    await client.send_return_video(_frame("ghost"))
    assert room_client.sources == []
    assert client._return_video == {}
