# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client roles, participant attributes, and the ``/clients`` roster."""
from __future__ import annotations

import asyncio

import jwt
import pytest
from device_io_hub.transport.livekit import _clients as clients_module
from device_io_hub.transport.livekit._clients import ClientAdmin, validated_role
from device_io_hub.transport.livekit._room_client import _application_attributes
from device_io_hub.transport.livekit._token import (
    CLIENT_ROLES,
    HUB_ROLE,
    INPUT_ATTRIBUTE,
    ROLE_ATTRIBUTE,
    make_client_token,
)
from device_io_hub.transport.livekit.config import LiveKitConnectorConfig
from fastapi import HTTPException
from livekit.protocol import models
from xr_ai_hub import MsgType, ParticipantAttributes, ParticipantEvent, decode, encode

_SECRET = "test-secret-at-least-32-bytes-long"


def _cfg() -> LiveKitConnectorConfig:
    return LiveKitConnectorConfig(
        api_key="key", api_secret=_SECRET,
        room_name="xr-room", identity="xr-hub-connector",
    )


def _participant(identity: str, *, role: str = "", tracks=(),
                 selects: str | None = None) -> models.ParticipantInfo:
    info = models.ParticipantInfo(identity=identity, sid=f"PA_{identity}")
    if role:
        info.attributes[ROLE_ATTRIBUTE] = role
    if selects is not None:
        info.attributes[INPUT_ATTRIBUTE] = selects
    info.tracks.extend(tracks)
    return info


def _track(name: str, source: int, *, muted: bool = False) -> models.TrackInfo:
    return models.TrackInfo(sid=f"TR_{name}", name=name, source=source, muted=muted)


class _FakeRoomService:
    def __init__(self, participants, *, error: Exception | None = None) -> None:
        self._participants = participants
        self._error = error
        self.removed: list[tuple[str, str]] = []

    async def list_participants(self, request):
        if self._error is not None:
            raise self._error
        assert request.room == "xr-room"
        return type("_Response", (), {"participants": self._participants})()

    async def remove_participant(self, request):
        if self._error is not None:
            raise self._error
        self.removed.append((request.room, request.identity))


def _admin(service: _FakeRoomService) -> ClientAdmin:
    admin = ClientAdmin(_cfg(), "http://127.0.0.1:7880")
    admin._lk = lambda: type("_Api", (), {"room": service})()
    return admin


async def _roster(service: _FakeRoomService) -> dict[str, dict]:
    return {entry["identity"]: entry for entry in await _admin(service).list_clients()}


@pytest.mark.asyncio
async def test_roster_reports_role_input_and_livestream() -> None:
    roster = await _roster(_FakeRoomService([
        _participant("xr-hub-connector", role=HUB_ROLE),
        _participant("operator", role="web-client",
                     tracks=[_track("cam", models.TrackSource.CAMERA)]),
        _participant("observer", role="watcher",
                     tracks=[_track("cam", models.TrackSource.CAMERA, muted=True)]),
    ]))

    assert roster["operator"] == dict(
        identity="operator", role="web-client", is_hub=False, input="",
        livestream=True,
    )
    # A muted camera is a published track, not a live stream.
    assert roster["observer"]["livestream"] is False
    assert roster["xr-hub-connector"]["is_hub"] is True
    assert roster["xr-hub-connector"]["role"] == HUB_ROLE


@pytest.mark.asyncio
async def test_livestream_follows_the_selected_input_participant() -> None:
    roster = await _roster(_FakeRoomService([
        _participant("operator", role="web-client", selects="glasses-01"),
        _participant("idle-operator", role="web-client", selects="glasses-off"),
        _participant("stale-operator", role="web-client", selects="departed"),
        _participant("glasses-01", tracks=[_track("cam", models.TrackSource.CAMERA)]),
        _participant("glasses-off",
                     tracks=[_track("cam", models.TrackSource.CAMERA, muted=True)]),
    ]))

    assert roster["operator"]["input"] == "glasses-01"
    assert roster["operator"]["livestream"] is True
    assert roster["idle-operator"]["livestream"] is False
    assert roster["stale-operator"]["input"] == "departed"
    assert roster["stale-operator"]["livestream"] is False


@pytest.mark.asyncio
async def test_roster_is_empty_when_the_room_does_not_exist() -> None:
    error = clients_module.TwirpError("not_found", "room does not exist", status=404)
    assert await _admin(_FakeRoomService([], error=error)).list_clients() == []


@pytest.mark.asyncio
async def test_a_livekit_failure_is_not_reported_as_an_empty_room() -> None:
    error = clients_module.TwirpError("internal", "boom", status=500)
    with pytest.raises(HTTPException) as excinfo:
        await _admin(_FakeRoomService([], error=error)).list_clients()
    assert excinfo.value.status_code == 502


@pytest.mark.asyncio
async def test_remove_client_targets_the_connector_room() -> None:
    service = _FakeRoomService([])
    await _admin(service).remove_client("operator")
    assert service.removed == [("xr-room", "operator")]


@pytest.mark.asyncio
async def test_remove_client_refuses_the_hub_itself() -> None:
    service = _FakeRoomService([])
    with pytest.raises(HTTPException) as excinfo:
        await _admin(service).remove_client("xr-hub-connector")
    assert excinfo.value.status_code == 409
    assert service.removed == []


@pytest.mark.asyncio
async def test_remove_client_is_idempotent() -> None:
    error = clients_module.TwirpError("not_found", "participant does not exist", status=404)
    await _admin(_FakeRoomService([], error=error)).remove_client("gone")


@pytest.mark.asyncio
async def test_token_stamps_a_declared_role_and_allows_attribute_updates() -> None:
    cfg = _cfg()
    claims = jwt.decode(
        make_client_token(cfg, identity="operator", role="web-client"),
        cfg.api_secret, algorithms=["HS256"],
    )
    assert claims["attributes"] == {ROLE_ATTRIBUTE: "web-client"}
    assert claims["video"]["canUpdateOwnMetadata"] is True


@pytest.mark.asyncio
async def test_token_without_a_role_declares_no_attributes() -> None:
    cfg = _cfg()
    claims = jwt.decode(
        make_client_token(cfg, identity="operator"),
        cfg.api_secret, algorithms=["HS256"],
    )
    assert not claims.get("attributes")


@pytest.mark.asyncio
async def test_unknown_roles_and_the_hub_role_are_rejected() -> None:
    assert validated_role("") == ""
    assert validated_role("web-client") == "web-client"
    assert HUB_ROLE not in CLIENT_ROLES
    for role in ("admin", HUB_ROLE):
        with pytest.raises(HTTPException) as excinfo:
            validated_role(role)
        assert excinfo.value.status_code == 400


def test_the_web_server_serves_the_roster_and_validates_roles() -> None:
    from device_io_hub.transport.livekit._web_server import _build_app
    from fastapi.testclient import TestClient

    app = _build_app(_cfg(), None)
    with TestClient(app) as client:
        refused = client.get("/token", params={"identity": "operator", "role": "nope"})
        assert refused.status_code == 400
        granted = client.get("/token", params={"identity": "operator", "role": "web-client"})
        assert granted.status_code == 200
        assert granted.json()["room"] == "xr-room"
    assert {"/clients", "/clients/{identity}"} <= {route.path for route in app.routes}


@pytest.mark.asyncio
async def test_reserved_livekit_attributes_are_not_forwarded() -> None:
    assert _application_attributes(
        {"xr.role": "wearer", "lk.agent.state": "listening"},
    ) == {"xr.role": "wearer"}


@pytest.mark.asyncio
async def test_participant_event_codec_carries_attributes() -> None:
    event = ParticipantEvent("alice", True, 1, "conn", "s-1", {"xr.role": "wearer"})
    assert decode(encode(MsgType.PARTICIPANT_EVENT, event))[1] == event
    change = ParticipantAttributes("alice", {"xr.input": "glasses"}, 2, "s-1")
    assert decode(encode(MsgType.PARTICIPANT_ATTRIBUTES, change))[1] == change


@pytest.mark.asyncio
async def test_processors_see_join_attributes_changes_and_roster_replay(
    hub, make_connector, make_processor, settle,
) -> None:
    proc = make_processor()
    changes: list[ParticipantAttributes] = []

    async def on_change(msg): changes.append(msg)

    proc.on_participant_attributes(on_change)
    await settle()
    conn = make_connector()
    await conn.register()
    await settle()
    await conn.notify_participant_joined(
        "alice", pts_us=1, attributes={ROLE_ATTRIBUTE: "wearer"},
    )
    for _ in range(40):
        if "alice" in proc.connected_participants:
            break
        await asyncio.sleep(0.05)
    assert proc.participant_attributes("alice") == {ROLE_ATTRIBUTE: "wearer"}

    await conn.notify_participant_attributes(
        "alice", {ROLE_ATTRIBUTE: "wearer", INPUT_ATTRIBUTE: "glasses"}, pts_us=2,
    )
    for _ in range(40):
        if changes:
            break
        await asyncio.sleep(0.05)
    assert [c.attributes for c in changes] == [
        {ROLE_ATTRIBUTE: "wearer", INPUT_ATTRIBUTE: "glasses"},
    ]

    late = make_processor()
    await settle()
    for _ in range(40):
        if late.participant_attributes("alice"):
            break
        await asyncio.sleep(0.05)
    assert late.participant_attributes("alice") == {
        ROLE_ATTRIBUTE: "wearer", INPUT_ATTRIBUTE: "glasses",
    }

    await conn.notify_participant_left("alice", pts_us=3)
    for _ in range(40):
        if "alice" not in proc.connected_participants:
            break
        await asyncio.sleep(0.05)
    assert proc.participant_attributes("alice") == {}
