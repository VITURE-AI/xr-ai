# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Room roster and participant removal for browser clients.

Serves:
  GET     /clients            — participants with their role, input and media state
  DELETE  /clients/{identity} — disconnect that participant

Both go through the LiveKit server API, for which only the hub holds
credentials. Any client can already displace a participant by requesting a
token for its identity, so removal adds no authority the token endpoint does
not already grant; it makes that action explicit and immediate.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from livekit.api import (
    ListParticipantsRequest,
    LiveKitAPI,
    RoomParticipantIdentity,
    TwirpError,
)
from livekit.protocol import models

from ._token import CLIENT_ROLES, INPUT_ATTRIBUTE, ROLE_ATTRIBUTE
from .config import LiveKitConnectorConfig

_LIVESTREAM_SOURCES = (models.TrackSource.CAMERA, models.TrackSource.MICROPHONE)

# Twirp code LiveKit answers with when the room or participant is absent.
_NOT_FOUND = "not_found"


def _unreachable(error: TwirpError) -> HTTPException:
    return HTTPException(status_code=502, detail=f"LiveKit refused the call: {error.message}")


class ClientAdmin:
    """LiveKit server-API access scoped to the connector's own room."""

    def __init__(self, cfg: LiveKitConnectorConfig, lk_internal_http: str) -> None:
        self._cfg = cfg
        self._url = lk_internal_http
        self._api: LiveKitAPI | None = None

    def _lk(self) -> LiveKitAPI:
        # Created on first use and then reused, so a hub that never serves
        # /clients opens no HTTP session and requests share one.
        if self._api is None:
            self._api = LiveKitAPI(
                url=self._url,
                api_key=self._cfg.api_key,
                api_secret=self._cfg.api_secret,
            )
        return self._api

    async def aclose(self) -> None:
        if self._api is not None:
            await self._api.aclose()
            self._api = None

    async def list_clients(self) -> list[dict]:
        try:
            response = await self._lk().room.list_participants(
                ListParticipantsRequest(room=self._cfg.room_name),
            )
        except TwirpError as error:
            if error.code != _NOT_FOUND:
                raise _unreachable(error) from error
            # No room yet: nobody has joined since the server started.
            return []

        by_identity = {p.identity: p for p in response.participants}
        clients = []
        for participant in response.participants:
            selected = participant.attributes.get(INPUT_ATTRIBUTE, "")
            # A client reading a peer's media is live when that peer streams.
            # A selection naming someone who left reports as not streaming.
            source = by_identity.get(selected) if selected else participant
            clients.append(dict(
                identity=participant.identity,
                role=participant.attributes.get(ROLE_ATTRIBUTE, ""),
                is_hub=participant.identity == self._cfg.identity,
                input=selected,
                livestream=source is not None and any(
                    track.source in _LIVESTREAM_SOURCES and not track.muted
                    for track in source.tracks
                ),
            ))
        return clients

    async def remove_client(self, identity: str) -> None:
        if identity == self._cfg.identity:
            raise HTTPException(
                status_code=409,
                detail="The hub's own participant cannot be removed.",
            )
        try:
            await self._lk().room.remove_participant(
                RoomParticipantIdentity(room=self._cfg.room_name, identity=identity),
            )
        except TwirpError as error:
            # Already gone, or the room does not exist: the outcome the caller
            # asked for.
            if error.code != _NOT_FOUND:
                raise _unreachable(error) from error


def mount_client_admin(app: FastAPI, admin: ClientAdmin) -> None:
    """Add the ``/clients`` roster and removal routes to *app*."""

    @app.get("/clients")
    async def list_clients() -> dict:
        return {"clients": await admin.list_clients()}

    @app.delete("/clients/{identity}", status_code=204)
    async def remove_client(identity: str) -> None:
        await admin.remove_client(identity)


def validated_role(role: str) -> str:
    """Return *role* if a client may claim it, else raise HTTP 400."""
    if not role:
        return ""
    if role not in CLIENT_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown role {role!r}; expected one of {sorted(CLIENT_ROLES)}.",
        )
    return role
