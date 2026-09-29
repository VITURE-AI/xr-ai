# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token generation helpers — used by both the room client and external callers."""
from __future__ import annotations

from datetime import timedelta

from livekit.api import AccessToken, VideoGrants

from .config import LiveKitConnectorConfig

#: Participant attribute carrying the role a client declared at ``/token``.
#: LiveKit replicates attributes to every participant, so peers read a role
#: instead of inferring one from an identity string.
ROLE_ATTRIBUTE = "xr.role"

#: Roles a client may claim at ``/token``.
CLIENT_ROLES = frozenset({"web-client", "watcher", "wearer"})

#: The connector's own role. Not in :data:`CLIENT_ROLES`, so no client can
#: claim it and appear in the roster as the hub.
HUB_ROLE = "hub"

#: Attribute a client sets itself to name the peer whose media it reads.
#: Empty or absent means the client reads its own media.
INPUT_ATTRIBUTE = "xr.input"


def make_client_token(
    cfg: LiveKitConnectorConfig,
    identity: str = "client",
    ttl: int | None = 3600 * 24,   # 24 h — long enough for dev; None → SDK default
    role: str = "",
) -> str:
    """
    Generate a signed LiveKit JWT for a browser or mobile client.

    On a local/HTTP network pass this token directly to the livekit-client SDK
    along with ws://<host>:<lk_port_ws> — no token server needed.

        token = make_client_token(cfg, identity="alice")
        # hand token + ws://10.x.x.x:7880 to the browser client

    Pass ``ttl=None`` to skip the explicit lifetime and use the LiveKit SDK's
    default token TTL — used by short-lived per-session web tokens.

    A non-empty ``role`` is stamped as the ``xr.role`` participant attribute.
    The grant lets the client update its own attributes, such as ``xr.input``;
    LiveKit merges attribute updates, so the role survives them.
    """
    builder = (
        AccessToken(cfg.api_key, cfg.api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(VideoGrants(room_join=True, room=cfg.room_name,
                                 can_update_own_metadata=True))
    )
    if role:
        builder = builder.with_attributes({ROLE_ATTRIBUTE: role})
    if ttl is not None:
        builder = builder.with_ttl(timedelta(seconds=ttl))
    return builder.to_jwt()
