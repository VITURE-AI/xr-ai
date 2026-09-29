# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared DashScope REST plumbing: auth headers and error extraction."""
from __future__ import annotations

import json
from typing import Any

import httpx


class DashScopeError(RuntimeError):
    """DashScope rejected a request or returned an unusable response.

    ``status_code`` is the HTTP status this shim should answer with: 502 for
    an upstream rejection or malformed reply, 504 for an upstream timeout.
    """

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


def headers(api_key: str, *, sse: bool = False) -> dict[str, str]:
    out = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "text/event-stream" if sse else "application/json; charset=utf-8",
    }
    if sse:
        out["X-DashScope-SSE"] = "enable"
    return out


def describe_error(body: Any, *, status: int | str, model: str, url: str) -> str:
    """Name the API's own code/message plus the model and endpoint.

    A regional key against the wrong host fails as "Invalid API-key", and a
    model a region does not publish fails as "Model not exist." -- neither says
    which endpoint was asked, so both are appended.
    """
    code = message = request_id = ""
    if isinstance(body, dict):
        code = str(body.get("code") or "")
        message = str(body.get("message") or body.get("msg") or "")
        request_id = str(body.get("request_id") or "")
    elif body:
        message = str(body)[:300]
    detail = " ".join(p for p in (code, message) if p)
    text = f"dashscope {status}" + (f": {detail}" if detail else "")
    suffix = f" (model={model!r}, endpoint={url}"
    if request_id:
        suffix += f", request_id={request_id}"
    return text + suffix + ")"


def error_from_response(resp: httpx.Response, *, model: str, body: bytes | None = None) -> DashScopeError:
    raw = resp.content if body is None else body
    try:
        parsed: Any = json.loads(raw) if raw else None
    except (ValueError, UnicodeDecodeError):
        parsed = raw.decode("utf-8", "replace") if raw else None
    return DashScopeError(describe_error(parsed, status=resp.status_code, model=model, url=str(resp.request.url)))


def transport_error(exc: httpx.HTTPError, *, model: str, url: str) -> DashScopeError:
    if isinstance(exc, httpx.TimeoutException):
        return DashScopeError(f"dashscope timeout ({type(exc).__name__}) (model={model!r}, endpoint={url})",
                              status_code=504)
    return DashScopeError(f"dashscope unreachable: {type(exc).__name__}: {exc} (model={model!r}, endpoint={url})")
