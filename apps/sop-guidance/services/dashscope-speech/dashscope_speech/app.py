# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI surface: the OpenAI-compatible routes the xr_ai_models clients call.

    GET  /health                   {"status": "ok", "stt": bool, "tts": bool}
    GET  /v1/models                configured DashScope models
    POST /v1/audio/transcriptions  multipart ``file`` (WAV) -> {"text": ...}
    POST /v1/audio/speech          {"input", "response_format", "stream"} -> audio

Voice, model and delivery instruction are server configuration; request
``model``/``voice``/``instructions`` fields are accepted for OpenAI surface
parity and ignored.
"""
# No `from __future__ import annotations`: FastAPI resolves the endpoint
# annotations at route registration.

from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from ._dashscope import DashScopeError
from .config import Settings
from .stt import AudioError, DashScopeSTT
from .tts import DashScopeTTS

_MEDIA_TYPES = {"wav": "audio/wav", "pcm": "audio/pcm", "mp3": "audio/mpeg"}


class SpeechRequest(BaseModel):
    input: str
    response_format: str = "wav"
    stream: bool = False
    # Accepted for OpenAI parity; configuration wins.
    model: str | None = None
    voice: str | None = None
    instructions: str | None = None
    speed: float | None = None


def _error(status: int, message: str, kind: str = "dashscope_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind, "code": status}}, status_code=status)


def _upstream_error(role: str, exc: DashScopeError) -> JSONResponse:
    logger.error("{} failed: {}", role, exc)
    return _error(exc.status_code, str(exc))


def _pcm_headers(sample_rate: int) -> dict[str, str]:
    # _PocketTTS.stream reads these; without them it falls back to WAV.
    return {"x-audio-sample-rate": str(sample_rate), "x-audio-channels": "1"}


def create_app(settings: Settings, *, client: httpx.AsyncClient | None = None) -> FastAPI:
    """Build the app. An injected *client* stays owned by the caller."""
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(settings.tts.endpoint.timeout_s))
    stt = DashScopeSTT(settings.stt, http)
    tts = DashScopeTTS(settings.tts, http)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            if owns_client:
                await http.aclose()

    app = FastAPI(title="DashScope speech shim", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "stt": settings.stt.enabled, "tts": settings.tts.enabled}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        data = []
        if settings.stt.enabled:
            data.append({"id": settings.stt.model, "object": "model", "owned_by": "dashscope", "role": "stt"})
        if settings.tts.enabled:
            data.append({"id": settings.tts.model, "object": "model", "owned_by": "dashscope",
                         "role": "tts", "voice": settings.tts.voice})
        return {"object": "list", "data": data}

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        response_format: str = Form("json"),
        model: str | None = Form(None),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
    ) -> Response:
        if not settings.stt.enabled:
            return _error(503, "speech-to-text is disabled in this service's config", "disabled")
        if response_format not in ("json", "text"):
            return _error(400, f"unsupported response_format {response_format!r}; use json or text",
                          "invalid_request_error")
        raw = await file.read()
        try:
            text = await stt.transcribe(raw)
        except AudioError as exc:
            return _error(400, f"bad audio payload: {exc}", "invalid_request_error")
        except DashScopeError as exc:
            return _upstream_error("stt", exc)
        if response_format == "text":
            return PlainTextResponse(text)
        return JSONResponse({"text": text})

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest) -> Response:
        if not settings.tts.enabled:
            return _error(503, "text-to-speech is disabled in this service's config", "disabled")
        text = req.input.strip()
        if not text:
            return _error(400, "empty input", "invalid_request_error")
        if req.response_format not in _MEDIA_TYPES:
            return _error(400, f"unsupported response_format {req.response_format!r}; "
                               f"use one of {', '.join(_MEDIA_TYPES)}", "invalid_request_error")
        if req.stream:
            if req.response_format != "pcm":
                return _error(400, "streaming requires response_format=pcm", "invalid_request_error")
            if not settings.tts.streaming_enabled:
                return _error(409, "streaming TTS is disabled in this service's config", "disabled")
            return await _stream(text)
        try:
            audio = await tts.synthesize(text, req.response_format)
        except DashScopeError as exc:
            return _upstream_error("tts", exc)
        extra = _pcm_headers(settings.tts.sample_rate) if req.response_format == "pcm" else None
        return Response(content=audio, media_type=_MEDIA_TYPES[req.response_format], headers=extra)

    async def _stream(text: str) -> Response:
        chunks = tts.stream_pcm(text)
        # Pull the first fragment before committing to a 200, so a rejected
        # key, model or voice surfaces as a 5xx the client raises on rather
        # than as an empty audio stream.
        try:
            first: bytes | None = await anext(chunks)
        except StopAsyncIteration:
            first = None
        except DashScopeError as exc:
            await chunks.aclose()
            return _upstream_error("tts", exc)

        async def body():
            try:
                if first is None:
                    return
                yield first
                async for chunk in chunks:
                    yield chunk
            except DashScopeError as exc:
                # Headers are already sent: abort the connection so the client
                # sees a broken stream instead of a silently short utterance.
                logger.error("tts stream failed mid-utterance: {}", exc)
                raise
            finally:
                await chunks.aclose()

        return StreamingResponse(body(), media_type="audio/pcm", headers=_pcm_headers(settings.tts.sample_rate))

    return app
