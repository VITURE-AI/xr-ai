# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Audio-TTS / CosyVoice over DashScope's HTTP SpeechSynthesizer API.

Non-streaming synthesis returns a short-lived audio URL, which is downloaded
and relayed. Streaming uses the same endpoint with ``X-DashScope-SSE`` and
yields each base64 PCM fragment as soon as its event arrives.
"""
from __future__ import annotations

import base64
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from loguru import logger

from ._dashscope import DashScopeError, describe_error, error_from_response, headers, transport_error
from .config import TTSConfig

_PATH = "/services/audio/tts/SpeechSynthesizer"


class DashScopeTTS:
    """One configured voice behind a shared HTTP client."""

    def __init__(self, cfg: TTSConfig, client: httpx.AsyncClient) -> None:
        self.cfg = cfg
        self._client = client
        self._url = cfg.endpoint.base_url + _PATH

    def build_payload(self, text: str, audio_format: str) -> dict[str, Any]:
        synth_input: dict[str, Any] = {
            "text": text,
            "voice": self.cfg.voice,
            "format": audio_format,
            "sample_rate": self.cfg.sample_rate,
        }
        if self.cfg.instruction and self.cfg.supports_instruction:
            synth_input["instruction"] = self.cfg.instruction
            if self.cfg.optimize_instruction:
                synth_input["optimize_instructions"] = True
        return {"model": self.cfg.model, "input": synth_input}

    def _api_key(self) -> str:
        api_key = self.cfg.endpoint.api_key
        if not api_key:
            raise DashScopeError(self.cfg.endpoint.missing_key_message("tts"), status_code=503)
        return api_key

    async def synthesize(self, text: str, audio_format: str = "wav") -> bytes:
        """Return the whole utterance encoded as *audio_format* (wav, pcm or mp3)."""
        api_key = self._api_key()
        t0 = time.monotonic()
        try:
            resp = await self._client.post(
                self._url,
                json=self.build_payload(text, audio_format),
                headers=headers(api_key),
                timeout=self.cfg.endpoint.timeout_s,
            )
        except httpx.HTTPError as exc:
            raise transport_error(exc, model=self.cfg.model, url=self._url) from exc
        if resp.status_code != 200:
            raise error_from_response(resp, model=self.cfg.model)
        try:
            data = resp.json()
            url = (data.get("output") or {}).get("audio", {}).get("url")
        except (ValueError, AttributeError):
            raise DashScopeError(f"dashscope tts returned an unexpected body: {resp.text[:300]}") from None
        if not url:
            raise DashScopeError(
                describe_error(data, status="200 without audio url", model=self.cfg.model, url=self._url)
            )
        # The URL is a pre-signed object-store link: no DashScope credential.
        try:
            audio = await self._client.get(url, timeout=self.cfg.endpoint.timeout_s)
        except httpx.HTTPError as exc:
            raise transport_error(exc, model=self.cfg.model, url="<audio url>") from exc
        if audio.status_code != 200:
            raise DashScopeError(f"dashscope tts audio download failed: HTTP {audio.status_code}")
        logger.info("synthesized {} chars in {:.2f}s -> {} KiB {}",
                    len(text), time.monotonic() - t0, len(audio.content) // 1024, audio_format)
        return audio.content

    async def stream_pcm(self, text: str) -> AsyncIterator[bytes]:
        """Yield signed 16-bit mono PCM at ``cfg.sample_rate`` as DashScope emits it."""
        api_key = self._api_key()
        t0 = time.monotonic()
        first = True
        total = 0
        try:
            async with self._client.stream(
                "POST",
                self._url,
                json=self.build_payload(text, "pcm"),
                headers=headers(api_key, sse=True),
                timeout=self.cfg.endpoint.timeout_s,
            ) as resp:
                if resp.status_code != 200:
                    raise error_from_response(resp, model=self.cfg.model, body=await resp.aread())
                async for chunk in self._iter_sse_audio(resp):
                    if first:
                        logger.debug("tts first audio after {:.2f}s", time.monotonic() - t0)
                        first = False
                    total += len(chunk)
                    yield chunk
        except httpx.HTTPError as exc:
            raise transport_error(exc, model=self.cfg.model, url=self._url) from exc
        logger.info("streamed {} chars in {:.2f}s -> {} KiB pcm", len(text), time.monotonic() - t0, total // 1024)

    async def _iter_sse_audio(self, resp: httpx.Response) -> AsyncIterator[bytes]:
        event = ""
        async for line in resp.aiter_lines():
            if not line:
                event = ""
                continue
            if line.startswith(":"):
                continue  # comment / heartbeat (e.g. ":HTTP_STATUS/200")
            field, _, value = line.partition(":")
            value = value.strip()
            if field == "event":
                event = value
                continue
            if field != "data":
                continue  # id:, status:, retry:
            try:
                msg = json.loads(value)
            except ValueError:
                raise DashScopeError(f"dashscope tts sent a non-JSON event: {value[:200]}") from None
            if event == "error" or (isinstance(msg, dict) and msg.get("code") and not msg.get("output")):
                raise DashScopeError(describe_error(msg, status="stream error", model=self.cfg.model, url=self._url))
            output = (msg.get("output") or {}) if isinstance(msg, dict) else {}
            # Same routing as the SDK's HttpSpeechSynthesizer: sentence-* events
            # carry fresh audio. The final finish_reason=stop event repeats the
            # concatenated utterance plus its URL; forwarding it would replay
            # the whole utterance.
            if str(output.get("type", "")).startswith("sentence-"):
                data = (output.get("audio") or {}).get("data")
                if data:
                    yield base64.b64decode(data)
            elif output.get("finish_reason") == "stop":
                return
