# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-ASR (``qwen3-asr-flash``) over DashScope's MultiModalConversation API.

The audio goes up inline as a ``data:audio/wav;base64,`` URL in the user turn
and the free-text vocabulary ``context`` rides as the system turn. The Qwen
family is the only hosted DashScope ASR whose vocabulary biasing actually
changes decoding (measured: bare "changed the feature Luma Ultra, no spam" vs.
"change the feature Luma Ultra nose pad" with context).
"""
from __future__ import annotations

import base64
import io
import re
import time
import wave
from typing import Any

import httpx
import numpy as np
from loguru import logger

from ._dashscope import DashScopeError, error_from_response, headers, transport_error
from .config import STTConfig

TARGET_SAMPLE_RATE = 16_000
_PATH = "/services/aigc/multimodal-generation/generation"


class AudioError(ValueError):
    """The uploaded audio is not a 16-bit PCM WAV file."""


def wav_to_16k_mono(wav_bytes: bytes) -> tuple[bytes, float]:
    """Re-frame arbitrary 16-bit PCM WAV as 16 kHz mono WAV.

    Returns ``(wav_bytes, duration_seconds)``. Linear resampling is adequate
    for ASR and keeps the upload small.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError) as exc:
        raise AudioError(f"not a WAV file: {exc}") from exc
    if sw != 2:
        raise AudioError(f"only 16-bit PCM WAV is supported, got sample width {sw} bytes")
    pcm = np.frombuffer(raw[: len(raw) - len(raw) % (2 * ch)], dtype=np.int16)
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
    duration_s = pcm.size / max(sr, 1)
    if sr != TARGET_SAMPLE_RATE and pcm.size:
        f32 = pcm.astype(np.float32)
        n_out = max(1, int(round(f32.size * TARGET_SAMPLE_RATE / sr)))
        f32 = np.interp(
            np.linspace(0.0, f32.size - 1, n_out, dtype=np.float32),
            np.arange(f32.size, dtype=np.float32),
            f32,
        )
        pcm = np.clip(f32, -32768, 32767).astype(np.int16)
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return out.getvalue(), duration_s


def _context_terms(context: str) -> list[str]:
    parts = re.split(r"[,;:.、。：，\n]+", context.lower())
    # Two characters or fewer match far too much ordinary speech.
    return [t for t in (p.strip() for p in parts) if len(t) > 2]


def is_context_echo(text: str, context: str) -> bool:
    """Whether *text* is the model reciting its own context instead of speech.

    qwen3-asr-flash is a chat model and sometimes transcribes the system turn
    rather than the audio; observed live, that echo reached the worker as a
    wearer utterance and restarted guidance from step 1. Counted rather than
    substring-matched because the echo comes back reflowed. A wearer may say
    two listed terms in one breath, not four unrelated ones.
    """
    if not context or not text:
        return False
    lowered = text.lower()
    return sum(1 for term in _context_terms(context) if term in lowered) >= 4


def _extract_text(data: Any) -> str:
    try:
        choices = data["output"]["choices"]
    except (KeyError, TypeError):
        raise DashScopeError(f"dashscope asr response has no output.choices: {str(data)[:300]}") from None
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content")
    # A list of parts on the happy path, a bare string in some API versions.
    # Anything else means the shape changed; silence beats invented words.
    if isinstance(content, list):
        return " ".join(
            str(part["text"]) for part in content if isinstance(part, dict) and part.get("text")
        ).strip()
    if isinstance(content, str):
        return content.strip()
    logger.warning("unexpected qwen-asr content type: {}", type(content).__name__)
    return ""


class DashScopeSTT:
    """One configured Qwen-ASR model behind a shared HTTP client."""

    def __init__(self, cfg: STTConfig, client: httpx.AsyncClient) -> None:
        self.cfg = cfg
        self._client = client
        self._url = cfg.endpoint.base_url + _PATH

    def build_payload(self, wav16k: bytes) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if self.cfg.context:
            messages.append({"role": "system", "content": [{"text": self.cfg.context}]})
        audio = "data:audio/wav;base64," + base64.b64encode(wav16k).decode("ascii")
        messages.append({"role": "user", "content": [{"audio": audio}]})
        parameters: dict[str, Any] = {"result_format": "message"}
        if self.cfg.asr_options:
            parameters["asr_options"] = dict(self.cfg.asr_options)
        return {"model": self.cfg.model, "input": {"messages": messages}, "parameters": parameters}

    async def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe one WAV upload. Raises :class:`AudioError` / :class:`DashScopeError`."""
        api_key = self.cfg.endpoint.api_key
        if not api_key:
            raise DashScopeError(self.cfg.endpoint.missing_key_message("stt"), status_code=503)
        wav16k, audio_s = wav_to_16k_mono(wav_bytes)
        t0 = time.monotonic()
        try:
            resp = await self._client.post(
                self._url,
                json=self.build_payload(wav16k),
                headers=headers(api_key),
                timeout=self.cfg.endpoint.timeout_s,
            )
        except httpx.HTTPError as exc:
            raise transport_error(exc, model=self.cfg.model, url=self._url) from exc
        if resp.status_code != 200:
            raise error_from_response(resp, model=self.cfg.model)
        try:
            data = resp.json()
        except ValueError as exc:
            raise DashScopeError(f"dashscope asr returned non-JSON: {resp.text[:300]}") from exc
        text = _extract_text(data)
        if self.cfg.drop_context_echo and is_context_echo(text, self.cfg.context):
            logger.warning("dropping context echo ({} chars): {}", len(text), text[:80])
            text = ""
        logger.info("transcribed {:.1f}s audio in {:.2f}s -> {} chars", audio_s, time.monotonic() - t0, len(text))
        return text
