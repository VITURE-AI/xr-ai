# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DashScope speech shim: routes, error mapping, config, and upstream-client wire tests.

No network: DashScope is an ``httpx.MockTransport``. The wire tests run the
app under a real uvicorn on a free loopback port and drive it with the
``xr_ai_models`` clients built from the recommended models.json entries.

Run from ``apps/sop-guidance/services/dashscope-speech``::

    uv --config-file ../../../../uv.toml sync --project .
    .venv/bin/python -m pytest ../../tests/test_dashscope_speech.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import struct
import sys
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

_SERVICE = Path(__file__).resolve().parents[1] / "services" / "dashscope-speech"
if str(_SERVICE) not in sys.path:
    sys.path.insert(0, str(_SERVICE))

from dashscope_speech import ConfigError, create_app, load_settings, settings_from_dict  # noqa: E402
from dashscope_speech.stt import is_context_echo  # noqa: E402

KEY = "sk-test-not-real"
BASE = "https://dashscope.example/api/v1"
ASR_PATH = "/api/v1/services/aigc/multimodal-generation/generation"
TTS_PATH = "/api/v1/services/audio/tts/SpeechSynthesizer"
AUDIO_URL = "https://oss.example/tmp/utterance.wav"

CONTEXT = "Expect this vocabulary: nose pad, solid saddle, wire butterfly, bridge slot, magnetic socket."


# ── fixtures and helpers ─────────────────────────────────────────────────────


def _settings(raw: dict[str, Any] | None = None, env: dict[str, str] | None = None):
    base: dict[str, Any] = {
        "base_url": BASE,
        "stt": {"context": CONTEXT},
        "tts": {"instruction": "Speak slowly."},
    }
    for key, value in (raw or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return settings_from_dict(base, env={"DASHSCOPE_API_KEY": KEY} if env is None else env)


def _wav(sample_rate: int = 48_000, channels: int = 2, seconds: float = 0.25) -> bytes:
    n = int(sample_rate * seconds)
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / sample_rate))) * channels
        for i in range(n)
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)
    return buf.getvalue()


def _asr_ok(text: str) -> httpx.Response:
    return httpx.Response(200, json={
        "output": {"choices": [{"finish_reason": "stop",
                                "message": {"role": "assistant", "content": [{"text": text}]}}]},
        "request_id": "req-asr",
    })


def _sse(events: list[dict[str, Any]], *, event: str | None = None) -> bytes:
    out = []
    for i, msg in enumerate(events):
        out.append(f"id:{i}\n")
        if event:
            out.append(f"event:{event}\n")
        out.append(":HTTP_STATUS/200\n")
        out.append(f"data:{json.dumps(msg)}\n\n")
    return "".join(out).encode()


def _sentence(pcm: bytes) -> dict[str, Any]:
    return {"output": {"type": "sentence-synthesis", "audio": {"data": base64.b64encode(pcm).decode()}}}


def _stop(all_pcm: bytes) -> dict[str, Any]:
    return {"output": {"finish_reason": "stop",
                       "audio": {"data": base64.b64encode(all_pcm).decode(), "url": AUDIO_URL}}}


class FakeDashScope:
    """Records requests; ``asr``/``tts``/``download`` produce the replies."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.asr: Callable[[httpx.Request], httpx.Response] = lambda r: _asr_ok("change the nose pad")
        self.tts: Callable[[httpx.Request], httpx.Response] = self._tts_default
        self.download: Callable[[httpx.Request], httpx.Response] = (
            lambda r: httpx.Response(200, content=_wav(24_000, 1))
        )

    @staticmethod
    def _tts_default(request: httpx.Request) -> httpx.Response:
        if request.headers.get("x-dashscope-sse") == "enable":
            chunks = [b"\x01\x00" * 100, b"\x02\x00" * 100, b"\x03\x00" * 100]
            body = _sse([*(_sentence(c) for c in chunks), _stop(b"".join(chunks))])
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"output": {"audio": {"url": AUDIO_URL, "id": "a1"}}})

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == ASR_PATH:
            return self.asr(request)
        if request.url.path == TTS_PATH:
            return self.tts(request)
        if str(request.url) == AUDIO_URL:
            return self.download(request)
        return httpx.Response(404, json={"code": "NotFound", "message": str(request.url)})

    def body(self, index: int = 0) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def _run(coro_fn: Callable[[httpx.AsyncClient], Any], fake: FakeDashScope, settings=None) -> Any:
    async def main() -> Any:
        upstream = httpx.AsyncClient(transport=httpx.MockTransport(fake))
        app = create_app(settings or _settings(), client=upstream)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://shim") as client:
            try:
                return await coro_fn(client)
            finally:
                await upstream.aclose()
    return asyncio.run(main())


def _post_wav(client: httpx.AsyncClient, wav: bytes, fmt: str = "json"):
    return client.post("/v1/audio/transcriptions",
                       files={"file": ("audio.wav", wav, "audio/wav")}, data={"response_format": fmt})


# ── STT ──────────────────────────────────────────────────────────────────────


def test_transcription_returns_text_and_sends_qwen_asr_request():
    fake = FakeDashScope()
    resp = _run(lambda c: _post_wav(c, _wav(48_000, 2)), fake)
    assert resp.status_code == 200
    assert resp.json() == {"text": "change the nose pad"}

    req = fake.requests[0]
    assert req.url.path == ASR_PATH
    assert req.headers["authorization"] == f"Bearer {KEY}"
    body = fake.body()
    assert body["model"] == "qwen3-asr-flash"
    assert body["parameters"] == {"result_format": "message"}
    system, user = body["input"]["messages"]
    assert system == {"role": "system", "content": [{"text": CONTEXT}]}
    audio_url = user["content"][0]["audio"]
    assert audio_url.startswith("data:audio/wav;base64,")
    with wave.open(io.BytesIO(base64.b64decode(audio_url.split(",", 1)[1])), "rb") as wf:
        assert (wf.getframerate(), wf.getnchannels(), wf.getsampwidth()) == (16_000, 1, 2)
        assert wf.getnframes() == 4000  # 0.25 s at 16 kHz


def test_transcription_text_format_and_asr_options():
    fake = FakeDashScope()
    settings = _settings({"stt": {"asr_options": {"language": "en"}}})
    resp = _run(lambda c: _post_wav(c, _wav(16_000, 1), "text"), fake, settings)
    assert resp.status_code == 200
    assert resp.text == "change the nose pad"
    assert fake.body()["parameters"]["asr_options"] == {"language": "en"}


def test_transcription_drops_context_echo():
    fake = FakeDashScope()
    fake.asr = lambda r: _asr_ok("nose pad, solid saddle, wire butterfly, bridge slot, magnetic socket")
    resp = _run(lambda c: _post_wav(c, _wav()), fake)
    assert resp.json() == {"text": ""}
    assert not is_context_echo("replace the nose pad with the solid saddle", CONTEXT)


def test_transcription_rejects_non_wav():
    fake = FakeDashScope()
    resp = _run(lambda c: _post_wav(c, b"not audio at all"), fake)
    assert resp.status_code == 400
    assert "bad audio payload" in resp.json()["error"]["message"]
    assert fake.requests == []


def test_transcription_upstream_error_maps_to_502_with_detail():
    fake = FakeDashScope()
    fake.asr = lambda r: httpx.Response(
        401, json={"code": "InvalidApiKey", "message": "Invalid API-key provided.", "request_id": "r-9"})
    resp = _run(lambda c: _post_wav(c, _wav()), fake)
    assert resp.status_code == 502
    message = resp.json()["error"]["message"]
    assert "401" in message and "InvalidApiKey" in message and "qwen3-asr-flash" in message
    assert "dashscope.example" in message and "r-9" in message
    assert KEY not in message


def test_transcription_timeout_maps_to_504():
    fake = FakeDashScope()

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    fake.asr = slow
    resp = _run(lambda c: _post_wav(c, _wav()), fake)
    assert resp.status_code == 504


def test_transcription_missing_key_is_clear_503():
    fake = FakeDashScope()
    resp = _run(lambda c: _post_wav(c, _wav()), fake, _settings(env={}))
    assert resp.status_code == 503
    assert "DASHSCOPE_API_KEY" in resp.json()["error"]["message"]
    assert fake.requests == []


# ── TTS ──────────────────────────────────────────────────────────────────────


def test_speech_wav_downloads_audio_url_without_credentials():
    fake = FakeDashScope()
    resp = _run(lambda c: c.post("/v1/audio/speech", json={"input": "Hello.", "response_format": "wav"}), fake)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(resp.content), "rb") as wf:
        assert wf.getframerate() == 24_000

    synth, download = fake.requests
    assert synth.headers["authorization"] == f"Bearer {KEY}"
    assert fake.body() == {
        "model": "qwen-audio-3.0-tts-flash",
        "input": {"text": "Hello.", "voice": "loongjohn", "format": "wav",
                  "sample_rate": 24000, "instruction": "Speak slowly."},
    }
    assert str(download.url) == AUDIO_URL
    assert "authorization" not in download.headers


def test_speech_ignores_request_voice_and_model():
    fake = FakeDashScope()
    _run(lambda c: c.post("/v1/audio/speech", json={
        "input": "Hi", "model": "tts-1", "voice": "alloy", "instructions": "shout"}), fake)
    assert fake.body()["model"] == "qwen-audio-3.0-tts-flash"
    assert fake.body()["input"]["voice"] == "loongjohn"
    assert fake.body()["input"]["instruction"] == "Speak slowly."


def test_speech_pcm_stream_yields_sentence_chunks_without_final_replay():
    fake = FakeDashScope()

    async def go(client: httpx.AsyncClient):
        async with client.stream("POST", "/v1/audio/speech",
                                 json={"input": "Hello.", "response_format": "pcm", "stream": True}) as resp:
            return resp.status_code, dict(resp.headers), b"".join([b async for b in resp.aiter_bytes()])

    status, headers, body = _run(go, fake)
    assert status == 200
    assert headers["x-audio-sample-rate"] == "24000"
    assert headers["x-audio-channels"] == "1"
    assert headers["content-type"] == "audio/pcm"
    assert body == b"\x01\x00" * 100 + b"\x02\x00" * 100 + b"\x03\x00" * 100
    req = fake.requests[0]
    assert req.headers["x-dashscope-sse"] == "enable"
    assert fake.body()["input"]["format"] == "pcm"


def test_speech_stream_rejection_before_audio_is_502():
    fake = FakeDashScope()
    fake.tts = lambda r: httpx.Response(400, json={"code": "InvalidParameter", "message": "voice not found"})
    resp = _run(lambda c: c.post("/v1/audio/speech",
                                 json={"input": "Hi", "response_format": "pcm", "stream": True}), fake)
    assert resp.status_code == 502
    assert "voice not found" in resp.json()["error"]["message"]


def test_speech_stream_sse_error_event_is_502():
    fake = FakeDashScope()
    fake.tts = lambda r: httpx.Response(
        200, content=_sse([{"code": "Throttling", "message": "rate limited"}], event="error"),
        headers={"content-type": "text/event-stream"})
    resp = _run(lambda c: c.post("/v1/audio/speech",
                                 json={"input": "Hi", "response_format": "pcm", "stream": True}), fake)
    assert resp.status_code == 502
    assert "Throttling" in resp.json()["error"]["message"]


def test_speech_upstream_error_and_missing_url_map_to_502():
    fake = FakeDashScope()
    fake.tts = lambda r: httpx.Response(400, json={"code": "InvalidParameter", "message": "Model not exist."})
    resp = _run(lambda c: c.post("/v1/audio/speech", json={"input": "Hi"}), fake)
    assert resp.status_code == 502
    assert "Model not exist." in resp.json()["error"]["message"]

    fake = FakeDashScope()
    fake.tts = lambda r: httpx.Response(200, json={"output": {"audio": {}}})
    resp = _run(lambda c: c.post("/v1/audio/speech", json={"input": "Hi"}), fake)
    assert resp.status_code == 502
    assert "without audio url" in resp.json()["error"]["message"]


def test_speech_request_validation():
    fake = FakeDashScope()

    async def go(client: httpx.AsyncClient):
        return [
            (await client.post("/v1/audio/speech", json={"input": "   "})).status_code,
            (await client.post("/v1/audio/speech", json={"input": "Hi", "response_format": "flac"})).status_code,
            (await client.post("/v1/audio/speech", json={"input": "Hi", "stream": True})).status_code,
        ]

    assert _run(go, fake) == [400, 400, 400]
    assert fake.requests == []


def test_speech_streaming_disabled_is_409():
    fake = FakeDashScope()
    resp = _run(lambda c: c.post("/v1/audio/speech", json={"input": "Hi", "response_format": "pcm", "stream": True}),
                fake, _settings({"tts": {"streaming_enabled": False}}))
    assert resp.status_code == 409


def test_speech_missing_key_is_clear_503():
    fake = FakeDashScope()
    resp = _run(lambda c: c.post("/v1/audio/speech", json={"input": "Hi"}), fake, _settings(env={}))
    assert resp.status_code == 503
    assert "DASHSCOPE_API_KEY" in resp.json()["error"]["message"]


def test_health_and_models():
    fake = FakeDashScope()

    async def go(client: httpx.AsyncClient):
        return (await client.get("/health")).json(), (await client.get("/v1/models")).json()

    health, models = _run(go, fake)
    assert health == {"status": "ok", "stt": True, "tts": True}
    assert [m["id"] for m in models["data"]] == ["qwen3-asr-flash", "qwen-audio-3.0-tts-flash"]


# ── config ───────────────────────────────────────────────────────────────────


def test_reference_yaml_loads_with_documented_defaults():
    settings = load_settings(_SERVICE / "dashscope_speech.yaml", env={"DASHSCOPE_API_KEY": KEY})
    assert settings.port == 8106
    assert settings.stt.model == "qwen3-asr-flash"
    assert "nose pad" in settings.stt.context
    assert settings.tts.model == "qwen-audio-3.0-tts-flash"
    assert settings.tts.voice == "loongjohn"
    assert settings.tts.instruction.startswith("Speak slowly")
    assert settings.stt.endpoint.base_url == "https://dashscope.aliyuncs.com/api/v1"
    assert settings.tts.endpoint.api_key == KEY  # falls back to the shared key
    assert settings.missing_keys() == []


def test_missing_keys_names_every_candidate_variable():
    settings = load_settings(_SERVICE / "dashscope_speech.yaml", env={})
    assert settings.missing_keys() == [
        "DashScope stt API key is not set: export DASHSCOPE_API_KEY",
        "DashScope tts API key is not set: export DASHSCOPE_TTS_API_KEY or DASHSCOPE_API_KEY",
    ]
    disabled = settings_from_dict({"stt": {"enabled": False}, "tts": {"enabled": False}}, env={})
    assert disabled.missing_keys() == []


def test_tts_borrows_key_and_region_without_moving_stt():
    env = {
        "DASHSCOPE_API_KEY": "us-key",
        "DASHSCOPE_API_URL": "https://ws-1.us-east-1.maas.aliyuncs.com/api/v1",
        "DASHSCOPE_TTS_API_KEY": "cn-key",
        "DASHSCOPE_TTS_API_URL": "https://dashscope.aliyuncs.com/api/v1/",
    }
    settings = load_settings(_SERVICE / "dashscope_speech.yaml", env=env)
    assert settings.stt.endpoint.api_key == "us-key"
    assert settings.stt.endpoint.base_url == "https://ws-1.us-east-1.maas.aliyuncs.com/api/v1"
    assert settings.tts.endpoint.api_key == "cn-key"
    assert settings.tts.endpoint.base_url == "https://dashscope.aliyuncs.com/api/v1"


def test_region_selection():
    s = settings_from_dict({"region": "intl", "tts": {"region": "cn"}}, env={})
    assert s.stt.endpoint.base_url == "https://dashscope-intl.aliyuncs.com/api/v1"
    assert s.tts.endpoint.base_url == "https://dashscope.aliyuncs.com/api/v1"
    with pytest.raises(ConfigError, match="unknown region"):
        settings_from_dict({"region": "mars"}, env={})


@pytest.mark.parametrize("raw, match", [
    ({"api_key": "sk-x"}, "not allowed"),
    ({"tts": {"api_key": "sk-x"}}, "not allowed"),
    ({"prot": 8106}, "unknown key"),
    ({"stt": {"model": "fun-asr-realtime"}}, "not supported"),
    ({"tts": {"model": "qwen3-tts-flash"}}, "not supported"),
    ({"timeout_s": 0}, "timeout_s"),
    ({"port": "8106"}, "port"),
])
def test_config_rejects(raw, match):
    with pytest.raises(ConfigError, match=match):
        settings_from_dict(raw, env={})


def test_instruction_dropped_for_models_that_reject_it():
    s = settings_from_dict({"tts": {"model": "cosyvoice-v3-flash", "instruction": "slow"}}, env={})
    assert s.tts.instruction == ""


# ── wire: upstream xr_ai_models clients against a real uvicorn ──────────────


def _models_json(port: int) -> dict[str, Any]:
    """The models.json entries the README recommends, pointed at *port*."""
    base_url = f"http://127.0.0.1:{port}"
    return {"models": {
        "stt": {"adapter": {"preset": "parakeet_stt"},
                "endpoint": {"base_url": base_url, "timeout": 30}},
        "tts": {"adapter": {"preset": "pocket_tts"},
                "endpoint": {"base_url": base_url, "timeout": 60}},
    }}


async def _serve(app) -> tuple[Any, asyncio.Task, int]:
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        assert not task.done(), "uvicorn exited before binding"
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, task, port


def test_upstream_clients_interoperate_and_stream_incrementally():
    xr_ai_models = pytest.importorskip("xr_ai_models")
    from xr_ai_models._openai_compat import _PocketTTS

    async def main() -> None:
        release = asyncio.Event()
        first_pcm, rest_pcm = b"\x10\x00" * 240, b"\x20\x00" * 480

        async def gated_sse():
            yield _sse([_sentence(first_pcm)])
            await asyncio.wait_for(release.wait(), timeout=5)
            yield _sse([_sentence(rest_pcm), _stop(first_pcm + rest_pcm)])

        fake = FakeDashScope()

        def tts(request: httpx.Request) -> httpx.Response:
            if request.headers.get("x-dashscope-sse") == "enable":
                return httpx.Response(200, content=gated_sse(), headers={"content-type": "text/event-stream"})
            return FakeDashScope._tts_default(request)

        fake.tts = tts
        upstream = httpx.AsyncClient(transport=httpx.MockTransport(fake))
        server, task, port = await _serve(create_app(_settings(), client=upstream))
        try:
            config = xr_ai_models.load_models_config_from_dict(_models_json(port))
            stt = xr_ai_models.make_stt(config, "stt")
            tts_client = xr_ai_models.make_tts(config, "tts")
            assert isinstance(tts_client, _PocketTTS)
            try:
                assert await stt.health() and await tts_client.health()

                # PCM path: the client wraps raw PCM into WAV itself.
                pcm = b"".join(struct.pack("<h", (i * 97) % 20000 - 10000) for i in range(8000))
                assert await stt.transcribe(pcm, sample_rate=16_000) == "change the nose pad"
                assert await stt.transcribe(_wav(44_100, 1)) == "change the nose pad"

                wav = await tts_client.synthesize("Hello.")
                with wave.open(io.BytesIO(wav), "rb") as wf:
                    assert wf.getframerate() == 24_000

                chunks = []
                async for chunk in tts_client.stream("Hello."):
                    chunks.append(chunk)
                    if len(chunks) == 1:
                        # Upstream is still holding the rest: the first
                        # fragment reached the client before DashScope finished.
                        assert chunk.data == first_pcm
                        release.set()
                assert {(c.sample_rate, c.channels) for c in chunks} == {(24_000, 1)}
                assert b"".join(c.data for c in chunks) == first_pcm + rest_pcm
            finally:
                await stt.close()
                await tts_client.close()
        finally:
            server.should_exit = True
            await task
            await upstream.aclose()

    asyncio.run(main())


def test_mid_stream_failure_breaks_the_client_stream():
    xr_ai_models = pytest.importorskip("xr_ai_models")

    async def main() -> None:
        fake = FakeDashScope()
        body = _sse([_sentence(b"\x05\x00" * 50)]) + _sse([{"code": "InternalError", "message": "x"}], event="error")
        fake.tts = lambda r: httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
        upstream = httpx.AsyncClient(transport=httpx.MockTransport(fake))
        server, task, port = await _serve(create_app(_settings(), client=upstream))
        try:
            tts_client = xr_ai_models.make_tts(xr_ai_models.load_models_config_from_dict(_models_json(port)), "tts")
            received = []
            try:
                with pytest.raises(httpx.HTTPError):
                    async for chunk in tts_client.stream("Hi"):
                        received.append(chunk.data)
            finally:
                await tts_client.close()
            # The fragment before the failure was delivered; the stream then
            # broke instead of ending as if the utterance were complete.
            assert received == [b"\x05\x00" * 50]
        finally:
            server.should_exit = True
            await task
            await upstream.aclose()

    asyncio.run(main())


def test_upstream_clients_raise_on_mapped_errors():
    xr_ai_models = pytest.importorskip("xr_ai_models")

    async def main() -> None:
        fake = FakeDashScope()
        fake.asr = lambda r: httpx.Response(500, json={"code": "InternalError", "message": "boom"})
        fake.tts = lambda r: httpx.Response(401, json={"code": "InvalidApiKey", "message": "bad key"})
        upstream = httpx.AsyncClient(transport=httpx.MockTransport(fake))
        server, task, port = await _serve(create_app(_settings(), client=upstream))
        try:
            config = xr_ai_models.load_models_config_from_dict(_models_json(port))
            stt = xr_ai_models.make_stt(config, "stt")
            tts_client = xr_ai_models.make_tts(config, "tts")
            try:
                with pytest.raises(httpx.HTTPStatusError) as stt_err:
                    await stt.transcribe(_wav())
                assert stt_err.value.response.status_code == 502
                with pytest.raises(httpx.HTTPStatusError) as tts_err:
                    await tts_client.synthesize("Hi")
                assert tts_err.value.response.status_code == 502
                with pytest.raises(httpx.HTTPStatusError) as stream_err:
                    async for _ in tts_client.stream("Hi"):
                        pass
                assert stream_err.value.response.status_code == 502
            finally:
                await stt.close()
                await tts_client.close()
        finally:
            server.should_exit = True
            await task
            await upstream.aclose()

    asyncio.run(main())
