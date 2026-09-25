<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# dashscope-speech

This is an app-owned service that puts hosted Alibaba DashScope speech models
behind the OpenAI-compatible routes the `xr_ai_models` STT and TTS clients
already use. STT and TTS share one port, 8106 by default. The service runs no
local model: every request goes over the network to DashScope. It is a
server-side shim, so it calls DashScope's REST/SSE API directly with `httpx`
and does not use the vendor SDK.

| Route | Behavior |
| --- | --- |
| `GET /health` | `{"status": "ok", "stt": bool, "tts": bool}` |
| `GET /v1/models` | Lists the configured DashScope models. |
| `POST /v1/audio/transcriptions` | Multipart `file` (16-bit WAV) plus `response_format` `json` or `text`. Returns `{"text": ...}`. Audio is resampled to 16 kHz mono and sent to `qwen3-asr-flash`, with the `context` vocabulary as the system turn. |
| `POST /v1/audio/speech` | `{"input", "response_format": wav\|pcm\|mp3, "stream"}`. With `stream: true` and `pcm`, it returns chunked 16-bit mono PCM fragments as DashScope emits them. The `x-audio-sample-rate` and `x-audio-channels` headers carry the format. |

The model, voice and delivery instruction come from the config file. Request
`model`, `voice` and `instructions` fields are ignored.

Error codes:

- **400**: bad input.
- **409**: streaming is disabled.
- **502**: DashScope rejected the request or sent a malformed reply. The
  message includes DashScope's code and message, the model, the endpoint and
  the request ID.
- **503**: no API key is set, or the role is disabled.
- **504**: the DashScope request timed out.

## Configuration

See the comments in [`dashscope_speech.yaml`](dashscope_speech.yaml). API keys
are read only from environment variables. The file names those variables with
`api_key_env` and may not contain an `api_key` field.

| Variable | Purpose |
| --- | --- |
| `DASHSCOPE_API_KEY` | Shared key, required |
| `DASHSCOPE_API_URL` | Optional. Overrides the region or workspace endpoint for both roles. |
| `DASHSCOPE_TTS_API_KEY`, `DASHSCOPE_TTS_API_URL` | Optional. Lets TTS use a key and endpoint from another region while ASR stays local. |

At startup the service exits with a clear message if an enabled role has no
key.

## Running

The launcher starts the service with `--config` and `--ready-file`. The
service creates the ready file only after uvicorn is listening:

```python
Process("speech", "services/dashscope-speech", "dashscope_speech",
        config="services/dashscope-speech/dashscope_speech.yaml", port=8106)
```

To point `yaml/models.json` at this service, set both roles to the same port:

```json
"stt": {"adapter": {"preset": "parakeet_stt"},
        "endpoint": {"base_url": "http://localhost:8106", "timeout": 30}},
"tts": {"adapter": {"preset": "pocket_tts"},
        "endpoint": {"base_url": "http://localhost:8106", "timeout": 60}}
```

These presets are generic `openai_compat` presets. `pocket_tts` sets
`capabilities.streaming: true`, which selects the streaming TTS client.

## Tests

```bash
uv --config-file ../../../../uv.toml sync --project .
.venv/bin/python -m pytest ../../tests/test_dashscope_speech.py
```
