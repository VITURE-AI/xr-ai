# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YAML configuration for the DashScope speech shim.

The file holds everything except credentials. API keys are read from the
environment variables the file *names* (``api_key_env``); a literal
``api_key`` in the file is rejected.

Endpoint, key and timeout settings may appear at the top level and again
inside the ``stt:`` / ``tts:`` sections. A section value wins over the top
level, which is how TTS borrows a key and endpoint from another region while
ASR stays local (a region can serve ASR and publish no TTS model at all).
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

DEFAULT_PORT = 8106

REGION_URLS: dict[str, str] = {
    # Beijing; the DashScope SDK default.
    "cn":   "https://dashscope.aliyuncs.com/api/v1",
    # Singapore (international).
    "intl": "https://dashscope-intl.aliyuncs.com/api/v1",
    # US (Virginia).
    "us":   "https://dashscope-us.aliyuncs.com/api/v1",
}

# Keys that may appear at the top level and be overridden per section.
_ENDPOINT_KEYS = ("region", "base_url", "base_url_env", "api_key_env", "timeout_s")
_TOP_KEYS = {"host", "port", "stt", "tts", *_ENDPOINT_KEYS}
_STT_KEYS = {"enabled", "model", "context", "asr_options", "drop_context_echo", *_ENDPOINT_KEYS}
_TTS_KEYS = {
    "enabled", "model", "voice", "instruction", "optimize_instruction",
    "sample_rate", "streaming_enabled", *_ENDPOINT_KEYS,
}

_STT_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "model": "qwen3-asr-flash",
    "context": "",
    "asr_options": {},
    "drop_context_echo": True,
}
_TTS_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "model": "qwen-audio-3.0-tts-flash",
    "voice": "loongjohn",
    "instruction": "",
    "optimize_instruction": False,
    "sample_rate": 24000,
    "streaming_enabled": True,
}
_TOP_DEFAULTS: dict[str, Any] = {
    "host": "0.0.0.0",
    "port": DEFAULT_PORT,
    "region": "cn",
    "base_url": "",
    "base_url_env": "",
    "api_key_env": "DASHSCOPE_API_KEY",
    "timeout_s": 30.0,
}


class ConfigError(ValueError):
    """The YAML file is malformed or names an unsupported model."""


@dataclass(frozen=True)
class Endpoint:
    """Resolved DashScope endpoint and credential for one role."""

    base_url: str
    """Native DashScope API root, e.g. ``https://dashscope.aliyuncs.com/api/v1``."""

    api_key: str | None = field(repr=False)
    """Bearer token, or ``None`` when no named variable is set."""

    api_key_envs: tuple[str, ...]
    """Environment variables consulted for the key, in precedence order."""

    timeout_s: float
    """Per-request DashScope timeout (idle timeout between stream chunks)."""

    def missing_key_message(self, role: str) -> str:
        names = " or ".join(self.api_key_envs) or "<no api_key_env configured>"
        return f"DashScope {role} API key is not set: export {names}"


@dataclass(frozen=True)
class STTConfig:
    enabled: bool
    model: str
    context: str
    asr_options: dict[str, Any]
    drop_context_echo: bool
    endpoint: Endpoint


@dataclass(frozen=True)
class TTSConfig:
    enabled: bool
    model: str
    voice: str
    instruction: str
    optimize_instruction: bool
    sample_rate: int
    streaming_enabled: bool
    endpoint: Endpoint

    @property
    def supports_instruction(self) -> bool:
        """Instruction control is a Qwen-Audio-3.0-TTS feature; CosyVoice rejects it."""
        return self.model.startswith("qwen-audio-3.0-tts")


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    stt: STTConfig
    tts: TTSConfig

    def missing_keys(self) -> list[str]:
        """Messages for every enabled role that has no API key."""
        out = []
        for role, section in (("stt", self.stt), ("tts", self.tts)):
            if section.enabled and not section.endpoint.api_key:
                out.append(section.endpoint.missing_key_message(role))
        return out


def load_settings(path: Path | str | None, env: Mapping[str, str] | None = None) -> Settings:
    """Load *path* (or built-in defaults when ``None``) and resolve credentials."""
    raw: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{p}: top level must be a mapping")
        raw = loaded
    return settings_from_dict(raw, env=os.environ if env is None else env)


def settings_from_dict(raw: Mapping[str, Any], env: Mapping[str, str] | None = None) -> Settings:
    """Build :class:`Settings` from a parsed YAML mapping."""
    env = os.environ if env is None else env
    _reject_secrets(raw, "top level")
    _check_keys(raw, _TOP_KEYS, "top level")
    stt_raw = _section(raw, "stt", _STT_KEYS)
    tts_raw = _section(raw, "tts", _TTS_KEYS)
    top = {**_TOP_DEFAULTS, **{k: v for k, v in raw.items() if k not in ("stt", "tts")}}

    stt_vals = {**_STT_DEFAULTS, **stt_raw}
    tts_vals = {**_TTS_DEFAULTS, **tts_raw}

    stt = STTConfig(
        enabled=_bool(stt_vals, "enabled", "stt"),
        model=_str(stt_vals, "model", "stt", allow_empty=False),
        context=_str(stt_vals, "context", "stt").strip(),
        asr_options=_mapping(stt_vals, "asr_options", "stt"),
        drop_context_echo=_bool(stt_vals, "drop_context_echo", "stt"),
        endpoint=_endpoint(top, stt_raw, env, "stt"),
    )
    tts = TTSConfig(
        enabled=_bool(tts_vals, "enabled", "tts"),
        model=_str(tts_vals, "model", "tts", allow_empty=False),
        voice=_str(tts_vals, "voice", "tts", allow_empty=False),
        instruction=_str(tts_vals, "instruction", "tts").strip(),
        optimize_instruction=_bool(tts_vals, "optimize_instruction", "tts"),
        sample_rate=_positive_int(tts_vals, "sample_rate", "tts"),
        streaming_enabled=_bool(tts_vals, "streaming_enabled", "tts"),
        endpoint=_endpoint(top, tts_raw, env, "tts"),
    )
    _check_models(stt, tts)
    if tts.enabled and tts.instruction and not tts.supports_instruction:
        logger.warning("tts.instruction is set but model {!r} does not accept it; dropping it", tts.model)
        tts = replace(tts, instruction="")
    return Settings(
        host=_str(top, "host", "top level", allow_empty=False),
        port=_positive_int(top, "port", "top level"),
        stt=stt,
        tts=tts,
    )


# ── helpers ──────────────────────────────────────────────────────────────────


def _reject_secrets(raw: Mapping[str, Any], where: str) -> None:
    if "api_key" in raw:
        raise ConfigError(
            f"{where}: 'api_key' is not allowed in the config file; "
            "name an environment variable with 'api_key_env' instead"
        )


def _check_keys(raw: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}")


def _section(raw: Mapping[str, Any], name: str, allowed: set[str]) -> dict[str, Any]:
    value = raw.get(name) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"{name}: must be a mapping")
    _reject_secrets(value, name)
    _check_keys(value, allowed, name)
    return dict(value)


def _endpoint(
    top: Mapping[str, Any], section: Mapping[str, Any], env: Mapping[str, str], where: str,
) -> Endpoint:
    # Section settings win over top-level ones; within one level an env
    # override (base_url_env) wins over base_url, which wins over region.
    base_url = _level_url(section, env, where) or _level_url(top, env, "top level")
    if not base_url:
        base_url = REGION_URLS["cn"]
    if not base_url.startswith(("https://", "http://")):
        raise ConfigError(f"{where}: base_url must be an http(s) URL, got {base_url!r}")

    envs: list[str] = []
    for level, label in ((section, where), (top, "top level")):
        names = level.get("api_key_env")
        if names is None or names == "":
            continue
        if isinstance(names, str):
            names = [names]
        if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
            raise ConfigError(f"{label}: api_key_env must be a variable name or a list of them")
        envs.extend(n for n in names if n not in envs)
    api_key = next((env[n].strip() for n in envs if env.get(n, "").strip()), None)

    timeout = section.get("timeout_s", top.get("timeout_s"))
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ConfigError(f"{where}: timeout_s must be a positive number")
    return Endpoint(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        api_key_envs=tuple(envs),
        timeout_s=float(timeout),
    )


def _level_url(level: Mapping[str, Any], env: Mapping[str, str], where: str) -> str:
    env_name = level.get("base_url_env") or ""
    if not isinstance(env_name, str):
        raise ConfigError(f"{where}: base_url_env must be a string")
    if env_name and env.get(env_name, "").strip():
        return env[env_name].strip()
    base_url = level.get("base_url") or ""
    if not isinstance(base_url, str):
        raise ConfigError(f"{where}: base_url must be a string")
    if base_url.strip():
        return base_url.strip()
    region = level.get("region")
    if region in (None, ""):
        return ""
    if region not in REGION_URLS:
        raise ConfigError(f"{where}: unknown region {region!r}; known: {', '.join(REGION_URLS)}")
    return REGION_URLS[region]


def _check_models(stt: STTConfig, tts: TTSConfig) -> None:
    if stt.enabled and not stt.model.lower().startswith("qwen"):
        raise ConfigError(
            f"stt.model {stt.model!r} is not supported: only the qwen*-asr family "
            "(MultiModalConversation HTTP API) is implemented; the fun-asr/paraformer "
            "realtime websocket family is not"
        )
    if tts.enabled and not tts.model.startswith(("qwen-audio-", "cosyvoice-")):
        raise ConfigError(
            f"tts.model {tts.model!r} is not supported: only models served by the "
            "HTTP SpeechSynthesizer API (qwen-audio-*, cosyvoice-*) are implemented"
        )


def _bool(vals: Mapping[str, Any], key: str, where: str) -> bool:
    value = vals[key]
    if not isinstance(value, bool):
        raise ConfigError(f"{where}: {key} must be true or false")
    return value


def _str(vals: Mapping[str, Any], key: str, where: str, *, allow_empty: bool = True) -> str:
    value = vals[key]
    if value is None and allow_empty:
        return ""
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ConfigError(f"{where}: {key} must be a {'non-empty ' if not allow_empty else ''}string")
    return value


def _positive_int(vals: Mapping[str, Any], key: str, where: str) -> int:
    value = vals[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{where}: {key} must be a positive integer")
    return value


def _mapping(vals: Mapping[str, Any], key: str, where: str) -> dict[str, Any]:
    value = vals[key]
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: {key} must be a mapping")
    return dict(value)
