# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose the SOP guidance worker from shared SDK primitives."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any

import uvicorn
from loguru import logger
from sop_guidance.backends.base import BackendServices, ModelHandles
from sop_guidance.backends.registry import resolve_backend
from sop_guidance.host import GuidanceHost, LoadedProcedure
from sop_guidance.procedures import ProcedureEntry, discover_procedures
from sop_guidance.recorder import SessionStore
from sop_guidance.vision import (
    DetectorProfile,
    FrameAnnotator,
    load_detector_profiles,
    load_geometry_plugin,
)
from xr_ai_logging import setup_logging
from xr_ai_models import ChatMessage, load_models_config, make_llm, make_stt, make_tts, make_vlm
from xr_ai_runtime import AgentRuntime
from xr_ai_voice import HubVoiceTransport, VadConfig, VoiceAgent
from xr_ai_voicegate import load_voice_gate_config

from .agent import (
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
    GuidanceAgent,
)
from .api import create_api
from .config import WorkerConfig
from .foreground import Foreground
from .interaction import Interaction
from .preview import FrameCache, PreviewManager
from .protocol import ClientProtocol, ClientRegistry, WorkerPorts
from .speech import PlaybackTracker, SpeechRouter, runtime_publisher


class _Models:
    """One client per model role, shared by every procedure that names it."""

    def __init__(self, config: WorkerConfig) -> None:
        self._config = load_models_config(config.models_config)
        self._llm: dict[str, Any] = {}
        self._vlm: dict[str, Any] = {}

    def stt(self) -> Any:
        return make_stt(self._config, "stt")

    def tts(self) -> Any:
        return make_tts(self._config, "tts")

    def llm(self, role: str) -> Any:
        if role not in self._llm:
            self._llm[role] = make_llm(self._config, role)
        return self._llm[role]

    def vlm(self, role: str) -> Any:
        if role not in self._vlm:
            self._vlm[role] = make_vlm(self._config, role)
        return self._vlm[role]

    def closeables(self) -> list[Any]:
        return [*self._llm.values(), *self._vlm.values()]


class _Annotators:
    """Build frame annotators from the shared detector profiles."""

    def __init__(self, profiles: dict[str, DetectorProfile], artifacts_dir: Path) -> None:
        self._profiles = profiles
        self._artifacts = artifacts_dir
        self.built: list[FrameAnnotator] = []

    def build(
        self,
        profile_name: str,
        *,
        overrides: dict[str, Any] | None = None,
        geometry_path: Path | None = None,
        spatial_context: bool = False,
    ) -> FrameAnnotator:
        profile = self._profiles.get(profile_name)
        if profile is None:
            raise ValueError(
                f"unknown detector profile {profile_name!r} "
                f"(have: {', '.join(sorted(self._profiles)) or 'none'})"
            )
        if overrides:
            profile = profile.model_copy(update=overrides)
        annotator = FrameAnnotator(
            profile,
            geometry=load_geometry_plugin(geometry_path) if geometry_path else None,
            spatial_context=spatial_context,
            artifacts_dir=self._artifacts / "overlays" / profile_name,
        )
        self.built.append(annotator)
        return annotator

    async def warmup(self) -> bool:
        for annotator in self.built:
            if annotator.profile.enabled and annotator.profile.preheat:
                if not await asyncio.to_thread(annotator.warmup):
                    return False
        return True


def _load_procedures(
    config: WorkerConfig, models: _Models, annotators: _Annotators, profiles: dict[str, Any],
) -> list[LoadedProcedure]:
    loaded: list[LoadedProcedure] = []
    problems: list[str] = []
    for entry in discover_procedures(config.procedures_dir, config.guidance_defaults):
        backend = _build_backend(entry, config, annotators, profiles)
        for problem in backend.validate():
            problems.append(f"{entry.id}: {problem}")
        spec = entry.spec
        handles = ModelHandles(llm=models.llm(spec.models.llm_role),
                               vlm=models.vlm(spec.models.vlm_role))
        loaded.append(LoadedProcedure(entry=entry, backend=backend, models=handles))
        logger.info("procedure {} backend={} steps={}", entry.id, backend.name,
                    len(backend.steps()))
    if problems:
        raise ValueError("procedures failed validation:\n  " + "\n  ".join(problems))
    return loaded


def _build_backend(entry: ProcedureEntry, config: WorkerConfig, annotators: _Annotators,
                   profiles: dict[str, Any]) -> Any:
    factory = resolve_backend(entry.spec.backend)
    return factory(BackendServices(
        entry=entry,
        config=entry.spec.backend_config,
        artifacts_dir=config.run_dir / "artifacts",
        detector_profiles=profiles,
        frame_annotator=annotators.build,
    ))


def _llm_text(llm: Any):
    async def ask(system: str, user: str, max_tokens: int, temperature: float) -> str:
        response = await llm.chat(
            (ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)),
            max_tokens=max_tokens, temperature=temperature,
        )
        return response.content

    return ask


async def _close_backends(procedures: Iterable[LoadedProcedure]) -> None:
    # Backends holding outside resources (a remote sidecar connection, shared
    # memory) expose aclose(); in-process ones have nothing to release.
    for procedure in procedures:
        aclose = getattr(procedure.backend, "aclose", None)
        if aclose is None:
            continue
        try:
            await aclose()
        except Exception:
            logger.exception("backend {} failed to close", procedure.id)


def _api_server(app: Any, host: str, port: int) -> uvicorn.Server:
    return uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))


async def _stop_api(server: uvicorn.Server, task: asyncio.Task[None]) -> None:
    # Ask uvicorn to finish its lifespan first: cancelling the serve task
    # outright makes Starlette log the lifespan's CancelledError as an error.
    server.should_exit = True
    done, _ = await asyncio.wait({task}, timeout=5.0)
    if not done:
        task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await task


async def run_app(config: WorkerConfig, *, ready_file: Path | None = None) -> None:
    """Run the worker until the voice session shuts down."""

    setup_logging("worker")
    models = _Models(config)
    profiles = load_detector_profiles(config.detectors_yaml)
    annotators = _Annotators(profiles, config.run_dir / "artifacts")
    procedures = _load_procedures(config, models, annotators, profiles)
    live_annotator = (annotators.build(config.preview.live_profile)
                      if config.preview.live_profile in profiles else None)

    foreground_llm = models.llm(config.foreground.llm_role)
    foreground_vlm = models.vlm(config.foreground.vlm_role)

    transport = HubVoiceTransport()
    endpoint = transport.endpoint
    tracker = PlaybackTracker()
    tracker.install(endpoint)

    store = SessionStore(
        config.run_dir / "guidance",
        level=config.debug.level,
        max_bytes=config.debug.max_bytes,
        clip_seconds=config.debug.clip_seconds,
        clip_fps=config.debug.clip_fps,
    )
    await store.start()
    interrupted = store.mark_running_interrupted()
    if interrupted:
        logger.info("marked {} sessions from a previous run as interrupted", interrupted)

    runtime = AgentRuntime()
    clients = ClientRegistry(endpoint, wake_in_live=config.wake.required_in_live)
    frames = FrameCache(endpoint, max_age_s=config.frame_max_age_s,
                        timeout_s=config.frame_timeout_s)
    host_ref: dict[str, GuidanceHost] = {}

    def recorder_for(owner: str):
        session = host_ref["host"].session_of(owner)
        return session.recorder if session is not None else None

    preview = PreviewManager(
        endpoint=endpoint, frames=frames, fps=config.preview.fps,
        live_annotator=lambda: live_annotator, recorder=recorder_for,
        box_painter=FrameAnnotator(DetectorProfile(enabled=False),
                                   artifacts_dir=config.run_dir / "artifacts" / "overlays" / "backend"),
    )
    speech = SpeechRouter(
        endpoint=endpoint, tracker=tracker, publish=runtime_publisher(runtime),
        selected_input=clients.input_of,
    )
    host = GuidanceHost(
        procedures=procedures,
        store=store,
        ports=WorkerPorts(speech=speech, frames=frames, preview=preview, clients=clients),
        settings=config.guidance,
        llm_text=_llm_text(foreground_llm),
    )
    host_ref["host"] = host
    foreground = Foreground(
        host=host, llm=foreground_llm, vlm=foreground_vlm, config=config,
        speech=speech, frames=frames, input_of=clients.input_of,
    )
    interaction = Interaction(
        host=host, foreground=foreground, speech=speech, clients=clients,
        endpoint=endpoint, llm=foreground_llm, config=config,
    )
    protocol = ClientProtocol(
        host=host, clients=clients, speech=speech, preview=preview,
        on_typed=interaction.on_typed, cancel_turn=interaction.cancel,
    )
    endpoint.on_data(protocol.on_data)

    async def joined(pid: str) -> None:
        logger.info("participant joined {}", pid)

    async def left(pid: str) -> None:
        logger.info("participant left {}", pid)
        await interaction.cancel(pid)
        await host.participant_left(pid)
        await preview.stop(pid)
        frames.release(pid)
        clients.forget(pid)

    voice = VoiceAgent(
        query_topic=USER_QUERY_TOPIC,
        stt=models.stt(),
        tts=models.tts(),
        vad=VadConfig(
            silence_duration=config.voice.silence_duration,
            min_speech=config.voice.min_speech,
            silero_threshold=config.voice.silero_threshold,
        ),
        voice_gate=load_voice_gate_config(config.voice_gate_yaml),
        probes={"detectors": annotators.warmup},
        ready_file=ready_file,
        closeables=models.closeables(),
        # Text lines are fanned out to the whole room by the speech router.
        text_topic="",
        idle_timeout_secs=config.voice.idle_timeout_secs or None,
        transport=transport,
        # Typed text is handled by the client protocol, without the gates.
        text_input=False,
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        interrupt_on_supersede=False,
    )
    runtime.register("guidance", GuidanceAgent(
        interaction=interaction, speech=speech, on_joined=joined, on_left=left,
    ))
    runtime.register("voice", voice)

    api_server = _api_server(create_api(host=host, store=store, config=config.api),
                             config.api.host, config.api.port)
    api_task = asyncio.create_task(api_server.serve(), name="sop-guidance-api")
    logger.info("sop-guidance worker starting: {} procedures, API on {}:{}",
                len(procedures), config.api.host, config.api.port)
    try:
        async with runtime:
            try:
                await voice.run(runtime)
            finally:
                await interaction.aclose()
                await host.shutdown()
                await preview.aclose()
                await _close_backends(procedures)
    finally:
        await _stop_api(api_server, api_task)
        await store.aclose()
    logger.info("sop-guidance worker stopped")


__all__ = ["run_app"]
