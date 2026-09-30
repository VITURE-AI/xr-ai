# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker layers: speech routing and playback, request gating, client protocol, API."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from conftest import HostHarness, eventually, make_harness, settle
from fastapi.testclient import TestClient
from sop_guidance.backends.base import RunCommand
from sop_guidance_worker.api import create_api
from sop_guidance_worker.config import ApiConfig, WorkerConfig
from sop_guidance_worker.interaction import Interaction
from sop_guidance_worker.protocol import ClientProtocol, ClientRegistry, WorkerPorts
from sop_guidance_worker.speech import PlaybackTracker, SpeechRouter
from xr_ai_hub import AudioChunk, DataMessage


@dataclass
class FakeEndpoint:
    connected_participants: frozenset[str] = frozenset({"alice", "bob", "glasses"})
    data: list[DataMessage] = field(default_factory=list)
    flushed: list[str] = field(default_factory=list)
    audio: list[AudioChunk] = field(default_factory=list)
    statuses: list[tuple[str, str]] = field(default_factory=list)

    async def send_return_data(self, msg: DataMessage) -> None:
        self.data.append(msg)

    async def flush_return_audio(self, participant_id: str) -> None:
        self.flushed.append(participant_id)

    async def send_return_audio(self, chunk: AudioChunk) -> None:
        self.audio.append(chunk)

    async def set_status(self, status: str, participant_id: str | None = None) -> None:
        self.statuses.append((status, participant_id or ""))

    def topics(self, topic: str) -> list[tuple[str, str]]:
        return [(m.participant_id, m.data.decode()) for m in self.data if m.topic == topic]


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


# ── speech ───────────────────────────────────────────────────────────────────


def _chunk(pid: str, seconds: float) -> AudioChunk:
    return AudioChunk(
        pts_us=0, sample_rate=24000, channels=1, samples=int(24000 * seconds), data=b"",
        participant_id=pid, track_id="tts",
    )


async def test_tracker_follows_real_audio() -> None:
    clock = Clock()
    tracker = PlaybackTracker(slack_s=0.3, clock=clock)
    endpoint = FakeEndpoint()
    tracker.install(endpoint)  # type: ignore[arg-type]

    tracker.enqueued("alice", "one two three four five six")
    assert tracker.remaining_s("alice") > 2.0  # estimate before audio flows

    # Audio longer than the estimate extends it.
    await endpoint.send_return_audio(_chunk("alice", 5.0))
    assert tracker.remaining_s("alice") == pytest.approx(5.3)
    clock.now += 1.0
    assert tracker.remaining_s("alice") == pytest.approx(4.3)

    await endpoint.flush_return_audio("alice")
    assert tracker.remaining_s("alice") == 0.0
    assert endpoint.flushed == ["alice"]


async def test_tracker_keeps_the_estimate_while_paced_audio_flows() -> None:
    # The voice runtime hands audio over in real time, a few chunks ahead of
    # playback: the first chunk must not make a long instruction look done.
    clock = Clock()
    tracker = PlaybackTracker(slack_s=0.3, clock=clock)
    endpoint = FakeEndpoint()
    tracker.install(endpoint)  # type: ignore[arg-type]

    tracker.enqueued("alice", " ".join(["word"] * 26))  # about 11 s at the default rate
    clock.now += 1.0
    await endpoint.send_return_audio(_chunk("alice", 0.12))
    assert tracker.remaining_s("alice") > 9.0
    for _ in range(40):
        clock.now += 0.1
        await endpoint.send_return_audio(_chunk("alice", 0.1))
    assert tracker.remaining_s("alice") > 5.0


async def test_tracker_calibrates_the_speaking_rate() -> None:
    clock = Clock()
    tracker = PlaybackTracker(slack_s=0.3, clock=clock)
    endpoint = FakeEndpoint()
    tracker.install(endpoint)  # type: ignore[arg-type]
    default = tracker.words_per_s

    # Twenty words that take ten seconds: a slower voice than the default.
    tracker.enqueued("alice", " ".join(["word"] * 20))
    for _ in range(100):
        await endpoint.send_return_audio(_chunk("alice", 0.1))
        clock.now += 0.1
    clock.now += 5.0
    tracker.enqueued("alice", "next line")
    assert 2.0 < tracker.words_per_s < default

    # An interrupted utterance does not count.
    rate = tracker.words_per_s
    tracker.enqueued("bob", " ".join(["word"] * 20))
    await endpoint.send_return_audio(_chunk("bob", 1.0))
    await endpoint.flush_return_audio("bob")
    clock.now += 5.0
    tracker.enqueued("bob", "again")
    assert tracker.words_per_s == rate


async def test_router_modes_and_interrupt() -> None:
    endpoint = FakeEndpoint()
    published: list[tuple[str, str, bool]] = []

    async def publish(output: Any, pid: str) -> None:
        published.append((pid, output.text, output.interrupt))

    router = SpeechRouter(
        endpoint=endpoint, tracker=PlaybackTracker(), publish=publish,  # type: ignore[arg-type]
        selected_input=lambda pid: "glasses" if pid == "alice" else pid,
    )
    await router.say("alice", "Hello.")
    await router.set_mode("bob", "selected_input", ["alice"])
    await router.say("alice", "To the glasses.")
    await router.interrupt("alice")
    await router.say("alice", "Supersedes.")
    await router.set_mode("bob", "web_client", [])
    await router.say("alice", "To bob's browser.")

    assert published == [
        ("alice", "Hello.", False),
        ("glasses", "To the glasses.", False),
        ("glasses", "Supersedes.", True),
        ("bob", "To bob's browser.", False),
    ]
    # Text goes to the whole room, not just the recipient.
    assert sorted(p for p, _ in endpoint.topics("agent.response")) == sorted(
        ["alice", "bob", "glasses"] * 4
    )
    with pytest.raises(ValueError):
        await router.set_mode("bob", "everyone", [])


# ── interaction ──────────────────────────────────────────────────────────────


class FakeForeground:
    def __init__(self) -> None:
        self.asked: list[tuple[str, str]] = []

    async def answer(self, pid: str, request: str, *, timestamp_us: int) -> None:
        self.asked.append((pid, request))


class FakeLlm:
    def __init__(self, reply: str = "yes") -> None:
        self.reply = reply

    async def chat(self, *_args: Any, **_kwargs: Any) -> Any:
        @dataclass
        class R:
            content: str

        return R(self.reply)


def _interaction(harness: HostHarness, *, classifier: str = "yes"):
    endpoint = FakeEndpoint()
    config = WorkerConfig.model_validate({
        "models_config": "m.json", "voice_gate_yaml": "v.yaml",
        "detectors_yaml": "d.yaml", "procedures_dir": ".", "run_dir": ".",
    })
    said: list[tuple[str, str]] = []

    async def publish(output: Any, pid: str) -> None:
        said.append((pid, output.text))

    speech = SpeechRouter(endpoint=endpoint, tracker=PlaybackTracker(), publish=publish,  # type: ignore[arg-type]
                          selected_input=lambda pid: pid)
    clients = ClientRegistry(endpoint, wake_in_live=True)  # type: ignore[arg-type]
    foreground = FakeForeground()
    interaction = Interaction(
        host=harness.host, foreground=foreground, speech=speech,  # type: ignore[arg-type]
        clients=clients, endpoint=endpoint, llm=FakeLlm(classifier),  # type: ignore[arg-type]
        config=config,
    )
    return interaction, foreground, clients, endpoint, said


async def _drain(interaction: Interaction) -> None:
    for task in list(interaction._tasks.values()):
        await asyncio.gather(task, return_exceptions=True)
    await settle()


async def test_live_mode_needs_the_wake_word(harness: HostHarness) -> None:
    interaction, foreground, clients, _, _ = _interaction(harness)

    await interaction.on_speech("alice", "what is on the table", 1)
    await interaction.on_speech("alice", "Hey Helix, what is on the table?", 2)
    clients.state("bob").wake_in_live = False
    await interaction.on_speech("bob", "what time is it", 3)
    await _drain(interaction)

    assert foreground.asked == [("alice", "what is on the table?"), ("bob", "what time is it")]


async def test_live_wake_mode_follows_the_client_driving_the_glasses(harness: HostHarness) -> None:
    # The glasses never announce a wake mode; the operator who picked their
    # camera turns the wake word off for them.
    interaction, foreground, clients, _, _ = _interaction(harness)
    clients.state("alice").input = "glasses"
    clients.set_wake_in_live("alice", False)

    await interaction.on_speech("glasses", "what is on the table", 1)
    await _drain(interaction)

    assert foreground.asked == [("glasses", "what is on the table")]


def test_live_wake_mode_resolution() -> None:
    endpoint = FakeEndpoint()
    clients = ClientRegistry(endpoint, wake_in_live=True)  # type: ignore[arg-type]
    assert clients.wake_required_in_live("glasses")

    clients.state("alice").input = "glasses"
    clients.state("bob").input = "glasses"
    clients.set_wake_in_live("alice", False)
    assert not clients.wake_required_in_live("glasses")
    clients.set_wake_in_live("bob", True)  # the newest driver's choice wins
    assert clients.wake_required_in_live("glasses")
    clients.set_wake_in_live("alice", False)
    assert not clients.wake_required_in_live("glasses")

    clients.set_wake_in_live("glasses", True)  # the speaker's own choice wins
    assert clients.wake_required_in_live("glasses")

    clients.forget("glasses")
    clients.state("ghost").input = "bob"
    clients.set_wake_in_live("ghost", False)  # a driver that is not connected
    assert clients.wake_required_in_live("bob")


async def test_noise_classifier_drops_unaddressed_chatter(harness: HostHarness) -> None:
    interaction, foreground, clients, _, _ = _interaction(harness, classifier="no")
    clients.state("alice").wake_in_live = False

    await interaction.on_speech("alice", "so anyway he said the thing", 1)
    await _drain(interaction)

    assert foreground.asked == []


async def test_spoken_entry_starts_without_a_model(harness: HostHarness) -> None:
    interaction, foreground, _, _, said = _interaction(harness)

    await interaction.on_speech("alice", "Hey Helix, guide me through the lid", 1)
    await _drain(interaction)
    # Offered first: guided mode takes the conversation and the camera.
    assert harness.host.session_of("alice") is None
    assert said[-1][1].endswith("Ready to start?")

    await interaction.on_speech("alice", "Hey Helix, yes", 2)
    await _drain(interaction)

    assert foreground.asked == []
    assert harness.host.session_of("alice") is not None
    assert harness.ports.texts("alice")[-1] == "Step 1 of 3: Open the lid."


async def test_a_declined_start_offer_starts_nothing(harness: HostHarness) -> None:
    interaction, foreground, _, _, said = _interaction(harness)

    await interaction.on_speech("alice", "Hey Helix, guide me through the lid", 1)
    await _drain(interaction)
    await interaction.on_speech("alice", "Hey Helix, not now", 2)
    await _drain(interaction)

    assert harness.host.session_of("alice") is None
    assert harness.host.start_offer("alice") is None
    assert said[-1][1].startswith("Okay, we'll leave it")
    # A yes after the decline has nothing to accept, so it goes to the model.
    await interaction.on_speech("alice", "Hey Helix, yes", 3)
    await _drain(interaction)
    assert harness.host.session_of("alice") is None and foreground.asked


async def test_start_offer_is_not_made_for_client_controls_or_within_a_run(
        harness: HostHarness) -> None:
    host = harness.host
    assert (await host.begin("alice", "lid-demo")).status == "started"
    # A step jump inside the run the wearer is already in is not a new start.
    moved = await host.begin("alice", "lid-demo", at_step=2, entry_mode="step",
                             intent_quote="step 2", request="go to step 2", confirm=True)
    assert moved.status == "started"


async def test_an_unanswered_start_offer_starts(tmp_path) -> None:
    from sop_guidance.host import HostSettings

    h = await make_harness(tmp_path, settings=HostSettings(step_ack_timeout_s=0,
                                                           start_confirm_s=0.05))
    try:
        offer = await h.host.begin("alice", "lid-demo", confirm=True)
        assert offer.status == "confirmation_required"
        assert h.host.session_of("alice") is None
        await eventually(lambda: h.host.session_of("alice") is not None)
        assert h.host.start_offer("alice") is None
        # A yes landing just after has nothing to accept and says nothing.
        assert (await h.host.confirm_start("alice")).message == ""

        # A no stops the clock.
        await h.host.stop("alice", reason="wearer_request")
        await h.host.begin("alice", "lid-demo", at_step=2, entry_mode="step",
                           intent_quote="step 2", request="go to step 2", confirm=True)
        h.host.cancel_start("alice")
        await asyncio.sleep(0.2)
        assert h.host.session_of("alice") is None
    finally:
        await h.host.shutdown()
        await h.store.aclose()


async def test_guidance_fast_paths(harness: HostHarness) -> None:
    interaction, foreground, _, _, _ = _interaction(harness)
    await harness.host.begin("alice", "lid-demo")

    await interaction.on_speech("alice", "next", 1)  # unaddressed: dropped
    await _drain(interaction)
    assert harness.backend.runs[0].step == 0

    await interaction.on_speech("alice", "Hey Helix, next", 2)
    await _drain(interaction)
    assert harness.backend.runs[0].step == 1

    await interaction.on_speech("alice", "how do I hold it", 3)  # unaddressed question
    await _drain(interaction)
    assert foreground.asked == []

    # A bare "stop" is the wake-free exit; the voice gate swallows it, so the
    # transcript path hands it over.
    assert interaction.gate_eats_exit("alice", "stop.")
    await interaction.on_speech("alice", "stop.", 4)
    await _drain(interaction)
    assert harness.host.session_of("alice") is None
    assert "Stopped guidance for 'lid demo' at step 2 of 3" in harness.ports.texts("alice")[-1]


async def test_bare_resume_after_a_stop(harness: HostHarness) -> None:
    # The stop message says "Say resume to pick up there": no procedure name.
    interaction, foreground, _, _, spoken = _interaction(harness)
    host = harness.host
    await host.begin("alice", "lid-demo")
    await host.command("alice", RunCommand("next"))
    await host.stop("alice", reason="wearer_request")

    for said in ("Hey Helix, resume guidance", "Hey Helix, resume"):
        await interaction.on_speech("alice", said, 1)
        await _drain(interaction)
        assert spoken[-1][1].endswith("at step 2 of 3. Ready to carry on?")
        await interaction.on_speech("alice", "Hey Helix, yes", 1)
        await _drain(interaction)
        assert foreground.asked == []
        assert harness.host.session_of("alice") is not None
        assert harness.backend.runs[-1].step == 1
        await host.stop("alice", reason="wearer_request")

    # Someone with nothing stopped gets the model, not someone else's session.
    await interaction.on_speech("bob", "Hey Helix, resume", 2)
    await _drain(interaction)
    assert harness.host.session_of("bob") is None
    assert foreground.asked


async def test_control_resume_defaults_to_the_latest_session(harness: HostHarness) -> None:
    host = harness.host
    nothing = await host.resume("alice", "")
    assert nothing.status == "error"

    await host.begin("alice", "lid-demo")
    await host.command("alice", RunCommand("next"))
    await host.stop("alice", reason="wearer_request")
    reply = await host.resume("alice", "")
    assert reply.status == "started"
    assert harness.backend.runs[-1].step == 1


async def test_wearer_glasses_drive_the_operator_session(harness: HostHarness) -> None:
    interaction, foreground, _, _, _ = _interaction(harness)
    await harness.host.begin("alice", "lid-demo")
    await harness.host.change_input("alice", "glasses")

    await interaction.on_speech("glasses", "Hey Helix, is this right?", 1)
    await _drain(interaction)

    assert foreground.asked == [("alice", "is this right?")]


async def test_other_participant_cannot_advance(harness: HostHarness) -> None:
    interaction, _, _, _, said = _interaction(harness)
    await harness.host.begin("alice", "lid-demo")

    await interaction.on_typed("bob", "next", 1)
    await _drain(interaction)

    assert harness.backend.runs[0].step == 0
    assert any('owned by participant "alice"' in text for _, text in said)


async def test_takeover_answered_by_voice(harness: HostHarness) -> None:
    interaction, _, _, _, _ = _interaction(harness)
    await harness.host.begin("alice", "lid-demo")
    offer = await harness.host.begin("bob", "lid-demo")
    assert offer.status == "confirmation_required"

    await interaction.on_typed("bob", "yes", 1)
    await _drain(interaction)

    assert harness.host.session_of("bob") is not None


# ── client protocol ──────────────────────────────────────────────────────────


class FakePreview:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def start_live(self, recipient: str, source: str) -> None:
        self.calls.append(("live", recipient, source))

    async def stop_live(self, recipient: str) -> None:
        self.calls.append(("stop", recipient, ""))


async def test_control_messages(harness: HostHarness) -> None:
    interaction, _, clients, endpoint, _ = _interaction(harness)
    preview = FakePreview()
    speech = interaction._speech
    protocol = ClientProtocol(
        host=harness.host, clients=clients, speech=speech, preview=preview,  # type: ignore[arg-type]
        on_typed=interaction.on_typed, cancel_turn=interaction.cancel,
    )

    def msg(topic: str, payload: dict[str, Any], pid: str = "alice") -> DataMessage:
        return DataMessage(participant_id=pid, topic=topic, pts_us=1,
                           data=json.dumps(payload).encode())

    await protocol.on_data(msg("xr.yolo_mode", {"mode": "default"}))
    await protocol.on_data(msg("xr.main_input", {"participant_id": "glasses"}))
    await protocol.on_data(msg("xr.main_input", {"participant_id": "ghost"}))
    await protocol.on_data(msg("xr.wake_mode", {"required_in_live": False}))
    await protocol.on_data(msg("guidance.control", {"action": "ready", "request_id": "r1"}))
    await protocol.on_data(msg("guidance.control", {
        "action": "start", "procedure_id": "lid-demo", "request_id": "r2",
        "input_participant": "glasses",
    }))
    session = harness.host.session_of("alice")
    # A client announcing itself gets the running session's state again.
    states_before = len(harness.ports.states)
    await protocol.on_data(msg("guidance.control", {"action": "ready", "request_id": "r4"},
                               pid="bob"))
    assert len(harness.ports.states) == states_before + 1
    assert harness.ports.states[-1]["session_id"] == session.session_id
    await protocol.on_data(msg("guidance.control", {
        "action": "stop", "session_id": session.session_id, "request_id": "r3",
    }, pid="bob"))

    assert preview.calls == [("live", "alice", "alice"), ("live", "alice", "glasses")]
    assert clients.input_of("alice") == "glasses"
    assert not clients.wake_required_in_live("alice")
    results = {json.loads(d)["request_id"]: json.loads(d)
               for _, d in endpoint.topics("guidance.result")}
    assert results["r1"]["status"] == "ok"
    assert results["r2"]["status"] == "ok" and results["r2"]["session_id"]
    assert results["r3"]["status"] == "ok" and harness.host.session_of("alice") is None


def test_ports_resolve_the_owner_selection() -> None:
    endpoint = FakeEndpoint()
    clients = ClientRegistry(endpoint, wake_in_live=True)  # type: ignore[arg-type]
    ports = WorkerPorts(speech=None, frames=None, preview=None, clients=clients)  # type: ignore[arg-type]
    clients.state("alice").input = "glasses"

    assert ports.resolve_input("alice", "") == "glasses"
    assert ports.resolve_input("alice", "bob") == "bob"  # a saved, connected input wins
    assert ports.resolve_input("alice", "ghost") == "glasses"
    clients.forget("glasses")
    assert ports.resolve_input("alice", "") == "alice"


# ── API ──────────────────────────────────────────────────────────────────────


def test_api_serves_procedures_and_requires_the_token(
    harness: HostHarness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("TEST_API_TOKEN", "s3cret")
    folder = harness.host.procedure("lid-demo").entry.directory
    (folder / "thumb.jpg").write_bytes(b"\xff\xd8jpeg")
    app = create_api(host=harness.host, store=harness.store,
                     config=ApiConfig(token_env="TEST_API_TOKEN"))
    client = TestClient(app)
    auth = {"authorization": "Bearer s3cret"}

    assert client.get("/api/procedures").status_code == 401
    # Any header byte is a wrong token, not a server error.
    assert client.get("/api/procedures",
                      headers={"authorization": "Bearer caf\xe9".encode("latin-1")}).status_code == 401
    assert client.get("/debug/index.json").status_code == 401
    listing = client.get("/api/procedures", headers=auth).json()
    assert [p["id"] for p in listing["procedures"]] == ["lid-demo"]
    detail = client.get("/api/procedures/lid-demo", headers=auth).json()
    assert [s["instruction"] for s in detail["step_list"]][0] == "Open the lid."
    assert {"requirements", "done_when"} <= set(detail["step_list"][0])
    assert client.get("/api/procedures/lid-demo/files/thumb.jpg", headers=auth).content \
        == b"\xff\xd8jpeg"
    assert client.get("/api/procedures/lid-demo/files/procedure.yaml",
                      headers=auth).status_code == 404
    assert client.get("/api/procedures/lid-demo/files/../../x.jpg",
                      headers=auth).status_code == 404
    assert client.get("/api/sessions", headers=auth).json()["sessions"] == []


def test_a_replaced_step_image_gets_a_new_url(tmp_path: Path) -> None:
    import os
    from types import SimpleNamespace

    from sop_guidance_worker.api import _frame_url

    procedure = SimpleNamespace(id="lid-demo", entry=SimpleNamespace(directory=tmp_path))
    image = tmp_path / "frames" / "step_01.jpg"
    image.parent.mkdir()
    image.write_bytes(b"old")
    first = _frame_url(procedure, str(image))
    assert first.startswith("/api/procedures/lid-demo/files/frames/step_01.jpg?v=")
    os.utime(image, ns=(1, 1))
    assert _frame_url(procedure, str(image)) != first
    assert _frame_url(procedure, str(tmp_path.parent / "elsewhere.jpg")) == ""


def test_the_thumbnail_is_a_versioned_file_url(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from sop_guidance_worker.api import _ui

    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "step_05.jpg").write_bytes(b"jpeg")
    spec = SimpleNamespace(ui={"thumbnail": "frames/step_05.jpg", "accent": "green"})
    procedure = SimpleNamespace(id="lid-demo", entry=SimpleNamespace(directory=tmp_path, spec=spec))
    ui = _ui(procedure)
    assert ui["thumbnail"].startswith("/api/procedures/lid-demo/files/frames/step_05.jpg?v=")
    assert ui["accent"] == "green" and spec.ui["thumbnail"] == "frames/step_05.jpg"


async def test_backend_overlay_reaches_clients_and_the_preview() -> None:
    from sop_guidance.backends.base import OverlayUpdate
    from sop_guidance.vision import Detection
    from sop_guidance_worker.protocol import WorkerPorts

    sent: list[tuple[str, str, object]] = []
    overlays: list[tuple[str, OverlayUpdate]] = []

    class Speech:
        async def send(self, owner: str, topic: str, payload: object) -> None:
            json.dumps(payload)  # must be JSON-safe: Detection dataclasses were not
            sent.append((owner, topic, payload))

    class Preview:
        def set_overlay(self, owner: str, update: OverlayUpdate) -> None:
            overlays.append((owner, update))

    ports = WorkerPorts(speech=Speech(), frames=None, preview=Preview(),  # type: ignore[arg-type]
                        clients=None)  # type: ignore[arg-type]
    update = OverlayUpdate(timestamp_us=5, detections=(Detection("lid", 1, 2, 3, 4, 0.9),),
                           extra={"holes": [1]})
    await ports.overlay_update("alice", update)

    assert overlays == [("alice", update)]
    owner, topic, payload = sent[0]
    assert (owner, topic) == ("alice", "guidance.overlay")
    assert payload["detections"][0]["label"] == "lid"  # type: ignore[index]


async def test_pronoun_entry_starts_the_only_procedure(harness: HostHarness) -> None:
    interaction, foreground, _, _, _ = _interaction(harness)

    await interaction.on_speech("alice", "Hey Helix, guide me through that procedure.", 1)
    await _drain(interaction)
    await interaction.on_speech("alice", "Hey Helix, yes", 2)
    await _drain(interaction)

    assert foreground.asked == []
    assert harness.host.session_of("alice") is not None
