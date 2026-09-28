# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Who sees the guidance preview, and what the idle conversation remembers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from conftest import HostHarness, eventually
from sop_guidance_worker.config import WorkerConfig
from sop_guidance_worker.foreground import Foreground
from sop_guidance_worker.preview import FrameCache, PreviewManager
from sop_guidance_worker.protocol import ClientRegistry, WorkerPorts
from xr_ai_hub import FrameData, FrameSignal, PixelFormat, ReturnVideoFrame

# ── preview audience ─────────────────────────────────────────────────────────


@dataclass
class VideoEndpoint:
    connected_participants: frozenset[str] = frozenset({"wearer", "operator", "other"})
    sent: list[str] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)
    seq: int = 0

    def on_frame(self, _cb: Any) -> None:
        return None

    def on_participant(self, _cb: Any) -> None:
        return None

    async def request_frame(self, signal: FrameSignal) -> FrameData:
        self.seq += 1
        return FrameData(seq=self.seq, pts_us=time.time_ns() // 1_000 + self.seq, width=4,
                         height=2, fmt=PixelFormat.RGB24, data=bytes(4 * 2 * 3),
                         participant_id=signal.participant_id)

    async def send_return_video(self, frame: ReturnVideoFrame) -> None:
        self.sent.append(frame.participant_id)

    async def stop_return_video(self, participant_id: str, track_id: str = "overlay") -> None:
        self.stopped.append(participant_id)


class Painter:
    """Stands in for the live COCO annotator."""

    @dataclass
    class Drawn:
        image: np.ndarray

    async def annotate_array(self, image: np.ndarray, *, stream: str = "") -> Drawn:
        return self.Drawn(image)


class FrameSource(FrameCache):
    """Always has a fresh signal: every tick is a new frame."""

    def signal(self, participant_id: str) -> FrameSignal | None:
        return FrameSignal(slot=0, seq=0, pts_us=time.time_ns() // 1_000, width=4, height=2,
                           fmt=PixelFormat.RGB24, data_sz=24, participant_id=participant_id)


def _preview(endpoint: VideoEndpoint, clients: ClientRegistry) -> PreviewManager:
    frames = FrameSource(endpoint)  # type: ignore[arg-type]
    return PreviewManager(
        endpoint=endpoint, frames=frames, fps=200.0,  # type: ignore[arg-type]
        live_annotator=lambda: Painter(), recorder=lambda _pid: None,  # type: ignore[arg-type,return-value]
        watchers=clients.watchers,
    )


async def test_operator_watching_the_wearer_sees_the_guidance_preview() -> None:
    endpoint = VideoEndpoint()
    clients = ClientRegistry(endpoint, wake_in_live=True)  # type: ignore[arg-type]
    clients.state("operator").yolo_mode = "default"
    clients.state("operator").input = "wearer"
    preview = _preview(endpoint, clients)
    ports = WorkerPorts(speech=None, frames=None, preview=preview, clients=clients)  # type: ignore[arg-type]

    await preview.start_live("operator", "wearer")
    assert not preview.running("operator").guidance

    # Guidance started by voice from the wearer's device: the wearer owns it.
    await preview.start_guidance("wearer", "wearer", None)
    assert preview.running("operator") is None  # its COCO loop yields
    endpoint.sent.clear()
    await eventually(lambda: {"wearer", "operator"} <= set(endpoint.sent))

    # A re-announced overlay mode does not bring the COCO loop back.
    await preview.start_live("operator", "wearer")
    assert preview.running("operator") is None

    await ports.stop_preview("wearer")
    assert "operator" in endpoint.stopped
    loop = preview.running("operator")
    assert loop is not None and not loop.guidance and loop.source == "wearer"
    await preview.aclose()


async def test_clients_without_an_overlay_mode_get_no_video() -> None:
    endpoint = VideoEndpoint()
    clients = ClientRegistry(endpoint, wake_in_live=True)  # type: ignore[arg-type]
    clients.state("other").input = "wearer"  # never announced an overlay mode
    preview = _preview(endpoint, clients)

    await preview.start_guidance("wearer", "wearer", None)
    endpoint.sent.clear()
    await eventually(lambda: len(endpoint.sent) >= 3)

    assert set(endpoint.sent) == {"wearer"}
    await preview.aclose()


# ── idle history ─────────────────────────────────────────────────────────────


class RecordingLlm:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls: list[list[tuple[str, Any]]] = []
        self.capabilities = None

    async def chat(self, messages: Any, **_kwargs: Any) -> Any:
        self.calls.append([(m.role, m.content) for m in messages])

        @dataclass
        class R:
            content: str
            tool_calls: Any = None
            finish_reason: str = "stop"
            reasoning: Any = None

        return R(self.replies.pop(0))


class Speech:
    def __init__(self) -> None:
        self.said: list[str] = []

    async def say(self, pid: str, text: str, **_kwargs: Any) -> None:
        self.said.append(text)

    async def progress(self, text: str) -> None:
        return None


async def test_idle_turns_remember_the_offer(harness: HostHarness) -> None:
    config = WorkerConfig.model_validate({
        "models_config": "m.json", "voice_gate_yaml": "v.yaml", "detectors_yaml": "d.yaml",
        "procedures_dir": ".", "run_dir": ".", "foreground": {"quick_ack": False},
    })
    llm = RecordingLlm(["Shall I start the lid demo?", "Starting."])
    foreground = Foreground(host=harness.host, llm=llm, vlm=None, config=config,  # type: ignore[arg-type]
                            speech=Speech(), frames=None, input_of=lambda p: p)  # type: ignore[arg-type]

    await foreground.answer("alice", "how do I open the lid", timestamp_us=1)
    await foreground.answer("alice", "yes", timestamp_us=2)

    second = llm.calls[1]
    assert [role for role, _ in second] == ["system", "user"]
    context = second[1][1]
    assert "[Recent conversation]\n  User: how do I open the lid\n  Agent: Shall I start" in context
    assert context.endswith("[User request]\nyes")

    # Another participant's conversation is their own.
    llm.replies.append("Hello.")
    await foreground.answer("bob", "hi", timestamp_us=3)
    assert "[Recent conversation]" not in llm.calls[2][1][1]


# ── guidance turn (the old fork's contract) ──────────────────────────────────


def _foreground(harness: HostHarness, replies: list[str]) -> tuple[Foreground, RecordingLlm]:
    config = WorkerConfig.model_validate({
        "models_config": "m.json", "voice_gate_yaml": "v.yaml", "detectors_yaml": "d.yaml",
        "procedures_dir": ".", "run_dir": ".", "foreground": {"quick_ack": False},
    })
    llm = RecordingLlm(replies)
    foreground = Foreground(host=harness.host, llm=llm, vlm=None, config=config,  # type: ignore[arg-type]
                            speech=Speech(), frames=None, input_of=lambda p: p)  # type: ignore[arg-type]
    return foreground, llm


async def test_guidance_turn_speaks_the_json_reply(harness: HostHarness) -> None:
    await harness.host.begin("alice", "lid-demo")
    foreground, llm = _foreground(harness, [
        '{"reply":"Size zero is the solid black saddle.","action":"none","reason":"asked"}',
        '{"reply":"Lift the solid one off the table.","action":"none","reason":"again"}',
    ])

    await foreground.answer("alice", "which one is size zero", timestamp_us=1)
    await foreground.answer("alice", "which one again", timestamp_us=2)

    system = llm.calls[0][0][1]
    assert system.startswith("You are guiding someone through 'lid demo'. THEY ARE ON STEP 1 OF 3")
    assert ">> Step 1: Open the lid." in system and "   Step 2: Lift the tray." in system
    assert 'Reply with exactly one JSON object: {"reply":"<spoken>"' in system
    assert "[Earlier on this step]\nWearer:" not in system
    assert "[Earlier on this step]\nWearer: which one is size zero\nAssistant: Size zero" \
        in llm.calls[1][0][1]
    assert harness.ports.texts("alice")[-2:] == [
        "Size zero is the solid black saddle.", "Lift the solid one off the table.",
    ]
    assert "still on step" not in " ".join(harness.ports.texts("alice"))


async def test_guidance_turn_actions(harness: HostHarness) -> None:
    await harness.host.begin("alice", "lid-demo")
    foreground, _ = _foreground(harness, [
        '{"reply":"","action":"restep","reason":"missed it"}',
        '{"reply":"Done.","action":"advance","reason":"finished"}',
        'Keep it level while you lift.',
        '{"reply":"","action":"exit","reason":"stop"}',
    ])

    await foreground.answer("alice", "say that again", timestamp_us=1)
    assert harness.ports.texts("alice")[-1] == "Step 1 of 3 is: Open the lid."

    await foreground.answer("alice", "I opened it", timestamp_us=2)
    assert harness.backend.runs[0].step == 1  # the run granted the advance

    await foreground.answer("alice", "how do I hold it", timestamp_us=3)
    assert harness.ports.texts("alice")[-1] == "Keep it level while you lift."  # salvaged prose

    await foreground.answer("alice", "I'm done for today", timestamp_us=4)
    assert harness.host.session_of("alice") is None
    assert harness.ports.texts("alice")[-1].startswith("Stopped guidance for 'lid demo'")


async def test_guidance_turn_falls_back_to_the_step_line(harness: HostHarness) -> None:
    await harness.host.begin("alice", "lid-demo")
    foreground, _ = _foreground(harness, ['{"reply": "unterminated'])

    await foreground.answer("alice", "hmm {what}", timestamp_us=1)

    assert harness.ports.texts("alice")[-1] == "Step 1 of 3 is: Open the lid."


# ── scene memory and the can't-see guard ─────────────────────────────────────


class ScriptedVlm:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.questions: list[str] = []

    async def ask_image(self, _image: Any, question: str, **_kwargs: Any) -> Any:
        self.questions.append(question)

        @dataclass
        class R:
            content: str

        return R(self.answers.pop(0))


class StillFrames:
    """A camera whose frames carry an increasing timestamp."""

    def __init__(self) -> None:
        self.ts = 1_000_000

    def latest(self, _pid: str) -> Any:
        return None

    async def fetch(self, pid: str) -> Any:
        from sop_guidance.backends.base import TimedFrame

        self.ts += 1_000_000
        return TimedFrame(participant_id=pid, timestamp_us=self.ts, width=4, height=2,
                          image=np.zeros((2, 4, 3), dtype=np.uint8))


async def test_observer_records_changes_and_condenses() -> None:
    from sop_guidance_worker.observer import SceneObserver

    vlm = ScriptedVlm(["hands lift the glasses at center", "unchanged",
                       "a nose pad is placed on the table at right"])
    llm = RecordingLlm(['{"overview":"Someone is handling glasses.","events":[]}'])
    observer = SceneObserver(frames=StillFrames(), vlm=vlm, llm=llm,  # type: ignore[arg-type]
                             sources=lambda: ["cam"], busy=lambda _s: False)

    assert (await observer.observe("cam")).description == "hands lift the glasses at center"
    assert await observer.observe("cam") is None  # "unchanged" is dropped
    await observer.observe("cam")
    assert "Previous observation: hands lift the glasses at center" in vlm.questions[1]
    assert await observer.condense("cam") == "Someone is handling glasses."

    block = observer.memory("cam").context_block(8)
    assert block.startswith("[Scene summary]\nSomeone is handling glasses.")
    assert "hands lift the glasses at center" in block and "nose pad is placed" in block
    assert observer.memory("empty").context_block(8) == (
        "[Scene summary]\nNo scene summary available yet.\n\n[Recent observations]\nNone yet."
    )


async def test_cant_see_reply_looks_through_the_camera(harness: HostHarness) -> None:
    config = WorkerConfig.model_validate({
        "models_config": "m.json", "voice_gate_yaml": "v.yaml", "detectors_yaml": "d.yaml",
        "procedures_dir": ".", "run_dir": ".", "foreground": {"quick_ack": False},
    })

    class Frames:
        async def fetch_jpeg(self, _pid: str) -> bytes:
            return b"jpeg"

    speech = Speech()
    foreground = Foreground(
        host=harness.host, llm=RecordingLlm(["I cannot see what you are looking at."]),  # type: ignore[arg-type]
        vlm=ScriptedVlm(["A pair of black glasses on a table."]), config=config,
        speech=speech, frames=Frames(), input_of=lambda p: p,  # type: ignore[arg-type]
        scene=lambda source: f"[Scene summary]\nscene of {source}",
    )

    await foreground.answer("alice", "what am I looking at", timestamp_us=1)

    assert speech.said == ["A pair of black glasses on a table."]
