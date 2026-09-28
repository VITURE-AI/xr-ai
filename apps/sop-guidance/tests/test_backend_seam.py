# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The host/backend seam, run once in-process and once through ``remote.py``.

The scripted backend stands in for a camera-judged procedure: frames decide
progress, voice cannot advance it, and it draws its own overlay and speaks its
own cues. Every scenario runs twice, against the backend itself and against
the same backend served from a sidecar process, and must look the same to the
host, the wearer and the clients either way.

The fake ports have no camera, so the host's own frame pump idles and
:func:`pump` hands frames over deterministically instead: the session's input
participant's frame, to that session's run. :func:`test_host_pumps_frames`
covers the host pump itself.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio
from conftest import HostHarness, eventually, make_harness
from fixtures.scripted_backend import (
    DONE,
    HAND,
    HAND_CUE,
    LID_BOX,
    STEPS,
    ScriptedBackend,
    labelled_frame,
)
from sop_guidance.backends import remote
from sop_guidance.backends.base import (
    BackendServices,
    Capabilities,
    Cue,
    OverlayUpdate,
    RunCommand,
    RunFinished,
    StepChanged,
    Verdict,
)
from sop_guidance.backends.registry import available_backends, resolve_backend
from sop_guidance.procedures import GuidanceDefaults, load_procedure

STUB = Path(__file__).resolve().parent / "fixtures" / "sidecar_stub.py"
SEAM_CAPS = Capabilities(voice_advance=False, provides_overlay=True, frame_hz=1.0)


@pytest.fixture(scope="module")
def sidecar(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A sidecar process serving the scripted backend; yields its endpoint."""

    folder = tmp_path_factory.mktemp("sidecar")
    endpoint = f"ipc://{folder / 'scripted.sock'}"
    with (folder / "sidecar.log").open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, str(STUB), "--endpoint", endpoint,
             "--capabilities", SEAM_CAPS.model_dump_json()],
            stdout=subprocess.DEVNULL, stderr=log,
        )
        try:
            yield endpoint
        finally:
            process.terminate()
            process.wait(timeout=10)


def _remote_backend(root: Path, endpoint: str, **config: object) -> remote.RemoteBackend:
    """Build the ``remote`` backend the way the worker does, from a procedure folder."""

    folder = root / "lid-demo"
    folder.mkdir(parents=True)
    (folder / "procedure.yaml").write_text(
        "id: lid-demo\ntitle: lid demo\nbackend: remote\n"
        f"backend_config: {{endpoint: '{endpoint}'}}\n"
    )
    entry = load_procedure(folder, GuidanceDefaults())
    factory = resolve_backend(entry.spec.backend)
    return factory(BackendServices(entry=entry, config={**entry.spec.backend_config, **config},
                                   artifacts_dir=root / "artifacts"))


@pytest_asyncio.fixture(params=["in_process", "remote"])
async def seam(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[HostHarness]:
    if request.param == "remote":
        backend = _remote_backend(tmp_path / "remote", request.getfixturevalue("sidecar"))
    else:
        backend = ScriptedBackend(SEAM_CAPS)
    h = await make_harness(tmp_path, backend=backend)
    try:
        yield h
    finally:
        await h.host.shutdown()
        await h.store.aclose()
        if isinstance(backend, remote.RemoteBackend):
            await backend.aclose()


async def pump(h: HostHarness, label: int, pid: str = "alice") -> None:
    session = h.host.session_for_input(pid)
    assert session is not None and session.run is not None
    await session.run.on_frame(labelled_frame(label, pid))


def _extra(h: HostHarness, owner: str = "alice") -> dict:
    return h.host.state_of(h.host.session_of(owner))["extra"]


# ── the same scenarios, in-process and remote ────────────────────────────────


async def test_frames_drive_progress_and_voice_cannot(seam: HostHarness) -> None:
    ports = seam.ports
    assert (await seam.host.begin("alice", "lid-demo")).status == "started"
    assert ports.said == [("alice", "Step 1 of 3: Open the lid.", "announcement")]
    state = ports.states[-1]
    assert state["backend"] == "scripted"
    assert state["capabilities"]["voice_advance"] is False
    assert state["capabilities"]["provides_overlay"] is True
    # The preview still runs; the worker draws the backend's boxes on it
    # instead of running the host detector.
    assert ports.previews == [("start", "alice", "alice")]

    await pump(seam, HAND)
    assert ports.said[-1] == ("alice", HAND_CUE, "correction")  # verbatim, as a cue
    overlay = ports.overlays[-1]
    assert overlay.detections == (LID_BOX,) and overlay.extra == {"step": 0}
    assert _extra(seam) == {"holes": [0], "cues": 1}
    assert seam.host.turn_context("alice").prompt_block == "\n\nScripted context."

    for kind in ("next", "advance"):
        result = await seam.host.command("alice", RunCommand(kind))
        assert not result.accepted and result.reason == "voice-advance-disabled"
        assert result.speech == ("The camera confirms each step, so I can't skip ahead. "
                                 "Step 1 of 3 is: Open the lid.")
    assert seam.host.session_of("alice").step_index == 0

    await pump(seam, DONE)
    assert ports.said[-1] == ("alice", "Step 2 of 3: Lift the tray.", "announcement")
    state = ports.states[-1]
    assert state["step"] == 2 and state["extra"] == {"holes": [1], "cues": 1}

    await pump(seam, DONE)
    await pump(seam, DONE)
    await eventually(lambda: seam.host.session_of("alice") is None)
    assert ports.texts("alice")[-2:] == [
        "Step 3 of 3: Close the lid.", "You've completed all steps in 'lid demo'. Well done!",
    ]
    assert ports.states[-1]["outcome"] == "completed"


async def test_reset_and_restart_give_fresh_state(seam: HostHarness) -> None:
    host, ports = seam.host, seam.ports
    await host.begin("alice", "lid-demo")
    await pump(seam, HAND)
    await pump(seam, DONE)
    assert _extra(seam) == {"holes": [1], "cues": 1}

    assert (await host.command("alice", RunCommand("reset"))).accepted
    assert ports.said[-1] == ("alice", "Step 1 of 3: Open the lid.", "announcement")
    assert ports.states[-1]["extra"] == {"holes": [0], "cues": 0}

    await pump(seam, HAND)
    await pump(seam, DONE)
    first = host.session_of("alice").session_id
    await host.stop("alice", reason="wearer_request")
    assert (await host.begin("alice", "lid-demo", explicit=True)).status == "started"

    session = host.session_of("alice")
    assert session.session_id != first
    assert ports.said[-1] == ("alice", "Step 1 of 3: Open the lid.", "announcement")
    assert ports.states[-1]["extra"] == {"holes": [0], "cues": 0}


async def test_takeover_behaves_like_any_backend(seam: HostHarness) -> None:
    host, ports = seam.host, seam.ports
    await host.begin("alice", "lid-demo")
    await pump(seam, DONE)
    old = host.session_of("alice")

    offer = await host.begin("bob", "lid-demo")
    assert offer.status == "confirmation_required"
    assert host.session_of("alice") is old

    assert (await host.confirm_takeover("bob", offer.token)).status == "started"
    assert host.session_of("alice") is None
    assert ports.texts("alice")[-1] == (
        'Your guidance was stopped because participant "bob" took over.')
    assert ports.texts("bob") == ["Step 1 of 3: Open the lid."]
    ended = [s for s in ports.states if s["session_id"] == old.session_id][-1]
    assert (ended["status"], ended["outcome"]) == ("ended", "superseded")

    # The superseded run is closed: its frames reach nobody.
    said = len(ports.said)
    await old.run.on_frame(labelled_frame(DONE, "alice"))
    assert len(ports.said) == said
    # The taken-over camera (alice's glasses) now drives bob's run, from a
    # fresh start.
    assert host.session_of("bob").input_pid == "alice"
    await pump(seam, DONE, "alice")
    assert ports.said[-1] == ("bob", "Step 2 of 3: Lift the tray.", "announcement")
    assert _extra(seam, "bob") == {"holes": [1], "cues": 0}


async def test_owner_disconnect_behaves_like_any_backend(seam: HostHarness) -> None:
    host, ports = seam.host, seam.ports
    await host.begin("alice", "lid-demo")
    run = host.session_of("alice").run

    await host.participant_left("alice")

    assert host.session_of("alice") is None
    assert (ports.states[-1]["status"], ports.states[-1]["outcome"]) == ("ended", "interrupted")
    assert ports.texts("alice") == ["Step 1 of 3: Open the lid."]  # nobody left to tell
    assert ports.previews == [("start", "alice", "alice"), ("stop", "alice", "")]
    if isinstance(run, remote.RemoteRun):
        assert not (Path("/dev/shm") / run.ring_name).exists()
    else:
        assert seam.backend.runs[0].closed == "owner_disconnected"


# ── remote specifics ─────────────────────────────────────────────────────────


def test_remote_is_a_registered_backend() -> None:
    assert "remote" in available_backends()
    assert resolve_backend("remote") is remote.create_backend


def test_remote_backend_mirrors_what_the_sidecar_serves(sidecar: str, tmp_path: Path) -> None:
    backend = _remote_backend(tmp_path, sidecar)

    assert (backend.name, backend.title) == ("scripted", "scripted")
    assert backend.capabilities == SEAM_CAPS
    assert [s.instruction for s in backend.steps()] == list(STEPS)
    assert backend.instructions_digest() == list(STEPS)
    assert backend.validate() == []
    assert backend.preview_annotator() is None


def test_unreachable_sidecar_stops_startup(tmp_path: Path) -> None:
    endpoint = f"ipc://{tmp_path / 'nobody.sock'}"

    with pytest.raises(ValueError, match="no remote backend answered at ipc://"):
        _remote_backend(tmp_path, endpoint, connect_timeout_s=0.2)


def test_bad_remote_config_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"procedure\.yaml: backend_config"):
        _remote_backend(tmp_path, "ipc:///unused", colour="blue")


def test_every_event_survives_the_wire() -> None:
    events = [
        StepChanged(2, reason="navigate", acknowledge=True),
        Cue("Turn it over.", kind="hint", priority=3),
        Verdict({"completed": True, "checks": [{"requirement": "lid", "visible": True}]}),
        OverlayUpdate(123, (LID_BOX,), extra={"holes": [1, 0]}),
        RunFinished("stopped", reason="operator"),
    ]

    assert [remote.decode_event(remote.encode_event(e)) for e in events] == events


async def test_host_pumps_frames(tmp_path: Path) -> None:
    # A frame-driven backend gets the input camera from the host, at its
    # frame_hz, with no one calling on_frame by hand.
    backend = ScriptedBackend(SEAM_CAPS.model_copy(update={"frame_hz": 50.0}))
    h = await make_harness(tmp_path, backend=backend)
    camera = {"label": HAND}
    h.ports.latest_frame = lambda pid: labelled_frame(camera["label"], pid)  # type: ignore[method-assign]
    try:
        await h.host.begin("alice", "lid-demo")
        await eventually(lambda: HAND_CUE in h.ports.texts("alice"))
        camera["label"] = DONE
        await eventually(lambda: h.host.session_of("alice") is None
                         or h.host.session_of("alice").step_index > 0)
    finally:
        await h.host.shutdown()
        await h.store.aclose()


def test_a_request_check_that_denies_the_request_does_not_count() -> None:
    from sop_guidance.backends.vlm.grading import REQUEST_CHECK_NAME, request_check_present

    def check(evidence: str) -> list[dict]:
        return [{"requirement": REQUEST_CHECK_NAME, "visible": True, "evidence": evidence}]

    # Seen live, three verdicts running, over a size-one pad after "size zero".
    for dismissed in ("No specific size was requested.",
                      "The user did not request a specific size, so any pad satisfies it.",
                      "The requested size was not specified."):
        assert not request_check_present(check(dismissed)), dismissed
    # The prompt's own wording for a request that does not apply still passes.
    for judged in ("wearer asked for size zero and holds the solid saddle",
                   "none of their requests bear on what is visible here"):
        assert request_check_present(check(judged)), judged
