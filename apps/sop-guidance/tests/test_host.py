# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The guidance host against a backend it knows nothing about."""

from __future__ import annotations

from pathlib import Path

from conftest import HostHarness, make_harness, settle
from sop_guidance.backends.base import Capabilities, RunCommand
from sop_guidance.host import HostSettings


async def test_start_announces_the_first_step_verbatim(harness: HostHarness) -> None:
    reply = await harness.host.begin("alice", "lid-demo")

    assert reply.status == "started"
    assert harness.ports.texts("alice") == ["Step 1 of 3: Open the lid."]
    assert harness.ports.previews == [("start", "alice", "alice")]
    state = harness.ports.states[-1]
    assert state["status"] == "running"
    assert state["step"] == 1 and state["total_steps"] == 3
    assert state["backend"] == "scripted"
    assert state["extra"] == {"holes": [0], "cues": 0}


async def test_unknown_procedure_is_refused(harness: HostHarness) -> None:
    reply = await harness.host.begin("alice", "nope")

    assert reply.status == "error"
    assert "'lid demo'" in reply.message
    assert harness.host.sessions() == []


async def test_advance_and_completion(harness: HostHarness) -> None:
    host = harness.host
    await host.begin("alice", "lid-demo")
    for _ in range(3):
        result = await host.command("alice", RunCommand("next"))
        assert result.accepted
    await settle()

    assert harness.ports.texts("alice") == [
        "Step 1 of 3: Open the lid.",
        "Step 2 of 3: Lift the tray.",
        "Step 3 of 3: Close the lid.",
        "You've completed all steps in 'lid demo'. Well done!",
    ]
    assert host.session_of("alice") is None
    assert harness.ports.states[-1]["status"] == "ended"
    assert harness.ports.states[-1]["outcome"] == "completed"
    assert harness.backend.runs[0].closed == "completed"


async def test_voice_advance_capability_is_enforced(tmp_path: Path) -> None:
    h = await make_harness(tmp_path, capabilities=Capabilities(voice_advance=False))
    try:
        await h.host.begin("alice", "lid-demo")
        result = await h.host.command("alice", RunCommand("next"))

        assert not result.accepted
        assert "camera confirms each step" in result.speech
        assert "Step 1 of 3 is: Open the lid." in result.speech
        assert h.backend.runs[0].step == 0
    finally:
        await h.host.shutdown()
        await h.store.aclose()


async def test_takeover_needs_confirmation(harness: HostHarness) -> None:
    host = harness.host
    await host.begin("alice", "lid-demo")

    offer = await host.begin("bob", "lid-demo")
    assert offer.status == "confirmation_required"
    assert host.session_of("alice") is not None

    reply = await host.confirm_takeover("bob", offer.token)
    assert reply.status == "started"
    assert host.session_of("alice") is None
    assert host.session_of("bob") is not None
    assert any('participant "bob" took over' in t for t in harness.ports.texts("alice"))


async def test_takeover_with_wrong_token_is_rejected(harness: HostHarness) -> None:
    host = harness.host
    await host.begin("alice", "lid-demo")
    await host.begin("bob", "lid-demo")

    reply = await host.confirm_takeover("bob", "forged")

    assert reply.status == "error"
    assert host.session_of("alice") is not None


async def test_two_sessions_when_policy_allows(tmp_path: Path) -> None:
    h = await make_harness(
        tmp_path,
        capabilities=Capabilities(max_concurrent_runs=2),
        settings=HostSettings(max_concurrent_sessions=2, step_ack_timeout_s=0),
    )
    try:
        assert (await h.host.begin("alice", "lid-demo")).status == "started"
        assert (await h.host.begin("bob", "lid-demo")).status == "started"
        assert len(h.host.sessions()) == 2
    finally:
        await h.host.shutdown()
        await h.store.aclose()


async def test_restart_is_confirmed_unless_explicit(harness: HostHarness) -> None:
    host = harness.host
    await host.begin("alice", "lid-demo")
    for _ in range(3):
        await host.command("alice", RunCommand("next"))
    await settle()

    ask = await host.begin("alice", "lid-demo")
    assert ask.status == "ok" and "go through it again" in ask.message
    assert host.session_of("alice") is None

    started = await host.begin("alice", "lid-demo", explicit=True)
    assert started.status == "started"


async def test_stop_then_resume_picks_up_the_saved_step(harness: HostHarness) -> None:
    host = harness.host
    await host.begin("alice", "lid-demo")
    await host.command("alice", RunCommand("next"))
    stopped = await host.stop("alice", reason="wearer_request")
    assert "at step 2 of 3" in stopped.message

    reply = await host.begin("alice", "lid-demo", request="resume")

    assert reply.status == "started"
    assert harness.backend.runs[-1].step == 1
    assert harness.ports.texts("alice")[-1] == "Step 2 of 3: Lift the tray."


async def test_model_cannot_invent_a_resume(harness: HostHarness) -> None:
    host = harness.host
    reply = await host.begin(
        "alice", "lid-demo", entry_mode="resume", intent_quote="resume it",
        request="guide me through the lid",
    )

    assert reply.status == "ok"
    assert "start from the beginning" in reply.message
    assert host.session_of("alice") is None


async def test_out_of_range_step_is_said_not_clamped(harness: HostHarness) -> None:
    reply = await harness.host.begin("alice", "lid-demo", at_step=7)

    assert "no step 7" in reply.message
    assert harness.host.session_of("alice") is None


async def test_owner_leaving_interrupts_the_session(harness: HostHarness) -> None:
    host = harness.host
    await host.begin("alice", "lid-demo")
    await host.participant_left("alice")

    assert host.session_of("alice") is None
    assert harness.ports.states[-1]["outcome"] == "interrupted"
    assert ("stop", "alice", "") in harness.ports.previews


async def test_wearer_request_revision_replaces(tmp_path: Path) -> None:
    h = await make_harness(tmp_path)
    try:
        await h.host.begin("alice", "lid-demo")
        h.llm_replies.extend(["size zero nose pad", "size one nose pad", "NONE"])
        await h.host.extract_wearer_request("alice", "I want the size zero nose pad")
        await h.host.extract_wearer_request("alice", "actually size one nose pad")
        await h.host.extract_wearer_request("alice", "is it this one?")

        assert h.host.session_of("alice").wearer_requests == ["size one nose pad"]
    finally:
        await h.host.shutdown()
        await h.store.aclose()


async def test_input_participant_follows_the_selection(harness: HostHarness) -> None:
    host = harness.host
    await host.begin("alice", "lid-demo")
    await host.change_input("alice", "glasses")

    assert host.session_for_input("glasses") is host.session_of("alice")
    assert harness.ports.states[-1]["input_participant"] == "glasses"


async def test_a_spoken_correction_stays_in_the_state_until_the_step_is_done(
    harness: HostHarness,
) -> None:
    from sop_guidance.backends.base import Cue, StepChanged, Verdict

    host = harness.host
    await host.begin("alice", "lid-demo")
    session = host.session_of("alice")
    assert harness.ports.states[-1]["correction"] == {}

    await host._on_event(session, Cue("Use the size zero pad.", kind="correction"))
    correction = harness.ports.states[-1]["correction"]
    assert (correction["step"], correction["text"]) == (1, "Use the size zero pad.")
    # A hint is spoken but is not a correction.
    await host._on_event(session, Cue("Turn it over.", kind="hint"))
    assert harness.ports.states[-1]["correction"]["text"] == "Use the size zero pad."
    await host._on_event(session, Verdict({"completed": False}))
    assert harness.ports.states[-1]["correction"]["text"] == "Use the size zero pad."
    await host._on_event(session, Verdict({"completed": True}))
    assert harness.ports.states[-1]["correction"] == {}

    await host._on_event(session, Cue("Wrong pad.", kind="correction"))
    await host._on_event(session, StepChanged(1, reason="advance"))
    assert harness.ports.states[-1]["correction"] == {}
