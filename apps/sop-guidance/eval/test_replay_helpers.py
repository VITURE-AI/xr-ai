# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-python checks of the replay helpers: no models, no recorded data.

Run with ``worker/.venv/bin/python -m pytest eval`` from ``apps/sop-guidance``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from metrics import Outcome, cohen_kappa, percentile, spoken_violation, summarize, vetoed
from recordings import (
    classify_tier,
    find_sessions,
    load_session,
    sample_checks,
    spatial_context,
    wearer_requests,
)

_COMPARE_Q = (
    "INSTRUCTION: Pick up a pad.\nImage 1 is the teacher's completed reference state.\n"
    "WHAT THE WEARER ASKED FOR (during this procedure, oldest first). Where two differ:\n"
    '  - "the wire one"\n'
    '  - "size zero"   <- MOST RECENT: this is what they want now\n'
    "  These are REQUESTS, not evidence.\n\n"
    "DETECTOR GEOMETRY for the live student image (advisory, computed from the same boxes):\n"
    "- 1 nose pad box is inside a hand box.\nThese relations are box arithmetic.\n\n"
    "Decide whether Image 2 matches Image 1."
)
_LIVE_Q = "INSTRUCTION: Pick up a pad.\nTEACHER CAPTION: x\nLook ONLY at this live student image."


def _write_session(root: Path) -> Path:
    session = root / "20260101_000000_demo"
    for rel in ("step_01/call_0001/in_00.png", "step_01/call_0001/in_01.png",
                "step_01/call_0002/in_00.png", "step_02/call_0004/in_00.png",
                "step_02/call_0004/in_01.png"):
        path = session / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
    calls = [
        {"ts_us": 2_000_000, "seq": 1, "kind": "vlm", "name": "ask_frames", "step": 1,
         "latency_ms": 1000.0, "request": {"question": _COMPARE_Q,
                                           "image_paths": ["/t/teacher.png", "/t/s1.png"]},
         "response": "{}", "artifacts": ["step_01/call_0001/in_00.png",
                                         "step_01/call_0001/in_01.png"]},
        {"ts_us": 3_000_000, "seq": 2, "kind": "vlm", "name": "ask_image", "step": 1,
         "latency_ms": 900.0, "request": {"question": _LIVE_Q, "image_path": "/t/s1.png"},
         "response": "{}", "artifacts": ["step_01/call_0002/in_00.png"]},
        {"ts_us": 3_100_000, "seq": 3, "kind": "llm", "name": "guidance_turn", "step": 1,
         "latency_ms": 500.0, "request": {}, "response": "", "artifacts": []},
        {"ts_us": 6_000_000, "seq": 4, "kind": "vlm", "name": "ask_frames", "step": 2,
         "latency_ms": 1200.0, "request": {"question": _LIVE_Q.replace("Look", "Decide"),
                                           "image_paths": ["/t/teacher2.png", "/t/s2.png"]},
         "response": "{}", "artifacts": ["step_02/call_0004/in_00.png",
                                         "step_02/call_0004/in_01.png"]},
        {"ts_us": 9_000_000, "seq": 5, "kind": "vlm", "name": "ask_image", "step": 2,
         "latency_ms": 800.0, "request": {"question": _LIVE_Q, "image_path": "/t/orphan.png"},
         "response": "{}", "artifacts": ["step_02/call_0005/in_00.png"]},
    ]
    events = [
        {"ts_us": 1_000_000, "event": "STEP", "step": 1},
        {"ts_us": 3_200_000, "event": "CHECK", "step": 1, "completed": False,
         "has_evidence": False, "observation": "no pad", "issue": "Pick one up.",
         "frame": "/t/s1.png"},
        {"ts_us": 4_000_000, "event": "CHECK", "step": 1, "completed": True,
         "has_evidence": True, "observation": "pad", "issue": "", "frame": "/t/unrecorded.png"},
        {"ts_us": 6_500_000, "event": "CHECK", "step": 2, "completed": True,
         "has_evidence": True, "observation": "held", "issue": "", "frame": "/t/s2.png"},
    ]
    session.mkdir(parents=True, exist_ok=True)
    (session / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls) + "\n")
    (session / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    (session / "meta.json").write_text(json.dumps({"steps": 2}))
    return session


def test_tiers_are_classified_from_the_question_and_image_count() -> None:
    assert classify_tier("ask_frames", _COMPARE_Q, 2) == "compare"
    assert classify_tier("ask_image", _LIVE_Q, 1) == "live"
    assert classify_tier("ask_image", "The user is trying to do this step: x", 1) == "diagnosis"
    assert classify_tier("ask_image", "Describe the scene.", 1) == ""


def test_prompt_blocks_are_recovered() -> None:
    assert wearer_requests(_COMPARE_Q) == ("the wire one", "size zero")
    block = spatial_context(_COMPARE_Q)
    assert block.startswith("DETECTOR GEOMETRY") and block.endswith("arithmetic.\n\n")
    assert wearer_requests(_LIVE_Q) == () and spatial_context(_LIVE_Q) == ""


def test_calls_pair_with_the_check_naming_their_frame(tmp_path: Path) -> None:
    session = load_session(_write_session(tmp_path))
    assert [c.step for c in session.checks] == [1, 2]
    first, second = session.checks
    assert [c.tier for c in first.calls] == ["compare", "live"]
    assert not first.passed and first.issue == "Pick one up."
    assert first.student_image.name == "in_00.png" and first.student_image.parent.name == "call_0002"
    assert [p.name for p in first.teacher_images] == ["in_00.png"]
    assert first.wearer_requests == ("the wire one", "size zero")
    # 3.2 s event minus the first call's start (2.0 s end - 1.0 s latency).
    assert math.isclose(first.latency_ms, 2200.0)
    assert math.isclose(first.vlm_ms, 1900.0)
    assert second.passed and [c.tier for c in second.calls] == ["compare"]
    assert session.skipped == {"calls-not-recorded": 1, "orphan-calls": 1}


def test_find_sessions_expands_a_parent_folder(tmp_path: Path) -> None:
    path = _write_session(tmp_path)
    assert find_sessions([tmp_path]) == [path]
    assert find_sessions([path, tmp_path]) == [path]


def test_sampling_round_robins_steps(tmp_path: Path) -> None:
    session = load_session(_write_session(tmp_path))
    checks = session.checks * 3
    picked = sample_checks(checks, 2)
    assert sorted(c.step for c in picked) == [1, 2]
    assert len(sample_checks(checks, 0)) == len(checks)


def test_percentile_and_kappa() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert math.isclose(percentile(list(range(101)), 95), 95.0)
    assert math.isnan(percentile([], 50))
    assert cohen_kappa([(True, True), (False, False)]) == 1.0
    assert cohen_kappa([(True, False), (False, True)]) < 0


def test_summary_counts_confusion_and_decisive_vetoes() -> None:
    outcomes = [
        Outcome(key="a", session="s", step=1, old_passed=True, new_passed=True),
        Outcome(key="b", session="s", step=1, old_passed=True, new_passed=False,
                new_issue="veto!", geometry_veto="veto!"),
        Outcome(key="c", session="s", step=2, old_passed=False, new_passed=False,
                old_issue="pull it off", new_issue="pull it straight off", geometry_veto="gate"),
        Outcome(key="d", session="s", step=2, old_passed=False, new_passed=None, error="timeout"),
    ]
    summary = summarize(outcomes)
    assert summary["graded"] == 3 and summary["errors"] == 1
    assert math.isclose(summary["agreement"], 2 / 3)
    assert summary["confusion_by_step"][1] == {"pp": 1, "pf": 1, "fp": 0, "ff": 0, "agreement": 0.5}
    assert summary["geometry_vetoes_new"] == 1
    assert vetoed(outcomes[1]) and not vetoed(outcomes[2])


def test_spoken_violation_flags_frame_talk_but_not_the_glasses_frame() -> None:
    assert spoken_violation("It still matches Image 1.")
    assert spoken_violation("The student has not picked it up.")
    assert not spoken_violation("Push the pad into the slot on the frame.")
