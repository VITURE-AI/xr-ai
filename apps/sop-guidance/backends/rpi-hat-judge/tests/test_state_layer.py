# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The old judge's self-tests (step_state.py, fpc_state.py), as pytest cases."""

from __future__ import annotations

from rpi_hat_judge.board import HOLE_ANCHORS
from rpi_hat_judge.fpc import FPCSeatState, iob
from rpi_hat_judge.holes import PerHoleTracker, StepDisplay

BOARD = [100.0, 100.0, 500.0, 380.0]
HZ = 10.0


def mk(installed=(), unseen=(), loose=0, filled_on=(), no_board=False):
    """One frame of boxes placed on the anchors."""
    bx, by = (BOARD[0] + BOARD[2]) / 2, (BOARD[1] + BOARD[3]) / 2
    bw, bh = BOARD[2] - BOARD[0], BOARD[3] - BOARD[1]
    boxes = [] if no_board else [{"cls": "board", "xyxy": list(BOARD), "conf": 0.97}]
    for k, (rx, ry) in HOLE_ANCHORS.items():
        if k in unseen:
            continue
        ax, ay = bx + rx * bw, by + ry * bh
        cls = "hole_filled" if (k in installed or k in filled_on) else "hole"
        boxes.append({"cls": cls, "xyxy": [ax - 8, ay - 8, ax + 8, ay + 8], "conf": 0.9})
    for i in range(loose):
        boxes.append({"cls": "screw", "xyxy": [600 + 30 * i, 500, 616 + 30 * i, 516], "conf": 0.88})
    return boxes


class V:
    def __init__(self, ok=True, reason=""):
        self.ok, self.reason = ok, reason


def run(frames, fpc_at=None):
    tracker = PerHoleTracker(n_fwd=5, n_back=15, m=7, need=5, slot_back_sec=3.0, tol=2,
                             hold_sec=1.5)
    last = None
    for i, frame in enumerate(frames):
        if fpc_at is not None and i >= fpc_at:
            tracker.fpc_done = True
        verdict = V() if any(b["cls"] == "board" for b in frame) else V(False, "no board")
        last = tracker.update(frame, verdict, i / HZ)
    return last, [(kind, fields) for _t, kind, fields in tracker.events]


def kinds(events):
    return [k for k, _ in events]


def test_normal_progress_through_all_four_and_the_cable():
    seq = [mk()] * 12
    for up_to in (1, 2, 3, 4):
        seq += [mk(installed=range(1, up_to + 1))] * 12
    last, events = run(seq, fpc_at=55)
    assert last["step"] == 5 and last["done"]
    assert kinds(events).count("advance") == 4
    assert not {"jump", "regress", "order_violation"} & set(kinds(events))


def test_a_one_frame_flicker_is_absorbed_by_the_hole_vote():
    seq = [mk(installed=[1])] * 12 + [mk(installed=[1, 2])] + [mk(installed=[1])] * 12
    last, events = run(seq)
    assert last["stable_filled"] == 1
    assert ("screw_installed", {"hole": 2}) not in events


def test_two_screws_in_while_occluded_jump_with_an_alert():
    seq = [mk(installed=[1])] * 12 + [mk(no_board=True)] * 10 + [mk(installed=[1, 2, 3])] * 26
    last, events = run(seq)
    assert last["stable_filled"] == 3
    assert "jump" in kinds(events)


def test_a_removed_screw_needs_three_seconds_of_evidence():
    seq = [mk(installed=[1, 2])] * 14 + [mk(installed=[1])] * 48
    last, events = run(seq)
    assert last["stable_filled"] == 1
    assert "rework" in kinds(events) and "regress" in kinds(events)

    seq = [mk(installed=[1, 2])] * 14 + [mk(installed=[1])] * 8 + [mk(installed=[1, 2])] * 14
    last, events = run(seq)
    assert last["stable_filled"] == 2
    assert "rework" not in kinds(events)


def test_a_filled_fixed_hole_voids_the_frame_when_no_sop_hole_is_free():
    seq = [mk(installed=[1, 2, 3])] * 14 + [mk(installed=[1, 2, 3], filled_on=[5])] * 10
    last, events = run(seq)
    assert not last["trusted"]
    assert any(k == "sentinel" and "fixed hole" in f["reason"] for k, f in events)


def test_an_ambiguous_filled_box_abstains_rather_than_faking_progress():
    seq = [mk(installed=[1])] * 12 + [mk(installed=[1, 2], filled_on=[5])] * 12
    last, events = run(seq)
    assert last["holes"][3] != "installed"
    assert ("screw_installed", {"hole": 3}) not in events


def test_too_many_loose_screws_void_the_frame():
    seq = [mk(installed=[1], loose=3)] * 12 + [mk(installed=[1], loose=4)] * 8
    last, events = run(seq)
    assert not last["trusted"] and "loose screws" in last["reason"]
    assert any(k == "sentinel" and "loose screws" in f["reason"] for k, f in events)


def test_out_of_order_raises_a_violation_and_points_back():
    seq = [mk()] * 12 + [mk(installed=[3])] * 14
    last, events = run(seq)
    assert ("order_violation", {"hole": 3, "expected": 1}) in events
    assert last["next_hole"] == 1


def test_display_hold_and_done_latch():
    d = StepDisplay(hold_sec=1.5)
    outs = [d.update(1, False, 0.0), d.update(2, False, 0.1), d.update(2, False, 2.0),
            d.update(5, True, 3.0), d.update(1, False, 9.0)]
    assert [o[0] for o in outs[:3]] == [1, 1, 2]
    assert outs[3] == (5, True) and outs[4] == (5, True)


def test_reset_clears_events_so_nothing_is_replayed():
    tracker = PerHoleTracker(n_fwd=5, n_back=15)
    for i in range(26):
        tracker.update(mk() if i < 12 else mk(installed=[3]), V(), i / HZ)
    assert tracker.events
    tracker.reset()
    assert tracker.events == [] and tracker.machine.k == 0


# ── the cable ────────────────────────────────────────────────────────────────


def cable(iob_target):
    if iob_target >= 0.999:
        box = [150.0, 150.0, 300.0, 250.0]
    else:
        inside = 200.0 * iob_target
        x1 = BOARD[2] - inside
        box = [x1, 150.0, x1 + 200.0, 250.0]
    return [{"cls": "board", "xyxy": list(BOARD), "conf": 0.95},
            {"cls": "fpc", "xyxy": box, "conf": 0.9}]


def test_cable_seats_on_the_tenth_passing_frame():
    st = FPCSeatState(n_need=10, m_win=15, unseat_ticks=30)
    for i in range(9):
        st.update(cable(1.0), BOARD, i / 10.0)
    assert not st.seated
    st.update(cable(1.0), BOARD, 1.0)
    assert st.seated


def test_sporadic_passes_do_not_seat_the_cable():
    st = FPCSeatState(n_need=10, m_win=15, unseat_ticks=30)
    for i in range(60):
        st.update(cable(1.0 if i % 12 == 0 else 0.5), BOARD, i / 10.0)
    assert not st.seated


def test_a_seated_cable_needs_sustained_evidence_to_unseat():
    st = FPCSeatState(n_need=5, m_win=7, unseat_ticks=30)
    for i in range(5):
        st.update(cable(1.0), BOARD, i / 10.0)
    for i in range(20):
        st.update(cable(0.4), BOARD, (5 + i) / 10.0)
    assert st.seated
    for i in range(12):
        st.update(cable(0.4), BOARD, (25 + i) / 10.0)
    assert not st.seated and any(k == "fpc_rework" for _t, k, _f in st.events)


def test_a_hidden_cable_abstains_without_clearing_votes():
    st = FPCSeatState(n_need=5, m_win=7, unseat_ticks=30)
    for i in range(4):
        st.update(cable(1.0), BOARD, i / 10.0)
    for i in range(20):
        st.update([{"cls": "board", "xyxy": list(BOARD), "conf": 0.9}], BOARD, (4 + i) / 10.0)
    assert not st.seated
    st.update(cable(1.0), BOARD, 3.0)
    assert st.seated


def test_iob():
    assert abs(iob([150, 150, 300, 250], BOARD) - 1.0) < 1e-6
    assert iob([600, 600, 700, 700], BOARD) == 0.0
    assert abs(iob([400, 150, 600, 250], BOARD) - 0.5) < 0.01
