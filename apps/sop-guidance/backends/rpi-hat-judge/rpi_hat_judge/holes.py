# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-hole state machine, ported from the judge's ``step_state.py``.

Three stages of noise suppression, so one hole flickering for a frame cannot
move the step:

    slot matching   hole / hole_filled boxes go to one of the six anchors;
                    boxes that match none, or a taken one, are dropped
    per-hole vote   each hole flips on a sliding-window majority (need of
                    window); an unseen hole abstains instead of voting wrong;
                    installed -> empty (rework) also needs back_sec of
                    sustained contrary evidence
    step confirm    k = screws installed among holes 1-4; a new k needs n_fwd
                    consecutive trusted ticks to move forward one, n_back to
                    jump or go back (rare and costly, so stricter)

Two frame-level sentinels void a whole frame without voting:

    hole 5 or 6 reported filled   they never get a screw, so the detector is
                                  wrong about this frame; sustained, it is an event
    too many loose screws         loose > 4 - k; fewer is normal (in hand, out of
                                  view), more means one came out or a false box

Events are ``(t, kind, fields)`` tuples with plain fields, not sentences; the
wording lives in :mod:`.texts`.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from .board import FIXED_SLOTS, HOLE_ANCHORS, SOP_SEQUENCE

ANCHOR_TOL = 0.16
"""Match radius as a fraction of the board's long side."""

AMBIG_RATIO = 1.4
"""How much nearer a SOP hole or a fixed hole must be for "installed" evidence
to count as that one. Holes 3/5 are 0.120 apart and 2/6 0.168, both inside the
match radius, so without this the two pairs always cross."""

INSTALLED_TOL = 0.12
"""A stricter radius for "installed" evidence, whose mistakes fake progress.
Distance from a hole_filled box to its nearest anchor, over the board's long
side: really installed p50 0.014-0.038, p75 0.024-0.084; false positives p50
0.051, p90 0.127. With conf >= 0.80, 0.12 kept 95.8% of real ones and 2.7% of
false ones."""

SCREW_AS_INSTALLED = False
"""Whether a screw on a hole counts as installed. Off since v7: hole_filled is
reliable on its own, and v7's fuller screw recall turned screws in hand or on
the desk into installed holes, which then got "removed"."""

Box = dict[str, Any]
Event = tuple[float, str, dict[str, Any]]


def _centre(box: Box) -> tuple[float, float]:
    b = box["xyxy"]
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def match_slots(
    boxes: list[Box],
    board_xyxy: list[float],
    anchor_px: dict[int, tuple[float, float]] | None = None,
    hf_conf_min: float = 0.0,
    installed_tol: float = INSTALLED_TOL,
    screw_as_installed: bool = SCREW_AS_INSTALLED,
) -> tuple[dict[int, str], int, dict[int, tuple[float, float]], int]:
    """Assign hole, hole_filled and screw boxes to the six anchor slots.

    Returns ``(obs, loose, anchor_px, n_ambig)``: this frame's observation per
    hole (``installed`` / ``empty`` / ``unseen``), screws on no slot, the
    anchors in pixels, and how many "installed" boxes abstained because a SOP
    hole and a fixed hole were equally close.
    """

    x1, y1, x2, y2 = board_xyxy
    bw, bh = x2 - x1, y2 - y1
    bx, by = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    tol = ANCHOR_TOL * max(bw, bh)
    tol_inst = installed_tol * max(bw, bh)
    if anchor_px is None:
        anchor_px = {k: (bx + rx * bw, by + ry * bh) for k, (rx, ry) in HOLE_ANCHORS.items()}

    def dist(cx: float, cy: float, k: int) -> float:
        ax, ay = anchor_px[k]
        return ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5

    def nearest_free(cx: float, cy: float, obs: dict[int, str],
                     only: tuple[int, ...] | None = None,
                     tol_use: float | None = None) -> int | None:
        best, bd = None, 1e18
        for k in anchor_px:
            if obs[k] != "unseen" or (only is not None and k not in only):
                continue
            d = dist(cx, cy, k)
            if d < bd:
                best, bd = k, d
        return best if bd < (tol if tol_use is None else tol_use) else None

    def nearest_installed(cx: float, cy: float, obs: dict[int, str]) -> tuple[int | None, bool]:
        """Slot for "installed" evidence; returns ``(slot, ambiguous)``.

        Holes 5 and 6 never get a screw, so a hole_filled box near 5 is almost
        certainly hole 3's screw carried by anchor drift. It still must not be
        handed to hole 3 outright: if the detector really did invent a filled
        box on 5, that would fake progress on 3, a worse failure than voiding
        the frame. So: clearly nearer the fixed hole -> the fixed hole (the
        sentinel fires); clearly nearer the SOP hole -> the SOP hole; neither ->
        abstain, which fakes no progress and voids nothing.
        """
        k_sop = nearest_free(cx, cy, obs, only=SOP_SEQUENCE, tol_use=tol_inst)
        k_fix = nearest_free(cx, cy, obs, only=FIXED_SLOTS, tol_use=tol_inst)
        if k_fix is None:
            return k_sop, False
        if k_sop is None:
            return k_fix, False
        d_sop, d_fix = dist(cx, cy, k_sop), dist(cx, cy, k_fix)
        if d_fix * AMBIG_RATIO < d_sop:
            return k_fix, False
        if d_sop * AMBIG_RATIO < d_fix:
            return k_sop, False
        return None, True

    obs = {k: "unseen" for k in HOLE_ANCHORS}
    n_ambig = 0
    by_cls: dict[str, list[Box]] = {"hole_filled": [], "hole": [], "screw": []}
    for b in boxes:
        if b["cls"] in by_cls:
            by_cls[b["cls"]].append(b)
    slot_conf: dict[int, float] = {}

    # Empty holes claim their slots first. Holes 5 and 6 are normally seen
    # empty, so once they are taken a filled box between 2 and 6 can only be 2.
    # With installed-first ordering the abstain rule fired on 90% of hole 2's
    # evidence and hole 2 never collected enough votes.
    for b in sorted(by_cls["hole"], key=lambda b: -b.get("conf", 0.0)):
        k = nearest_free(*_centre(b), obs)
        if k is not None:
            obs[k] = "empty"
            slot_conf[k] = b.get("conf", 0.0)

    for b in sorted(by_cls["hole_filled"], key=lambda b: -b.get("conf", 0.0)):
        # hole_filled's own confidence floor: its false positives fake progress,
        # and they separate on confidence.
        if b.get("conf", 0.0) < hf_conf_min:
            continue
        k, ambig = nearest_installed(*_centre(b), obs)
        if k is not None:
            obs[k] = "installed"
            slot_conf[k] = b.get("conf", 0.0)
            continue
        if ambig:
            n_ambig += 1
            continue
        # No free slot: may it take one an empty-hole box holds? They are two
        # exclusive readings of one hole, and the more confident one wins (in
        # 27 of 30 contested frames the hole box was the more confident).
        cx, cy = _centre(b)
        k2, d2 = None, None
        for kk, st in obs.items():
            if st != "empty":
                continue
            d = dist(cx, cy, kk)
            if d < tol_inst and (d2 is None or d < d2):
                k2, d2 = kk, d
        if k2 is not None and b.get("conf", 0.0) > slot_conf.get(k2, 1.0):
            obs[k2] = "installed"
            slot_conf[k2] = b.get("conf", 0.0)

    loose = 0
    for b in sorted(by_cls["screw"], key=lambda b: -b.get("conf", 0.0)):
        k, ambig = nearest_installed(*_centre(b), obs)
        if not screw_as_installed:
            # A screw on a hole is not loose on the desk; counting it would
            # trip the loose-screw sentinel.
            if k is None and not ambig:
                loose += 1
            continue
        if k is not None:
            obs[k] = "installed"
        elif ambig:
            n_ambig += 1
        else:
            loose += 1
    return obs, loose, anchor_px, n_ambig


class SlotVoter:
    """One hole's sliding-window majority vote.

    ``unseen`` abstains; a flip clears the votes (hysteresis); installed ->
    empty also needs the contrary evidence to last ``back_sec``.
    """

    def __init__(self, m: int = 7, need: int = 5, back_sec: float = 3.0) -> None:
        self.m, self.need, self.back_sec = m, need, back_sec
        self.state = "empty"
        self.buf: deque[str] = deque(maxlen=m)
        self.opp_since: float | None = None

    def observe(self, obs: str, t: float) -> str | None:
        """Feed one observation; returns ``installed`` / ``removed`` on a flip."""

        if obs == "unseen":
            return None
        self.buf.append(obs)
        if obs == self.state:
            self.opp_since = None  # one agreeing frame breaks the contrary run
        elif self.opp_since is None:
            self.opp_since = t
        opp = "installed" if self.state == "empty" else "empty"
        if self.buf.count(opp) < self.need:
            return None
        if opp == "empty" and (self.opp_since is None or t - self.opp_since < self.back_sec):
            return None
        self.state = opp
        self.buf.clear()
        self.opp_since = None
        return "installed" if opp == "installed" else "removed"


class StepMachine:
    """k (screws installed) confirmed asymmetrically: +1 on n_fwd ticks, else n_back."""

    def __init__(self, n_fwd: int = 5, n_back: int = 15) -> None:
        self.n_fwd, self.n_back = n_fwd, n_back
        self.k = 0
        self.cand: int | None = None
        self.cnt = 0

    def reset_cand(self) -> None:
        self.cand, self.cnt = None, 0

    def update(self, k_obs: int) -> tuple[str, int, int] | None:
        """Feed one trusted tick's k; returns ``(kind, old, new)`` when k moves."""

        if k_obs == self.k:
            self.reset_cand()
            return None
        if k_obs == self.cand:
            self.cnt += 1
        else:
            self.cand, self.cnt = k_obs, 1
        need = self.n_fwd if self.cand == self.k + 1 else self.n_back
        if self.cnt < need:
            return None
        kind = ("advance" if self.cand == self.k + 1
                else "jump" if self.cand > self.k + 1 else "regress")
        old, self.k = self.k, self.cand
        self.reset_cand()
        return kind, old, self.k


class StepDisplay:
    """A minimum hold between step changes, and a done latch only a reset clears."""

    def __init__(self, hold_sec: float = 1.5) -> None:
        self.hold_sec = hold_sec
        self.shown: int | None = None
        self.t_changed: float | None = None
        self.done = False

    def update(self, step: int, done: bool, t: float) -> tuple[int | None, bool]:
        if self.done:
            return self.shown, True
        if done:
            self.done, self.shown = True, step
            return self.shown, True
        if step != self.shown and (self.t_changed is None or t - self.t_changed >= self.hold_sec):
            self.shown, self.t_changed = step, t
        return self.shown, False

    def reset(self) -> None:
        self.shown, self.t_changed, self.done = None, None, False


class PerHoleTracker:
    """The per-hole state layer.

    ``n_fwd`` / ``n_back`` / ``tol`` are in detection ticks; the caller turns
    seconds into ticks at its tick rate.
    """

    def __init__(self, n_fwd: int = 5, n_back: int = 15, m: int = 7, need: int = 5,
                 slot_back_sec: float = 3.0, tol: int = 2, screws_expected: int | None = None,
                 hold_sec: float = 1.5, hf_conf_min: float = 0.0,
                 installed_tol: float = INSTALLED_TOL,
                 screw_as_installed: bool = SCREW_AS_INSTALLED) -> None:
        self.n_fwd = n_fwd
        self.machine = StepMachine(n_fwd, n_back)
        self.voters = {k: SlotVoter(m, need, slot_back_sec) for k in SOP_SEQUENCE}
        self.display = StepDisplay(hold_sec)
        self.tol = tol
        self.screws_expected = screws_expected if screws_expected is not None else len(SOP_SEQUENCE)
        self.miss = 0
        self.sentinel_hits = 0
        self.sentinel_fired = False
        self.fpc_done = False
        self.events: list[Event] = []
        self.hole_px: dict[int, tuple[float, float]] = {}
        self.n_ambig = 0
        self.hf_conf_min = hf_conf_min
        self.installed_tol = installed_tol
        self.screw_as_installed = screw_as_installed

    def _emit(self, t: float, kind: str, **fields: Any) -> None:
        self.events.append((t, kind, fields))

    def _k(self) -> int:
        return sum(1 for k in SOP_SEQUENCE if self.voters[k].state == "installed")

    def _expected_next(self) -> int | None:
        return next((k for k in SOP_SEQUENCE if self.voters[k].state != "installed"), None)

    def _sentinel(self, obs: dict[int, str], loose: int, t: float) -> str | None:
        """Frame-level sentinel: a reason voids the frame; sustained, it becomes an event."""

        bad = [k for k in FIXED_SLOTS if obs[k] == "installed"]
        expected_loose = self.screws_expected - self.machine.k
        if bad:
            reason = ("fixed hole %s reported as filled (false-positive sentinel)"
                      % "/".join(map(str, bad)))
        elif loose > expected_loose:
            reason = (f"{loose} loose screws, expected at most {expected_loose} "
                      "(removed, or a false positive)")
        else:
            self.sentinel_hits, self.sentinel_fired = 0, False
            return None
        self.sentinel_hits += 1
        if self.sentinel_hits >= self.n_fwd and not self.sentinel_fired:
            self._emit(t, "sentinel", reason=reason)
            self.sentinel_fired = True
        return reason

    def update(self, boxes: list[Box], verdict: Any, t: float) -> dict[str, Any]:
        """*boxes*: smoothed detections, board included; *verdict*: an OcclusionVerdict."""

        counts = {c: 0 for c in ("board", "hole", "screw", "hole_filled")}
        for b in boxes:
            if b["cls"] in counts:
                counts[b["cls"]] += 1
        if verdict is not None:
            trusted, reason = verdict.ok, verdict.reason
        else:
            trusted = counts["board"] == 1
            reason = "" if trusted else ("no board detected" if counts["board"] == 0
                                         else f"{counts['board']} boards detected (false positives)")
        obs: dict[int, str] = {}
        if trusted:
            board = next(b for b in boxes if b["cls"] == "board")
            obs, loose, self.hole_px, self.n_ambig = match_slots(
                boxes, board["xyxy"], hf_conf_min=self.hf_conf_min,
                installed_tol=self.installed_tol, screw_as_installed=self.screw_as_installed)
            sentinel = self._sentinel(obs, loose, t)
            if sentinel is not None:
                trusted, reason = False, sentinel

        if trusted:
            self.miss = 0
            for k in SOP_SEQUENCE:  # 5 and 6 never vote; they only feed the sentinel
                ev = self.voters[k].observe(obs[k], t)
                if ev == "installed":
                    expected = next((s for s in SOP_SEQUENCE
                                     if s != k and self.voters[s].state != "installed"
                                     and SOP_SEQUENCE.index(s) < SOP_SEQUENCE.index(k)), None)
                    self._emit(t, "screw_installed", hole=k)
                    if expected is not None:
                        self._emit(t, "order_violation", hole=k, expected=expected)
                elif ev == "removed":
                    self._emit(t, "rework", hole=k)
            moved = self.machine.update(self._k())
            if moved is not None:
                kind, old, new = moved
                self._emit(t, kind, old=old, new=new)
        else:
            # 43% of misses last one frame; clearing the candidate on every
            # miss would mean confirmation almost never completes.
            self.miss += 1
            if self.miss > self.tol:
                self.machine.reset_cand()

        k = self.machine.k
        step = min(k + 1, len(SOP_SEQUENCE) + 1)
        done = bool(step == len(SOP_SEQUENCE) + 1 and self.fpc_done)
        step_shown, done_shown = self.display.update(step, done, t)
        return {
            "trusted": trusted, "reason": reason,
            "raw": (counts["board"], counts["hole"], counts["screw"], counts["hole_filled"]),
            "stable_filled": k, "step": step, "done": done,
            "holes": {kk: ("installed" if kk in SOP_SEQUENCE
                           and self.voters[kk].state == "installed"
                           else ("fixed" if kk in FIXED_SLOTS else "empty"))
                      for kk in HOLE_ANCHORS},
            "hole_px": dict(self.hole_px),
            "next_hole": self._expected_next(),
            "step_shown": step_shown, "done_shown": done_shown,
        }

    def reset(self) -> None:
        self.machine = StepMachine(self.machine.n_fwd, self.machine.n_back)
        for voter in self.voters.values():
            voter.state, voter.opp_since = "empty", None
            voter.buf.clear()
        self.display.reset()
        self.miss = self.sentinel_hits = 0
        self.sentinel_fired = self.fpc_done = False
        # The old judge kept these across a reset and replayed stale alerts.
        self.events.clear()
        self.hole_px, self.n_ambig = {}, 0


__all__ = ["PerHoleTracker", "SlotVoter", "StepDisplay", "StepMachine", "match_slots"]
