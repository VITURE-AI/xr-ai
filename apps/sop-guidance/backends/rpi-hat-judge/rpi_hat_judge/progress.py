# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracker output to "which step is the wearer on", ported from the judge's ``Guidance``.

This decides nothing about completion: done, the step count and every event
come from :class:`~.holes.PerHoleTracker`. It only picks the step to show,
which steps count as completed, and which events a wearer should hear.

The step shown is the step of the NEXT hole to fill, not ``k + 1``. With
holes 1 and 3 in, k is 2 but the step to do is 2 (the bottom-right screw);
the old HUD mixed the two once and titled step 3 with step 2's name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .board import SOP_SEQUENCE
from .spec import JudgeSpec
from .texts import OPERATOR_EVENTS


@dataclass(slots=True)
class Tick:
    """What one tracker update means for the wearer."""

    active: int | None
    """0-based index of the step shown, or None once done."""

    completed: list[int] = field(default_factory=list)
    outstanding: list[int] = field(default_factory=list)
    """Skipped steps: a later hole is filled while this one is not."""

    newly_completed: list[int] = field(default_factory=list)
    alerts: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    """Operator events raised this tick, oldest first."""

    done: bool = False
    finished_now: bool = False
    """The done latch closed on this tick."""


class Progress:
    """Stateful per run; call :meth:`update` exactly once per detection tick."""

    def __init__(self, spec: JudgeSpec, hold_display: float = 1.5) -> None:
        self.spec = spec
        self.hold_display = hold_display
        self.completed_prev: list[int] = []
        self.ev_seen = 0
        self.shown: int | None = None
        self.t_shown: float | None = None
        self.done = False

    def _step_list(self, holes: dict[int, str]) -> tuple[list[int], list[int]]:
        """``(completed, outstanding)`` by position, not by count."""

        done, owed = [], []
        installed = [holes.get(h) == "installed" for h in SOP_SEQUENCE]
        for i, hole in enumerate(SOP_SEQUENCE):
            index = self.spec.index_of_hole(hole)
            if installed[i]:
                done.append(index)
            elif any(installed[i + 1:]):
                owed.append(index)
        return done, owed

    def update(self, res: dict[str, Any], events: list[tuple[float, str, dict[str, Any]]],
               t: float) -> Tick:
        holes = res.get("holes", {})
        next_hole = res.get("next_hole")
        done_shown = bool(res.get("done_shown"))
        completed, outstanding = self._step_list(holes)
        if done_shown:
            completed = list(range(self.spec.total))

        alerts = [(kind, fields) for _t, kind, fields in events[self.ev_seen:]
                  if kind in OPERATOR_EVENTS]
        self.ev_seen = len(events)

        newly = [i for i in completed if i not in self.completed_prev]
        self.completed_prev = completed
        finished_now = done_shown and not self.done
        self.done = self.done or done_shown

        if done_shown:
            target = None
        elif next_hole is not None:
            target = self.spec.index_of_hole(next_hole)
        elif holes and all(holes.get(h) == "installed" for h in SOP_SEQUENCE):
            target = self.spec.fpc_index
        else:
            # A missing next hole is not "all four in". Point at the first
            # step: the safe side, never the last.
            target = 0
        # Minimum hold after a change, against advance/regress bouncing.
        if target != self.shown and (self.t_shown is None or target is None or self.shown is None
                                     or t - self.t_shown >= self.hold_display):
            self.shown, self.t_shown = target, t

        return Tick(active=self.shown, completed=completed, outstanding=outstanding,
                    newly_completed=newly, alerts=alerts, done=self.done,
                    finished_now=finished_now)


__all__ = ["Progress", "Tick"]
