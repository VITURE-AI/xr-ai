# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 5, the FPC ribbon cable, ported from the judge's ``fpc_state.py``.

Decided from the detector's ``fpc`` class by geometry: the fraction of the
cable box that lies inside the board box (IoB). A seated cable folds back
over the board, so its box sits wholly inside; an unseated one lies on the
desk or in a hand and sticks far out. Measured on glasses footage:

    before seating   IoB p50 0.57  p90 0.83   frames >= 0.95:  6.8%
    after seating    IoB p50 1.00  p10 0.87   frames >= 0.95: 84.3%

"Cable box covers the connector anchor" was tried and flickers throughout:
a cable lying on the desk often crosses the board edge the connector is on.

Hysteresis as for the holes: ``n_need`` passing frames in a ``m_win`` window
seat it; once seated, ``unseat_ticks`` consecutive failing frames unseat it; a
frame with no cable box abstains (a hand covering the cable is normal).
"""

from __future__ import annotations

from collections import deque
from typing import Any

IOB_SEATED = 0.95
"""The judge's own default; the glasses demo ran with 0.90."""


def iob(fpc_xyxy: list[float], board_xyxy: list[float]) -> float:
    """Intersection of the cable box with the board box, over the cable box's area."""
    ix1, iy1 = max(fpc_xyxy[0], board_xyxy[0]), max(fpc_xyxy[1], board_xyxy[1])
    ix2, iy2 = min(fpc_xyxy[2], board_xyxy[2]), min(fpc_xyxy[3], board_xyxy[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    area = (fpc_xyxy[2] - fpc_xyxy[0]) * (fpc_xyxy[3] - fpc_xyxy[1])
    return (ix2 - ix1) * (iy2 - iy1) / max(area, 1.0)


def frame_evidence(boxes: list[dict[str, Any]], board_xyxy: list[float] | None,
                   iob_min: float = IOB_SEATED) -> tuple[bool | None, float, int]:
    """``(passes, best IoB, cable boxes)``; ``passes`` is None when there is no evidence.

    The best IoB is taken: another cable misdetected in view must not hide the
    one lying on the board.
    """
    cables = [b for b in boxes if b["cls"] == "fpc"]
    if not cables or board_xyxy is None:
        return None, 0.0, 0
    best = max(iob(b["xyxy"], board_xyxy) for b in cables)
    return best >= iob_min, best, len(cables)


class FPCSeatState:
    """Hysteresis over the per-frame cable evidence, one instance per run."""

    def __init__(self, n_need: int = 10, m_win: int = 15, unseat_ticks: int = 30,
                 iob_min: float = IOB_SEATED) -> None:
        self.n_need = max(1, n_need)
        self.m_win = max(self.n_need, m_win)
        self.unseat_ticks = max(1, unseat_ticks)
        self.iob_min = iob_min
        self.buf: deque[bool] = deque(maxlen=self.m_win)
        self.seated = False
        self.opp = 0
        self.events: list[tuple[float, str, dict[str, Any]]] = []
        self.last_iob = 0.0

    def update(self, boxes: list[dict[str, Any]], board_xyxy: list[float] | None,
               t: float) -> bool:
        """Feed one trusted frame, once all four screws are confirmed; returns ``seated``."""

        ok, best, _n = frame_evidence(boxes, board_xyxy, self.iob_min)
        self.last_iob = best
        if ok is None:
            return self.seated
        if not self.seated:
            self.buf.append(bool(ok))
            if self.buf.count(True) >= self.n_need:
                self.seated = True
                self.buf.clear()
                self.opp = 0
                self.events.append((t, "fpc_seated", {"iob": best}))
            return self.seated
        if ok:
            self.opp = 0
        else:
            self.opp += 1
            if self.opp >= self.unseat_ticks:
                self.seated = False
                self.opp = 0
                self.buf.clear()
                self.events.append((t, "fpc_rework", {"iob": best}))
        return self.seated

    def reset(self) -> None:
        self.buf.clear()
        self.seated = False
        self.opp = 0
        self.last_iob = 0.0
        self.events.clear()


__all__ = ["FPCSeatState", "frame_evidence", "iob"]
