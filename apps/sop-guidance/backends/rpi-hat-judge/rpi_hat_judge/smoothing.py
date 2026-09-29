# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame-to-frame smoothing, ported from the judge's ``smoothing.py``.

Turns per-frame detections into observations that are continuous in time,
before the occlusion gate reads them. The detector jitters three ways, measured
on 13346 frames:

    board missed          43% for 1 frame, 75% <= 4, 82% <= 6    hold the last box briefly
                          the 23 gaps >= 40 frames were the board really leaving view
    >= 2 boards found     53% for 1 frame, 82% <= 4              claim one by IoU with the track
    hole count jumps +-1  |d| <= 1 in 95.7% of frames            sliding median

Adjacent boards overlap with IoU median 0.982, 1st percentile 0.807, so the
0.50 claim gate has a wide margin. Every duration is in seconds, not frames.

A box is a dict: ``{"cls": str, "xyxy": [x1, y1, x2, y2], "conf": float}``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

HOLD_SEC = 0.20
"""How long a missed board keeps its last box."""

IOU_GATE = 0.50
"""IoU with the track a candidate needs to be claimed as the same board."""

WIN = 5
"""Sliding-median window (odd) for hole and screw counts and board confidence."""

EMA_ALPHA = 0.5
"""Exponential smoothing of the board box coordinates; 1.0 disables it."""

PREFER_HOLES = True
"""With several boards, keep the ones with the most holes inside first."""

_HOLE_LIKE = ("hole", "hole_filled", "screw")

Box = dict[str, Any]


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1.0)


def median(xs: deque[float] | list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


def _inside(box: Box, board: list[float]) -> bool:
    cx = (box["xyxy"][0] + box["xyxy"][2]) / 2
    cy = (box["xyxy"][1] + box["xyxy"][3]) / 2
    return board[0] <= cx <= board[2] and board[1] <= cy <= board[3]


@dataclass
class SmoothedFrame:
    """One smoothed frame; field names match what the occlusion gate takes."""

    boxes: list[Box] = field(default_factory=list)
    """The board replaced by its smoothed (or held) box, then every other box."""

    hand_boxes: list[list[float]] = field(default_factory=list)
    counts: dict[str, float] = field(default_factory=lambda: {
        "board": 0, "hole": 0, "screw": 0, "hole_filled": 0})
    visible_holes: float = 0
    """Median of holes (hole + hole_filled + screw) inside the board."""

    board_conf: float = 0.0
    held: bool = False
    """The board box was held over from an earlier frame, not detected in this one."""

    hold_age: float = 0.0
    raw_boards: int = 0


class BoardTrack:
    """One board's track: claim, smooth, and briefly hold.

    Only one board is tracked; in the task there is one, and extra boxes are
    false positives. The first criterion for claiming a candidate is how many
    holes lie inside it. On first-person glasses footage 12.4% of frames held
    two or more boards, nearly always the real board plus a checked shirt cuff
    or a watch, and the false box often out-scored the real one on confidence
    (up to 0.81). A cuff box holds no holes; the real board holds 3-6. When
    every candidate holds none (both hands covering), IoU and confidence decide.
    """

    def __init__(self, hold_sec: float = HOLD_SEC, iou_gate: float = IOU_GATE,
                 alpha: float = EMA_ALPHA, prefer_holes: bool = PREFER_HOLES) -> None:
        self.hold_sec, self.iou_gate, self.alpha = hold_sec, iou_gate, alpha
        self.prefer_holes = prefer_holes
        self.box: list[float] | None = None
        self.conf = 0.0
        self.t_seen: float | None = None

    @staticmethod
    def _holes_inside(board: list[float], others: list[Box]) -> int:
        return sum(1 for b in others if b["cls"] in _HOLE_LIKE and _inside(b, board))

    def update(self, cands: list[Box], t: float,
               others: list[Box] | None = None) -> tuple[list[float] | None, float, bool]:
        """Claim this frame's board; returns ``(box, conf, held)``."""

        if self.prefer_holes and cands and others and len(cands) > 1:
            scored = [(self._holes_inside(b["xyxy"], others), b) for b in cands]
            top = max(n for n, _ in scored)
            if top > 0:
                cands = [b for n, b in scored if n == top]
        pick = None
        if cands:
            if self.box is None:
                pick = max(cands, key=lambda b: b.get("conf", 0.0))
            else:
                best_iou, best = max(((iou(self.box, b["xyxy"]), b) for b in cands),
                                     key=lambda s: s[0])
                # A board that no longer matches the track (moved away and back)
                # starts a new track rather than being forced onto the old one.
                if best_iou >= self.iou_gate:
                    pick = best
                else:
                    pick = max(cands, key=lambda b: b.get("conf", 0.0))
                    self.box = None

        if pick is not None:
            nb = list(pick["xyxy"])
            if self.box is not None and self.alpha < 1.0:
                nb = [self.alpha * n + (1 - self.alpha) * o for n, o in zip(nb, self.box, strict=True)]
            self.box, self.conf, self.t_seen = nb, float(pick.get("conf", 0.0)), t
            return self.box, self.conf, False

        if self.box is not None and self.t_seen is not None and t - self.t_seen <= self.hold_sec:
            return self.box, self.conf, True

        self.box, self.conf, self.t_seen = None, 0.0, None
        return None, 0.0, False


class TemporalSmoother:
    """Detections to smoothed observations, fed once per detection tick (seconds)."""

    def __init__(self, hold_sec: float = HOLD_SEC, win: int = WIN, iou_gate: float = IOU_GATE,
                 alpha: float = EMA_ALPHA, prefer_holes: bool = PREFER_HOLES) -> None:
        self.track = BoardTrack(hold_sec, iou_gate, alpha, prefer_holes)
        self.hold_sec = hold_sec
        self.win = max(1, win)
        self.nvis_buf: deque[float] = deque(maxlen=self.win)
        self.conf_buf: deque[float] = deque(maxlen=self.win)
        self.cnt_buf = {c: deque(maxlen=self.win) for c in _HOLE_LIKE}
        self.hands: list[list[float]] = []
        self.t_hands: float | None = None

    def update(self, dets: list[Box], hand_boxes: list[list[float]] | None = None,
               t: float = 0.0) -> SmoothedFrame:
        cands = [b for b in dets if b["cls"] == "board"]
        others = [b for b in dets if b["cls"] != "board"]
        box, conf, held = self.track.update(cands, t, others)

        out = SmoothedFrame(hand_boxes=[], raw_boards=len(cands), held=held)
        out.hold_age = 0.0 if not held or self.track.t_seen is None else t - self.track.t_seen

        # Hands drop out now and then too, and are held the same way. Holding a
        # hand only makes the gate more conservative.
        if hand_boxes:
            self.hands, self.t_hands = list(hand_boxes), t
        elif self.t_hands is not None and t - self.t_hands <= self.hold_sec:
            pass
        else:
            self.hands = []
        out.hand_boxes = self.hands

        if box is None:
            # No board: sample nothing, so the gap's zeros do not poison the medians.
            self._clear_bufs()
            out.boxes = others
            return out

        board = {"cls": "board", "xyxy": list(box), "conf": conf}
        self.nvis_buf.append(sum(1 for b in others if b["cls"] in _HOLE_LIKE and _inside(b, box)))
        self.conf_buf.append(conf)
        for c in _HOLE_LIKE:
            self.cnt_buf[c].append(sum(1 for b in others if b["cls"] == c))

        out.boxes = [board] + others
        out.visible_holes = median(self.nvis_buf)
        out.board_conf = median(self.conf_buf)
        out.counts = {"board": 1, **{c: median(self.cnt_buf[c]) for c in _HOLE_LIKE}}
        board["conf"] = out.board_conf
        return out

    def _clear_bufs(self) -> None:
        self.nvis_buf.clear()
        self.conf_buf.clear()
        for d in self.cnt_buf.values():
            d.clear()

    def reset(self) -> None:
        self.track = BoardTrack(self.track.hold_sec, self.track.iou_gate,
                                self.track.alpha, self.track.prefer_holes)
        self._clear_bufs()
        self.hands, self.t_hands = [], None


__all__ = ["BoardTrack", "SmoothedFrame", "TemporalSmoother", "iou", "median"]
