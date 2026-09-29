# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The occlusion gate, ported from the judge's ``occlusion.py``.

Decides whether one frame's detections may move the assembly state. Three
criteria, any of which rejects the frame, with thresholds from 475 measured
frames (visible holes inside the board box as the ground truth for how
occluded it is):

    criterion               separation   threshold
    hand area over board    1.56 sigma   th_hand
    board confidence        0.86 sigma   th_conf
    visible holes           exact        min_holes

Only hands are boxed, not arms: a COCO person box separated better but spans
the whole arm, and two hands either side of the board then boxed the empty
space between them as "covering" it.

The old judge boxed hands with MediaPipe landmarks padded by 12%. This port
takes them from the shared hand detector instead, whose boxes are drawn
around the whole hand; ``th_hand`` was tuned on the landmark boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TH_HAND = 0.40
"""Hand-over-board fraction. 0.20 was best on fixed top-down phone footage; the
first-person glasses view puts hands much nearer the lens, where 0.20 rejected
61-75% of frames that passed both other criteria."""

TH_CONF = 0.75
"""Board confidence floor, the value the glasses demo ran with."""

MIN_VISIBLE_HOLES = 2
"""Holes that must be visible inside the board. The v1 rule was 5; since v2 an
unseen hole abstains per slot instead, and on glasses footage 5 confirmed only
1-2 of 4 screws where 2 confirmed 3-4 with no false positive."""

TOTAL_HOLES = 6


def inter_over_b(a: list[float], b: list[float]) -> float:
    """The fraction of *b*'s area that *a* covers."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return (ix2 - ix1) * (iy2 - iy1) / max(area_b, 1.0)


def _centre_inside(box: dict[str, Any], board: list[float]) -> bool:
    cx = (box["xyxy"][0] + box["xyxy"][2]) / 2
    cy = (box["xyxy"][1] + box["xyxy"][3]) / 2
    return board[0] <= cx <= board[2] and board[1] <= cy <= board[3]


def count_visible_holes(boxes: list[dict[str, Any]], board: list[float]) -> int:
    """hole + hole_filled + screw inside the board: how much of it is exposed."""
    return sum(1 for b in boxes
               if b["cls"] in ("hole", "hole_filled", "screw") and _centre_inside(b, board))


def count_hole_slots(boxes: list[dict[str, Any]], board: list[float]) -> int:
    """hole + hole_filled inside the board, without screws.

    ``hole + hole_filled == 6`` means no hole is covered, which is sufficient
    evidence of an unoccluded frame. Loose screws on the board would push
    :func:`count_visible_holes` past 6, so they are left out here.
    """
    return sum(1 for b in boxes
               if b["cls"] in ("hole", "hole_filled") and _centre_inside(b, board))


@dataclass
class OcclusionVerdict:
    ok: bool
    """The frame may move the state."""

    reason: str = ""
    hand_cov: float = 0.0
    board_conf: float = 0.0
    visible_holes: float = 0
    all_slots: bool = False
    """All six holes were seen in this frame, which accepts it outright."""


class OcclusionFilter:
    """The three-way gate. Without hand boxes the hand criterion is skipped."""

    def __init__(self, th_hand: float = TH_HAND, th_conf: float = TH_CONF,
                 min_holes: int = MIN_VISIBLE_HOLES, use_hand: bool = True,
                 total_holes: int = TOTAL_HOLES) -> None:
        self.th_hand = th_hand
        self.th_conf = th_conf
        self.min_holes = min_holes
        self.use_hand = use_hand
        self.total_holes = total_holes

    def check(self, boxes: list[dict[str, Any]], hand_boxes: list[list[float]] | None = None,
              visible_holes: float | None = None) -> OcclusionVerdict:
        """*visible_holes* replaces this frame's own count (the smoother's median)."""

        boards = [b for b in boxes if b["cls"] == "board"]
        if not boards:
            return OcclusionVerdict(False, "no board detected")
        if len(boards) > 1:
            return OcclusionVerdict(False, f"{len(boards)} boards detected (false positives)")
        board = boards[0]
        bx, conf = board["xyxy"], board.get("conf", 1.0)
        nvis = count_visible_holes(boxes, bx) if visible_holes is None else visible_holes
        cov = 0.0
        if self.use_hand and hand_boxes:
            cov = max((inter_over_b(p, bx) for p in hand_boxes), default=0.0)

        # "All six holes seen, take it" comes before the confidence and hole-count
        # criteria, and uses THIS frame's count rather than the median: on glasses
        # footage the hole count swings 0 <-> 6, so a frame showing all six could
        # sit under a median of 3 (27.7% of such frames were lost that way, at a
        # board confidence of p50 0.97). The hand criterion still applies: it is
        # about a hand pressing on the board, not about what is visible.
        all_slots = count_hole_slots(boxes, bx) >= self.total_holes
        verdict = OcclusionVerdict(True, "", cov, conf, nvis, all_slots)
        if self.use_hand and hand_boxes and cov >= self.th_hand:
            verdict.ok, verdict.reason = False, f"hand covers {100 * cov:.0f}% of the board"
        elif all_slots:
            pass
        elif conf < self.th_conf:
            verdict.ok, verdict.reason = False, f"board confidence {conf:.2f} is low"
        elif nvis < self.min_holes:
            verdict.ok, verdict.reason = False, f"only {nvis:.0f} of {self.total_holes} holes visible"
        return verdict


__all__ = ["OcclusionFilter", "OcclusionVerdict", "count_hole_slots", "count_visible_holes",
           "inter_over_b"]
