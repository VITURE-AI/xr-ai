# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Facts about the Raspberry Pi M.2 HAT+ board the judge reads.

The detector (v7, five classes) finds the board and every hole on it. Which
hole is which comes from these anchors: the median offset of each of the six
holes from the board box centre, as a fraction of the box size, measured on
226 hand-labelled frames. They assume the "Raspberry Pi" silkscreen is roughly
upright in the frame.
"""

from __future__ import annotations

CLASSES = ("board", "hole", "screw", "hole_filled", "fpc")
"""The detector's classes, in checkpoint order."""

HOLE_ANCHORS: dict[int, tuple[float, float]] = {
    1: (-0.445, -0.420),  # top-left
    2: (+0.207, +0.362),  # bottom-right
    3: (+0.181, -0.436),  # top-right
    4: (-0.431, +0.342),  # bottom-left
    5: (+0.185, -0.298),  # mid-right upper: M.2 mounting hole, never screwed
    6: (+0.205, +0.169),  # mid-right lower: M.2 mounting hole, never screwed
}

SOP_SEQUENCE: tuple[int, ...] = (1, 2, 3, 4)
"""The order the four corner screws go in: diagonals, so the board loads evenly."""

FIXED_SLOTS: tuple[int, ...] = tuple(k for k in HOLE_ANCHORS if k not in SOP_SEQUENCE)
"""Holes 5 and 6. They never get a screw, which makes them a known-answer test."""

HOLE_NAMES: dict[int, str] = {
    1: "top-left", 2: "bottom-right", 3: "top-right", 4: "bottom-left",
    5: "mid-right (fixed)", 6: "mid-right (fixed)",
}


__all__ = ["CLASSES", "FIXED_SLOTS", "HOLE_ANCHORS", "HOLE_NAMES", "SOP_SEQUENCE"]
