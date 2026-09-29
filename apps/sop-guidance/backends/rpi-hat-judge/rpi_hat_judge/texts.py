# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the judge says, ported from the English half of ``guidance_text.py``.

The state layer reports events as plain fields; everything a wearer hears or
an operator reads is worded here, and nowhere else.
"""

from __future__ import annotations

import re
from typing import Any

from .board import HOLE_NAMES

OPERATOR_EVENTS: dict[str, tuple[str, str]] = {
    "order_violation": ("error", "Out of order"),
    "rework": ("warn", "Rework"),
    "jump": ("warn", "Step skipped"),
    "regress": ("warn", "Progress regressed"),
}
"""The only events a wearer hears. A sentinel means the detector contradicted
itself in a frame: a model-quality signal, never the wearer's mistake."""


def tts(text: str) -> str:
    """Screen text made speakable: circled digits, ``#``, arrows and brackets."""

    if not text:
        return ""
    for a, b in (("①", "1"), ("②", "2"), ("③", "3"), ("④", "4"), ("⑤", "5"), ("⑥", "6"),
                 ("#", "number "), ("→", " to "), ("·", ", "), ("✓", ""), ("▶", ""),
                 ("（", ", "), ("(", ", "), ("）", ""), (")", "")):
        text = text.replace(a, b)
    # A bracket turned comma leaves "hole 2 , bottom-right", an abrupt pause.
    text = text.replace(" ,", ",").replace(" .", ".")
    return re.sub(r"\s+", " ", text).strip(" ,.").strip()


def event_detail(kind: str, fields: dict[str, Any]) -> str:
    """One event's detail, as the operator reads it."""

    if kind in ("screw_installed", "rework"):
        hole = int(fields["hole"])
        verb = "was removed" if kind == "rework" else ""
        return f"screw {hole} ({HOLE_NAMES[hole]}) {verb}".strip()
    if kind == "order_violation":
        expected = int(fields["expected"])
        return (f"screw {fields['hole']} went in while hole {expected} "
                f"({HOLE_NAMES[expected]}) is still empty")
    if kind in ("advance", "jump", "regress"):
        return f"installed count {fields['old']} to {fields['new']}"
    if kind in ("fpc_seated", "fpc_rework"):
        state = "seated" if kind == "fpc_seated" else "pulled out"
        return f"ribbon cable {state} (IoB {float(fields['iob']):.2f})"
    return str(fields.get("reason", kind.replace("_", " ")))


def alert_label(kind: str) -> str:
    return OPERATOR_EVENTS.get(kind, ("", kind.replace("_", " ")))[1]


def alert_cue(kind: str, fields: dict[str, Any], next_name: str | None) -> str:
    """An alert, spoken: what happened, then what to do next."""

    tail = f". Next, {next_name}" if next_name else ""
    return tts(f"{alert_label(kind)}. {event_detail(kind, fields)}{tail}")


def done_phrase(position: int, name: str, next_name: str | None = None) -> str:
    """"Step N, X, done." -- with what comes next when it is not announced separately."""

    head = f"Step {position}, {name}, done"
    if next_name is None:
        return tts(head) + "."
    return tts(f"{head}. Next, {next_name}")


__all__ = ["OPERATOR_EVENTS", "alert_cue", "alert_label", "done_phrase", "event_detail",
           "tts"]
