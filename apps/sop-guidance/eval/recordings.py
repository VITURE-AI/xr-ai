# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read old-fork guidance recordings into replayable grounded checks.

A session recorded at debug level ``full`` by the old glasses worker is a
folder ``<YYYYmmdd_HHMMSS>_<procedure>/`` holding:

- ``events.jsonl``: one ``CHECK`` event per grounded check, with the verdict
  (``completed``, ``has_evidence``), the observation, the issue, and the host
  path of the graded student frame (``frame``).
- ``calls.jsonl``: every model call. A grading call is ``kind: vlm`` named
  ``ask_frames`` (the teacher comparison tier, 2-3 images) or ``ask_image``
  (the live tier, or the tier-3 diagnosis), with the exact question, the raw
  response, its latency and ``artifacts``: the images the model saw, in order,
  hardlinked into ``step_NN/call_NNNN/in_NN.png``. The student frame is always
  the last image.
- ``step_NN/clip_NNNN/``: 1 fps preview strips; not needed for replay.

A check is the calls sharing one student frame, closed by the ``CHECK`` event
naming that frame. Checks whose calls were not recorded (the old recorder
stopped recording calls past a budget) cannot be replayed and are counted as
skipped. Only the standard library is used here, so this module is testable
without the model stack.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GRADING_TIERS = ("compare", "live", "diagnosis")

_DIAGNOSIS_PREFIX = "The user is trying to do this step"
_GEOMETRY_HEADER = "DETECTOR GEOMETRY for the live student image"
_WEARER_HEADER = "WHAT THE WEARER ASKED FOR"
_REQUEST_LINE = re.compile(r'^\s+- "(.*)"(?:\s+<- MOST RECENT.*)?$')


def classify_tier(name: str, question: str, image_count: int) -> str:
    """The grading tier of one recorded VLM call, or '' for a non-grading call."""

    if question.startswith(_DIAGNOSIS_PREFIX):
        return "diagnosis"
    if not question.startswith("INSTRUCTION:"):
        return ""
    if name == "ask_frames" or image_count > 1:
        return "compare"
    return "live"


def wearer_requests(question: str) -> tuple[str, ...]:
    """The wearer's spoken requests quoted in a grading prompt, oldest first."""

    start = question.find(_WEARER_HEADER)
    if start < 0:
        return ()
    requests: list[str] = []
    for line in question[start:].splitlines()[1:]:
        match = _REQUEST_LINE.match(line)
        if match is None:
            if requests:
                break
            continue
        requests.append(match.group(1))
    return tuple(requests)


def spatial_context(question: str) -> str:
    """The detector-geometry prose block of a grading prompt, or ''."""

    start = question.find(_GEOMETRY_HEADER)
    if start < 0:
        return ""
    end = question.find("\n\n", start)
    if end < 0:
        return question[start:].rstrip() + "\n\n"
    return question[start:end] + "\n\n"


@dataclass(frozen=True, slots=True)
class RecordedCall:
    """One recorded grading call."""

    seq: int
    ts_us: int
    step: int
    tier: str
    question: str
    response: str
    latency_ms: float
    error: str
    images: tuple[Path, ...]
    """Local copies of the images the model saw, in order; the student frame is last."""

    sources: tuple[str, ...]
    """The host paths those images had on the recording machine."""

    @property
    def student_source(self) -> str:
        return self.sources[-1] if self.sources else ""


@dataclass(frozen=True, slots=True)
class RecordedCheck:
    """One grounded check of the old worker: its inputs, calls and verdict."""

    session: str
    step: int
    """1-based step number the check graded."""

    instruction: str
    ts_us: int
    completed: bool
    has_evidence: bool
    observation: str
    issue: str
    frame_source: str
    calls: tuple[RecordedCall, ...]

    @property
    def key(self) -> str:
        """Stable id: session plus the seq of the check's first call."""

        return f"{self.session}#{self.calls[0].seq:04d}" if self.calls else self.session

    @property
    def passed(self) -> bool:
        """The verdict the old monitor counted towards the streak."""

        return self.completed and self.has_evidence

    def call(self, tier: str) -> RecordedCall | None:
        for recorded in self.calls:
            if recorded.tier == tier:
                return recorded
        return None

    @property
    def student_image(self) -> Path:
        return self.calls[-1].images[-1]

    @property
    def teacher_images(self) -> tuple[Path, ...]:
        """The annotated teacher frames of the comparison tier: (after,) or (before, after)."""

        compare = self.call("compare")
        return compare.images[:-1] if compare is not None else ()

    @property
    def wearer_requests(self) -> tuple[str, ...]:
        return wearer_requests(self.calls[0].question) if self.calls else ()

    @property
    def spatial_context(self) -> str:
        return spatial_context(self.calls[0].question) if self.calls else ""

    @property
    def latency_ms(self) -> float:
        """End-to-end check time: first call's start to the CHECK event."""

        if not self.calls:
            return 0.0
        first = self.calls[0]
        started_us = first.ts_us - first.latency_ms * 1000.0
        return max(0.0, (self.ts_us - started_us) / 1000.0)

    @property
    def vlm_ms(self) -> float:
        return sum(c.latency_ms for c in self.calls)


@dataclass(slots=True)
class RecordedSession:
    """The replayable checks of one recorded session."""

    path: Path
    meta: dict[str, Any]
    checks: list[RecordedCheck] = field(default_factory=list)
    skipped: Counter[str] = field(default_factory=Counter)

    @property
    def id(self) -> str:
        return self.path.name


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _grading_call(session_dir: Path, row: dict[str, Any]) -> RecordedCall | None:
    if row.get("kind") != "vlm":
        return None
    request = row.get("request") if isinstance(row.get("request"), dict) else {}
    question = str(request.get("question", ""))
    sources = request.get("image_paths")
    if not isinstance(sources, list):
        single = request.get("image_path")
        sources = [single] if single else []
    artifacts = [str(a) for a in row.get("artifacts") or []]
    tier = classify_tier(str(row.get("name", "")), question, len(artifacts))
    if not tier:
        return None
    return RecordedCall(
        seq=int(row.get("seq", 0) or 0),
        ts_us=int(row.get("ts_us", 0) or 0),
        step=int(row.get("step", 0) or 0),
        tier=tier,
        question=question,
        response=str(row.get("response") or ""),
        latency_ms=float(row.get("latency_ms", 0.0) or 0.0),
        error=str(row.get("error") or ""),
        images=tuple(session_dir / a for a in artifacts),
        sources=tuple(str(s) for s in sources),
    )


def load_session(session_dir: Path) -> RecordedSession:
    """Pair a session's grading calls with the ``CHECK`` events they produced."""

    session_dir = Path(session_dir)
    meta_path = session_dir / "meta.json"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            meta = loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            meta = {}
    session = RecordedSession(path=session_dir, meta=meta)

    timeline: list[tuple[int, int, str, Any]] = []
    for row in _read_jsonl(session_dir / "calls.jsonl"):
        call = _grading_call(session_dir, row)
        if call is not None:
            timeline.append((call.ts_us, 0, "call", call))
    for row in _read_jsonl(session_dir / "events.jsonl"):
        if row.get("event") == "CHECK":
            timeline.append((int(row.get("ts_us", 0) or 0), 1, "check", row))
    timeline.sort(key=lambda item: (item[0], item[1]))

    pending: dict[tuple[int, str], list[RecordedCall]] = {}
    for _ts, _order, kind, item in timeline:
        if kind == "call":
            pending.setdefault((item.step, item.student_source), []).append(item)
            continue
        step = int(item.get("step", 0) or 0)
        frame = str(item.get("frame", "") or "")
        calls = pending.pop((step, frame), [])
        if not calls:
            session.skipped["calls-not-recorded"] += 1
            continue
        if any(not image.is_file() for c in calls for image in c.images) or not calls[-1].images:
            session.skipped["missing-artifact"] += 1
            continue
        session.checks.append(RecordedCheck(
            session=session.id,
            step=step,
            instruction=str(item.get("instruction") or "") or _instruction_of(calls[0]),
            ts_us=int(item.get("ts_us", 0) or 0),
            completed=bool(item.get("completed", False)),
            has_evidence=bool(item.get("has_evidence", False)),
            observation=str(item.get("observation", "") or ""),
            issue=str(item.get("issue", "") or ""),
            frame_source=frame,
            calls=tuple(calls),
        ))
    session.skipped["orphan-calls"] += sum(len(v) for v in pending.values())
    if not session.skipped["orphan-calls"]:
        del session.skipped["orphan-calls"]
    return session


def _instruction_of(call: RecordedCall) -> str:
    first = call.question.split("\n", 1)[0]
    return first.removeprefix("INSTRUCTION:").strip()


def find_sessions(paths: list[Path]) -> list[Path]:
    """Expand CLI paths: a session folder itself, or a folder of session folders."""

    found: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if (path / "events.jsonl").is_file():
            found.append(path)
            continue
        if path.is_dir():
            found.extend(sorted(p for p in path.iterdir() if (p / "events.jsonl").is_file()))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def sample_checks(
    checks: list[RecordedCheck], limit: int, *, per_session: int = 0,
) -> list[RecordedCheck]:
    """A deterministic subset that keeps every (session, step) represented.

    Round-robin over (session, step) buckets in chronological order, so a small
    ``limit`` still covers every step rather than the first minutes of one
    session.
    """

    buckets: dict[tuple[str, int], list[RecordedCheck]] = {}
    for check in sorted(checks, key=lambda c: (c.session, c.ts_us)):
        buckets.setdefault((check.session, check.step), []).append(check)
    taken: list[RecordedCheck] = []
    per: Counter[str] = Counter()
    order = sorted(buckets)
    while order and (limit <= 0 or len(taken) < limit):
        remaining = []
        for key in order:
            bucket = buckets[key]
            if not bucket or (per_session and per[key[0]] >= per_session):
                continue
            taken.append(bucket.pop(0))
            per[key[0]] += 1
            if limit > 0 and len(taken) >= limit:
                break
            if bucket:
                remaining.append(key)
        order = remaining
    return sorted(taken, key=lambda c: (c.session, c.ts_us))


__all__ = [
    "GRADING_TIERS",
    "RecordedCall",
    "RecordedCheck",
    "RecordedSession",
    "classify_tier",
    "find_sessions",
    "load_session",
    "sample_checks",
    "spatial_context",
    "wearer_requests",
]
