# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agreement, confusion, issue-text and latency summaries for a replay run.

Standard library only, so the numbers can be unit tested without models.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

# What a spoken correction must never contain (see grading.spoken_issue_rule):
# image numbers, the teacher/reference, or third-person "student"/"user".
# Not "frame": the glasses frame is a part the corrections name.
_SPOKEN_VIOLATION = re.compile(
    r"\bimages?\s*\d\b|\bteacher\b|\breference\b|\bstudent\b|\bthe user\b|\bphoto\b",
    re.I,
)
_WORD = re.compile(r"[a-z0-9']+")


@dataclass(slots=True)
class Outcome:
    """One replayed check: the recorded verdict next to the new one."""

    key: str
    session: str
    step: int
    old_passed: bool
    new_passed: bool | None
    """None when the new evaluator errored or timed out."""

    old_issue: str = ""
    new_issue: str = ""
    old_observation: str = ""
    new_observation: str = ""
    new_tier: str = ""
    geometry_veto: str = ""
    old_ms: float = 0.0
    new_ms: float = 0.0
    old_call_ms: list[float] = field(default_factory=list)
    new_call_ms: list[float] = field(default_factory=list)
    baseline_passed: bool | None = None
    """The recorded (old) prompts re-asked verbatim, when ``--baseline`` ran."""

    baseline_ms: float = 0.0
    wearer_requests: list[str] = field(default_factory=list)
    student_image: str = ""
    prompt_similarity: float | None = None
    """How close the new comparison prompt is to the recorded one, 0..1."""

    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, *q* in 0..100; NaN for no values."""

    data = sorted(v for v in values if v is not None and not math.isnan(v))
    if not data:
        return math.nan
    if len(data) == 1:
        return data[0]
    rank = (len(data) - 1) * q / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    return data[low] + (data[high] - data[low]) * (rank - low)


def cohen_kappa(pairs: Iterable[tuple[bool, bool]]) -> float:
    """Cohen's kappa for two binary raters; NaN when undefined."""

    pairs = list(pairs)
    n = len(pairs)
    if not n:
        return math.nan
    observed = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else math.nan
    return (observed - expected) / (1 - expected)


def token_jaccard(a: str, b: str) -> float:
    ta, tb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def spoken_violation(text: str) -> bool:
    return bool(text) and _SPOKEN_VIOLATION.search(text) is not None


def vetoed(outcome: Outcome) -> bool:
    """Whether the geometry veto overturned a VLM pass on this check."""

    return bool(outcome.geometry_veto) and outcome.new_issue == outcome.geometry_veto


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def summarize(outcomes: Sequence[Outcome]) -> dict[str, Any]:
    """The headline numbers of a run."""

    graded = [o for o in outcomes if o.new_passed is not None]
    pairs = [(o.old_passed, bool(o.new_passed)) for o in graded]
    agree = sum(a == b for a, b in pairs)
    summary: dict[str, Any] = {
        "checks": len(outcomes),
        "graded": len(graded),
        "errors": len(outcomes) - len(graded),
        "agreement": _rate(agree, len(graded)),
        "kappa": cohen_kappa(pairs),
        "old_pass_rate": _rate(sum(a for a, _ in pairs), len(pairs)),
        "new_pass_rate": _rate(sum(b for _, b in pairs), len(pairs)),
    }

    baseline = [o for o in graded if o.baseline_passed is not None]
    if baseline:
        summary["baseline_checks"] = len(baseline)
        summary["baseline_agreement"] = _rate(
            sum(o.old_passed == o.baseline_passed for o in baseline), len(baseline))
        summary["new_agreement_on_baseline_set"] = _rate(
            sum(o.old_passed == o.new_passed for o in baseline), len(baseline))
        summary["new_vs_baseline_agreement"] = _rate(
            sum(o.new_passed == o.baseline_passed for o in baseline), len(baseline))

    by_step: dict[int, dict[str, int]] = {}
    for o in graded:
        cell = by_step.setdefault(o.step, {"pp": 0, "pf": 0, "fp": 0, "ff": 0})
        cell[("p" if o.old_passed else "f") + ("p" if o.new_passed else "f")] += 1
    summary["confusion_by_step"] = {
        step: {**cell, "agreement": _rate(cell["pp"] + cell["ff"], sum(cell.values()))}
        for step, cell in sorted(by_step.items())
    }

    both_failed = [o for o in graded if not o.old_passed and not o.new_passed]
    summary["issue_jaccard_both_failed"] = (
        sum(token_jaccard(o.old_issue, o.new_issue) for o in both_failed) / len(both_failed)
        if both_failed else math.nan
    )
    failed_old = [o for o in graded if not o.old_passed and o.old_issue]
    failed_new = [o for o in graded if not o.new_passed and o.new_issue]
    summary["spoken_violations_old"] = f"{sum(spoken_violation(o.old_issue) for o in failed_old)}/{len(failed_old)}"
    summary["spoken_violations_new"] = f"{sum(spoken_violation(o.new_issue) for o in failed_new)}/{len(failed_new)}"
    # A veto string is computed for every gated frame; it only DECIDED the
    # check when it replaced a VLM pass, i.e. became the issue.
    summary["geometry_vetoes_new"] = sum(1 for o in graded if vetoed(o))
    summary["new_tiers"] = _count(o.new_tier or "-" for o in graded)

    def lat(values: list[float]) -> dict[str, float]:
        return {"n": len(values), "p50": percentile(values, 50), "p95": percentile(values, 95)}

    summary["latency_ms"] = {
        "old_check": lat([o.old_ms for o in graded if o.old_ms]),
        "new_check": lat([o.new_ms for o in graded if o.new_ms]),
        "old_call": lat([ms for o in graded for ms in o.old_call_ms]),
        "new_call": lat([ms for o in graded for ms in o.new_call_ms]),
    }
    if baseline:
        summary["latency_ms"]["baseline_check"] = lat([o.baseline_ms for o in baseline if o.baseline_ms])
    summary["calls_per_check"] = {
        "old": _rate(sum(len(o.old_call_ms) for o in graded), len(graded)),
        "new": _rate(sum(len(o.new_call_ms) for o in graded), len(graded)),
    }
    similarities = [o.prompt_similarity for o in outcomes if o.prompt_similarity is not None]
    if similarities:
        summary["prompt_similarity"] = {
            "min": min(similarities),
            "mean": sum(similarities) / len(similarities),
            "identical": sum(1 for s in similarities if s >= 0.9999),
            "n": len(similarities),
        }
    return summary


def _count(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _fmt(value: Any, pct: bool = False) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.1%}" if pct else f"{value:.0f}"
    return str(value)


def _clip(text: str, limit: int = 110) -> str:
    text = " ".join(text.split()).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_markdown(summary: dict[str, Any], outcomes: Sequence[Outcome], *, title: str) -> str:
    """A human-readable report of one run."""

    calls = summary["calls_per_check"]
    rates = f"{_fmt(summary['old_pass_rate'], True)} / {_fmt(summary['new_pass_rate'], True)}"
    violations = f"{summary['spoken_violations_old']} / {summary['spoken_violations_new']}"
    lines = [f"# {title}", ""]
    lines += [
        "| metric | value |",
        "|---|---|",
        f"| checks replayed (graded / errors) | {summary['checks']} ({summary['graded']} / {summary['errors']}) |",
        f"| agreement new vs recorded | {_fmt(summary['agreement'], True)} (kappa {summary['kappa']:.2f}) |",
        f"| pass rate recorded / new | {rates} |",
    ]
    if "baseline_agreement" in summary:
        lines += [
            f"| baseline: recorded prompts re-asked vs recorded ({summary['baseline_checks']}) "
            f"| {_fmt(summary['baseline_agreement'], True)} |",
            f"| new vs recorded, same checks | {_fmt(summary['new_agreement_on_baseline_set'], True)} |",
            f"| new vs baseline | {_fmt(summary['new_vs_baseline_agreement'], True)} |",
        ]
    lines += [
        f"| VLM passes overturned by the geometry veto (new) | {summary['geometry_vetoes_new']} |",
        f"| new deciding tier | {summary['new_tiers']} |",
        f"| issue token-Jaccard when both fail | {_fmt(summary['issue_jaccard_both_failed'], True)} |",
        f"| spoken-rule violations recorded / new | {violations} |",
        f"| VLM calls per check recorded / new | {calls['old']:.2f} / {calls['new']:.2f} |",
    ]
    if "prompt_similarity" in summary:
        sim = summary["prompt_similarity"]
        lines.append(
            f"| comparison prompt vs recorded (identical / mean / min) "
            f"| {sim['identical']}/{sim['n']} / {sim['mean']:.3f} / {sim['min']:.3f} |")
    lines += ["", "## Latency (ms)", "", "| | n | p50 | p95 |", "|---|---|---|---|"]
    for name, row in summary["latency_ms"].items():
        lines.append(f"| {name} | {row['n']} | {_fmt(row['p50'])} | {_fmt(row['p95'])} |")
    lines += [
        "", "## Confusion by step (recorded x new; p = pass, f = fail)", "",
        "| step | pp | pf | fp | ff | agreement |", "|---|---|---|---|---|---|",
    ]
    for step, cell in summary["confusion_by_step"].items():
        lines.append(
            f"| {step} | {cell['pp']} | {cell['pf']} | {cell['fp']} | {cell['ff']} "
            f"| {_fmt(cell['agreement'], True)} |")
    disagreements = [o for o in outcomes if o.new_passed is not None and o.old_passed != o.new_passed]
    lines += ["", f"## Disagreements ({len(disagreements)})", ""]
    if disagreements:
        lines += [
            "| check | step | recorded | new | tier | recorded issue | new issue / observation |",
            "|---|---|---|---|---|---|---|",
        ]
        for o in disagreements:
            new_text = o.new_issue or o.new_observation
            if vetoed(o):
                new_text = f"[veto] {new_text}"
            lines.append(
                f"| {o.key} | {o.step} | {'pass' if o.old_passed else 'fail'} "
                f"| {'pass' if o.new_passed else 'fail'} | {o.new_tier} "
                f"| {_clip(o.old_issue or o.old_observation)} | {_clip(new_text)} |")
    errors = [o for o in outcomes if o.new_passed is None]
    if errors:
        lines += ["", f"## Errors ({len(errors)})", ""]
        lines += [f"- {o.key}: {_clip(o.error, 200)}" for o in errors]
    return "\n".join(lines) + "\n"


__all__ = [
    "Outcome",
    "cohen_kappa",
    "percentile",
    "render_markdown",
    "spoken_violation",
    "summarize",
    "token_jaccard",
    "vetoed",
]
