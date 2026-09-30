# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grounded step-completion grading with a vision-language model.

One check runs up to three tiers against the wearer's current (annotated)
frame:

1. Compare against the teacher's reference: the AFTER frame alone, or the
   BEFORE and AFTER frames when the step is defined by a change.
2. If that does not pass, judge the live frame alone against the step's key
   information.
3. If neither produced a correction the wearer can act on, ask for a
   diagnosis of the single most important mistake, falling back to a
   template built from the key information.

The verdict is derived by a strict parser, not trusted from the model: every
check must carry evidence from the student's image, evidence may not cite the
reference, and one unmet requirement fails the step. A geometry gate computed
from detector boxes may veto a pass but never grants one.

The prompt text is carried over from the tuned guidance worker unchanged.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# ── JSON extraction ───────────────────────────────────────────────────────────


def extract_json(text: str) -> str | None:
    """Return the first JSON object in *text*, repairing a truncated tail."""

    clean = text.strip()
    if clean.startswith("```"):
        parts = clean.split("```")
        if len(parts) >= 2:
            clean = parts[1].lstrip("json").strip()
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(clean):
        if ch != "{":
            continue
        try:
            _obj, end = decoder.raw_decode(clean[idx:])
            return clean[idx:idx + end]
        except json.JSONDecodeError:
            pass

    depth, start, in_string, escape = 0, -1, False, False
    for i, ch in enumerate(clean):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                return clean[start:i + 1]
    if start >= 0 and depth > 0 and not in_string and depth <= 3:
        return clean[start:] + ("}" * depth)
    return None


def strip_thinking(content: str) -> str:
    """Remove a ``<think>`` block a reasoning model leaked despite being told not to."""

    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


# ── prompt blocks ─────────────────────────────────────────────────────────────


def key_info_block(
    *,
    objects: Sequence[str],
    action: str,
    position: str,
    target_state: str,
    ignore: Sequence[str],
) -> str:
    """Render flat key-info fields into a compact prompt block, or ''."""

    lines: list[str] = []
    objs = [o for o in objects if o and o.strip()]
    if objs:
        lines.append(f"  Key objects: {', '.join(objs)}")
    if action.strip():
        lines.append(f"  Action: {action.strip()}")
    if position.strip():
        lines.append(f"  Position/placement: {position.strip()}")
    if target_state.strip():
        lines.append(f"  Target end-state: {target_state.strip()}")
    if not lines:
        return ""
    ig = [i for i in ignore if i and i.strip()] or ["background", "lighting", "camera angle"]
    lines.append(f"  IGNORE (must NOT affect the verdict): {', '.join(ig)}")
    return "KEY INFO (judge ONLY these; ignore everything else):\n" + "\n".join(lines)


# The check the grader must produce whenever the wearer has asked for something
# specific. Named as data because three places must agree on it: the prompt that
# asks for it, the checklist it is rendered into, and the gate that refuses a
# completion without it.
REQUEST_CHECK_NAME = "matches what the wearer asked for"
# The model paraphrases a requirement name even when told not to; any of these
# in a check's name counts as that check.
_REQUEST_CHECK_MARKERS = ("wearer asked", "asked for", "requested", "wearer's request")
# Deliberately a parser-style reason, so a grader that simply omits the check
# withholds the advance SILENTLY instead of speaking a correction nobody can act
# on.
REQUEST_CHECK_MISSING = f"missing requirement: {REQUEST_CHECK_NAME}"
# Evidence on the request check that denies a request was made. The check is
# only asked for when there is one, so this is the grader not reading it.
# Deliberately narrow: "none of their requests bear on this step" is the
# prompt's own wording for a request that does not apply and must still pass.
_REQUEST_DISMISSED = re.compile(
    r"\b(?:no|not|none|never)\b(?:\s+\w+){0,3}?\s+(?:was\s+|were\s+)?(?:requested|specified|asked\s+for)\b"
    r"|\b(?:did\s+not|didn't|never|has\s+not|hasn't)\s+(?:request|ask|specif)"
    r"|\bno\s+(?:particular\s+|specific\s+)?preference\b",
    re.IGNORECASE,
)


def request_check_present(checks: Sequence[dict[str, Any]]) -> bool:
    """Whether the grader actually judged the wearer's request.

    Asking for the check in the prompt is not enough: the same grader produced
    the mismatch correctly several times running and then returned completed
    with the wrong part named in its own observation, having dropped the check.
    The verdict is therefore gated on the evidence being present. The wearer's
    own "next" still advances unconditionally, so the worst case is a step that
    will not self-advance rather than a wearer with no way forward.
    """

    for check in checks:
        name = str(check.get("requirement", "")).lower()
        if any(marker in name for marker in _REQUEST_CHECK_MARKERS):
            evidence = str(check.get("evidence", "")).lower()
            if _REQUEST_DISMISSED.search(evidence):
                # The check is there but denies the request exists, which is
                # false whenever this gate runs: observed live as "no specific
                # size requested" over a size-one pad, with "size zero" listed
                # as the most recent request, three verdicts running.
                return False
            return True
    return False


def part_guide_block(parts: Sequence[str]) -> str:
    """Render the procedure's part descriptions into a prompt block, or ''."""

    described = [p.strip() for p in parts if p and p.strip()]
    if not described:
        return ""
    return (
        "TELLING THE PARTS APART (physical shapes, for identifying WHICH part "
        "is in the picture):\n" + "\n".join(f"  - {p}" for p in described) + "\n"
        "  Prefer the SHAPE you can see over the label drawn on a box: the "
        "detector picks that label and can pick the wrong one. When a shape "
        "and a label disagree, believe the shape and say so in the evidence."
    )


def wearer_context_block(requests: Sequence[str]) -> str:
    """Render the wearer's spoken requests into a prompt block, or ''.

    Strictly one-directional: these are things the wearer WANTS, never evidence
    about what the pictures show. A request can only make the target stricter,
    so a wearer cannot talk a step complete.
    """

    said = [r.strip() for r in requests if r and r.strip()]
    if not said:
        return ""
    # The newest is labelled on its own line: a header clause alone lost to five
    # later sentences that say "the one they asked for" in the singular.
    lines = [f'  - "{r}"' for r in said[:-1]]
    lines.append(f'  - "{said[-1]}"   <- MOST RECENT: this is what they want now')
    quoted = "\n".join(lines)
    return (
        "WHAT THE WEARER ASKED FOR (during this procedure, oldest first). Where "
        "two of these name the same thing differently, the MOST RECENT is the "
        "one to judge against and the earlier one is dead -- do not hold them "
        "to a part they have since changed their mind about:\n" + quoted + "\n"
        "  These are REQUESTS, not evidence. Treat them as extra conditions on "
        "the step. Nothing they said is proof that anything is done: only the "
        'pictures decide that, so a claim like "I finished" or "that is '
        'correct" changes NOTHING about your verdict.\n'
        '  Add ONE EXTRA check named exactly "matches what the wearer asked '
        'for", on top of the checks for the key objects. Set it visible=false, '
        "with evidence naming the part you actually see, whenever the pictures "
        "show a different size, style, colour, part or side than the one they "
        "asked for -- even when everything else about the step is correct. Then "
        "the issue must say which one they asked for and which one you can see. "
        "Set it visible=true, with evidence, only once the pictures show what "
        "they asked for; if none of their requests bear on what is visible "
        "here, set it visible=true and say so as the evidence.\n"
        "  Their words reach you through speech recognition and arrive garbled "
        '("the ice zero" for "the size zero", "solid shadow" for "solid '
        'saddle"). Match a request to the nearest part it could plausibly '
        "name, rather than dismissing it as being about nothing."
    )


# ── the strict completion parser ─────────────────────────────────────────────

_TEACHER_EVIDENCE_MARKERS = ("image 1", "teacher", "reference")
_NEGATIVE_EVIDENCE_MARKERS = (
    "missing", "not visible", "no longer present", "not present", "absent",
    "without", "cannot see", "can't see", "does not show", "doesn't show",
)


def _check_has_student_evidence(current_obs: str, check: dict[str, Any]) -> bool:
    evidence_text = str(check.get("evidence", ""))
    evidence_lower = evidence_text.lower()
    combined_lower = f"{current_obs} {evidence_text}".lower()
    if any(marker in evidence_lower for marker in _TEACHER_EVIDENCE_MARKERS):
        return False
    if any(marker in combined_lower for marker in _NEGATIVE_EVIDENCE_MARKERS):
        return False
    return bool(evidence_text.strip())


def normalize_checks(obj: dict[str, Any]) -> list[dict[str, Any]]:
    """Checks from the flat ``requirements`` map or the legacy ``checks`` list."""

    out: list[dict[str, Any]] = []
    flat = obj.get("requirements")
    if isinstance(flat, dict):
        for req, payload in flat.items():
            if not isinstance(payload, dict):
                continue
            out.append({
                "requirement": str(req).strip(),
                "visible": bool(payload.get("visible", False)),
                "evidence": str(payload.get("evidence", "")).strip(),
            })
        return out
    nested = obj.get("checks", [])
    if isinstance(nested, list):
        for entry in nested:
            if not isinstance(entry, dict):
                continue
            out.append({
                "requirement": str(entry.get("requirement", "")).strip(),
                "visible": bool(entry.get("visible", False)),
                "evidence": str(entry.get("evidence", "")).strip(),
            })
    return out


def parse_grounded_completion(
    raw: str,
) -> tuple[bool, str, list[dict[str, Any]], list[str], str]:
    """Return ``(completed, observation, checks, missing, reject_reason)``.

    ``completed`` is derived: every check the model produced must be visible
    with non-empty evidence from the student's image. An empty observation,
    visible-without-evidence, malformed JSON, or any unmet requirement
    collapses to not completed with a reason for the trace.
    """

    json_str = extract_json(raw)
    if not json_str:
        return False, "", [], [], "vlm returned non-json"
    try:
        obj = json.loads(json_str)
    except Exception:
        return False, "", [], [], "vlm returned non-json"
    if not isinstance(obj, dict):
        return False, "", [], [], "vlm returned non-json"

    current_obs = str(obj.get("observation") or obj.get("current_observation") or "").strip()
    issue_text = str(obj.get("issue") or obj.get("correction") or "").strip()
    checks = normalize_checks(obj)
    missing = [c["requirement"] for c in checks if not c["visible"] and c["requirement"]]

    if not current_obs:
        return False, "", checks, missing, "vlm omitted observation"
    if issue_text and issue_text.lower() not in {"none", "n/a", "na", "no issue"}:
        return False, current_obs, checks, missing, issue_text

    for c in checks:
        if c["visible"] and not c["evidence"]:
            return False, current_obs, checks, missing, "visible without evidence"
        if c["visible"] and not _check_has_student_evidence(current_obs, c):
            return (False, current_obs, checks, missing,
                    f"no grounded evidence for: {c['requirement']}")

    # A requirement marked visible=false is UNMET, e.g. the wrong object is in
    # place. A sibling visible check must not complete the step on its own.
    if missing:
        return False, current_obs, checks, missing, f"requirement not met: {missing[0]}"

    if not any(c["visible"] and c["evidence"] for c in checks):
        return False, current_obs, checks, missing, "no grounded evidence"
    return True, current_obs, checks, missing, ""


# ── spoken corrections ───────────────────────────────────────────────────────

_SPOKEN_EXAMPLE_DEFAULT = "your hand is covering the part that needs to move"
_CONTRADICTION_DEFAULT = (
    "Do not describe the target state as reached and then report that it is not"
)


def spoken_issue_rule(example: str = "", contradiction: str = "") -> str:
    """The rules shaping every ``issue`` string that is read to the wearer.

    Written to be spoken verbatim with nothing prepended, so the sentence has
    to carry the fix as well as the fault.
    """

    illustration = (example or _SPOKEN_EXAMPLE_DEFAULT).strip().rstrip(".")
    conflict = (contradiction or _CONTRADICTION_DEFAULT).strip().rstrip(".")
    return (
        '- The "issue" text is SPOKEN ALOUD to the person doing the step, word '
        "for word, and it is the ONLY thing they hear. Nothing is added in front "
        "of it. Write one complete spoken sentence, addressed to them, that "
        "stands on its own.\n"
        "- Say what is wrong AND what to do about it, in that order, in one "
        f'breath: "{illustration}". A correction they can act on, not a report '
        "about a discrepancy. Openings like \"Not yet —\" or \"Almost —\" are fine "
        "when the fix follows immediately.\n"
        '- Address them directly in the second person as "you". Never call them '
        '"the student" or "the user", and do not describe them from the outside '
        '— "your hand is covering the bridge", not "the student\'s hand is '
        'covering the bridge".\n'
        "- They cannot see any of these images and do not know what "
        '\"Image 1\", \"Image 2\" or \"Image 3\" means. Never mention an image '
        "number, a photo, a frame, the teacher, the reference, or comparing "
        f'anything: "{illustration}", never "it still matches Image 1".\n'
        "- Do not give the step number, do not repeat the instruction back to "
        'them, and do not open with a label like "Issue:", "Correction:" or '
        '"Problem:". Keep it under about fifteen words — it is heard, not read.\n'
        "- Your observation and your verdict must agree. If your own observation "
        "describes the target state as already present, then the step IS complete: "
        f"leave issue empty. {conflict}, and do not fail the step over something "
        "the KEY INFO lists as ignorable. Re-read your observation before writing "
        "issue.\n"
    )


# Third-person 3sg -> second-person base form. Rewriting the subject to "you"
# leaves the verb disagreeing ("you has placed") unless it is conjugated too.
_THIRD_PERSON_VERBS = {
    "has": "have", "is": "are", "was": "were", "does": "do", "goes": "go",
    "hasn't": "haven't", "isn't": "aren't", "wasn't": "weren't",
    "doesn't": "don't", "has'nt": "haven't",
}

# Regular 3sg endings, longest suffix first. Applied only to the token directly
# after a replaced subject, so a plural noun elsewhere is never touched.
_VERB_SUFFIXES = (
    ("ies", "y"),
    ("sses", "ss"),
    ("shes", "sh"),
    ("ches", "ch"),
    ("xes", "x"),
    ("zes", "ze"),
)

_STUDENT_SUBJECT = re.compile(
    r"\b(the\s+)?(student|user)(?:'s|s')(?=\s)|\b(?:the\s+)?(student|user)\b",
    re.I,
)


def _to_second_person(verb: str) -> str | None:
    """*verb* conjugated for "you", or None when it is not a 3sg verb.

    None means leave the sentence in the third person: a wrong guess is spoken
    aloud as broken grammar.
    """

    lowered = verb.lower()
    if lowered in _THIRD_PERSON_VERBS:
        return _THIRD_PERSON_VERBS[lowered]
    if not verb.isalpha() or not lowered.endswith("s") or lowered.endswith("ss"):
        return None
    for suffix, replacement in _VERB_SUFFIXES:
        if lowered.endswith(suffix):
            return lowered[: -len(suffix)] + replacement
    if lowered.endswith(("us", "is")):
        return None
    return lowered[:-1]


_IMAGE_PARENTHETICAL = re.compile(r"\s*[(\[][^()\[\]]*\bimages?\s*\d+[^()\[\]]*[)\]]", re.I)
_IMAGE_REFERENCE = re.compile(r"\b(in|on|from|of|to|per)?\s*\bimage\s*(\d+)\b", re.I)


def _second_person(text: str) -> str:
    """Rewrite "the student ..." as "you ...", conjugating the verb with it.

    Degrades rather than guesses: an unrecognised verb leaves the sentence in
    the third person with "student" swapped for "user".
    """

    pending_verbs: list[tuple[str, str]] = []

    def _rewrite(match: re.Match[str]) -> str:
        whole = match.group(0)
        possessive = bool(match.group(2))
        capitalised = whole[:1].isupper() or whole.lower().startswith("the ")
        leading_the = whole.lower().startswith("the ")
        if possessive:
            return "Your" if (capitalised and not leading_the) or leading_the else "your"

        def _degrade() -> str:
            noun = "user" if whole.lower().endswith("student") else whole.split()[-1]
            return f"the {noun}" if leading_the else noun

        rest = text[match.end():]
        verb_match = re.match(r"(\s+)([A-Za-z']+)", rest)
        if verb_match is None:
            return _degrade()
        conjugated = _to_second_person(verb_match.group(2))
        if conjugated is None:
            return _degrade()
        pending_verbs.append((verb_match.group(2), conjugated))
        return "you"

    rewritten = _STUDENT_SUBJECT.sub(_rewrite, text)
    for original, conjugated in reversed(pending_verbs):
        rewritten = re.sub(
            rf"\byou(\s+){re.escape(original)}\b",
            lambda m, c=conjugated: f"you{m.group(1)}{c}",
            rewritten,
            count=1,
        )
    return rewritten


def spoken_issue(text: str, *, student_label: str = "") -> str:
    """Rewrite a model correction into something safe to say out loud.

    The comparison prompts label their inputs "Image 1/2/3" and the model
    reaches for those labels; the wearer never saw the frames, so a frame
    reference becomes plain language here.
    """

    if not text:
        return ""
    student_index = ""
    match = re.search(r"\d+", student_label or "")
    if match:
        student_index = match.group(0)

    cleaned = _IMAGE_PARENTHETICAL.sub("", text)

    def _replace(m: re.Match[str]) -> str:
        preposition, index = m.group(1), m.group(2)
        is_student = bool(student_index) and index == student_index
        if preposition:
            return " right now" if is_student else f" {preposition.lower()} the reference"
        return " what I can see" if is_student else " the reference"

    cleaned = _IMAGE_REFERENCE.sub(_replace, cleaned)
    cleaned = _second_person(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:]){2,}", r"\1", cleaned)
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    cleaned = cleaned.strip(" ,;:")
    return re.sub(
        r"(^|(?<=[.!?] ))([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        cleaned,
    )


def human_issue_from_raw(raw: str, *, student_label: str = "") -> str:
    json_str = extract_json(raw)
    if not json_str:
        return ""
    try:
        obj = json.loads(json_str)
    except Exception:
        return ""
    if not isinstance(obj, dict):
        return ""
    issue = str(obj.get("issue") or obj.get("correction") or "").strip()
    return spoken_issue(issue, student_label=student_label)


def issue_from_failure(
    raw: str, reject_reason: str, missing: Sequence[str], *, student_label: str = "",
) -> str:
    human_issue = human_issue_from_raw(raw, student_label=student_label)
    if human_issue:
        return human_issue
    if reject_reason:
        return reject_reason
    if missing:
        return (
            f"{missing[0]} not visible"
            if len(missing) == 1
            else f"{missing[0]} and {missing[1]} not visible"
        )
    return ""


def key_info_correction(
    objects: Sequence[str], action: str, position: str, target_state: str,
) -> str:
    """A second-person "what's wrong" line built from the step's key facts.

    The one correction on this path the model does not write, so it follows
    the same spoken rules as model output.
    """

    obj = (objects[0] if objects else "").strip()
    if target_state.strip():
        ts = target_state.strip()
        return f"You still need the {obj} {ts}" if obj else f"I don't see {ts} yet"
    if position.strip():
        return f"Get the {obj or 'it'} {position.strip()}"
    if action.strip():
        return f"You haven't {action.strip()} yet"
    return ""


# Reasons produced by the machinery rather than written for the wearer. The
# correction path must never read one aloud, and the advance gate treats a
# verdict carrying one as no evidence.
PARSER_REASON_PREFIXES = (
    "vlm ", "yolo overlay", "missing requirement", "visible without",
    "no grounded", "i cannot see", "waiting for a fresh student frame",
    "unreliable-reference", "text-video-mismatch",
)


def is_parser_issue(issue: str) -> bool:
    return issue.strip().lower().startswith(PARSER_REASON_PREFIXES)


def _tier_parser_issue(issue: str) -> bool:
    return issue.lower().startswith((
        "vlm ", "yolo overlay", "missing requirement", "visible without",
        "no grounded", "vlm omitted",
    ))


# ── one check ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class CheckResult:
    """The outcome of one grounded check, in the shape the monitor reads."""

    completed: bool = False
    current_observation: str = ""
    checks: list[dict[str, Any]] = field(default_factory=list)
    missing_or_mismatched: list[str] = field(default_factory=list)
    image_path: str = ""
    teacher_image_path: str = ""
    overlay_applied: bool = False
    timestamp_us: int = 0
    issue: str = ""
    raw_vlm: str = ""
    geometry_veto: str = ""
    tier: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "completed": self.completed,
            "current_observation": self.current_observation,
            "checks": list(self.checks),
            "missing_or_mismatched": list(self.missing_or_mismatched),
            "image_path": self.image_path,
            "teacher_image_path": self.teacher_image_path,
            "overlay_applied": self.overlay_applied,
            "timestamp_us": self.timestamp_us,
            "issue": self.issue,
            "raw_vlm": self.raw_vlm,
            "geometry_veto": self.geometry_veto,
            "tier": self.tier,
        }

    @property
    def has_evidence(self) -> bool:
        """Observation plus at least one seen check with evidence.

        The single definition of "evidence": the monitor needs it apart from
        ``completed`` for the streak, and the advance gate needs both.
        """

        if not self.current_observation.strip():
            return False
        return any(
            c.get("visible") and str(c.get("evidence", "")).strip() for c in self.checks
        )


@dataclass(frozen=True, slots=True)
class StudentImage:
    """The frame being graded."""

    path: str
    """JPEG or PNG on disk: what the model is shown and the recorder links."""

    timestamp_us: int
    overlay_applied: bool = False
    spatial_context: str = ""
    geometry: object = None


@dataclass(frozen=True, slots=True)
class StepFacts:
    """Everything about the current step and procedure the grader reads."""

    instruction: str
    teacher_image_path: str = ""
    teacher_before_image_path: str = ""
    teacher_caption: str = ""
    expected_requirements: tuple[str, ...] = ()
    key_objects: tuple[str, ...] = ()
    key_action: str = ""
    key_position: str = ""
    key_target_state: str = ""
    key_ignore: tuple[str, ...] = ()
    geometry_gate: str = ""
    wearer_context: tuple[str, ...] = ()
    part_guide: tuple[str, ...] = ()


AskImages = Callable[[Sequence[str], str], Awaitable[str]]
"""Ask the VLM one question about 1-4 image files and return its text."""

AnnotateTeacher = Callable[[str], Awaitable[tuple[str, bool]]]
"""Annotate a teacher frame with the detector: returns (path, applied)."""


@dataclass(frozen=True, slots=True)
class OverlayPrompting:
    """Detector-derived prompt material for one check."""

    enabled: bool = False
    """Whether the detector overlay is on; then every image must be annotated."""

    live_guide: str = ""
    comparison_guide: Callable[[list[str]], str] | None = None
    spoken_example: str = ""
    contradiction_example: str = ""
    veto: Callable[[object, str], str] | None = None
    request_veto: Callable[[object, str, Sequence[str]], str] | None = None
    """Geometry's answer to "is that the part they asked for?"; see ``check_step``."""


async def _diagnose_mistake(
    ask: AskImages,
    image_path: str,
    *,
    instruction: str,
    objects: Sequence[str],
    action: str,
    position: str,
    target_state: str,
    overlay_guide: str = "",
) -> str:
    """Tier 3: name the single most important thing the student is doing wrong."""

    facts: list[str] = []
    objs = [o for o in objects if o and o.strip()]
    if objs:
        facts.append(f"required object(s): {', '.join(objs)}")
    if action.strip():
        facts.append(f"action: {action.strip()}")
    if position.strip():
        facts.append(f"placement: {position.strip()}")
    if target_state.strip():
        facts.append(f"done when: {target_state.strip()}")
    facts_block = ("\n".join(f"  - {f}" for f in facts) + "\n") if facts else ""

    question = (
        f"The user is trying to do this step: {instruction}\n"
        f"{facts_block}"
        f"{overlay_guide}"
        "The step is NOT complete. Look ONLY at this live image and say, in ONE "
        "short second-person sentence, the single most important thing that is "
        "wrong — most often the WRONG OBJECT (name what you actually see vs what "
        "the step needs), otherwise wrong placement or that it is not done yet. "
        "Do NOT claim it is correct.\n"
        "This sentence is SPOKEN ALOUD to the user with nothing added in "
        "front of it, so it has to stand on its own: say what is wrong AND what "
        "to do about it, addressed to them as \"you\", and never mention images, "
        "photos, frames, the teacher, or a reference to compare against.\n\n"
        'Output ONLY this JSON: {"issue": "<one short correction to the user>"}'
    )
    try:
        raw = await ask([image_path], question)
    except Exception:
        logger.exception("guidance tier-3 diagnosis failed")
        return ""
    return human_issue_from_raw(raw)


async def check_step(
    *,
    facts: StepFacts,
    student: StudentImage,
    ask: AskImages,
    overlay: OverlayPrompting,
    annotate_teacher: AnnotateTeacher | None = None,
    tier2_parallel: bool = False,
) -> CheckResult:
    """Run one grounded completion check for the current step."""

    image_path = student.path
    frame_ts = student.timestamp_us
    student_overlay_applied = student.overlay_applied
    student_spatial_context = student.spatial_context

    geometry_issue = (
        overlay.veto(student.geometry, facts.geometry_gate)
        if overlay.veto is not None and facts.geometry_gate else ""
    )
    # The request check below is the VLM grading itself, and it has passed a
    # step whose own evidence said the detector saw the other size in the hand.
    # Where the detector can tell the sizes apart, that is box arithmetic.
    if (not geometry_issue and overlay.request_veto is not None
            and facts.geometry_gate and facts.wearer_context):
        geometry_issue = overlay.request_veto(
            student.geometry, facts.geometry_gate, facts.wearer_context,
        )

    key_block = key_info_block(
        objects=facts.key_objects,
        action=facts.key_action,
        position=facts.key_position,
        target_state=facts.key_target_state,
        ignore=facts.key_ignore,
    )
    key_lines = f"{key_block}\n\n" if key_block else ""

    wearer_block = wearer_context_block(facts.wearer_context)
    wearer_lines = f"{wearer_block}\n\n" if wearer_block else ""
    # Gated on there being a spoken request: without one the step never asks
    # which interchangeable part is in frame, and the shapes would invite the
    # model to fail a part choice the SOP left open on purpose.
    part_block = part_guide_block(facts.part_guide) if wearer_block else ""
    part_lines = f"{part_block}\n\n" if part_block else ""

    expected = [r.strip() for r in facts.expected_requirements if r and str(r).strip()]
    if wearer_block and REQUEST_CHECK_NAME not in expected:
        expected = [*expected, REQUEST_CHECK_NAME]
    if expected:
        checklist_block = "\n".join(f"  - {r}" for r in expected)
        checklist_lines = (
            "REQUIREMENTS (use as visual hints, not as stricter wording than the "
            f"instruction):\n{checklist_block}\n\n"
        )
    else:
        checklist_lines = (
            "No predefined requirements were provided. Derive 1-2 short, "
            "visually checkable requirements from the INSTRUCTION and include "
            "those requirement texts in the JSON.\n\n"
        )

    teacher_path = facts.teacher_image_path
    teacher_overlay_applied = False
    if teacher_path and annotate_teacher is not None:
        annotated, teacher_overlay_applied = await annotate_teacher(teacher_path)
        if teacher_overlay_applied:
            teacher_path = annotated
        elif overlay.enabled:
            # Never compare an annotated student image against an unannotated
            # teacher image under an overlay-aware prompt.
            teacher_path = ""

    before_path = facts.teacher_before_image_path if teacher_path else ""
    if before_path and annotate_teacher is not None:
        annotated, applied = await annotate_teacher(before_path)
        if applied:
            before_path = annotated
        elif overlay.enabled:
            # A raw before-frame under an overlay-aware prompt reads as "the
            # boxes vanished", itself a difference the model would explain.
            before_path = ""

    comparison_overlay_guide = ""
    live_overlay_guide = ""
    if student_overlay_applied:
        live_overlay_guide = overlay.live_guide
        if teacher_overlay_applied and overlay.comparison_guide is not None:
            names = ["Image 1", "Image 2"] + (["Image 3"] if before_path else [])
            comparison_overlay_guide = overlay.comparison_guide(names)

    rule = spoken_issue_rule(overlay.spoken_example, overlay.contradiction_example)

    live_question = (
        f"INSTRUCTION: {facts.instruction}\n"
        f"TEACHER CAPTION: {facts.teacher_caption or 'not available'}\n"
        f"{key_lines}"
        f"{part_lines}"
        f"{wearer_lines}"
        f"{checklist_lines}"
        f"{live_overlay_guide}"
        f"{student_spatial_context}"
        "Look ONLY at this live student image.\n"
        "Decide whether it satisfies the KEY INFO — the key OBJECTS (by "
        "type/identity) in the target end-state/placement, performing the action.\n"
        "OBJECT IDENTITY IS STRICT: if the student is using a DIFFERENT object "
        "than the step requires (e.g. a phone instead of an AirPod case), the "
        "step is NOT satisfied — even if it is placed correctly.\n"
        "You MAY ignore color shade, exact brand, background, lighting, camera "
        "angle, and clothing; you may NOT ignore the object's type/identity, its "
        "placement, or the action.\n\n"
        "Output ONLY this JSON, no prose, no markdown:\n"
        "{\n"
        '  "observation": "<one sentence naming the object(s) you actually see>",\n'
        '  "requirements": {\n'
        '    "<each key object/placement, named>": {"visible": true|false, "evidence": "<live-image cue, or empty>"}\n'
        "  },\n"
        '  "issue": "<if any check is false, a concrete correction naming the wrong/missing object vs the expected one; else empty>"\n'
        "}\n\n"
        "Rules:\n"
        "- observation MUST be a non-empty sentence about THIS live image, naming the object(s) present.\n"
        "- Add one check per key object/placement; set visible=false when the object is the wrong type, missing, or misplaced.\n"
        "- visible=true REQUIRES non-empty evidence from THIS live image; do not invent evidence.\n"
        "- If any check is false, issue MUST name what is wrong vs what the step expects.\n"
        + rule
    )

    def _result(
        *, completed: bool, obs: str, checks: list[dict[str, Any]], missing: list[str],
        issue: str, raw: str, tier: str,
    ) -> CheckResult:
        return CheckResult(
            completed=completed,
            current_observation=obs,
            checks=checks,
            missing_or_mismatched=missing,
            image_path=image_path,
            teacher_image_path=teacher_path,
            overlay_applied=student_overlay_applied,
            timestamp_us=frame_ts,
            issue=issue,
            raw_vlm=raw,
            geometry_veto=geometry_issue,
            tier=tier,
        )

    live_task: Any = None
    if tier2_parallel:
        import asyncio

        live_task = asyncio.ensure_future(ask([image_path], live_question))

    teacher_failure: CheckResult | None = None
    try:
        if teacher_path:
            if before_path:
                # A step defined by a CHANGE cannot be judged from the after-frame
                # alone: pinching a pad still attached and one just removed are
                # near-identical pixels. Naming the before-state makes it "which
                # of these two is this?", which the model answers reliably.
                framing = (
                    "Image 1 is the teacher BEFORE performing this step.\n"
                    "Image 2 is the teacher AFTER performing this step — the target.\n"
                    "Image 3 is the student's current state.\n"
                    "The step is about the CHANGE from Image 1 to Image 2. First name "
                    "what actually differs between them; that difference IS the step.\n"
                    "Then decide which one Image 3 resembles on that difference alone. "
                    "If Image 3 still matches Image 1, or sits part-way between the "
                    "two, the step is NOT complete — say so, even when Image 3 "
                    "otherwise looks a lot like Image 2.\n"
                )
                student_label = "Image 3"
                target_label = "Image 2"
                image_paths = [before_path, teacher_path, image_path]
            else:
                framing = (
                    "Image 1 is the teacher's completed reference state for this step.\n"
                    "Image 2 is the student's current state.\n"
                )
                student_label = "Image 2"
                target_label = "Image 1"
                image_paths = [teacher_path, image_path]
            comparison_question = (
                f"INSTRUCTION: {facts.instruction}\n"
                f"{framing}"
                f"Teacher reference caption: {facts.teacher_caption or 'not available'}\n"
                f"{key_lines}"
                f"{part_lines}"
                f"{wearer_lines}"
                f"{checklist_lines}"
                f"{comparison_overlay_guide}"
                f"{student_spatial_context}"
                f"Decide whether {student_label} matches {target_label} for the KEY "
                "INFO — the key OBJECTS (by type/identity), the action, and the "
                "spatial placement.\n"
                "OBJECT IDENTITY IS STRICT: the student must be using the SAME kind "
                "of object the step calls for. A DIFFERENT object in the right place "
                "is a MISMATCH, not a match (e.g. a phone where an AirPod case is "
                "required — fail it).\n"
                "You MAY ignore: color shade, exact brand, background, lighting, "
                "camera angle, distance, clothing, hand pose. You may NOT ignore the "
                "object's type/identity, its placement, or the action.\n\n"
                "Output ONLY this JSON, no prose, no markdown:\n"
                "{\n"
                f'  "observation": "<one sentence naming the object(s) you actually see in {student_label}>",\n'
                '  "requirements": {\n'
                f'    "<each key object/placement, named>": {{"visible": true|false, "evidence": "<{student_label} cue, or empty>"}}\n'
                "  },\n"
                '  "issue": "<if any check is false, a concrete correction naming the wrong/missing object vs the expected one; else empty>"\n'
                "}\n\n"
                "Rules:\n"
                "- Add one check per key object/placement; set visible=false when the "
                "object is the wrong type, missing, or misplaced.\n"
                f"- Evidence must come from {student_label}, not from the teacher images.\n"
                f"- If every key check is visible=true with {student_label} evidence, leave issue empty.\n"
                "- If any check is false, issue MUST name what is wrong (e.g. "
                '"that looks like a phone, but this step needs the AirPod case").\n'
                + rule
            )
            try:
                compare_raw = await ask(image_paths, comparison_question)
            except Exception as exc:
                logger.warning("comparison check failed; using the live-image tier: {}", exc)
                compare_raw = ""
            if compare_raw:
                cmp_completed, cmp_obs, cmp_checks, cmp_missing, cmp_reject = (
                    parse_grounded_completion(compare_raw)
                )
                cmp_issue = issue_from_failure(
                    compare_raw, cmp_reject, cmp_missing, student_label=student_label,
                )
                if cmp_completed and wearer_block and not request_check_present(cmp_checks):
                    logger.info("GUIDANCE_REQUEST_CHECK_MISSING tier=compare obs={!r}",
                                cmp_obs[:60])
                    cmp_completed = False
                    cmp_issue = REQUEST_CHECK_MISSING
                if cmp_completed and geometry_issue:
                    # Return rather than fall through: the live tier is a weaker
                    # question than the one geometry just answered, and letting
                    # every vetoed frame pay for it would add a round trip to the
                    # common "not done yet" case.
                    logger.info("GUIDANCE_GEOMETRY_VETO gate={} tier=compare issue={!r}",
                                facts.geometry_gate, geometry_issue)
                    return _result(completed=False, obs=cmp_obs, checks=cmp_checks,
                                   missing=cmp_missing, issue=geometry_issue,
                                   raw=compare_raw, tier="compare")
                if cmp_completed:
                    return _result(completed=True, obs=cmp_obs, checks=cmp_checks,
                                   missing=cmp_missing, issue="", raw=compare_raw,
                                   tier="compare")
                teacher_failure = _result(completed=False, obs=cmp_obs, checks=cmp_checks,
                                          missing=cmp_missing, issue=cmp_issue,
                                          raw=compare_raw, tier="compare")

        if live_task is not None:
            live_raw = await live_task
            live_task = None
        else:
            live_raw = await ask([image_path], live_question)
    finally:
        if live_task is not None:
            live_task.cancel()

    completed, current_obs, checks, missing, reject_reason = parse_grounded_completion(live_raw)
    live_issue = issue_from_failure(live_raw, reject_reason, missing, student_label="")
    if completed and wearer_block and not request_check_present(checks):
        logger.info("GUIDANCE_REQUEST_CHECK_MISSING tier=live obs={!r}", current_obs[:60])
        completed = False
        live_issue = REQUEST_CHECK_MISSING
    if completed and geometry_issue:
        logger.info("GUIDANCE_GEOMETRY_VETO gate={} tier=live issue={!r}",
                    facts.geometry_gate, geometry_issue)
        completed = False
        live_issue = geometry_issue
        teacher_failure = None
    if completed:
        return _result(completed=True, obs=current_obs, checks=checks, missing=missing,
                       issue="", raw=live_raw, tier="live")

    final_issue = (
        teacher_failure.issue
        if (
            teacher_failure is not None
            and teacher_failure.issue
            and not _tier_parser_issue(teacher_failure.issue)
        )
        else live_issue
    )
    tier = "live"
    if not final_issue or _tier_parser_issue(final_issue):
        diagnosed = await _diagnose_mistake(
            ask, image_path,
            instruction=facts.instruction,
            objects=facts.key_objects, action=facts.key_action,
            position=facts.key_position, target_state=facts.key_target_state,
            overlay_guide=live_overlay_guide,
        )
        if diagnosed:
            final_issue = diagnosed
            tier = "diagnosis"
        else:
            ki_issue = key_info_correction(
                facts.key_objects, facts.key_action, facts.key_position,
                facts.key_target_state,
            )
            if ki_issue:
                final_issue = ki_issue
                tier = "key-info"
    return _result(completed=False, obs=current_obs, checks=checks, missing=missing,
                   issue=final_issue, raw=live_raw, tier=tier)


def unreliable_reference_result() -> CheckResult:
    """The verdict for a step the automatic monitor can never grade."""

    return CheckResult(issue="unreliable-reference")


def timed_out(started: float, timeout_s: float) -> bool:
    return (time.monotonic() - started) >= timeout_s


__all__ = [
    "PARSER_REASON_PREFIXES",
    "REQUEST_CHECK_MISSING",
    "REQUEST_CHECK_NAME",
    "AnnotateTeacher",
    "AskImages",
    "CheckResult",
    "OverlayPrompting",
    "StepFacts",
    "StudentImage",
    "check_step",
    "extract_json",
    "human_issue_from_raw",
    "is_parser_issue",
    "issue_from_failure",
    "key_info_block",
    "key_info_correction",
    "normalize_checks",
    "parse_grounded_completion",
    "part_guide_block",
    "request_check_present",
    "spoken_issue",
    "spoken_issue_rule",
    "strip_thinking",
    "unreliable_reference_result",
    "wearer_context_block",
]
