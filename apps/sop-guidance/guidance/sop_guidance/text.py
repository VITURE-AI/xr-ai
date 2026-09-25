# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic text rules for spoken guidance: wake word, commands, entry.

Everything here runs without a model, so it holds regardless of what a model
decides: a bare "next" advances, "stop guidance" always ends a run, and the
wake word is matched the same way on every path.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# ── wake word ────────────────────────────────────────────────────────────────

# Pure filler tokens. A transcript made only of these said nothing useful.
FILLER: frozenset[str] = frozenset({
    "uh", "uhh", "um", "umm", "huh", "hmm", "mhm", "mm",
    "ah", "oh", "eh", "er", "err",
    "yeah", "yep", "yup", "no", "nope", "ok", "okay",
    "thanks", "thank", "you", "please",
    "the", "a", "an",
})

_WORD = re.compile(r"[A-Za-z']+")
_HAS_LETTER = re.compile(r"[A-Za-z]")
_SALUTATIONS = frozenset({
    "hey", "hay", "hei", "hi", "ok", "okay", "hello", "yo", "a", "ah", "eh", "he",
})


def _levenshtein_within(left: str, right: str, limit: int) -> bool:
    """Whether two words differ by at most *limit* edits."""

    if abs(len(left) - len(right)) > limit:
        return False
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for col, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[col] + 1,
                previous[col - 1] + (left_char != right_char),
            ))
        if min(current) > limit:
            return False
        previous = current
    return previous[-1] <= limit


def match_wake_word(
    text: str,
    *,
    names: Iterable[str],
    max_probe: int = 3,
    max_distance: int = 2,
) -> tuple[bool, str]:
    """Match a wake name near the start of *text*; return (addressed, remainder)."""

    token_re = re.compile(r"[a-z0-9']+")
    tokens = list(token_re.finditer(text.lower()))
    wake_names = {str(name).strip().lower() for name in names if str(name).strip()}
    if not tokens or not wake_names:
        return False, text

    # Fuzzy matching needs a name long enough that max_distance edits still
    # leave something distinctive: at distance 2 "hex" is reachable from every
    # four-letter token, which made "next" and "exit" read as the wake word.
    # Short aliases are in the list precisely because they are outside fuzzy
    # reach, so exact matching is their whole job.
    fuzzy_names = [n for n in wake_names if len(n) >= max_distance + 3]

    def is_name(token: str) -> bool:
        return token in wake_names or (
            len(token) >= 4
            and any(_levenshtein_within(token, name, max_distance) for name in fuzzy_names)
        )

    # STT prefixes acknowledgements constantly ("Yeah. Hey Helix, ..."), so
    # filler before the salutation still leaves the name at the start.
    prefix_ok = _SALUTATIONS | FILLER
    for index, token_match in enumerate(tokens[:max_probe]):
        token = token_match.group()
        matched = is_name(token)
        if not matched:
            for salutation in _SALUTATIONS:
                suffix = token[len(salutation):] if token.startswith(salutation) else ""
                if suffix and is_name(suffix):
                    matched = True
                    break
        if not matched:
            continue
        if index and any(preceding.group() not in prefix_ok for preceding in tokens[:index]):
            continue
        return True, text[token_match.end():].lstrip(" \t,.!?;:-")
    return False, text


def is_shape_noise(text: str) -> bool:
    """Whether *text* is obviously not a request: too short, no letters, all filler."""

    s = text.strip()
    if len(s) < 3:
        return True
    if not _HAS_LETTER.search(s):
        return True
    toks = [t.lower() for t in _WORD.findall(s)]
    if not toks:
        return True
    return all(t in FILLER for t in toks)


# ── bare commands ────────────────────────────────────────────────────────────

# Words that may surround a bare command without changing it: "okay, stop." is
# a stop; "stop notes pad" is not.
_COMMAND_FILLER = frozenset({
    "a", "ah", "alright", "and", "er", "hmm", "just", "kay", "now", "ok",
    "okay", "please", "right", "so", "uh", "um", "well", "yeah", "yes",
})

FAST_ADVANCE = ("next", "next step", "continue")
FAST_EXIT = ("stop", "exit")

GUIDANCE_STOP_PHRASES = (
    "stop guidance",
    "cancel guidance",
    "exit guidance",
    "stop guiding",
    "end guidance",
)

STOP_SPEAKING_PHRASES = (
    "stop talking",
    "be quiet",
    "shut up",
    "stop speaking",
    "quiet",
    "shush",
    "hush",
    "stop",
    "cancel",
)

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall(s)


def strip_filler(text: str) -> str:
    """*text* lowercased, punctuation removed, surrounding filler words dropped."""

    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    words = [w for w in cleaned.split() if w]
    while words and words[0] in _COMMAND_FILLER:
        words.pop(0)
    while words and words[-1] in _COMMAND_FILLER:
        words.pop()
    return " ".join(words)


def is_fast_advance(text: str) -> bool:
    """Bare "next" / "next step" / "continue", modulo filler.

    Strict equality: "next time I see this" is a comment, not a command. The
    doubled form is kept because STT bundles a repeated word into one
    utterance ("next next"), which must still advance exactly one step.
    """

    core = strip_filler(text)
    if core in FAST_ADVANCE:
        return True
    tokens = tokenize(core)
    half = len(tokens) // 2
    return bool(
        half and len(tokens) == half * 2 and tokens[:half] == tokens[half:]
        and " ".join(tokens[:half]) in FAST_ADVANCE
    )


def is_guidance_stop(text: str) -> bool:
    """An explicit "stop guidance" style phrase anywhere in *text*."""

    s = text.lower().strip()
    return any(phrase in s for phrase in GUIDANCE_STOP_PHRASES)


def is_guidance_voice_exit(text: str) -> bool:
    """Exits are the only commands exempt from the wake word during guidance."""

    core = strip_filler(text)
    return is_guidance_stop(core) or core in FAST_EXIT


def is_stop_speaking(text: str) -> bool:
    """Generic "be quiet" outside guidance: flush speech, start nothing."""

    s = text.lower().strip().rstrip(".!?")
    words = set(s.replace(",", " ").replace(".", " ").split())
    for phrase in STOP_SPEAKING_PHRASES:
        if s == phrase:
            return True
        if " " in phrase and phrase in s:
            return True
        if " " not in phrase and phrase in words and len(words) <= 3:
            return True
    return False


# ── takeover confirmation ────────────────────────────────────────────────────

_AFFIRMATIVE = re.compile(
    r"(?:yes(?: please)?|yeah|yep|sure|ok|okay|go ahead"
    r"|yes[, ]+(?:go ahead|stop (?:it|that session)|start (?:mine|my session)))"
)
_NEGATIVE = re.compile(r"(?:no|no thanks|no thank you|cancel|never mind|nevermind)")


def confirmation_answer(text: str) -> bool | None:
    """True for a yes, False for a no, None when *text* is neither."""

    lower = text.lower().strip().rstrip(".!?,")
    if _AFFIRMATIVE.fullmatch(lower):
        return True
    if _NEGATIVE.fullmatch(lower):
        return False
    return None


# ── explicit entry controls ──────────────────────────────────────────────────

_GUIDANCE_REQUEST = re.compile(
    r"^(?:(?:please|can you|could you|would you|can we|could we|i want to)\s+)*"
    r"(?P<command>(?:guide|walk|take|get) me through|show me how to|"
    r"resume|continue|carry on|pick up where we left off|"
    r"start over|start again|restart|go to|jump to|go back to|start from|start at|"
    r"skip (?:ahead |forward )?to|take me to|move (?:on )?to)\b"
    r"(?P<target>.*)$"
)
_STEP_NUMBER = re.compile(r"\bstep\s+(-?\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b")
_NUMBER_WORDS = "zero one two three four five six seven eight nine ten".split()


def guidance_request(text: str) -> tuple[str, int, str] | None:
    """Recognise an explicit entry control as ``(mode, step, target)``.

    ``mode`` is ``start``, ``resume`` or ``step``; ``step`` is 1-based for
    ``step`` mode; ``target`` is the procedure name as spoken, or "".
    """

    normalized = re.sub(r"[^a-z0-9' -]", " ", text.lower())
    normalized = " ".join(normalized.split())
    match = _GUIDANCE_REQUEST.fullmatch(normalized)
    if match is None:
        return None
    command, target = match.group("command", "target")
    # A requested restart wins over a remembered or merely mentioned step.
    beginning = bool(re.search(
        r"\b(?:from the beginning|from scratch|start over|start again|restart)\b", normalized,
    ))
    number = _STEP_NUMBER.search(target)
    # Jumps name a step: "go to the sink" or "move on to the next part" are not.
    jumps = {"go to", "jump to", "go back to", "start from", "start at", "skip to",
             "skip ahead to", "skip forward to", "take me to", "move to", "move on to"}
    if command in jumps and not (number or beginning):
        return None
    if beginning and not re.search(r"\bnot from the beginning\b", normalized):
        mode, step = "start", 0
    elif number and re.search(r"\b(?:from|at|in|to) step\b", normalized):
        value = number.group(1)
        mode = "step"
        step = int(value) if value.lstrip("-").isdigit() else _NUMBER_WORDS.index(value)
    elif command in {"resume", "continue", "carry on", "pick up where we left off"}:
        mode, step = "resume", 0
    else:
        mode, step = "start", 0
    target = re.sub(
        r"\b(?:(?:from|at|in|to) )?step\s+(?:-?\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        "", target,
    )
    target = re.sub(
        r"\b(?:from the beginning|from scratch|where we (?:left off|stopped)|again|please)\b",
        "", target,
    )
    target = " ".join(target.split())
    if target in {"", "it", "this", "that", "the procedure", "the guide", "it from there",
                  "from there", "guidance", "the guidance", "my guidance", "guiding"}:
        target = ""
    return mode, step, target


# ── wearer requests ──────────────────────────────────────────────────────────

# Words that tell two variants of one choice APART rather than naming the
# choice. Stripped before comparing requests, so "size 0 nose pad" and "size one
# nose pad" reduce to the same subject and the second revises the first.
DEFAULT_REQUEST_QUALIFIERS = frozenset({
    "a", "an", "the", "my", "one", "ones", "please", "use", "want", "with",
    "size", "sized", "number", "no",
    "zero", "two", "three", "four", "five", "first", "second",
    "small", "big", "large", "left", "right",
})


def request_subject(text: str, qualifiers: frozenset[str]) -> frozenset[str]:
    """The words in *text* that say WHAT it is about, qualifiers removed."""

    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return frozenset(
        w for w in cleaned.split() if w and not w.isdigit() and w not in qualifiers
    )


# ── announcements ────────────────────────────────────────────────────────────

# The host's own announcement shape, "Step 3 of 4: ...". Requiring a NUMBER
# after "of" keeps it off talk about a procedure ("step 3 of the nosepad
# replacement"). Only the host announces steps, from authored text.
STEP_ANNOUNCEMENT = re.compile(r"\bsteps?\s+\d+\s+of\s+\d+", re.IGNORECASE)


def step_announcement(index: int, total: int, instruction: str) -> str:
    return f"Step {index + 1} of {total}: {instruction}"


__all__ = [
    "DEFAULT_REQUEST_QUALIFIERS",
    "FAST_ADVANCE",
    "FAST_EXIT",
    "FILLER",
    "GUIDANCE_STOP_PHRASES",
    "STEP_ANNOUNCEMENT",
    "STOP_SPEAKING_PHRASES",
    "confirmation_answer",
    "guidance_request",
    "is_fast_advance",
    "is_guidance_stop",
    "is_guidance_voice_exit",
    "is_shape_noise",
    "is_stop_speaking",
    "match_wake_word",
    "request_subject",
    "step_announcement",
    "strip_filler",
    "tokenize",
]
