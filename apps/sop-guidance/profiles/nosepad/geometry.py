# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detector-geometry reasoning for the nose-pad replacement SOP.

Loaded by path with ``sop_guidance.vision.load_geometry_plugin`` — read
``sop_guidance/vision/geometry.py`` for the hook contract and why ``veto``
returns a reason string rather than a verdict. It is the reference
implementation of that contract as much as it is this task's logic.

The vote windows below are module state, and that is per plugin INSTANCE, not
per process: every ``load_geometry_plugin`` call executes this file into a
fresh module, so two loads keep two independent histories.

Answers the one question the boxes answer better than prose — is a pad ON the
glasses, or merely held NEAR them — which is precisely where the VLM was
failing: a pad pinched at the bridge looks, to a language model reading a
caption, much like a pad seated in it.

Two rules do the work. Per pad, a hand beats the glasses: a pad inside a hand box
is held even when it also overlaps the glasses. Across pads, a held pad casts
doubt on any pad reported on the glasses at the same moment — with the
replacement still in hand the bridge is usually empty, and a box there is more
often the socket detected as a pad than a fitted one.

Known blind spot, confirmed on a real frame: once a pad is in a hand the boxes
cannot tell attached from detached, because the detector emits no pad box at all
for the pinch pose in ``step_02_a.jpg``. Step 2's gate contributes nothing there
and the SOP's before/after reference pair carries the step instead. Do not try to
make geometry solve it.

The counts a gate reads are voted over a short window of recent frames rather
than taken from the one frame in hand — see "temporal persistence" below.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, replace

# The vision package exports the box primitives; only the task meaning lives here.
from sop_guidance.vision.overlay import Detection, union_coverage

# Detector class labels this task reasons about, as they appear after
# `overlay.class_labels` renaming. A checkpoint with a different vocabulary
# needs its own plugin, not a wider one.
GLASSES_LABEL = "glasses"
# Every label that means "a nose pad". The v3 checkpoint splits pads by SIZE
# into nosepad_0 / nosepad_1, and the overlay keeps those apart so the VLM can
# read the distinction off the drawn boxes. Size is irrelevant to every relation
# computed here -- what matters is where a pad is and whose hand is on it -- so
# this matches a set and the plugin never learns which variant it is looking at.
#
# That is the whole reason the split can exist in the overlay without a second
# plugin: the two consumers of `overlay.class_labels` want different things, and
# only one of them cares about the difference.
PAD_LABELS = frozenset({"nosepad", "nosepad_0", "nosepad_1"})
HAND_LABEL = "hand"

# Relation thresholds, as a fraction of the nose pad's own box area. Measured on
# the five teacher frames of the nosepad SOP, which cover every state the task
# has (pad installed / pinched in hand / spares on the table / held at the bridge
# / seated):
#
#   step_01 installed        in_glasses 1.00  in_hand 0.00   gap   0%
#   step_02 pinched in hand  in_glasses 0.00  in_hand 0.96   gap   6%
#   step_03 spares on table  in_glasses 0.00  in_hand 0.02   gap  50-58%
#   step_04 held at bridge   in_glasses 1.00  in_hand 1.00   gap   0%
#   step_05 seated           in_glasses 1.00  in_hand 0.00   gap   0%
#
# step_04 and step_05 are the pair that matters: both put the pad wholly inside
# the glasses box, and only the hand overlap separates "about to fit it" from
# "fitted". Hence a pad in a hand is reported as held even when it also overlaps
# the glasses — the hand wins.
_SEATED_IN_GLASSES = 0.60
_HELD_IN_HAND      = 0.60
# How far a pad in a hand must be from the glasses before it stops counting as
# part of this step, as a percentage of the glasses cluster's width.
#
# A pad in a hand normally means "still in progress", and that reading is what
# covers the empty bridge socket being detected as a pad while the real one is
# being lined up. But it is wrong for a spare lying on the table that a hand
# happens to pass in front of: in the 2D projection the spare falls inside the
# hand box, and treating it as in-progress unseats a pad that is already fitted.
#
# Distance separates the two with a wide margin on the teacher frames: the pad
# being lined up at the bridge (step_02) is 6% away, the spares on the table
# (step_03) are 50-58%. Anything in a hand that is not provably past this line
# is still read as in-progress, so a bad measurement costs a retry rather than a
# false pass.
_HELD_AWAY_MIN_GAP_PCT = 25.0

# ── the bridge, and when a hand has covered it ───────────────────────────────
#
# The one case boxes genuinely cannot answer, stated in this module's own
# docstring: "once a pad is in a hand the boxes cannot tell attached from
# detached, because the detector emits no pad box at all for the pinch pose".
# What that leaves behind is a frame with the glasses in view, a hand across the
# bridge, and NO pad box anywhere -- which every count here reports as zero, and
# `pad_on_glasses` therefore answers with "I do not see a nose pad resting in the
# bridge yet. Push it into the slot". At a wearer whose fingers are on the pad
# they just seated, that is not a missed detection producing silence, it is a
# missed detection producing a confident instruction to redo finished work.
#
# So the absence is distinguished from a clear view of an empty bridge, and the
# distinction is told to the VLM as prose -- which then judges from the pixels
# the boxes do not have. NO GATE READS IT, and that is the measured part.
#
# It was first built with both bridge gates vetoing on it and speaking "move your
# hand away so I can check". On the v3 test split at the shipped thresholds that
# fired on 9% of frames and was right 16% of the time: 6 frames where a pad truly
# was at the bridge, against 32 where the bridge really was clear and the
# ordinary "push it into the slot" had been CORRECT. Five needless requests per
# avoided false correction is worse than the thing it replaced, so the argument
# that carried it -- asking is cheaper than correcting wrongly -- is simply false
# at this accuracy.
#
# Swept before giving up on it: 4 band widths x 4 coverage bars, all 16 cells
# between 10% and 22% right. Tightening cuts firings proportionally and never
# improves the ratio, so the band is not the problem. "A hand over the bridge
# with no pad box" is dominated by frames where the bridge is genuinely empty and
# a hand is merely in front of it. Conditioning on a pad being visible elsewhere
# in the frame gave 27% against 9%, on n=15 and n=23 -- no separation worth
# having.
#
# What would make it actionable is a temporal precondition: ask only if a pad was
# confirmed AT the bridge shortly before the hand arrived, which targets those 6
# frames directly and which `_confirmed` already records. That cannot be measured
# on this split -- `prepare.split_blocks` samples blocks within source videos, so
# there is no continuous footage in it. Needs a held-out recording session, not
# another threshold.
#
# Do not re-add a veto here without that measurement. The prose costs nothing
# because the model can look; a gate speaking on this evidence costs trust.
#
# The bridge is the centre band of the glasses cluster, full height. There is no
# bridge class in the checkpoint and adding one is a retrain; the band is where a
# nose pad sits on every frame in the dataset. Wide, because it only has to be
# good enough to caveat a prompt with.
_BRIDGE_BAND_FRAC = 0.34
# Fraction of that band a hand must cover. Hands are large and confidently
# detected -- median hand box is ~40% of frame width against a pad's ~5% -- so
# this is not a marginal measurement.
_BRIDGE_OBSCURED_COVERAGE = 0.50
# `glasses_lens_area` covers the whole eyewear in one box — it is not per lens —
# but the detector sometimes emits extra overlapping boxes for the same pair
# (two on step_04, three confident ones on step_02). Relations therefore run
# against the best box plus any box touching it, with overlap measured against
# that cluster's UNION (see union_coverage): a duplicate that splits the frame
# leaves a bridge pad at ~0.5 against either piece while it sits wholly on the
# glasses. Boxes that touch nothing are left out, and low-confidence ones are
# dropped first — step_02's run down to 0.26, a real detection scores 0.93-0.96.
_GLASSES_MIN_CONF  = 0.50

# Per-step gates a SOP step may opt into via ``geometry_gate``. These only ever
# VETO a positive VLM verdict; nothing here can turn a "no" into a "yes", so a
# missed detection costs a retry rather than a false pass.
GATES = ("pad_on_glasses", "no_pad_on_glasses", "pad_in_hand")

# ── temporal persistence ─────────────────────────────────────────────────────
#
# The detector is scored one frame at a time but it is FED a 15 fps stream, so a
# single-frame count throws away almost all the evidence on hand. On the held-out
# test split `nosepad` carries 88 of the checkpoint's 121 false positives at conf
# 0.25 (~42 of ~52 at 0.6), and the confusion matrix has no class-to-class error
# at all — every one of them is a box invented on background rather than a pad
# mistaken for something else.
#
# A sporadic invented box does not survive being asked again; a real pad does. At
# the measured per-frame recall of ~0.83, a real pad clears "a majority of the
# last half-second" ~98% of the time, while a box appearing in one frame of seven
# contributes nothing. Voting the COUNTS rather than tracking boxes is deliberate:
# no association, no IoU matching, no identity to lose when a pad is occluded —
# the k-th largest count over the window is exactly "the number of pads at least
# k frames agree on".
#
# What this does NOT fix, and must not be claimed to: an empty bridge socket read
# as a pad is the same pixels every frame, so it votes as steadily as a real pad.
# Persistence removes flicker, not systematic confusion. The cross-pad `held`
# discount in `pads_on_glasses` is still the only thing covering that case.
#
# Voting is keyed by stream and only happens when the caller supplies one. A
# one-off `analyze` on an arbitrary still has no temporal context and must not
# pretend to have any, so it falls through to the raw per-frame counts — which is
# also what the check does when the preview loop has stalled and the window has
# aged out from under it.
_VOTE_WINDOW_S = 0.5
# Below this many samples "a majority" is too weak a claim to be worth making,
# and the window is still filling after a stream starts.
_VOTE_MIN_SAMPLES = 3

# ── coasting ─────────────────────────────────────────────────────────────────
#
# The median above can only ever SUBTRACT. That is the right shape for an
# invented box, and it is why the docstring above can claim a real pad clears a
# majority ~98% of the time -- but that claim rests on ~0.83 recall, which is
# the number for a pad lying loose on the bench. Measured per state on the v4b
# checkpoint (v3 test split, conf 0.50, class-agnostic because PAD_LABELS is):
#
#   loose    0.897      majority holds
#   held     0.727      majority usually holds
#   mounted  0.693      majority is marginal
#   mounted, size-0 alone   0.481   <- majority DOES NOT hold
#
# Below one half the median is not smoothing the dropout, it is reproducing it:
# fewer than half the frames carry the pad, so the voted count is 0 and stays 0
# while the pad is plainly seated. `pad_on_glasses` vetoes on a zero count, so
# the wearer who has just fitted a size-0 pad gets told "I do not see a nose pad
# resting in the bridge yet" -- a spoken contradiction, which costs far more
# trust than the silent lost veto a false positive costs. (`veto` is
# one-directional: no box here can create a pass, so an invented one only ever
# loses a catch, while a missed one manufactures a correction.)
#
# So a category that a majority has confirmed stays asserted for a short while
# after losing that majority, at the count that was confirmed. Two properties
# make this safe, and both must survive any edit here:
#
#   1. Arming requires a MAJORITY. A count no window ever confirmed is never
#      coasted, so this cannot invent a category -- exactly the argument the
#      median itself rests on. Coasting re-asserts; it does not detect.
#   2. It only ever raises a count back to a confirmed value, never above it,
#      and it expires. A pad genuinely removed is gone after _COAST_S.
#
# Checked against each gate's own conservative direction, which is not the same
# direction for all of them (see `veto`): a coasted `seated` makes
# `pad_on_glasses` veto LESS -- the false correction this exists to stop -- and
# makes `no_pad_on_glasses` veto MORE, which is the safe way for that one to be
# wrong. A coasted `held_at_bridge` holds the `pads_on_glasses` discount longer,
# also conservative. A coasted `held` stops `pad_in_hand` vetoing, but only once
# a majority saw a pad in a hand, which means the wearer did pick one up. All
# four move safely under one rule, so there is deliberately no per-category
# special case here.
#
# What this does NOT fix, same as the median: a steady phantom. The empty bridge
# socket is majority-confirmed on its own merits and needs no help from coasting.
# Its five instances in the test split score 0.57-0.77 -- above every threshold
# under discussion -- so it is not a `conf` problem either. The cross-pad `held`
# discount remains the only thing covering it.
_COAST_S = 0.6

# (monotonic, seated, held_at_bridge, held_away, loose, obscured)
_history: dict[str, deque[tuple[float, int, int, int, int, int]]] = {}
# Per stream, one (monotonic, count) per category: the last count a majority of
# that category confirmed, and when. Lives and dies with that stream's ring in
# `_history` -- a stalled preview loop whose window has aged out must not coast
# either, because it has no idea what the scene is doing.
_confirmed: dict[str, tuple[tuple[float, int], ...]] = {}
_history_lock = threading.Lock()

# Field order of the four counts everywhere in this module, and the words the
# prose uses for them. One list so a reordering cannot desynchronise the two.
_CATEGORIES = ("seated", "held at the bridge", "held away", "loose")


def _vote(
    stream: str, seated: int, held_at_bridge: int, held_away: int, loose: int,
    obscured: bool = False,
) -> tuple[tuple[int, int, int, int], tuple[str, ...], bool]:
    """Record this frame's counts; answer the voted ones and what was coasted.

    Returns the counts unchanged for an unkeyed call, or while the window holds
    fewer than ``_VOTE_MIN_SAMPLES``. Falling back to raw rather than to zero is
    the safe direction and not a detail: `pad_on_glasses` vetoes when its count
    is 0, so a cold window that voted everything to zero would veto every fit
    step until it filled.

    The second element names the categories whose answer came from a recent
    confirmation rather than from this window (see "coasting"). It is empty in
    the ordinary case and exists so the prose can avoid claiming a box the
    detector did not just produce -- nothing gates on it.

    *obscured* rides the same window because it needs the same smoothing -- one
    frame of a hand sweeping past the bridge must not make the system ask the
    wearer to move their fingers. It is voted and deliberately NOT coasted: the
    counts are a claim about an object, which persists and so is worth
    re-asserting through a dropout, while this is a claim about the CURRENT
    view, which does not. Coasting it would keep asking after the hand had
    already moved away.
    """
    if not stream:
        return (seated, held_at_bridge, held_away, loose), (), obscured
    now = time.monotonic()
    cutoff = now - _VOTE_WINDOW_S
    with _history_lock:
        window = _history.setdefault(stream, deque())
        window.append(
            (now, seated, held_at_bridge, held_away, loose, int(obscured)),
        )
        # Prune every stream, not just this one: a participant who leaves never
        # calls again, and its ring would otherwise sit here for the process
        # lifetime. The dict is one entry per live wearer, so this is free.
        for key in list(_history):
            ring = _history[key]
            while ring and ring[0][0] < cutoff:
                ring.popleft()
            if not ring:
                del _history[key]
                # Dies with the ring, deliberately: coasting past a stalled
                # preview loop would re-assert a scene nobody has looked at.
                _confirmed.pop(key, None)
        samples = list(_history.get(stream, ()))
        if len(samples) < _VOTE_MIN_SAMPLES:
            return (seated, held_at_bridge, held_away, loose), (), obscured
        need = len(samples) // 2 + 1
        # Majority of the window, on the same k-th-largest rule as the counts.
        obscured_voted = bool(
            sorted((sample[5] for sample in samples), reverse=True)[need - 1],
        )
        voted = tuple(
            sorted((sample[i] for sample in samples), reverse=True)[need - 1]
            for i in (1, 2, 3, 4)
        )
        # Timed PER CATEGORY, because they expire independently: a seated pad
        # confirmed a moment ago and a spare last seen a second ago are not one
        # fact with one deadline.
        previous = _confirmed.get(stream, ((0.0, 0),) * 4)
        counts: list[int] = []
        coasted: list[str] = []
        standing: list[tuple[float, int]] = []
        for i in range(4):
            confirmed_at, confirmed_count = previous[i]
            if confirmed_count > voted[i] and now - confirmed_at <= _COAST_S:
                counts.append(confirmed_count)
                coasted.append(_CATEGORIES[i])
                # Carried forward WITHOUT refreshing the clock. Coasting off a
                # coast would renew itself every frame and never expire, which
                # is a pad welded to the bridge for the rest of the session.
                standing.append((confirmed_at, confirmed_count))
            else:
                counts.append(voted[i])
                # A majority of zero is the absence of a confirmation, not a
                # confirmation of absence, so it stores nothing to coast from.
                standing.append((now, voted[i]) if voted[i] else (0.0, 0))
        _confirmed[stream] = (standing[0], standing[1], standing[2], standing[3])
        return (
            (counts[0], counts[1], counts[2], counts[3]),
            tuple(coasted),
            obscured_voted,
        )


@dataclass(frozen=True)
class PadGeometry:
    """Counts behind the geometry prose, for steps that gate on them.

    Opaque to the caller: it is handed back to ``describe`` and ``veto`` and
    nothing else ever reads a field, so the shape is this plugin's business.
    ``prose`` is carried here so ``analyze`` classifies once and ``describe``
    only hands back the result.
    """

    glasses_seen: bool = False
    # `seated` is pads overlapping the glasses cluster with no hand on them.
    # A pad in a hand splits by where that hand is: `held_at_bridge` is at or
    # near the glasses (the in-progress pose), `held_away` is elsewhere in the
    # scene (a spare a hand is passing in front of). Only the first says
    # anything about whether this step is finished — see `pads_on_glasses`.
    #
    # These four are the VOTED counts (see "temporal persistence"); they are
    # what a gate reads, and on a keyed stream they can be lower than what this
    # frame alone showed.
    seated: int = 0
    held_at_bridge: int = 0
    held_away: int = 0
    loose: int = 0
    # What this one frame showed, before voting. Nothing may gate on these — they
    # exist so the prose can describe the image the VLM is actually looking at,
    # and so a disagreement between the two is visible in a log or a test repr.
    raw_seated: int = 0
    raw_held_at_bridge: int = 0
    raw_held_away: int = 0
    raw_loose: int = 0
    # Categories whose voted count above came from a recent confirmation rather
    # than from the current window -- see "coasting". Provenance for the prose
    # and for a log line; nothing gates on it, and a gate reading it would be
    # asking "how do I know this?" of a count that already answers "how many?".
    coasted: tuple[str, ...] = ()
    # A hand is across the bridge and no pad box was found there, voted over the
    # window. Means "the boxes cannot answer this", never "no pad is fitted" --
    # see `_BRIDGE_BAND_FRAC`. Both bridge gates read it to ask for a clear view
    # instead of asserting an empty bridge they cannot see.
    bridge_obscured: bool = False
    prose: str = ""

    @property
    def held(self) -> int:
        """Every pad in a hand, wherever that hand is.

        For prose and for choosing which correction to speak. A gate wanting
        "is this step still in progress?" must read `held_at_bridge`.
        """
        return self.held_at_bridge + self.held_away

    @property
    def raw_held(self) -> int:
        """`held` for this frame alone, before voting. Never gate on it."""
        return self.raw_held_at_bridge + self.raw_held_away

    @property
    def pads_on_glasses(self) -> int:
        """Seated pads, discounted to zero while a pad is held at the bridge.

        With a pad being positioned at the bridge the step is usually still in
        progress, and a box on the glasses at that moment is more often the
        empty socket that pad came out of, read as a pad, than a fitted one.
        The two are geometrically indistinguishable, so a gate reading this must
        accept the conservative answer.

        The discount is deliberately NOT triggered by every held pad. It needs
        the held pad to be at or near the glasses, because that is the only pose
        that can have left an exposed socket next to a spurious box. A pad in a
        hand a quarter of the frame away — a spare on the table that a hand is
        passing in front of, which in 2D falls inside the hand box — is not
        positioned to have left one, and letting it discount unseats a pad that
        is genuinely fitted. See `_HELD_AWAY_MIN_GAP_PCT`.

        This SOP fits one pad, so any other pad box in a step-4 frame is a
        spare. Do not reason about fitting a pair here; nothing in this task
        does that.

        Read this ONLY to ask "is a pad definitely fitted?". It is the wrong
        count for "is the bridge definitely clear?": the discount is what makes
        an attached pad vanish the moment a pad is picked up at the bridge,
        which is a *weaker* claim about the bridge, not a stronger one. Ask
        `seated` for that direction. See `veto`.

        The discount reads the raw count as well as the voted one, and either
        firing is enough. Voting is a median, so a single frame that saw a hand
        on the pad is outvoted by a majority that did not -- observed live as
        `held_at_bridge=0(raw 1)` while a thumb was on the pad. For flicker in
        the counts a gate reads that is the right direction, but not for the
        discount: this one exists to withhold a completion, so the frame that
        saw the hand is the informative sample and the majority that missed it
        is the noise. Ignoring a real observation is the expensive error here --
        it walks a false pass through, while acting on it costs a retry.
        """
        return 0 if (self.held_at_bridge or self.raw_held_at_bridge) else self.seated


# Appended only when at least one prose line actually carries a class note: a
# checkpoint without the size split would otherwise get a paragraph explaining
# notation that never appears.
_CLASS_NOTE = (
    "A `the detector reads it as ...` note names which SIZE variant the "
    "detector classified that box as. It is a trained classifier on exactly "
    "that distinction and it is steady from frame to frame, where judging the "
    "shape from a small blurry object is not -- so when you have to say which "
    "pad someone is holding, weigh it heavily and do not contradict it on a "
    "hunch. It is still a guess: say so if the pixels clearly disagree.\n"
)


def _which(pad: Detection) -> str:
    """Name the detector's class for one pad, for the prose only.

    The geometry itself is label-agnostic on purpose (see PAD_LABELS) -- size
    says nothing about where a pad is or whose hand is on it. But the VLM has to
    decide WHICH pad the wearer picked up, and left to the pixels it flips: on
    one frame it read the same held pad as "a wire-style nose pad" and passed the
    step, on the next as "a solid saddle shape rather than the requested wire
    butterfly" and failed it, one second apart. The detector is a classifier
    trained on exactly that distinction and it is stable across those frames, so
    stating its answer gives the model something better than a guess.

    Advisory, like everything else in this prose: the class can be wrong, and
    the caller's wording says so.
    """
    return f" (the detector reads it as {pad.label})" if pad.label != "nosepad" else ""


def _bridge_is_obscured(
    cluster: list[Detection], hands: list[Detection], pads_near_bridge: int,
) -> bool:
    """Is a hand covering the bridge with no pad box found there?

    Both halves are required, and the second is what keeps this from firing on
    the ordinary in-progress pose. A pad box AT the bridge -- seated or held --
    means the detector answered, and `held_at_bridge` already carries the right
    reading for it. This is for the frame where it answered nothing.

    Synthesises the bridge band as a Detection so `union_coverage` applies: a
    hand split into two adjacent boxes must count as one cover, which is exactly
    the seam problem that function exists for.
    """
    if not cluster or not hands or pads_near_bridge:
        return False
    x1 = min(g.x1 for g in cluster)
    x2 = max(g.x2 for g in cluster)
    y1 = min(g.y1 for g in cluster)
    y2 = max(g.y2 for g in cluster)
    middle = (x1 + x2) / 2.0
    half = (x2 - x1) * _BRIDGE_BAND_FRAC / 2.0
    band = Detection(
        label="_bridge", x1=middle - half, y1=y1, x2=middle + half, y2=y2,
        confidence=1.0,
    )
    return union_coverage(band, hands) >= _BRIDGE_OBSCURED_COVERAGE


def _state_lines(geometry: PadGeometry) -> list[str]:
    """Every prose line that comes from the VOTED state rather than this frame.

    Both of `analyze`'s exits call this, and that is the point. Lines derived
    from the window describe things the current frame may not show, so they are
    exactly the lines the no-pad-boxes exit needs -- and it is the exit that gets
    forgotten. Add a window-derived line here, never at a call site.
    """
    return _coast_lines(geometry.coasted) + _obscured_lines(geometry.bridge_obscured)


def _obscured_lines(obscured: bool) -> list[str]:
    """The prose for a bridge the boxes cannot see past.

    Says the boxes ABSTAINED. Without it the model sees a frame with no pad box
    and reads the absence as evidence of an empty bridge -- the same mistake the
    gate is being stopped from making, one layer up.
    """
    if not obscured:
        return []
    return [
        "- A hand is across the bridge and the detector found no nose pad box "
        "there. That means the boxes CANNOT say whether a pad is seated -- it is "
        "not evidence that the bridge is empty. Judge this one from the pixels "
        "alone, and if the fingers hide the slot, say the view is blocked rather "
        "than guessing either way."
    ]


def _coast_lines(coasted: tuple[str, ...]) -> list[str]:
    """The prose for a count that came from a confirmation, not from this frame.

    The mirror image of the flicker note: a gate is about to count a pad the
    prose did NOT list, because this frame has no box for it. Left unsaid, the
    model reads the absence as evidence and argues the pad is gone -- so it is
    told the detector blinked, not that the pad moved.

    Deliberately names no position. The confirmation carried a count, never a
    box, and inventing coordinates for the VLM to go and look at would be worse
    than saying nothing.

    A function rather than an inline block because the frame shape that needs it
    MOST -- the glasses in view and no pad box at all -- takes `analyze`'s
    early return and never reaches the prose builder at the bottom.
    """
    if not coasted:
        return []
    plural = len(coasted) > 1
    return [
        f"- A nose pad confirmed {' and '.join(coasted)} moments ago has no box "
        f"in THIS frame{'; the same is true of more than one' if plural else ''}. "
        "The detector loses small pads for a frame at a time, especially a "
        "size-0 pad seated in the bridge, so this is much more often a dropped "
        "detection than a pad that has moved. Do not read the missing box as "
        "the pad being gone."
    ]


def _wrap_prose(lines: list[str]) -> str:
    """The advisory envelope every geometry block is delivered in.

    Shared by both of `analyze`'s exits for the same reason `_coast_lines` is a
    function: the no-pad-boxes path can now have something to say.
    """
    if not lines:
        return ""
    body = "\n".join(lines)
    return (
        "DETECTOR GEOMETRY for the live student image (advisory, computed from "
        "the same boxes):\n"
        + body
        + "\nThese relations are box arithmetic, not ground truth: the detector "
        "misses pads, invents them, and draws boxes larger than the object. Use "
        "them to decide WHICH pad to look at, then confirm the actual state from "
        "the pixels. Do not let a relation here override what you can plainly "
        "see, and do not report a pad as fitted on this basis alone.\n"
        + (_CLASS_NOTE if "the detector reads it as" in body else "")
        + "\n"
    )


def analyze(detections: list[Detection], stream: str = "") -> PadGeometry:
    """Classify each pad against the glasses and the hands.

    ``prose`` is '' when the geometry says nothing useful (no pads, or no
    confident glasses box), so the prompt stays unchanged rather than gaining an
    empty section. Deliberately hedged in wording: these are detector boxes, and
    the detector is wrong often enough that a confident claim here would be
    worse than none. The counts carry no hedging, which is why the only thing
    allowed to read them is a veto (``veto``).

    *stream* identifies the continuous video this frame belongs to — callers
    pass the participant id. Given one, the counts returned are voted across a
    short window of that stream's recent frames instead of taken from this frame
    alone; see "temporal persistence". Left empty, the behaviour is exactly what
    it was before voting existed, which is what every one-off call wants.
    """
    pads = [d for d in detections if d.label in PAD_LABELS]
    hands = [d for d in detections if d.label == HAND_LABEL]
    glasses = [
        d for d in detections
        if d.label == GLASSES_LABEL and d.confidence >= _GLASSES_MIN_CONF
    ]
    if not glasses:
        # Not a judgeable frame, so it is deliberately NOT recorded in the vote
        # window: with no confident glasses box there is nothing to be on or off,
        # and folding a zero in here would let the wearer looking away vote a
        # real pad off the bridge. Same principle `veto` states — absence of
        # evidence is not evidence.
        return PadGeometry(glasses_seen=False)
    if not pads:
        # The pinch pose lands HERE -- glasses in view, a hand on the bridge, no
        # pad box anywhere -- so the occlusion test has to run on this path, and
        # this is the path it exists for. The cluster is rebuilt rather than
        # shared with the branch below, which runs after this return.
        anchor_box = max(glasses, key=lambda d: d.confidence)
        obscured = _bridge_is_obscured(
            [anchor_box] + [
                g for g in glasses
                if g is not anchor_box and g.intersects(anchor_box)
            ],
            hands,
            pads_near_bridge=0,
        )
        # This one IS recorded, and it has to be: a frame showing the glasses and
        # no pad boxes is the only evidence that ever outvotes an invented pad.
        # `glasses_seen` still reports honestly — a gate demanding an empty bridge
        # is satisfied here, and must not be silenced just because the prose has
        # nothing to say.
        # Coasting matters MOST here. This is the frame shape a dropout takes --
        # the glasses in view and no pad box at all -- so without it a fitted
        # size-0 pad reads as an empty bridge every time the detector blinks.
        voted, coasted, obscured = _vote(stream, 0, 0, 0, 0, obscured)
        counts = PadGeometry(
            glasses_seen=True,
            seated=voted[0], held_at_bridge=voted[1], held_away=voted[2],
            loose=voted[3], coasted=coasted, bridge_obscured=obscured,
        )
        # Built from the geometry rather than from the locals, so it picks up
        # every window-derived line by construction. Still '' in the ordinary
        # case -- a frame with no pads, nothing coasted and a clear bridge gains
        # no section, which is what this path always did.
        return replace(counts, prose=_wrap_prose(_state_lines(counts)))
    # One pair of glasses, so take the best box and only those touching it.
    # Overlapping duplicates of the same frame belong together; a box off on its
    # own is a spurious detection, and folding it in would stretch the region
    # across the table — step_02's confident boxes span [9..703] between them,
    # wide enough to call a pad lying on the table "on the glasses".
    anchor = max(glasses, key=lambda d: d.confidence)
    cluster = [anchor] + [
        g for g in glasses if g is not anchor and g.intersects(anchor)
    ]
    reference_width = (
        max(g.x2 for g in cluster) - min(g.x1 for g in cluster)
    ) or 1.0

    seated: list[Detection] = []
    held_at_bridge: list[tuple[Detection, float]] = []
    held_away: list[tuple[Detection, float]] = []
    loose: list[tuple[Detection, float]] = []
    for pad in pads:
        in_hand = union_coverage(pad, hands)
        in_glasses = union_coverage(pad, cluster)
        distance_pct = min(pad.gap_to(g) for g in cluster) / reference_width * 100.0
        if in_hand >= _HELD_IN_HAND:
            # Off the glasses AND provably far is the only combination that
            # reads as "a different pad, elsewhere in the scene". Everything
            # else in a hand stays in-progress: the fallthrough is the
            # conservative direction on purpose.
            if (in_glasses < _SEATED_IN_GLASSES
                    and distance_pct >= _HELD_AWAY_MIN_GAP_PCT):
                held_away.append((pad, distance_pct))
            else:
                held_at_bridge.append((pad, distance_pct))
        elif in_glasses >= _SEATED_IN_GLASSES:
            seated.append(pad)
        else:
            loose.append((pad, distance_pct))

    # Pads held at the bridge are stated first: when one is being positioned the
    # step is usually still in progress, which colours how the on-glasses box
    # should be read.
    lines: list[str] = []
    for pad, distance_pct in held_at_bridge:
        where = (
            "touching the glasses" if distance_pct <= 1.0
            else f"about {distance_pct:.0f}% of the glasses width away from them"
        )
        lines.append(
            f"- 1 nose pad box{_which(pad)} is inside a hand box, {where}. A pad "
            "in a hand is being held, even when it also overlaps the glasses — "
            "that is what positioning a pad at the bridge looks like, not a "
            "fitted pad."
        )
    for pad, distance_pct in held_away:
        # Deliberately does NOT say "still in progress". This box is a pad a
        # hand happens to be in front of, well away from the glasses, and
        # reusing the wording above would tell the model the step is unfinished
        # on the strength of a spare lying on the table.
        lines.append(
            f"- 1 nose pad box{_which(pad)} is inside a hand box but not on the "
            f"glasses, about {distance_pct:.0f}% of the glasses width away from "
            "them. That is a pad in or behind a hand -- it may be the one just "
            "picked up, or a spare the hand is passing in front of -- and it "
            "says nothing about whether a pad is already on the glasses."
        )
    if seated:
        plural = len(seated) > 1
        which = "".join(_which(p) for p in seated) if not plural else ""
        subject = (
            f"{len(seated)} nose pad box{'es' if plural else ''}{which} "
            f"{'overlap' if plural else 'overlaps'} the glasses and "
            f"{'are' if plural else 'is'} NOT inside a hand box"
        )
        if held_at_bridge:
            # A pad being positioned at the bridge is the one this step is
            # about, so a second box sitting on the glasses at the same moment
            # is more likely the empty socket read as a pad than a fitted one.
            # Narrowed to held_at_bridge: a spare behind a hand somewhere else
            # in the frame is no reason to doubt a seated pad. Stated as doubt
            # rather than as a verdict — the model can still see the pixels.
            lines.append(
                f"- {subject}. Read this one with suspicion: a pad is being held "
                "(above), so the step is probably still in progress and this box "
                "is more likely the empty bridge socket detected as a pad. Trust "
                f"the held pad, and treat {'these' if plural else 'this'} as a "
                "probable false positive unless the pixels plainly show a pad "
                "seated in the bridge."
            )
        else:
            lines.append(
                f"- {subject} — consistent with "
                f"{'pads' if plural else 'a pad'} already fitted to the glasses."
            )
    if loose:
        nearest = min(distance for _pad, distance in loose)
        plural = len(loose) > 1
        lines.append(
            f"- {len(loose)} nose pad box{'es' if plural else ''} "
            f"{'are' if plural else 'is'} neither on the glasses nor in a hand "
            f"(nearest about {nearest:.0f}% of the glasses width away) — "
            f"{'spare pads' if plural else 'a spare pad'} resting nearby, not "
            "part of this step."
        )
    voted, coasted, obscured = _vote(
        stream, len(seated), len(held_at_bridge), len(held_away), len(loose),
        _bridge_is_obscured(
            cluster, hands, pads_near_bridge=len(seated) + len(held_at_bridge),
        ),
    )
    counts = PadGeometry(
        glasses_seen=True,
        seated=voted[0], held_at_bridge=voted[1], held_away=voted[2],
        loose=voted[3], coasted=coasted, bridge_obscured=obscured,
        raw_seated=len(seated), raw_held_at_bridge=len(held_at_bridge),
        raw_held_away=len(held_away), raw_loose=len(loose),
    )
    if counts.seated < len(seated) or counts.held < counts.raw_held:
        # The prose describes THIS frame, because this frame is the image the VLM
        # is looking at — but a gate is about to ignore a box the prose just
        # pointed at, and a prompt that names evidence the system is privately
        # discounting invites the model to lean on it. Said in the same register
        # as the cross-pad doubt above rather than as a correction.
        lines.append(
            "- Not every box above held still across the last half-second of "
            "video. A box that appears in a single frame and then vanishes is "
            "usually the detector inventing one, so weigh what has been steadily "
            "visible over what flickered."
        )
    lines.extend(_state_lines(counts))
    if not lines:
        return counts
    prose = _wrap_prose(lines)
    return PadGeometry(
        glasses_seen=True,
        seated=counts.seated,
        held_at_bridge=counts.held_at_bridge,
        held_away=counts.held_away,
        loose=counts.loose,
        raw_seated=counts.raw_seated,
        raw_held_at_bridge=counts.raw_held_at_bridge,
        raw_held_away=counts.raw_held_away,
        raw_loose=counts.raw_loose,
        # Both carried through explicitly, or this exit silently answers the
        # defaults on every frame that HAS pad boxes.
        coasted=counts.coasted,
        bridge_obscured=counts.bridge_obscured,
        prose=prose,
    )


def describe(geometry: PadGeometry) -> str:
    """The prompt block alone; ``analyze`` already built it."""
    return geometry.prose


def veto(geometry: PadGeometry, gate: str) -> str:
    """Why *gate* rejects *geometry*, or '' when it does not object.

    Deliberately one-directional: a gate can only overturn a positive VLM
    verdict. The detector misses pads often enough that letting it *create* a
    pass would be worse than the false passes this exists to stop. The caller
    enforces this structurally — it only ever reads a non-empty return as a
    rejection — but the wording below assumes it too.

    Silent whenever the evidence is absent rather than contrary — no gate, no
    geometry, or no confident glasses box. Absence of evidence is not evidence,
    and a wearer holding the glasses out of frame should get the VLM's answer,
    not a veto.

    The gates deliberately read DIFFERENT counts, because "conservative" points
    a different way for each. `pad_on_glasses` must be sure a pad is fitted, so
    it uses the hand-discounted count and stays unconvinced while a pad is being
    positioned at the bridge. `no_pad_on_glasses` must be sure the bridge is
    clear, so it uses the raw seated count and objects to any pad box on the
    glasses at all. Sharing the discounted count between those two is exactly
    the hole a wearer walks through by leaving the old pad attached and picking
    up the replacement: one pad in a hand would erase the other from the bridge.
    `pad_in_hand` must be sure a pad has been picked UP, so it reads both halves
    of `held` and does not care where the hand is.

    No gate reads `bridge_obscured`, and that is a measured decision rather than
    an omission -- see the note on `_BRIDGE_BAND_FRAC`. The flag reaches the VLM
    as prose and stops there.
    """
    gate = (gate or "").strip().lower()
    if not gate or geometry is None or not geometry.glasses_seen:
        return ""
    if gate not in GATES:
        return ""
    if gate == "pad_on_glasses" and geometry.pads_on_glasses == 0:
        # held_at_bridge, not held: reaching here with only a distant spare in a
        # hand means seated is genuinely 0, and "it is still in your hand" would
        # be a lie about a pad the wearer is not holding at the bridge.
        if geometry.held_at_bridge:
            return (
                "The pad still looks like it is in your hand rather than in the "
                "bridge. Press it into the slot and let go of it."
            )
        return (
            "I do not see a nose pad resting in the bridge yet. Push it into the "
            "slot until it stays there on its own."
        )
    if gate == "pad_in_hand" and geometry.held == 0:
        # `held`, both halves: picking a pad up is done wherever the hand is,
        # and a spare lifted at arm's length from the glasses is `held_away`
        # rather than `held_at_bridge`. Reading only the bridge half would
        # refuse the one pose this step actually asks for.
        #
        # This gate exists because the step's requirements were satisfiable by
        # doing nothing: "replacement nose pads resting on table" and "hand
        # positioned near nose pads" are both true before the wearer moves, and
        # the grader duly passed the step with every pad still on the desk. The
        # prose is fixed too, but "is a pad off the table and in a hand" is box
        # arithmetic, so it should not rest on prose at all.
        return (
            "You have not picked up a replacement nose pad yet. Lift one off "
            "the table and hold it."
        )
    if gate == "no_pad_on_glasses" and geometry.seated > 0:
        if geometry.held:
            # Holding a pad does not clear the bridge. Called out separately
            # because the wearer is plainly doing something with a pad and
            # would otherwise read the generic wording as the system not
            # having noticed.
            return (
                "There is still a nose pad on the glasses. Holding a different "
                "pad does not count — take the one on the bridge off first."
            )
        return (
            "The nose pad still looks attached to the bridge. Pull it straight "
            "back off the frame, not just hold it."
        )
    return ""


def overlay_guide() -> str:
    """Task-specific tail for the box-legend prompt block."""
    return (
        "including whether the nosepad is aligned with or seated in the glasses "
        "area and how the hand is interacting with them"
    )


def spoken_example() -> str:
    """A concrete correction, for the rule that shapes spoken output."""
    return "the nose pad is still attached to the bridge; pull it straight back off"


def contradiction_example() -> str:
    """The observation-vs-verdict failure, in this task's own vocabulary.

    Verbatim from the fix for step 4 rejecting a correctly seated pad, where the
    model produced a passing ``observation`` and a failing ``issue`` in the same
    response. The concrete nouns are the point: the generic phrasing this
    replaces did not stop it, and a model handed an abstract rule alongside a
    concrete one follows the concrete one. Reword only against a live re-test of
    the both-temples pose.
    """
    return "Do not describe a pad as seated and then report that it is not seated"
