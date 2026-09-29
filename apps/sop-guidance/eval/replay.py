# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay old-fork guidance recordings through the new ``vlm`` backend evaluator.

Every recorded grounded check (see ``recordings.py``) is re-graded by the
shipping code path: the procedure's ``VlmBackend`` is built from its
``procedure.yaml`` exactly as the worker builds it, a ``VlmRun`` is opened on
the recorded step with a stand-in run context, and ``command("check")`` runs
``run.py`` -> ``grading.check_step`` with the backend's overlay prompting,
teacher annotation and geometry veto. The student frame the old model saw is
handed to the run as the preview's annotated frame, so the VLM sees the same
pixels; the new detector and geometry profile re-read that frame for the
prompt's geometry prose and the veto (``--geometry``).

Modes:

- ``live`` (default): the new evaluator asks the configured VLM. With
  ``--baseline`` the recorded prompts are also re-asked verbatim with the
  recorded images, which measures how often the OLD evaluator agrees with its
  own recording on a second draw; the goal is new agreement >= that.
- ``offline``: no API calls. The recorded raw responses answer the new
  evaluator's questions tier by tier, which checks parser/tier/veto parity and
  reports how far the new prompts drifted from the recorded ones.

Outputs ``results.jsonl``, ``summary.json`` and ``report.md`` under ``--out``
(default ``run/replay/<timestamp>/``, gitignored). Recorded data is only read.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import difflib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent
_APP = _EVAL_DIR.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from loguru import logger  # noqa: E402
from metrics import Outcome, render_markdown, summarize  # noqa: E402
from recordings import RecordedCheck, find_sessions, load_session, sample_checks  # noqa: E402
from sop_guidance.backends.base import (  # noqa: E402
    BackendServices,
    ModelHandles,
    RunCommand,
    TimedFrame,
    Verdict,
)
from sop_guidance.backends.registry import resolve_backend  # noqa: E402
from sop_guidance.backends.vlm.grading import (  # noqa: E402
    parse_grounded_completion,
    request_check_present,
    strip_thinking,
)
from sop_guidance.backends.vlm.run import VlmRun  # noqa: E402
from sop_guidance.procedures import discover_procedures  # noqa: E402
from sop_guidance.vision import (  # noqa: E402
    FrameAnnotator,
    load_bgr,
    load_detector_profiles,
    load_geometry_plugin,
)
from sop_guidance_worker.config import load_config  # noqa: E402

MAX_CONCURRENCY = 4


def _now_us() -> int:
    return time.time_ns() // 1_000


# ── stand-ins for the host services a run uses ───────────────────────────────


class _Recorder:
    """Collects what the run would have written to the session recorder."""

    recording = True

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.notes: list[tuple[str, dict[str, Any]]] = []

    def record_call(self, **fields: Any) -> None:
        self.calls.append(fields)

    def note(self, event: str, **fields: Any) -> None:
        self.notes.append((event, fields))

    def capture_clip(self, reason: str) -> None:
        return None

    def set_step(self, index: int, instruction: str) -> None:
        return None


class _ReplayContext:
    """A ``RunContext`` whose camera is one recorded, already-annotated frame."""

    owner = "replay"

    def __init__(self, *, session_id: str, frame: TimedFrame, requests: tuple[str, ...],
                 vlm: Any) -> None:
        self.session_id = session_id
        self._frame = frame
        self._requests = requests
        self._models = ModelHandles(vlm=vlm)
        self._recorder = _Recorder()
        self.verdicts: list[dict[str, Any]] = []

    @property
    def input_participant(self) -> str:
        return self.owner

    @property
    def models(self) -> ModelHandles:
        return self._models

    @property
    def recorder(self) -> _Recorder:
        return self._recorder

    async def emit(self, event: Any) -> None:
        if isinstance(event, Verdict):
            self.verdicts.append(dict(event.result))

    def speech_remaining_s(self) -> float:
        return 0.0

    def last_heard_us(self) -> int:
        return 0

    def wearer_requests(self) -> tuple[str, ...]:
        return self._requests

    def latest_frame(self) -> TimedFrame:
        # Re-stamped on every read so the run's 500 ms reuse window accepts it:
        # the recorded frame IS the fresh preview frame of this replay.
        self._frame.timestamp_us = _now_us()
        return self._frame

    async def fetch_frame(self) -> TimedFrame:
        return self.latest_frame()


class _BackendView:
    """The shared backend, with teacher frames and the veto optionally taken from the recording."""

    def __init__(self, backend: Any, teacher_map: dict[str, str],
                 recorded_veto: str | None = None) -> None:
        self._backend = backend
        self._teacher_map = teacher_map
        self._recorded_veto = recorded_veto

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def overlay_prompting(self) -> Any:
        prompting = self._backend.overlay_prompting()
        if self._recorded_veto is None:
            return prompting
        veto = self._recorded_veto
        return dataclasses.replace(prompting, veto=lambda _geometry, _gate: veto)

    async def annotate_teacher(self, path: str) -> tuple[str, bool]:
        if path in self._teacher_map:
            return self._teacher_map[path], True
        return await self._backend.annotate_teacher(path)


class _RecordedVlm:
    """Answers each tier's question with the response recorded for that tier."""

    def __init__(self, check: RecordedCheck) -> None:
        self._responses = {c.tier: c.response for c in check.calls}
        self.questions: dict[str, str] = {}

    def _answer(self, tier: str, question: str) -> SimpleNamespace:
        self.questions.setdefault(tier, question)
        return SimpleNamespace(content=self._responses.get(tier, ""))

    async def ask_image(self, image: Any, question: str, **_: Any) -> SimpleNamespace:
        tier = "diagnosis" if question.startswith("The user is trying") else "live"
        return self._answer(tier, question)

    async def ask_images(self, images: Any, question: str, **_: Any) -> SimpleNamespace:
        return self._answer("compare", question)


# Colours the old and new overlays draw with (detectors.yaml class_colors_rgb).
_OVERLAY_COLORS_RGB = ("#2F80ED", "#F4A6B5", "#27AE60", "#F2C94C")


def strip_overlay(image: Any, *, tolerance: int = 24) -> Any:
    """Inpaint the recorded frame's drawn boxes and label tabs before re-detecting.

    The recorded student frame already carries the old overlay; a label tab
    painted over a small pad hides it from a second detector pass. Only the
    detector sees the stripped copy: the VLM keeps the recorded pixels.
    """

    import cv2
    import numpy as np

    mask = np.zeros(image.shape[:2], np.uint8)
    pixels = image.astype(np.int16)
    for hex_rgb in _OVERLAY_COLORS_RGB:
        r, g, b = (int(hex_rgb[i:i + 2], 16) for i in (1, 3, 5))
        distance = np.abs(pixels - np.array([b, g, r], np.int16)).max(axis=2)
        mask |= (distance <= tolerance).astype(np.uint8) * 255
    # Close over the label text inside each filled tab, then cover the
    # anti-aliased fringe of the 1 px box lines.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def recorded_veto(check: RecordedCheck) -> str:
    """The old geometry veto of *check*, recovered from its recording, or ''.

    The old worker wrote no veto field; a veto shows up as a recorded
    response the parser accepts while the check itself failed with an issue.
    """

    if check.passed or not check.issue:
        return ""
    for call in check.calls:
        if call.tier not in ("compare", "live"):
            continue
        completed, _obs, checks, _missing, _reason = parse_grounded_completion(
            strip_thinking(call.response))
        if completed and "WHAT THE WEARER ASKED FOR" in call.question:
            completed = request_check_present(checks)
        if completed:
            return check.issue
    return ""


# ── building the procedure exactly as the worker does ────────────────────────


def _build_backend(args: argparse.Namespace, artifacts: Path) -> tuple[Any, Any]:
    config = load_config(args.worker_config)
    entries = {e.id: e for e in discover_procedures(config.procedures_dir, config.guidance_defaults)}
    if args.procedure not in entries:
        raise SystemExit(f"unknown procedure {args.procedure!r} (have: {', '.join(sorted(entries))})")
    entry = entries[args.procedure]
    profiles = load_detector_profiles(config.detectors_yaml)
    built: list[FrameAnnotator] = []

    def frame_annotator(profile_name: str, *, overrides: dict[str, Any] | None = None,
                        geometry_path: Path | None = None,
                        spatial_context: bool = False) -> FrameAnnotator:
        profile = profiles[profile_name]
        if overrides:
            profile = profile.model_copy(update=overrides)
        annotator = FrameAnnotator(
            profile,
            geometry=load_geometry_plugin(geometry_path) if geometry_path else None,
            spatial_context=spatial_context,
            artifacts_dir=artifacts / "overlays" / profile_name,
        )
        built.append(annotator)
        return annotator

    backend = resolve_backend(entry.spec.backend)(BackendServices(
        entry=entry,
        config=entry.spec.backend_config,
        artifacts_dir=artifacts,
        detector_profiles=profiles,
        frame_annotator=frame_annotator,
    ))
    problems = backend.validate()
    if problems:
        raise SystemExit("procedure failed validation:\n  " + "\n  ".join(problems))
    return config, backend


def _make_vlm(config: Any, entry_role: str, models_path: Path | None) -> Any:
    from xr_ai_models import load_models_config, make_vlm

    path = models_path or config.models_config
    return make_vlm(load_models_config(path), entry_role)


# ── one check ────────────────────────────────────────────────────────────────


class Replayer:
    def __init__(self, args: argparse.Namespace, backend: Any, vlm: Any) -> None:
        self._args = args
        self._backend = backend
        self._vlm = vlm
        self._annotator = backend.preview_annotator()
        self._semaphore = asyncio.Semaphore(args.concurrency)
        self.prompt_samples: dict[int, tuple[str, str]] = {}

    async def _student_frame(self, check: RecordedCheck) -> TimedFrame:
        image = await asyncio.to_thread(load_bgr, check.student_image)
        height, width = image.shape[:2]
        annotated = None
        if self._annotator is not None:
            mode = self._args.geometry
            if mode == "redetect":
                # Detect on a stripped copy: the drawn result is discarded, the VLM
                # keeps seeing the recorded pixels (already carrying the old overlay).
                source = await asyncio.to_thread(strip_overlay, image)
                fresh = await self._annotator.annotate_array(source, stream="")
                annotated = dataclasses.replace(fresh, image=image)
            else:
                from sop_guidance.vision import AnnotatedFrame

                annotated = AnnotatedFrame(
                    image=image, width=width, height=height, detections=(), boxes=(),
                    spatial_context=check.spatial_context if mode == "recorded" else "",
                    geometry=None,
                )
        return TimedFrame(participant_id="replay", timestamp_us=_now_us(), width=width,
                          height=height, image=image, annotated=annotated)

    def _teacher_map(self, check: RecordedCheck) -> dict[str, str]:
        if self._args.teacher != "recorded":
            return {}
        step = self._backend.sop.steps[check.step - 1]
        recorded = [str(p) for p in check.teacher_images]
        mapping: dict[str, str] = {}
        if len(recorded) == 2 and step.before_image_path:
            mapping[step.before_image_path] = recorded[0]
        if recorded and step.image_path:
            mapping[step.image_path] = recorded[-1]
        return mapping

    async def replay(self, check: RecordedCheck) -> Outcome:
        async with self._semaphore:
            outcome = Outcome(
                key=check.key, session=check.session, step=check.step,
                old_passed=check.passed, new_passed=None, old_issue=check.issue,
                old_observation=check.observation, old_ms=check.latency_ms,
                old_call_ms=[c.latency_ms for c in check.calls],
                wearer_requests=list(check.wearer_requests),
                student_image=str(check.student_image),
            )
            if not 1 <= check.step <= len(self._backend.sop.steps):
                outcome.error = f"step {check.step} is not in the procedure"
                return outcome
            try:
                await self._replay_new(check, outcome)
                if self._args.baseline and self._args.mode == "live":
                    await self._replay_baseline(check, outcome)
            except Exception as exc:  # one bad check must not end the run
                logger.exception("replay of {} failed", check.key)
                outcome.error = f"{type(exc).__name__}: {exc}"
                outcome.new_passed = None
            return outcome

    async def _replay_new(self, check: RecordedCheck, outcome: Outcome) -> None:
        offline = self._args.mode == "offline"
        vlm = _RecordedVlm(check) if offline else self._vlm
        frame = await self._student_frame(check)
        ctx = _ReplayContext(session_id=check.key.replace("#", "_"), frame=frame,
                             requests=check.wearer_requests, vlm=vlm)
        veto = recorded_veto(check) if self._args.geometry == "recorded" else None
        view = _BackendView(self._backend, self._teacher_map(check), veto)
        run = VlmRun(view, ctx, start_step=check.step - 1)
        started = time.monotonic()
        try:
            await run.command(RunCommand("check"))
        finally:
            await run.close("replay")
        outcome.new_ms = 0.0 if offline else (time.monotonic() - started) * 1000.0
        if not ctx.verdicts:
            outcome.error = "the run emitted no verdict"
            return
        verdict = ctx.verdicts[-1]
        timeouts = [n for n, _ in ctx.recorder.notes if n in ("CHECK_TIMEOUT", "CHECK_ERROR")]
        if timeouts:
            outcome.error = ",".join(timeouts)
            return
        checks = verdict.get("checks") or []
        has_evidence = bool(str(verdict.get("current_observation", "")).strip()) and any(
            c.get("visible") and str(c.get("evidence", "")).strip() for c in checks
        )
        outcome.new_passed = bool(verdict.get("completed")) and has_evidence
        outcome.new_issue = str(verdict.get("issue", ""))
        outcome.new_observation = str(verdict.get("current_observation", ""))
        outcome.new_tier = str(verdict.get("tier", ""))
        outcome.geometry_veto = str(verdict.get("geometry_veto", ""))
        outcome.new_call_ms = [] if offline else [
            float(c.get("latency_ms", 0.0)) for c in ctx.recorder.calls
        ]
        new_questions = (
            vlm.questions if offline else {
                ("compare" if len(c.get("images") or ()) > 1 else "live"):
                    str((c.get("request") or {}).get("question", ""))
                for c in reversed(ctx.recorder.calls)
            }
        )
        recorded = check.call("compare")
        if recorded is not None and "compare" in new_questions:
            new_q = new_questions["compare"]
            outcome.prompt_similarity = difflib.SequenceMatcher(
                None, recorded.question, new_q, autojunk=False).ratio()
            self.prompt_samples.setdefault(check.step, (recorded.question, new_q))

    async def _replay_baseline(self, check: RecordedCheck, outcome: Outcome) -> None:
        """Re-ask the recorded prompts, in the old tier order, with the recorded images."""

        started = time.monotonic()
        passed = False
        for tier in ("compare", "live"):
            recorded = check.call(tier)
            if recorded is None:
                if tier == "live" and check.call("compare") is not None:
                    # The old run passed on the comparison and never asked the
                    # live question, so a failed re-ask has no tier to fall to:
                    # leave this check out of the baseline instead of guessing.
                    outcome.baseline_passed = None
                    outcome.baseline_ms = (time.monotonic() - started) * 1000.0
                    return
                continue
            images = list(recorded.images)
            if len(images) == 1:
                response = await self._vlm.ask_image(images[0], recorded.question)
            else:
                response = await self._vlm.ask_images(images, recorded.question)
            raw = strip_thinking(response.content or "")
            completed, _obs, checks, _missing, _reason = parse_grounded_completion(raw)
            if completed and "WHAT THE WEARER ASKED FOR" in recorded.question:
                completed = request_check_present(checks)
            if completed and outcome.geometry_veto:
                completed = False
            if completed:
                passed = True
                break
        outcome.baseline_passed = passed
        outcome.baseline_ms = (time.monotonic() - started) * 1000.0


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay recorded guidance checks through the new vlm evaluator.")
    parser.add_argument("sessions", nargs="*", type=Path,
                        help="session folders, or folders holding session folders")
    parser.add_argument("--rescore", type=Path, default=None, metavar="OUT",
                        help="rebuild summary.json and report.md from OUT/results.jsonl; no replay")
    parser.add_argument("--procedure", default="nosepad-replacement")
    parser.add_argument("--worker-config", type=Path,
                        default=_APP / "yaml" / "sop_guidance_worker.yaml")
    parser.add_argument("--models", type=Path, default=None,
                        help="models JSON; default: the worker config's models_config")
    parser.add_argument("--mode", choices=("live", "offline"), default="live")
    parser.add_argument("--baseline", action="store_true",
                        help="live mode: also re-ask the recorded prompts verbatim")
    parser.add_argument("--geometry", choices=("redetect", "recorded", "off"), default="redetect",
                        help="new detector+geometry on the recorded frame (overlay "
                             "stripped), the recorded geometry prose and veto, or none")
    parser.add_argument("--teacher", choices=("sop", "recorded"), default="sop",
                        help="annotate the SOP's reference frames with the new detector, "
                             "or reuse the recorded annotated teacher frames")
    parser.add_argument("--limit", type=int, default=40,
                        help="maximum checks to replay (0 = all); sampled across sessions and steps")
    parser.add_argument("--per-session", type=int, default=0)
    parser.add_argument("--steps", default="", help="comma-separated 1-based steps to keep")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        parser.error(f"--concurrency must be 1..{MAX_CONCURRENCY}")
    if not args.sessions and args.rescore is None:
        parser.error("give at least one session folder (or --rescore OUT)")
    return args


def _finite(value: Any) -> Any:
    """*value* with NaN replaced by None, so the JSON stays strict."""

    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _write_summary(out: Path, outcomes: list[Outcome], run: dict[str, Any], title: str) -> str:
    summary = summarize(outcomes)
    summary["run"] = run
    (out / "summary.json").write_text(json.dumps(_finite(summary), indent=2), encoding="utf-8")
    report = render_markdown(summary, outcomes, title=title)
    (out / "report.md").write_text(report, encoding="utf-8")
    return report


def _rescore(out: Path) -> int:
    fields = {f.name for f in dataclasses.fields(Outcome)}
    outcomes = []
    with (out / "results.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                outcomes.append(Outcome(**{k: v for k, v in row.items() if k in fields}))
    run: dict[str, Any] = {}
    if (out / "summary.json").is_file():
        run = json.loads((out / "summary.json").read_text(encoding="utf-8")).get("run", {})
    title = f"Replay: {run.get('procedure', '?')} ({run.get('mode', '?')})"
    print(_write_summary(out, outcomes, run, title))
    return 0


async def _main(args: argparse.Namespace) -> int:
    logger.remove()
    logger.add(sys.stderr, level="INFO" if args.verbose else "WARNING")

    sessions = [load_session(p) for p in find_sessions(args.sessions)]
    if not sessions:
        raise SystemExit("no recorded sessions found (expected folders holding events.jsonl)")
    checks = [c for s in sessions for c in s.checks]
    if args.steps:
        wanted = {int(s) for s in args.steps.split(",") if s.strip()}
        checks = [c for c in checks if c.step in wanted]
    selected = sample_checks(checks, args.limit, per_session=args.per_session)
    for session in sessions:
        skipped = ", ".join(f"{k}={v}" for k, v in sorted(session.skipped.items())) or "none"
        print(f"{session.id}: {len(session.checks)} replayable checks, skipped: {skipped}")
    print(f"replaying {len(selected)} of {len(checks)} checks, mode={args.mode} "
          f"geometry={args.geometry} teacher={args.teacher} concurrency={args.concurrency}")

    out = args.out or _APP / "run" / "replay" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    config, backend = _build_backend(args, out / "artifacts")
    vlm = None
    if args.mode == "live":
        vlm = _make_vlm(config, "vlm", args.models)
    replayer = Replayer(args, backend, vlm)
    try:
        outcomes: list[Outcome] = []
        tasks = [asyncio.create_task(replayer.replay(c)) for c in selected]
        for done, task in enumerate(asyncio.as_completed(tasks), start=1):
            outcome = await task
            outcomes.append(outcome)
            state = "ERR " if outcome.new_passed is None else (
                "SAME" if outcome.new_passed == outcome.old_passed else "DIFF")
            print(f"[{done}/{len(tasks)}] {state} {outcome.key} step={outcome.step} "
                  f"old={'pass' if outcome.old_passed else 'fail'} "
                  f"new={'-' if outcome.new_passed is None else ('pass' if outcome.new_passed else 'fail')}"
                  f" {outcome.new_ms:.0f}ms {outcome.error}", flush=True)
    finally:
        if vlm is not None and hasattr(vlm, "close"):
            await vlm.close()

    outcomes.sort(key=lambda o: o.key)
    with (out / "results.jsonl").open("w", encoding="utf-8") as stream:
        for outcome in outcomes:
            stream.write(json.dumps(outcome.as_dict(), ensure_ascii=False) + "\n")
    run = {
        "mode": args.mode, "geometry": args.geometry, "teacher": args.teacher,
        "baseline": args.baseline, "procedure": args.procedure,
        "sessions": [s.id for s in sessions],
    }
    report = _write_summary(out, outcomes, run, f"Replay: {args.procedure} ({args.mode})")
    diffs = out / "prompt_diffs"
    diffs.mkdir(exist_ok=True)
    for step, (old_q, new_q) in sorted(replayer.prompt_samples.items()):
        (diffs / f"step_{step:02d}.diff").write_text("".join(difflib.unified_diff(
            old_q.splitlines(keepends=True), new_q.splitlines(keepends=True),
            fromfile="recorded", tofile="new")), encoding="utf-8")
    print()
    print(report)
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.rescore is not None:
        return _rescore(args.rescore)
    if args.mode == "live":
        config = load_config(args.worker_config)
        del config
        if not os.environ.get("DASHSCOPE_API_KEY") and args.models is None:
            print("warning: DASHSCOPE_API_KEY is not set; the hosted VLM will reject calls",
                  file=sys.stderr)
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
