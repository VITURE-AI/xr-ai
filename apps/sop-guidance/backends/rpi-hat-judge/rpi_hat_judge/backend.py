# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``rpi_hat_judge`` backend: the assembly judge behind the guidance seam.

Each pushed frame runs the v7 detector and the shared hand detector, then
the judge's pipeline, unchanged in substance from the old process:

    detections -> TemporalSmoother -> OcclusionFilter -> FPCSeatState (once
    four screws are in) -> PerHoleTracker -> Progress -> events

Everything the old judge did over HTTP is an event now. The step it moves to
is a :class:`StepChanged` the host announces; an out-of-order screw or a
removed one is a correction :class:`Cue`; its boxes are an
:class:`OverlayUpdate` drawn on the wearer's return video; the hole map rides
in ``guidance.state.extra``; and the last confirmed step ends the run.

Voice never moves the judge. "Next" is answered by the host from the
capabilities; starting the procedure again is a fresh run, so every run
begins with fresh state and the model stays loaded.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import ValidationError
from sop_guidance.backends.base import (
    BackendServices,
    Capabilities,
    CommandResult,
    Cue,
    OverlayUpdate,
    RunCommand,
    RunContext,
    RunFinished,
    RunSnapshot,
    StepChanged,
    StepInfo,
    TimedFrame,
    TurnContext,
    Verdict,
)
from sop_guidance.vision import Detection, bgr_to_jpeg, weight_problems

from .board import CLASSES, FIXED_SLOTS, HOLE_ANCHORS, HOLE_NAMES, SOP_SEQUENCE, anchor_pixels
from .config import RpiHatJudgeConfig
from .fpc import FPCSeatState
from .holes import PerHoleTracker
from .occlusion import OcclusionFilter
from .progress import Progress, Tick
from .smoothing import TemporalSmoother
from .spec import JudgeSpec, SpecError, load_spec
from .texts import alert_cue, done_phrase, event_detail

_HAND_LABEL = "hand"
_NEXT_LABEL = "next"
_FULL_HD = (1920, 1080)

_RULES = (
    "Progress on this procedure is judged by a detector watching the camera, "
    "not by you: never say a step is done, never announce the next step, and "
    "never tell the wearer to say next. Answer their questions; the step cues "
    "are spoken for you."
)


def _ticks(hz: float, seconds: float) -> int:
    return max(1, int(round(hz * seconds)))


class RpiHatJudgeBackend:
    """Judges the Raspberry Pi M.2 HAT+ assembly from the camera alone."""

    name = "rpi_hat_judge"

    def __init__(self, *, procedure_id: str, spec: JudgeSpec, config: RpiHatJudgeConfig,
                 annotator: Any = None, problems: Sequence[str] = (),
                 artifacts_dir: Path | None = None) -> None:
        self.procedure_id = procedure_id
        self.spec = spec
        self.config = config
        self.artifacts_dir = artifacts_dir
        self._annotator = annotator
        self._problems = list(problems)
        self.capabilities = Capabilities(
            voice_advance=False,
            jump_to_step=False,
            # The judge reads the board: a new run re-derives where it is.
            resume=False,
            wearer_requests=False,
            check_on_demand=False,
            frame_hz=config.tick_hz,
            min_frame_size=_FULL_HD,
            provides_overlay=True,
            max_concurrent_runs=1,
        )

    # ── description ──────────────────────────────────────────────────────────

    @property
    def title(self) -> str:
        return self.spec.name

    def steps(self) -> list[StepInfo]:
        return [
            StepInfo(number=index + 1, instruction=self.spec.spoken_instruction(index),
                     title=step.title, gradeable=True, done_when=step.tip,
                     reference_images=(step.image,) if step.image else ())
            for index, step in enumerate(self.spec.steps)
        ]

    def parts(self) -> tuple[str, ...]:
        return ()

    def instructions_digest(self) -> list[str]:
        return [self.spec.spoken_instruction(i) for i in range(self.spec.total)]

    def preview_annotator(self) -> Any:
        # Paints the boxes this backend sends; its detector is never run by the preview.
        return self._annotator

    def validate(self) -> list[str]:
        problems = list(self._problems)
        annotator = self._annotator
        if annotator is None:
            return problems
        profile = annotator.profile
        if not profile.enabled:
            problems.append(f"detector profile {self.config.detector.profile!r} is disabled")
        labels = set(profile.overlay.class_labels.values()) or set(profile.overlay.class_labels)
        missing = [c for c in CLASSES if c not in labels]
        if missing:
            problems.append(f"detector profile {self.config.detector.profile!r} does not label "
                            f"{', '.join(missing)} under the judge's class names")
        if self.config.occlusion.use_hand and not profile.hands.enabled:
            problems.append("occlusion.use_hand needs the detector profile's `hands` detector")
        problems.extend(weight_problems(profile))
        return problems

    # ── runs ─────────────────────────────────────────────────────────────────

    async def detect(self, image: Any) -> list[Detection]:
        return await self._annotator.detect_array(image)

    def draw(self, image: Any, detections: Sequence[Detection]) -> Any:
        return self._annotator.draw(image, list(detections))

    async def open_run(self, ctx: RunContext, *, start_step: int,
                       checkpoint: Mapping[str, Any] | None) -> RpiHatJudgeRun:
        if start_step != 0:
            raise ValueError(f"'{self.spec.name}' always starts from step 1")
        return RpiHatJudgeRun(self, ctx)


class RpiHatJudgeRun:
    """One session's judge: fresh smoothing, votes and latches."""

    def __init__(self, backend: RpiHatJudgeBackend, ctx: RunContext) -> None:
        self._backend = backend
        self._ctx = ctx
        self._spec = backend.spec
        self._cfg = backend.config
        self._closed = False
        self._finished = False
        self._lock = asyncio.Lock()
        # Judged frames for the session recorder: a small ring of names, as the
        # vlm backend keeps. The recorder hardlinks each one, and os.replace
        # gives every write a new inode, so a link keeps its bytes.
        root = backend.artifacts_dir or Path("run") / "artifacts" / backend.procedure_id
        stem = "".join(c if c.isalnum() or c in "-_." else "_"
                       for c in f"{ctx.session_id}_{ctx.owner}")
        self._frames_dir = root / "checks" / stem
        self._frame_slot = 0
        self._build()

    def _build(self) -> None:
        cfg, hz = self._cfg, self._cfg.tick_hz
        tracker = cfg.tracker
        n_fwd = _ticks(hz, tracker.stable_sec)
        self._tracker = PerHoleTracker(
            n_fwd=n_fwd, n_back=max(n_fwd, _ticks(hz, tracker.back_sec)),
            m=tracker.slot_window, need=tracker.slot_need, slot_back_sec=tracker.slot_back_sec,
            tol=_ticks(hz, tracker.tol_sec), screws_expected=len(SOP_SEQUENCE),
            hold_sec=tracker.hold_display_sec, hf_conf_min=tracker.hf_conf_min,
            installed_tol=tracker.installed_tol, screw_as_installed=tracker.screw_as_installed,
        )
        self._smoother = TemporalSmoother(hold_sec=cfg.smoothing.hold_sec, win=cfg.smoothing.window)
        occ = cfg.occlusion
        self._occlusion = OcclusionFilter(occ.th_hand, occ.th_conf, occ.min_holes,
                                          use_hand=occ.use_hand)
        self._fpc = FPCSeatState(n_need=_ticks(hz, cfg.fpc.need_sec),
                                 m_win=_ticks(hz, cfg.fpc.window_sec),
                                 unseat_ticks=_ticks(hz, cfg.fpc.unseat_sec),
                                 iob_min=cfg.fpc.iob_min)
        self._progress = Progress(self._spec, hold_display=tracker.hold_display_sec)
        self._index = 0
        self._pending_done: int | None = None
        self._fpc_events_seen = 0
        self._res: dict[str, Any] = {}
        self._tick: Tick | None = None
        self._fpc_armed = False
        self._summary: tuple[Any, ...] | None = None
        self._last_image: Any = None
        self._last_boxes: tuple[Detection, ...] = ()
        self._warned_size = False
        self._detect_ms = 0.0
        self._counts: dict[str, int] = {}
        self._box_diag: list[dict[str, Any]] = []
        self._last_record_t: float | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._ctx.recorder.set_step(0, self._spec.spoken_instruction(0))
        await self._ctx.emit(StepChanged(0, reason="start"))

    async def on_frame(self, frame: TimedFrame) -> None:
        if self._closed or self._finished:
            return
        if not self._warned_size and (frame.width < _FULL_HD[0] or frame.height < _FULL_HD[1]):
            self._warned_size = True
            logger.warning("RPI_JUDGE frame {}x{} is below the {}x{} the anchors were "
                           "calibrated on; keep the board at least a third of the frame wide",
                           frame.width, frame.height, *_FULL_HD)
        started = time.monotonic()
        detections = await self._backend.detect(frame.image)
        self._detect_ms = (time.monotonic() - started) * 1000.0
        async with self._lock:
            if self._closed or self._finished:
                return
            await self._judge(frame, detections)

    async def input_changed(self) -> None:
        # Another camera: its board track and medians start over; the votes
        # already cast stand.
        async with self._lock:
            self._smoother.reset()

    async def close(self, reason: str) -> None:
        self._closed = True
        # Every recorded frame was hardlinked into the session as it was
        # recorded (same run volume), so the ring is only this run's scratch.
        # The lock waits out a frame still being written.
        async with self._lock:
            if self._frames_dir.exists():
                await asyncio.to_thread(shutil.rmtree, self._frames_dir, True)

    # ── commands ─────────────────────────────────────────────────────────────

    async def command(self, command: RunCommand) -> CommandResult:
        if command.kind == "repeat":
            return CommandResult(True, speech=self._status_line())
        if command.kind == "reset":
            async with self._lock:
                self._build()
            self._ctx.recorder.set_step(0, self._spec.spoken_instruction(0))
            await self._ctx.emit(StepChanged(0, reason="reset"))
            return CommandResult(True)
        return CommandResult(False, reason=f"unsupported command {command.kind!r}")

    # ── state for the host ───────────────────────────────────────────────────

    def snapshot(self) -> RunSnapshot:
        return RunSnapshot(
            step_index=self._index,
            total_steps=self._spec.total,
            instruction=self._spec.spoken_instruction(self._index),
            extra=self._hole_map(),
        )

    def turn_context(self) -> TurnContext:
        frame = b""
        if self._last_image is not None:
            drawn = self._backend.draw(self._last_image.copy(), self._last_boxes)
            frame = bgr_to_jpeg(drawn)
        return TurnContext(prompt_block=f"\n\n[Assembly judge]\n{_RULES}\n{self._judge_line()}",
                           frame_jpeg=frame)

    # ── the judge ────────────────────────────────────────────────────────────

    async def _judge(self, frame: TimedFrame, detections: list[Detection]) -> None:
        t = frame.timestamp_us / 1_000_000
        dets = [{"cls": d.label, "xyxy": [d.x1, d.y1, d.x2, d.y2], "conf": d.confidence}
                for d in detections if d.label in CLASSES]
        self._counts = {c: sum(1 for d in detections if d.label == c)
                        for c in (*CLASSES, _HAND_LABEL)}
        hands = [[d.x1, d.y1, d.x2, d.y2] for d in detections if d.label == _HAND_LABEL]
        smoothed = self._smoother.update(dets, hands or None, t)
        verdict = self._occlusion.check(smoothed.boxes, smoothed.hand_boxes,
                                        visible_holes=smoothed.visible_holes)
        board = next((b["xyxy"] for b in smoothed.boxes if b["cls"] == "board"), None)
        tracker = self._tracker
        # The cable is judged only once all four screws are confirmed (it was
        # mistaken for seated while lying beside the connector) and only on a
        # frame that passed the gate.
        self._fpc_armed = tracker.machine.k == len(SOP_SEQUENCE) and verdict.ok
        if self._fpc_armed:
            tracker.fpc_done = self._fpc.update(smoothed.boxes, board, t)
        res = tracker.update(smoothed.boxes, verdict, t)
        tick = self._progress.update(res, tracker.events, t)
        self._res, self._tick = res, tick
        self._last_image = frame.image
        self._last_boxes = self._overlay_boxes(smoothed, res)
        self._box_diag = self._diagnose(smoothed.boxes, res)
        cable_changed = self._record_cable_events()
        summary = self._summary_key()
        changed = summary != self._summary

        # Recorded under the step being judged, before a step change moves the
        # recorder on: the frame that completed a step belongs to that step.
        advancing = tick.finished_now or (tick.active is not None and tick.active != self._index)
        if self._ctx.recorder.recording and (
            advancing or tick.alerts or cable_changed
            or (changed and (self._last_record_t is None
                             or t - self._last_record_t >= self._cfg.record_interval_s))
        ):
            self._last_record_t = t
            completed = tick.finished_now or (advancing and tick.active is not None
                                              and tick.active > self._index)
            await self._record_check(frame, completed=bool(completed))

        await self._ctx.emit(OverlayUpdate(frame.timestamp_us, self._last_boxes,
                                           extra=self._hole_map()))
        await self._speak(tick)
        if self._finished:
            return
        if changed:
            self._summary = summary
            await self._ctx.emit(Verdict({
                **self._hole_map(),
                "completed": tick.done,
            }))

    async def _speak(self, tick: Tick) -> None:
        spec = self._spec
        recorder = self._ctx.recorder
        for kind, fields in tick.alerts:
            logger.info("RPI_JUDGE_ALERT kind={} {}", kind, event_detail(kind, fields))
        if tick.newly_completed:
            self._pending_done = tick.newly_completed[-1]
            logger.info("RPI_JUDGE_COMPLETED steps={}", [i + 1 for i in tick.newly_completed])

        if tick.finished_now:
            last = spec.total - 1
            await self._ctx.emit(Cue(done_phrase(last + 1, spec.steps[last].speech), kind="status"))
            self._finished = True
            recorder.note("JUDGE_DONE")
            await self._ctx.emit(RunFinished("completed", reason="all steps confirmed"))
            return

        active = tick.active
        changed = active is not None and active != self._index
        alert = tick.alerts[-1] if tick.alerts else None
        if alert is not None:
            # An alert is said first and names what to do next, unless a step
            # announcement follows at once and says it.
            next_name = None if changed or active is None else spec.steps[active].speech
            kind, fields = alert
            spoken = alert_cue(kind, fields, next_name)
            await self._ctx.emit(Cue(spoken, kind="correction"))
            # One correction per spoken alert, as the vlm backend records them;
            # alerts the same tick outranked are kept beside it, unspoken.
            for other, other_fields in tick.alerts[:-1]:
                recorder.note("CORRECTION", kind=other, issue=event_detail(other, other_fields))
            recorder.note("CORRECTION", kind=kind, issue=event_detail(kind, fields),
                          spoken=spoken, count=1)
            recorder.capture_clip("correction")
        if not changed or active is None:
            return
        lead = ""
        if alert is None and self._pending_done is not None:
            done = self._pending_done
            lead = done_phrase(done + 1, spec.steps[done].speech)
        self._pending_done = None
        reason = "advance" if active > self._index else "restep"
        logger.info("RPI_JUDGE_STEP {} -> {} ({})", self._index + 1, active + 1, reason)
        recorder.capture_clip("step-advance")
        self._index = active
        recorder.set_step(active, spec.spoken_instruction(active))
        await self._ctx.emit(StepChanged(active, reason=reason, lead=lead))

    def _record_cable_events(self) -> bool:
        events = self._fpc.events[self._fpc_events_seen:]
        self._fpc_events_seen = len(self._fpc.events)
        for _t, kind, fields in events:
            logger.info("RPI_JUDGE_CABLE {}", event_detail(kind, fields))
            self._ctx.recorder.note("JUDGE_CABLE", kind=kind, iob=round(float(fields["iob"]), 3))
        return bool(events)

    # ── the session log ──────────────────────────────────────────────────────

    async def _record_check(self, frame: TimedFrame, *, completed: bool) -> None:
        """Log one judged frame: a ``judge`` call with the annotated frame, and
        the ``CHECK`` event the sessions viewer shows per step."""

        recorder = self._ctx.recorder
        try:
            path, raw = await asyncio.to_thread(self._write_frame, frame.image, self._last_boxes)
        except Exception:
            logger.exception("RPI_JUDGE could not write the judged frame")
            path, raw = "", ""
        res = self._res
        trusted = bool(res.get("trusted"))
        reason = "" if trusted else str(res.get("reason") or "")
        hole_map = self._hole_map()
        frames = recorder.record_call(
            kind="judge",
            name="judge_frame",
            request={
                "frame": {"width": frame.width, "height": frame.height,
                          "timestamp_us": frame.timestamp_us},
                "detections": self._counts,
            },
            response={
                "trusted": trusted,
                "reason": reason,
                "screws_confirmed": self._tracker.machine.k,
                "holes": hole_map["holes"],
                "next_hole": hole_map["next_hole"],
                "cable": {"armed": hole_map["fpc_armed"], "seated": hole_map["fpc_seated"],
                          "iob": round(self._fpc.last_iob, 3)},
                "steps_done": hole_map["steps_done"],
                "steps_owed": hole_map["steps_owed"],
                "hf_conf_min": self._cfg.tracker.hf_conf_min,
                "boxes": self._box_diag,
            },
            latency_ms=self._detect_ms,
            # The annotated frame, then the same pixels without boxes: the raw
            # one can be detected again offline when thresholds are tuned.
            images=[p for p in (path, raw) if p],
        )
        recorder.note("CHECK", completed=completed, has_evidence=trusted,
                      observation=self._observation(), issue=reason,
                      frame=frames[0] if frames else path)

    def _write_frame(self, image: Any, boxes: Sequence[Detection]) -> tuple[str, str]:
        """Write the annotated and the raw frame; returns both paths."""

        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._frame_slot = (self._frame_slot + 1) % 8
        paths = []
        for name, pixels in (("judged", self._backend.draw(image.copy(), boxes)), ("raw", image)):
            final = self._frames_dir / f"{name}_{self._frame_slot}.jpg"
            tmp = final.with_suffix(".tmp")
            tmp.write_bytes(bgr_to_jpeg(pixels, max_width=100_000, quality=90))
            os.replace(tmp, final)
            paths.append(str(final))
        return paths[0], paths[1]

    def _diagnose(self, boxes: list[dict[str, Any]], res: dict[str, Any]) -> list[dict[str, Any]]:
        """Each hole, screw and cable box with its confidence and nearest hole,
        as the old judge's /api/boxes reported them, for threshold tuning."""

        board = next((b["xyxy"] for b in boxes if b["cls"] == "board"), None)
        anchors = res.get("hole_px") or {}
        if not anchors and board is not None:
            anchors = anchor_pixels(board)
        long_side = max(board[2] - board[0], board[3] - board[1]) if board is not None else 0.0
        gate = self._cfg.tracker.hf_conf_min
        out = []
        for b in boxes:
            if b["cls"] not in ("hole", "hole_filled", "screw", "fpc"):
                continue
            x = (b["xyxy"][0] + b["xyxy"][2]) / 2
            y = (b["xyxy"][1] + b["xyxy"][3]) / 2
            slot, dist = None, None
            for k, (ax, ay) in anchors.items():
                d = ((x - ax) ** 2 + (y - ay) ** 2) ** 0.5
                if dist is None or d < dist:
                    slot, dist = k, d
            conf = float(b.get("conf", 0.0))
            out.append({
                "cls": b["cls"],
                "conf": round(conf, 3),
                "xyxy": [round(v, 1) for v in b["xyxy"]],
                "slot": slot,
                # Distance to that hole over the board's long side: the
                # judge's match radii are 0.16 (hole) and 0.12 (filled).
                "dist": round(dist / long_side, 3) if dist is not None and long_side else None,
                "passes_hf_gate": b["cls"] != "hole_filled" or conf >= gate,
            })
        return sorted(out, key=lambda d: (d["cls"], -d["conf"]))

    def _observation(self) -> str:
        """The judge's reading in one line, for the check row."""

        holes = self._res.get("holes") or {}
        installed = [HOLE_NAMES[k] for k in SOP_SEQUENCE if holes.get(k) == "installed"]
        parts = [f"{len(installed)} of {len(SOP_SEQUENCE)} screws confirmed"
                 + (f" ({', '.join(installed)})" if installed else "")]
        next_hole = self._res.get("next_hole")
        if next_hole:
            parts.append(f"next: {HOLE_NAMES[next_hole]}")
        if self._fpc.seated:
            parts.append("cable seated")
        elif len(installed) == len(SOP_SEQUENCE):
            parts.append(f"cable not seated yet (IoB {self._fpc.last_iob:.2f})")
        return "; ".join(parts) + "."

    # ── what clients and the foreground see ──────────────────────────────────

    def _overlay_boxes(self, smoothed: Any, res: dict[str, Any]) -> tuple[Detection, ...]:
        boxes = [Detection(label=b["cls"], x1=b["xyxy"][0], y1=b["xyxy"][1], x2=b["xyxy"][2],
                           y2=b["xyxy"][3], confidence=float(b.get("conf", 0.0)))
                 for b in smoothed.boxes]
        boxes += [Detection(label=_HAND_LABEL, x1=h[0], y1=h[1], x2=h[2], y2=h[3], confidence=1.0)
                  for h in smoothed.hand_boxes]
        board = next((b["xyxy"] for b in smoothed.boxes if b["cls"] == "board"), None)
        next_hole = res.get("next_hole")
        anchor = (res.get("hole_px") or {}).get(next_hole) if next_hole else None
        if res.get("trusted") and board is not None and anchor is not None:
            half = 0.04 * max(board[2] - board[0], board[3] - board[1])
            boxes.append(Detection(label=_NEXT_LABEL, x1=anchor[0] - half, y1=anchor[1] - half,
                                   x2=anchor[0] + half, y2=anchor[1] + half, confidence=1.0))
        return tuple(boxes)

    def _hole_map(self) -> dict[str, Any]:
        res, tick = self._res, self._tick
        holes = res.get("holes") or {k: ("fixed" if k in FIXED_SLOTS else "empty")
                                     for k in HOLE_ANCHORS}
        return {
            "holes": {str(k): v for k, v in holes.items()},
            "hole_names": {str(k): HOLE_NAMES[k] for k in HOLE_ANCHORS},
            "hole_layout": {str(k): list(v) for k, v in HOLE_ANCHORS.items()},
            "fixed_holes": [str(k) for k in FIXED_SLOTS],
            "next_hole": res.get("next_hole", SOP_SEQUENCE[0]),
            "trusted": bool(res.get("trusted")),
            "reject_reason": "" if res.get("trusted") else str(res.get("reason") or ""),
            "fpc_armed": self._fpc_armed,
            "fpc_seated": bool(self._fpc.seated),
            "steps_done": [i + 1 for i in tick.completed] if tick else [],
            "steps_owed": [i + 1 for i in tick.outstanding] if tick else [],
        }

    def _summary_key(self) -> tuple[Any, ...]:
        m = self._hole_map()
        return (tuple(sorted(m["holes"].items())), m["next_hole"], m["trusted"],
                m["reject_reason"], m["fpc_armed"], m["fpc_seated"], tuple(m["steps_done"]))

    def _judge_line(self) -> str:
        res = self._res
        holes = res.get("holes") or {}
        installed = [HOLE_NAMES[k] for k in SOP_SEQUENCE if holes.get(k) == "installed"]
        lines = [f"Screws confirmed: {', '.join(installed) or 'none'} "
                 f"({len(installed)} of {len(SOP_SEQUENCE)})."]
        tick = self._tick
        if tick and tick.outstanding:
            lines.append("Skipped and still owed: "
                         + ", ".join(self._spec.steps[i].title for i in tick.outstanding) + ".")
        next_hole = res.get("next_hole")
        if next_hole:
            lines.append(f"Next hole: {HOLE_NAMES[next_hole]}.")
        cable = ("seated" if self._fpc.seated else "not seated yet" if self._fpc_armed
                 or len(installed) == len(SOP_SEQUENCE) else "waiting for the four screws")
        lines.append(f"Ribbon cable: {cable}.")
        if not res:
            lines.append("Camera view: no frame judged yet.")
        elif res.get("trusted"):
            lines.append("Camera view: clear enough to judge.")
        else:
            lines.append(f"Camera view: not judged right now ({res.get('reason')}).")
        return "\n".join(lines)

    def _status_line(self) -> str:
        index = self._index
        return (f"Step {index + 1} of {self._spec.total}: {self._spec.steps[index].title}. "
                f"{self._spec.spoken_instruction(index)}")


def create_backend(services: BackendServices) -> RpiHatJudgeBackend:
    """Build the ``rpi_hat_judge`` backend for one procedure folder."""

    entry = services.entry
    try:
        config = RpiHatJudgeConfig.model_validate(dict(services.config))
    except ValidationError as exc:
        raise ValueError(f"{entry.directory / 'procedure.yaml'}: backend_config: {exc}") from exc
    try:
        spec = load_spec(Path(entry.resolve(config.spec)))
    except (OSError, SpecError, ValueError) as exc:
        raise ValueError(f"procedure {entry.id!r}: cannot load {config.spec}: {exc}") from exc
    problems: list[str] = []
    annotator = None
    if services.frame_annotator is None:
        problems.append("the judge needs vision support to run its detector")
    else:
        annotator = services.frame_annotator(config.detector.profile, overrides={},
                                             geometry_path=None, spatial_context=False)
    return RpiHatJudgeBackend(procedure_id=entry.id, spec=spec, config=config,
                              annotator=annotator, problems=problems,
                              artifacts_dir=services.artifacts_dir / entry.id)


__all__ = ["RpiHatJudgeBackend", "RpiHatJudgeRun", "create_backend"]
