# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One session's progress through a VLM-graded procedure.

The monitor launches a grounded check about once a second, never while the
wearer is still hearing the step, and never twice on the same frame. A step
advances only after ``pass_streak`` consecutive grounded passes that also held
for the step's ``hold_seconds``; any failure restarts both. A failure with a
usable correction is spoken, with the gap between repeats of the same problem
growing each time it has actually been said.

Every check is tagged with the step it was launched on and dropped if the step
moved while it ran, so a late verdict can never grade the wrong question.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from ..base import (
    CommandResult,
    Cue,
    RunCommand,
    RunContext,
    RunFinished,
    RunSnapshot,
    StepChanged,
    TimedFrame,
    TurnContext,
    Verdict,
)
from .grading import (
    CheckResult,
    StepFacts,
    StudentImage,
    check_step,
    is_parser_issue,
    strip_thinking,
    unreliable_reference_result,
)

if TYPE_CHECKING:
    from .backend import VlmBackend


def _now_us() -> int:
    return time.time_ns() // 1_000


def _safe_stem(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


class VlmRun:
    """The monitor, progression and corrections for one session."""

    def __init__(
        self,
        backend: VlmBackend,
        ctx: RunContext,
        *,
        start_step: int,
    ) -> None:
        self._backend = backend
        self._ctx = ctx
        self._cfg = backend.config
        self._sop = backend.sop
        self._step = start_step
        self._closed = False
        self._advancing = False
        self._lock = asyncio.Lock()
        self._monitor: asyncio.Task[None] | None = None
        self._started_at_us = 0
        self._reset_step_state()
        stem = _safe_stem(f"{ctx.session_id}_{ctx.owner}")
        self._frames_dir = backend.artifacts_dir / "checks" / stem
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._frame_slot = 0

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._started_at_us = _now_us()
        self._arm_step()
        await self._ctx.emit(StepChanged(self._step, reason="start"))
        self._start_monitor()

    async def on_frame(self, frame: TimedFrame) -> None:
        # The vlm backend pulls frames on its own cadence.
        return None

    async def input_changed(self) -> None:
        await self._stop_monitor()
        async with self._lock:
            self._arm_step()
        self._start_monitor()

    async def close(self, reason: str) -> None:
        self._closed = True
        await self._stop_monitor()

    # ── commands ──────────────────────────────────────────────────────────────

    async def command(self, command: RunCommand) -> CommandResult:
        kind = command.kind
        if kind == "repeat":
            return CommandResult(True, speech=self.step_line())
        if kind == "next":
            await self._advance()
            return CommandResult(True)
        if kind == "advance":
            granted, why = self._advance_is_grounded()
            logger.info("GUIDANCE_TURN_ADVANCE granted={} why={}", granted, why)
            if not granted:
                return CommandResult(False, speech=self._cached_spoken_issue(), reason=why)
            await self._advance()
            return CommandResult(True, reason=why)
        if kind in ("go_to", "reset"):
            index = 0 if kind == "reset" else (command.step_index or 0)
            total = len(self._sop.steps)
            if not 0 <= index < total:
                return CommandResult(
                    False,
                    speech=(f"'{self._sop.name}' has {total} steps, so there is no step "
                            f"{index + 1}. Which one did you mean?"),
                    reason="out-of-range",
                )
            await self._stop_monitor()
            async with self._lock:
                self._step = index
                self._arm_step()
            await self._ctx.emit(StepChanged(index, reason="navigate"))
            self._start_monitor()
            return CommandResult(True)
        if kind == "check":
            step_at_start = self._step
            result = await self._run_check(require_reliable_reference=False)
            if self._step == step_at_start:
                self._last_result = result
            return CommandResult(True, reason=self._verdict_summary())
        return CommandResult(False, reason=f"unsupported command {kind!r}")

    # ── state for the host ───────────────────────────────────────────────────

    def snapshot(self) -> RunSnapshot:
        return RunSnapshot(
            step_index=self._step,
            total_steps=len(self._sop.steps),
            instruction=self._sop.instruction(self._step),
            extra={
                "verdict": self._verdict_extra(),
                "gradeable": self._sop.steps[self._step].reference_reliable,
            },
        )

    def step_line(self) -> str:
        """The deterministic "here is where you are" sentence; mutates nothing."""

        instruction = self._sop.instruction(self._step).rstrip()
        tail = "" if instruction[-1:] in ".!?" else "."
        return (f"Step {self._step + 1} of {len(self._sop.steps)} is: "
                f"{instruction}{tail}")

    def turn_context(self) -> TurnContext:
        step = self._sop.steps[self._step]
        verdict = self._last_result
        verdict_text = (
            "Nothing yet — you have not looked at what they are doing on this step."
        )
        if verdict is not None and verdict.timestamp_us and not is_parser_issue(verdict.issue):
            age_s = max(0.0, (_now_us() - verdict.timestamp_us) / 1_000_000)
            fresh = self._cfg.progression.verdict_fresh_s
            age = "a moment ago" if age_s < 2 else (
                f"{age_s:.0f} seconds ago" if age_s <= fresh
                else f"{age_s:.0f} seconds ago — this may be out of date"
            )
            check_lines = "\n".join(
                f"- {c.get('requirement', '')}: {'yes' if c.get('visible') else 'no'}; "
                f"{c.get('evidence', '')}"
                for c in verdict.checks
            )
            verdict_text = (
                f"Grounded check ({age}): completed={'yes' if verdict.completed else 'no'}\n"
                f"Observation: {verdict.current_observation}"
                + (f"\n{check_lines}" if check_lines else "")
            )
        reference_note = ""
        if not step.reference_reliable:
            reference_note = (
                "\nYou cannot check this step by looking at a reliable reference; "
                "take their word for whether it is done."
            )
        key_info = f"\n\n{step.key_info.as_prompt_block()}" if step.key_info else ""
        parts = ""
        if self._sop.parts:
            parts = (
                "\n\nTelling the parts apart, for when they ask which is which:\n"
                + "\n".join(f"- {p}" for p in self._sop.parts)
                + "\nDescribe a part by its SHAPE, in these words. A detector "
                "label like nosepad_0 is drawn on your view of the scene and "
                "not on the object, so the wearer cannot see it -- naming one "
                "at them (\"the pad labeled 0\") points at something they have "
                "no access to. Say what the piece looks like instead."
            )
        block = key_info + parts + "\n\n" + verdict_text + reference_note
        return TurnContext(prompt_block=block, frame_jpeg=self._turn_frame())

    # ── step arming and advancing ────────────────────────────────────────────

    def _reset_step_state(self) -> None:
        self._step_spoken_at_us = 0
        self._consecutive_yes = 0
        self._yes_since_us = 0
        self._last_result: CheckResult | None = None
        self._correction_state: dict[str, Any] = {}
        self._last_checked_live_ts = 0
        self._last_speaking_us = 0

    def _arm_step(self) -> None:
        """Reset per-step grading state for the step now being entered."""

        recorder = self._ctx.recorder
        recorder.capture_clip("step-advance")
        recorder.set_step(self._step, self._sop.instruction(self._step))
        self._reset_step_state()
        self._step_spoken_at_us = _now_us()
        logger.info("GUIDANCE_STEP {}/{} {}", self._step + 1, len(self._sop.steps),
                    self._sop.instruction(self._step)[:60])

    async def _advance(self) -> None:
        if self._closed or self._advancing:
            return
        self._advancing = True
        try:
            async with self._lock:
                self._step += 1
                finished = self._step >= len(self._sop.steps)
                if finished:
                    self._step = len(self._sop.steps) - 1
                else:
                    self._arm_step()
            if finished:
                await self._ctx.emit(RunFinished("completed"))
            else:
                await self._ctx.emit(StepChanged(self._step, reason="advance",
                                                 acknowledge=True))
        finally:
            self._advancing = False

    def _hold_seconds(self, step_idx: int) -> float:
        step = self._sop.steps[step_idx]
        return step.hold_seconds or self._cfg.progression.default_hold_s

    def _advance_is_grounded(self) -> tuple[bool, str]:
        step = self._sop.steps[self._step]
        result = self._last_result
        # The cached verdict first, so an ungradeable step that was checked
        # anyway traces as grounded rather than hiding real evidence.
        if (
            result is not None and result.completed and result.has_evidence
            and not is_parser_issue(result.issue)
        ):
            max_age = self._cfg.progression.grounded_verdict_max_age_s
            if result.timestamp_us and (_now_us() - result.timestamp_us) / 1e6 <= max_age:
                return True, "grounded"
            return False, "stale-verdict"
        if not step.reference_reliable:
            return True, "ungradeable"
        return False, "no-evidence"

    def _cached_spoken_issue(self) -> str:
        result = self._last_result
        if result is None:
            return ""
        issue = result.issue.strip()
        if not issue or is_parser_issue(issue):
            return ""
        return issue if issue[-1:] in ".!?" else f"{issue}."

    def _verdict_summary(self) -> str:
        result = self._last_result
        if result is None or not result.timestamp_us:
            return "no fresh view of the work"
        state = "done" if (result.completed and result.has_evidence) else "not done yet"
        observation = result.current_observation or "nothing clear"
        issue = "" if is_parser_issue(result.issue) else result.issue
        return f"{state}; observed: {observation}" + (f"; problem: {issue}" if issue else "")

    def _verdict_extra(self) -> dict[str, Any]:
        result = self._last_result
        if result is None:
            return {}
        return {
            "completed": result.completed and result.has_evidence,
            "observation": result.current_observation,
            "issue": "" if is_parser_issue(result.issue) else result.issue,
            "timestamp_us": result.timestamp_us,
        }

    # ── the monitor ──────────────────────────────────────────────────────────

    def _start_monitor(self) -> None:
        if self._closed:
            return
        self._monitor = asyncio.create_task(self._monitor_loop(), name="guidance-monitor")

    async def _stop_monitor(self) -> None:
        task = self._monitor
        self._monitor = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _monitor_loop(self) -> None:
        monitor = self._cfg.monitor
        interval = monitor.check_interval_s
        tick = min(interval, monitor.tick_s)
        need = self._cfg.progression.pass_streak
        pending: asyncio.Task[CheckResult] | None = None
        pending_step = 0
        last_launch_us = 0
        logger.info("GUIDANCE_MONITOR start step={}", self._step)
        try:
            while not self._closed:
                if pending is None:
                    await asyncio.sleep(tick)
                else:
                    await asyncio.wait({pending}, timeout=tick)
                if self._closed:
                    break
                step_idx = self._step

                if pending is not None:
                    if not pending.done():
                        continue
                    finished, pending = pending, None
                    if pending_step != step_idx:
                        logger.info("GUIDANCE_MONITOR drop=stale-check was={} now={}",
                                    pending_step, step_idx)
                        continue
                    await self._apply_check(finished, step_idx, need)
                    continue

                # Do not grade the wearer while they are still being told what to
                # do: the first checks of a step would land before they could act.
                still_speaking = self._ctx.speech_remaining_s()
                if still_speaking > 0.0:
                    # The projected END of playback: the settle window and the
                    # correction gap both run from when the wearer stopped hearing.
                    self._last_speaking_us = _now_us() + int(still_speaking * 1_000_000)
                lead = monitor.speech_lead_s
                if still_speaking > lead:
                    continue
                settle = monitor.settle_s
                if settle > 0.0 and self._last_speaking_us:
                    left_s = settle - (_now_us() - self._last_speaking_us) / 1_000_000
                    if left_s > lead:
                        continue
                if last_launch_us and (_now_us() - last_launch_us) / 1_000_000 < interval:
                    continue
                # Same pixels, same verdict: skip the expensive compare.
                if monitor.skip_static_frames and self._last_checked_live_ts:
                    latest = self._ctx.latest_frame()
                    if latest is not None and latest.timestamp_us == self._last_checked_live_ts:
                        continue
                last_launch_us = _now_us()
                pending_step = self._step
                pending = asyncio.create_task(
                    self._run_check(require_reliable_reference=(
                        self._cfg.evaluator.require_reliable_reference
                    )),
                    name="guidance-check",
                )
        except asyncio.CancelledError:
            if pending is not None and not pending.done():
                pending.cancel()
            raise
        except Exception:
            logger.exception("guidance monitor error")
        if pending is not None and not pending.done():
            pending.cancel()
        logger.info("GUIDANCE_MONITOR exit")

    async def _apply_check(
        self, finished: asyncio.Task[CheckResult], step_idx: int, need: int,
    ) -> None:
        """Fold one completed check into the streak, the hold, and corrections."""

        try:
            result = finished.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("guidance check failed")
            return
        self._last_result = result
        if result.timestamp_us:
            self._last_checked_live_ts = result.timestamp_us
        if result.completed and result.has_evidence:
            self._consecutive_yes += 1
            self._correction_state = {}
            now_us = _now_us()
            if self._yes_since_us == 0:
                self._yes_since_us = now_us
            held_s = (now_us - self._yes_since_us) / 1_000_000
            hold_required = self._hold_seconds(step_idx)
            logger.info("GUIDANCE_MONITOR yes-count={}/{} step={} held={:.1f}s/{:.1f}s",
                        self._consecutive_yes, need, step_idx, held_s, hold_required)
            if self._consecutive_yes >= need and held_s < hold_required:
                # Count satisfied, time not yet. Keeping the streak is what lets
                # the clock run; dropping it would mean the step never passes.
                return
            if self._consecutive_yes >= need:
                self._consecutive_yes = 0
                self._yes_since_us = 0
                logger.info("GUIDANCE_MONITOR vlm-advance step={} held={:.1f}s",
                            step_idx, held_s)
                await self._advance()
            return
        self._consecutive_yes = 0
        # Any failure restarts the hold: a step that must stay done for N seconds
        # has to be continuously done, or a pad that fell out and was put back
        # would pass.
        self._yes_since_us = 0
        await self._maybe_correct(step_idx, result)

    async def _maybe_correct(self, step_idx: int, result: CheckResult) -> None:
        current_obs = result.current_observation.strip()
        missing_key = (
            str(result.missing_or_mismatched[0]).strip() if result.missing_or_mismatched else ""
        )
        issue = result.issue.strip()
        if not issue and missing_key:
            issue = f"{missing_key} not visible"
        if not issue:
            return
        if is_parser_issue(issue):
            if missing_key and current_obs:
                issue = f"{missing_key} not visible"
            elif issue.lower().startswith("unreliable-reference"):
                issue = (
                    "I cannot reliably check this step from the demo video. "
                    "Please re-record this step or say next to continue."
                )
            else:
                return

        now_us = _now_us()
        state = self._correction_state
        key = missing_key or issue
        same_key = state.get("step_idx") == step_idx and state.get("key") == key
        # Times this problem has been SPOKEN, not offered: counting offers drove
        # the back-off to its cap within seconds and muted corrections.
        spoken_count = int(state.get("count", 0)) if same_key else 0
        # The gap is per STEP, not per wording: the model phrases one unchanged
        # problem differently each cycle, and keying on the text restarted it.
        same_step = state.get("step_idx") == step_idx
        last_spoken_us = int(state.get("last_spoken_us", 0)) if same_step else 0
        state.update({"step_idx": step_idx, "key": key, "issue": issue,
                      "count": spoken_count, "last_spoken_us": last_spoken_us})

        corrections = self._cfg.corrections
        gap_s = corrections.first_gap_s
        if spoken_count:
            gap_s = min(corrections.max_gap_s, corrections.backoff_s * spoken_count)
        # The gap runs from the last thing the wearer HEARD or said, so a step is
        # never told and corrected in one breath.
        heard_us = max(last_spoken_us, self._last_speaking_us, self._ctx.last_heard_us())
        if heard_us and now_us - heard_us < int(gap_s * 1_000_000):
            return
        response = issue if issue[-1:] in ".!?" else f"{issue}."
        if self._closed or self._step != step_idx:
            return
        await self._ctx.emit(Cue(response, kind="correction"))
        state["last_spoken_us"] = now_us
        state["count"] = spoken_count + 1
        logger.info("GUIDANCE_CORRECTION step={} said={} issue={}", step_idx,
                    spoken_count + 1, issue[:80])
        recorder = self._ctx.recorder
        recorder.note("CORRECTION", issue=issue, spoken=response, count=spoken_count + 1)
        recorder.capture_clip("correction")

    # ── one grounded check ───────────────────────────────────────────────────

    async def _run_check(self, *, require_reliable_reference: bool) -> CheckResult:
        step_idx = self._step
        step = self._sop.steps[step_idx]
        recorder = self._ctx.recorder
        if require_reliable_reference and not step.reference_reliable:
            return unreliable_reference_result()
        min_ts = max(self._started_at_us, self._step_spoken_at_us)
        timeout_s = self._cfg.monitor.check_timeout_s
        try:
            result = await asyncio.wait_for(
                self._grade(step_idx, min_ts), timeout=timeout_s,
            )
        except TimeoutError:
            # Distinct from a negative verdict on purpose: without this an
            # overrunning VLM looks like a wearer who has not done the step.
            logger.info("GUIDANCE_CHECK_TIMEOUT step={} after={:.1f}s", step_idx, timeout_s)
            recorder.note("CHECK_TIMEOUT", after_s=timeout_s)
            recorder.capture_clip("check-timeout")
            result = CheckResult()
        except Exception as exc:
            logger.exception("guidance check error")
            recorder.note("CHECK_ERROR", error=f"{type(exc).__name__}: {exc}")
            result = CheckResult()
        logger.info("GUIDANCE_MONITOR vlm-check {!r} -> {} evidence={} obs={!r}",
                    self._sop.instruction(step_idx)[:40],
                    "YES" if result.completed else "NO",
                    "YES" if result.has_evidence else "NO",
                    result.current_observation[:40])
        recorder.note("CHECK", completed=result.completed, has_evidence=result.has_evidence,
                      observation=result.current_observation, issue=result.issue,
                      frame=result.image_path, tier=result.tier)
        await self._ctx.emit(Verdict(result.as_dict()))
        return result

    async def _grade(self, step_idx: int, min_ts: int) -> CheckResult:
        student = await self._student_image(min_ts)
        if isinstance(student, CheckResult):
            return student
        facts = self._facts(step_idx)
        return await check_step(
            facts=facts,
            student=student,
            ask=self._ask,
            overlay=self._backend.overlay_prompting(),
            annotate_teacher=self._backend.annotate_teacher,
            tier2_parallel=self._cfg.evaluator.tier2_parallel,
        )

    def _facts(self, step_idx: int) -> StepFacts:
        step = self._sop.steps[step_idx]
        info = step.key_info
        reliable = step.reference_reliable
        requests = (
            tuple(self._ctx.wearer_requests())
            if self._backend.capabilities.wearer_requests else ()
        )
        return StepFacts(
            instruction=self._sop.instruction(step_idx),
            teacher_image_path=step.image_path if reliable else "",
            # Only meaningful next to a usable after-frame: alone it would be
            # handed to the VLM as if it were the target.
            teacher_before_image_path=step.before_image_path if reliable else "",
            teacher_caption=step.teacher_caption,
            expected_requirements=step.expected_requirements,
            key_objects=info.objects if info else (),
            key_action=info.action if info else "",
            key_position=info.position if info else "",
            key_target_state=info.target_state if info else "",
            key_ignore=info.ignore if info else (),
            geometry_gate=step.geometry_gate,
            wearer_context=requests,
            part_guide=self._sop.parts,
        )

    async def _student_image(self, min_ts: int) -> StudentImage | CheckResult:
        """The frame to grade: the preview's newest annotation when fresh.

        Reusing it makes the wearer's preview and the model's evidence the same
        pixels. Only a frame newer than the step's freshness floor and within
        the reuse window qualifies; otherwise a frame is fetched and annotated.
        """

        evaluator = self._cfg.evaluator
        annotator = self._backend.preview_annotator()
        latest = self._ctx.latest_frame()
        max_age_us = int(evaluator.frame_reuse_max_age_ms * 1000)
        frame: TimedFrame | None = None
        if (
            latest is not None
            and latest.annotated is not None
            and latest.timestamp_us >= min_ts
            and _now_us() - latest.timestamp_us <= max_age_us
        ):
            frame = latest
        if frame is None:
            frame = await self._ctx.fetch_frame()
            if frame is None:
                return CheckResult(issue="I cannot see a current frame.")
            if min_ts and frame.timestamp_us and frame.timestamp_us < min_ts:
                return CheckResult(timestamp_us=frame.timestamp_us,
                                   issue="Waiting for a fresh student frame.")
            if annotator is not None and frame.annotated is None:
                try:
                    frame.annotated = await annotator.annotate_array(
                        frame.image.copy(), stream=self._ctx.input_participant,
                    )
                except Exception as exc:
                    if annotator.profile.required:
                        return CheckResult(timestamp_us=frame.timestamp_us,
                                           issue=f"YOLO overlay unavailable: {exc}")
                    logger.warning("student frame annotation failed: {}", exc)
        annotated = frame.annotated
        image = annotated.image if annotated is not None else frame.image
        path = await asyncio.to_thread(self._write_student_frame, image, frame.timestamp_us)
        spatial = ""
        if annotated is not None and self._backend.config.spatial_context:
            spatial = annotated.spatial_context
        return StudentImage(
            path=path,
            timestamp_us=frame.timestamp_us,
            overlay_applied=annotated is not None,
            spatial_context=spatial,
            geometry=annotated.geometry if annotated is not None else None,
        )

    def _write_student_frame(self, image: Any, timestamp_us: int) -> str:
        from ...vision.frames import bgr_to_jpeg

        data = bgr_to_jpeg(image, max_width=100_000, quality=self._cfg.evaluator.jpeg_quality)
        # A small ring of names: the recorder hardlinks what the model saw, and
        # os.replace gives each write a new inode, so a link keeps its bytes.
        self._frame_slot = (self._frame_slot + 1) % 8
        final = self._frames_dir / f"student_{self._frame_slot}.jpg"
        tmp = final.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, final)
        return str(final)

    async def _ask(self, images: list[str] | tuple[str, ...], question: str) -> str:
        vlm = self._ctx.models.vlm
        started = time.monotonic()
        error = ""
        raw = ""
        try:
            if len(images) == 1:
                response = await vlm.ask_image(Path(images[0]), question)
            else:
                response = await vlm.ask_images([Path(p) for p in images], question)
            raw = strip_thinking(response.content or "")
            return raw
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            recorder = self._ctx.recorder
            if recorder.recording:
                recorder.record_call(
                    kind="vlm",
                    name="ask_image" if len(images) == 1 else "ask_frames",
                    request={"question": question, "image_paths": list(images)},
                    response=raw,
                    latency_ms=(time.monotonic() - started) * 1000.0,
                    error=error,
                    images=list(images),
                )

    def _turn_frame(self) -> bytes:
        """The wearer's current annotated view for the foreground, or b"".

        The same picture the monitor grades, already in memory, so showing it
        costs prompt tokens and no extra model call. A missing frame is never
        an error: text only is the working fallback.
        """

        latest = self._ctx.latest_frame()
        if latest is None:
            return b""
        from ...vision.frames import bgr_to_jpeg

        image = latest.annotated.image if latest.annotated is not None else latest.image
        with suppress(Exception):
            return bgr_to_jpeg(image, max_width=1280, quality=80)
        return b""

    def checkpoint_state(self) -> Mapping[str, Any]:
        return {}


__all__ = ["VlmRun"]
