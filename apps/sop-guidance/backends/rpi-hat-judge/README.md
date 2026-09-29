<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Raspberry Pi HAT judge

The `rpi_hat_judge` procedure backend judges the Raspberry Pi M.2 HAT+
assembly from the camera alone. There are four corner screws, put in
diagonally (top-left, bottom-right, top-right, bottom-left), then the FPC
ribbon cable. The procedure is `procedures/rpi-hat-assembly/`.

It is the judge from the old fork's `v0.1.3-rpi-judge-beta.4` tag
(`agent-samples/sop-guidance/judge/`), moved behind the guidance backend
seam. The decision logic is ported unchanged. What changed is how the judge
talks to the rest of the system.

## How it judges

Each frame the host pushes (3 Hz) runs through:

1. **Detection.** The v7 detector (`rpi-hat-v7` in `yaml/detectors.yaml`) is
   a 5-class YOLO26m int8 IR run at 1280. The shared hand detector runs
   alongside it.
2. **`TemporalSmoother`.** Tracks one board box, holds it through short
   misses, and takes a median of the hole counts.
3. **`OcclusionFilter`.** Rejects the frame when a hand covers the board,
   board confidence is low, or too few holes are visible. A frame that shows
   all six holes is accepted outright.
4. **`FPCSeatState`.** Judges the cable, but only once all four screws are
   confirmed and only on accepted frames. It uses the cable box's overlap with
   the board and needs a majority over time to count it seated.
5. **`PerHoleTracker`.** Matches boxes to the six hole anchors and runs a
   per-hole majority vote. Removing a screw needs 3 s of contrary evidence.
   Two sentinels void a frame: the fixed holes 5 and 6 reported filled, or too
   many loose screws. A step counter then moves forward on a short
   confirmation, and jumps or goes back only on a long one.
6. **`Progress`.** Picks the step to show: the step of the next hole to fill.
   It also works out which steps are done and which alerts to speak.

The judge's output reaches the host as events:

- **Next step confirmed:** a `StepChanged`. The host announces it, led by
  the judge's own "Step 1, install the top-left screw, done."
- **Screw out of order, screw removed, step skipped, or count regressed:** a
  correction `Cue`. It says what happened and what to do next. The UI marks
  the step red until the step changes.
- **Boxes:** an `OverlayUpdate` every frame. The preview draws it on the
  wearer's return video in the profile's colours; the next hole is boxed as
  `next`.
- **Hole map:** in `guidance.state.extra`. It carries `holes`, `next_hole`,
  `trusted`, `reject_reason`, `fpc_armed`, `fpc_seated`, `steps_done`,
  `steps_owed`, `hole_layout` and `hole_names`.
- **Last step confirmed:** a `RunFinished`, which ends the session.

Voice never moves the judge. The capabilities are `voice_advance`,
`jump_to_step`, `resume` and `check_on_demand`, all false. So:

- "next" is answered with "The camera confirms each step, so I can't skip
  ahead" and the current step.
- "repeat" and "which step" read the current step.
- "start over" starts a fresh run, with fresh votes and latches and the model
  still loaded.

During the run, the guidance turn prompt carries the judge's state and a rule
not to declare a step done.

## What changed from the old judge

| Old judge | Here |
|---|---|
| Separate process, HTTP on :8020, gate file, shared-memory ingest worker | In-process backend; the host pushes the session's own input camera |
| Identity had to be `web-client` | Whichever participant the session's camera is |
| MediaPipe hand landmarks (`mediapipe<0.10.22`, own venv) | The shared `hand_yolov8n` detector, in the worker's own env |
| `reset` kept old events and replayed stale alerts | A new run, or `reset`, clears everything |
| Own TTS and HUD, `sop.*` topics, `/api/state` polling | Host speech, return-video boxes, `guidance.state.extra` |
| Wall clock in the state layer | Frame timestamps |
| Chinese strings translated by regex | Events carry fields; `texts.py` words them in English |
| Always-on mp4 and trace recording | The host recorder's debug levels |

The `th_hand` threshold (0.40) was tuned on MediaPipe landmark boxes padded
by 12%. The YOLO hand boxes are drawn a little differently, so check the
threshold on live footage: `hand covers N% of the board` in `reject_reason`
is the reading to watch.

## Configure

All settings live in `rpi_hat_judge/config.py`. The defaults are the values
the old judge ran live with. The procedure's `backend_config`, or
`guidance_defaults.backend_config.rpi_hat_judge` in the worker yaml,
overrides any key. The ones that matter most:

| Key | Default | What it does |
|---|---|---|
| `tick_hz` | 3.0 | Detection rate; vote counts assume it is met |
| `tracker.hf_conf_min` | 0.65 | Confidence floor for a filled hole |
| `occlusion.th_hand` | 0.40 | Hand-over-board fraction that voids a frame |
| `fpc.iob_min`, `fpc.need_sec` | 0.90, 2.5 | Cable-on-board overlap, and how long it must hold |

Keep the board flat and at least a third of the frame wide. Below that, the
detector misses the holes in about two frames of three. The hole anchors were
calibrated on 1920x1080 with the "Raspberry Pi" silkscreen upright.

## Test

```bash
cd apps/sop-guidance
worker/.venv/bin/python -m pytest -q backends/rpi-hat-judge/tests
```

`test_state_layer.py` is the old judge's own self-test, ported.
`test_backend.py` drives the backend through the real guidance host with
scripted detections.
