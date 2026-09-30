<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Adding a guidance task

A guidance task is called a **procedure**. Each one is a folder under
`procedures/`. The folder's `procedure.yaml` names a **backend**: the code
that decides when a step is done.

The guidance host, the voice pipeline, the tools, the speech and the UI are
shared by every procedure. A new task never changes them.

Pick the lightest path that can judge your task:

| Path | Use it when | You write | Worked example |
|---|---|---|---|
| [A. VLM, data only](#a-vlm-data-only) | A vision model can tell "done" from a reference photo | `procedure.yaml`, `sop.json`, frames | `tests/fixtures/procedures/filter-swap/` |
| [B. VLM + detector](#b-add-a-detector-to-a-vlm-procedure) | Boxes around the parts help the VLM, or you want an overlay | Path A, plus a detector profile and weights | `procedures/nosepad-replacement/` |
| [C. VLM + geometry plugin](#c-add-a-geometry-plugin) | Box geometry can prove a step is *not* done | Path B, plus one Python module | `profiles/nosepad/geometry.py` |
| [D. Custom backend](#d-write-a-custom-backend) | Your own model or state machine decides, not a VLM | A Python package | `backends/rpi-hat-judge/` |
| [E. Sidecar](#e-run-a-backend-in-its-own-process) | Path D, but its dependencies clash with the worker's | Path D, plus a small launcher | `tests/fixtures/sidecar_stub.py` |

Every path starts with [the procedure folder](#the-procedure-folder). The
UI works for any procedure without changes, and
[a procedure-specific UI view](#optional-a-procedure-specific-ui-view) is
optional. Finish with [the checklist](#checklist).

## The procedure folder

```text
procedures/<id>/
  procedure.yaml    # always
  ...               # whatever the backend reads: sop.json, frames/, its own files
```

`procedure.yaml`:

```yaml
id: water-filter-swap          # must equal the folder name: a-z, 0-9, hyphens
title: water filter swap       # what the host says: "Stopped guidance for '<title>' ..."
aliases: [water filter, jug filter, change the filter]
description: Replace the filter cartridge in a water jug.
backend: vlm                   # vlm | rpi_hat_judge | remote | <your backend>
enabled: true

backend_config:                # read and validated by the backend named above
  sop: sop.json

# Optional:
request_qualifiers: [solid, wire]   # words that tell interchangeable parts apart
foreground:
  active_prompt_file: prompts/active.txt   # task rules added to the guidance-turn prompt
  reminder_interval_s: 60                  # "we are still on step N" after Q&A; 0 = off
models:
  llm_role: llm                 # roles from yaml/models.*.json
  vlm_role: vlm
ui:
  thumbnail: frames/step_04_b.jpg          # served by the app API; shown by xr-ai-ui
```

**How the wearer starts it.** Two routes reach the same `GuidanceHost.begin`:

- **Fast path.** "Guide me through X" and similar phrasings are matched by
  `text.guidance_request`, then `GuidanceHost.match_procedure`. X must be
  exactly the id, the title or one of the aliases, with or without a leading
  "the". List every name people will actually say, including likely speech
  recognition spellings ("m2 hat" beside "m.2 hat").
- **LLM path.** Anything else goes to the normal-mode LLM. It sees every
  enabled procedure, with its title, aliases and description, as an exact
  `procedure_id` enum on `guidance__start`, so it cannot invent an id.

**How settings layer.** Later layers win: mappings merge key by key, lists
and scalars replace.

1. The backend's own defaults in code.
2. `guidance_defaults` in `yaml/sop_guidance_worker.yaml`. Its
   `backend_config` block is keyed by backend name, so
   `guidance_defaults.backend_config.vlm` only reaches `vlm` procedures.
3. `procedure.yaml`.

**Validation.** Every folder is validated when the worker starts. A typo stops
the worker with the file and field named, rather than failing mid-session.

## A. VLM, data only

The built-in `vlm` backend (`guidance/sop_guidance/backends/vlm/`) runs a
monitor loop per step:

1. It grades the newest camera frame with a vision-language model, in up to
   three tiers: compare with the teacher's frames, check the live frame
   against the step's key info, then diagnose the mistake.
2. It advances after two passes in a row, plus the step's `hold_seconds`.
3. It speaks corrections in between, with a back-off.

You supply only data: `sop.json` and its reference frames.

```json
{
  "schema_version": 1,
  "name": "water filter swap",
  "summary": "Replace the filter cartridge in a water jug.",
  "parts": [],
  "steps": [
    { "description": "Lift the old filter cartridge out of the jug." },
    {
      "description": "Press the new filter cartridge into the jug until it clicks.",
      "image": "frames/step_02.jpg",
      "teacher_caption": "A white filter cartridge seated in the funnel of a water jug.",
      "expected_requirements": ["New cartridge seated in the funnel"],
      "key_info": {
        "objects": ["filter cartridge", "water jug"],
        "action": "press the cartridge into the funnel",
        "target_state": "cartridge seated flush in the funnel"
      }
    }
  ]
}
```

| Step field | What it does |
|---|---|
| `description` | The instruction, read out verbatim as "Step n of N: ..." |
| `image` | The teacher's AFTER frame. **A step without one cannot advance on its own**: the wearer says "next" to move on. |
| `before_image` | The BEFORE frame. Add it when a step is defined by a change rather than a look, so the grader compares two states. |
| `teacher_caption` | One sentence describing the AFTER frame |
| `expected_requirements` | Short conditions that must all be visible; one unmet one fails the step |
| `key_info` | `objects`, `action`, `position`, `target_state`, `ignore`: the only facts the grader checks |
| `reference_images` | Extra frames shown to the UI tutorial |
| `hold_seconds` | How long a pass must hold before advancing (clamped to 60) |
| `geometry_gate` | Path C only: a gate name from the geometry plugin |

Top-level `parts` lists descriptions that tell interchangeable parts apart,
for example "Size 0 nose pad (detector label nosepad_0): ...". Wearer
requests such as "the size zero one" are graded against it.

**Reference frames.** Take them from the camera the wearer will use (the
glasses), at the same distance and lighting. Frame paths are relative to the
procedure folder and served to the UI by `/api/procedures/<id>/files/...`.

Tunables such as monitor cadence, streak, correction gaps and evaluator
flags live in `backends/vlm/config.py` (`VlmBackendConfig`). Override them
under `backend_config:` in `procedure.yaml`, or for every VLM procedure under
`guidance_defaults.backend_config.vlm` in the worker yaml.

## B. Add a detector to a VLM procedure

A detector draws boxes on the frame the VLM grades, and a legend of what each
colour means goes into the prompt. The same boxes are drawn on the wearer's
return video.

1. **Export the model** to an OpenVINO IR directory. The directory name
   **must** end in `_openvino_model`; ultralytics picks the backend from the
   name. A `.pt` file works too.
2. **Put it under `detectors/`**, not `models/`, which the repo ignores. The
   weights go through Git LFS by `.gitattributes`; check with `git lfs status`
   before committing.
3. **Add a profile** to `yaml/detectors.yaml`:

   ```yaml
   profiles:
     filter-v1:
       enabled: true
       required: true            # a failed annotation fails the check
       preheat: true             # warm up at startup
       model: ../detectors/filter-v1-int8_openvino_model
       imgsz: 640                # the export size
       device: intel:cpu         # intel:gpu, a CUDA ordinal, or cpu
       conf: 0.30
       overlay:
         class_labels:           # keys MUST be exactly the checkpoint's classes
           cartridge: cartridge
           jug: jug
         class_colors_rgb:
           cartridge: "#27AE60"
           jug: "#2F80ED"
       hands:                    # optional shared hand detector
         enabled: true
         model: ../detectors/hand/hand_yolov8n.pt
         replaces_class: ""      # a checkpoint class it supersedes, if any
   ```

   Colours must be ones the prompt legend can name: blue `#2F80ED`, light
   pink `#F4A6B5`, green `#27AE60`, yellow `#F2C94C`. Otherwise the legend
   tells the VLM to look for a hex code. Add a name to `_COLOR_NAMES` in
   `vision/overlay.py` if you need another.

4. **Point the procedure at it:**

   ```yaml
   backend_config:
     sop: sop.json
     detector:
       profile: filter-v1
   ```

The worker checks at startup that the weights exist and are not LFS pointer
files, and that `class_labels` matches the checkpoint's classes.

## C. Add a geometry plugin

A geometry plugin turns boxes into task facts, such as "the pad is in a
hand". It can add prose to the prompt and **veto** a pass on a step. It can
never grant one, so a missed box cannot complete a step.

Write a module, for example `profiles/filter/geometry.py`. Every hook is
optional; declare only what you need (the full contract is in
`vision/geometry.py`):

```python
GATES = ("cartridge_seated",)

def analyze(detections, stream=""):      # -> anything; handed back to describe/veto
    ...

def describe(geometry) -> str:          # prose for the VLM prompt, or ""
    ...

def veto(geometry, gate: str) -> str:   # a reason to reject, or "" to allow
    ...

def request_veto(geometry, gate: str, requests) -> str:
    ...                                 # the part in play is not the one the wearer asked for
```

Then reference it from the procedure and name a gate on a step:

```yaml
backend_config:
  detector: {profile: filter-v1}
  geometry: ../../profiles/filter/geometry.py
  spatial_context: true          # add describe() prose to prompts
```

```json
{ "description": "...", "image": "...", "geometry_gate": "cartridge_seated" }
```

A step naming a gate the plugin does not declare stops the worker at
startup. The plugin is executed from disk, so it is trusted application
code; keep it in the app's `profiles/`.

## D. Write a custom backend

Write your own backend when something other than a VLM decides: your own
detector, a state machine, a sensor, a rule engine. It is a separate Python
package that the worker installs and imports. It runs in the worker process;
there is no container or port. `backends/rpi-hat-judge/` is the complete
example.

### Layout

```text
backends/<name>/
  pyproject.toml
  README.md
  <package>/
    __init__.py      # exports create_backend
    backend.py       # the backend and its run
    config.py        # a pydantic model for backend_config
  tests/
```

`pyproject.toml` registers the backend under the `sop_guidance.backends`
entry point. That name is what `backend:` in `procedure.yaml` refers to.

```toml
[project]
name = "sop-guidance-<name>"
dependencies = ["xr-ai-sop-guidance[vision]"]   # [vision] only if you use the detectors

[project.entry-points."sop_guidance.backends"]
my_backend = "my_backend:create_backend"

[tool.uv.sources]
xr-ai-sop-guidance = { path = "../../guidance", editable = true }
```

### Install it into the worker

1. **`worker/pyproject.toml`:** add `"sop-guidance-<name>"` to `dependencies`
   and a path source:

   ```toml
   sop-guidance-<name> = { path = "../backends/<name>", editable = true }
   ```

2. **`docker/worker.Dockerfile`:** in the dependency layer, copy the new
   `pyproject.toml` and create an empty `<package>/__init__.py`, as is done for
   `rpi-hat-judge`. The first `uv sync` needs every path dependency's
   `pyproject.toml` before the code is copied.
3. **`pyproject.toml` at the app root:** add the package folder to `pythonpath`
   and its `tests` folder to `testpaths`.
4. Run `uv sync --project worker`, then check that it is found:

   ```bash
   worker/.venv/bin/python -c "from sop_guidance.backends.registry import available_backends as a; print(a())"
   ```

### Implement the interface

Everything is in `guidance/sop_guidance/backends/base.py`. The backend is
built once per procedure folder at startup; a run is created per session, so
every run starts from fresh state.

```python
def create_backend(services: BackendServices) -> MyBackend:
    config = MyConfig.model_validate(dict(services.config))   # backend_config, merged
    spec = load_my_steps(services.entry.resolve(config.steps_file))
    annotator = services.frame_annotator(config.detector_profile, overrides={},
                                         geometry_path=None, spatial_context=False)
    return MyBackend(spec, config, annotator)
```

**`ProcedureBackend`** (one per procedure):

| Member | What it must do |
|---|---|
| `name`, `capabilities` | The entry-point name, and what the host may let the wearer do (below) |
| `title`, `steps()` | `StepInfo` per step: `number`, `instruction` (read out verbatim), `title`, `reference_images`… |
| `parts()` | Descriptions of interchangeable parts, or `()` |
| `instructions_digest()` | The step instructions; a checkpoint whose digest differs is not resumed |
| `validate()` | Problems that must stop startup (missing weights, bad config), as strings |
| `preview_annotator()` | The detector the preview draws with; with `provides_overlay`, only its colours are used |
| `open_run(ctx, start_step, checkpoint)` | A fresh `ProcedureRun` |

**`ProcedureRun`** (one per session):

| Method | What it must do |
|---|---|
| `start()` | Emit the first `StepChanged` |
| `on_frame(frame)` | Called at `capabilities.frame_hz` with the session camera's frame (`TimedFrame`: BGR `image`, `timestamp_us`). With `frame_hz: 0` the run pulls instead, from `ctx.latest_frame()` or `ctx.fetch_frame()`. |
| `command(cmd)` | `repeat`, `advance`, `next`, `go_to`, `check`, `reset`. Return `CommandResult(accepted, speech=...)`. |
| `snapshot()` | The current step, total and instruction; `extra` is JSON for clients; `state` is JSON saved in the checkpoint |
| `turn_context()` | Text for the guidance-turn prompt, plus an optional JPEG of the view |
| `input_changed()` | The session's camera moved to another participant |
| `close(reason)` | Stop; no more events |

**Events** go through `await ctx.emit(...)`, the run's only way out:

| Event | What the host does |
|---|---|
| `StepChanged(index, reason, lead="")` | Announces "Step n of N: instruction", led by `lead` if given; checkpoints; publishes state |
| `Cue(text, kind="correction")` | Speaks it. A `correction` also turns the UI step bar red until the step changes or a `Verdict` reports `completed`. |
| `Verdict(result)` | Records it and republishes `guidance.state`, so `snapshot().extra` reaches the UI |
| `OverlayUpdate(timestamp_us, detections, extra)` | Draws the boxes on the wearer's return video (needs `provides_overlay`) |
| `RunFinished(outcome)` | Ends the session with the completion message |

`ctx` also gives you `ctx.recorder` (`note`, `capture_clip`),
`ctx.models.llm` / `ctx.models.vlm`, `ctx.speech_remaining_s()` (to avoid
talking over an announcement), `ctx.last_heard_us()` and
`ctx.wearer_requests()`.

**Capabilities** decide what voice and tools may do. The host enforces them
before your run sees a command:

| Capability | Off means |
|---|---|
| `voice_advance` | "next" is refused with "The camera confirms each step…" |
| `jump_to_step` | No starting at, or navigating to, step N |
| `resume` | No resume offer; "start over" is a fresh run |
| `wearer_requests` | Spoken choices are not extracted for grading |
| `check_on_demand` | The guidance turn cannot ask for an immediate check |
| `frame_hz` | 0: the run pulls frames. >0: the host pushes them at that rate. |
| `provides_overlay` | On: the preview draws your `OverlayUpdate` boxes instead of running a detector |
| `min_frame_size`, `max_concurrent_runs` | Operator hints and a session limit |

**Detectors.** Use `services.frame_annotator(profile_name, ...)` with a
profile in `yaml/detectors.yaml`, as in path B, rather than loading models
yourself. It is preheated at startup, it checks the weights, and it runs off
the event loop. `FrameAnnotator.detect_array(image)` returns boxes without
drawing.

**Config.** Validate `backend_config` with your own pydantic model
(`extra="forbid"` catches typos). Workers can set defaults under
`guidance_defaults.backend_config.<name>`.

### Test it

Drive it through the real host with scripted inputs. See
`backends/rpi-hat-judge/tests/test_backend.py`, which uses `make_harness` from
`tests/conftest.py`, and the smaller `tests/fixtures/scripted_backend.py`:

```python
harness = await make_harness(tmp_path, backend=MyBackend(...))
await harness.host.begin("alice", "lid-demo")
await harness.host.session_of("alice").run.on_frame(frame)
assert harness.ports.texts("alice")[-1].startswith("Step 2 of")
```

## E. Run a backend in its own process

If your backend's dependencies cannot live in the worker's environment, keep
the same backend class and serve it from a separate process:

```python
# its own uv project
from sop_guidance.backends.remote import serve
asyncio.run(serve(MyBackend(...), "ipc:///tmp/my-backend.sock"))
```

```yaml
backend: remote
backend_config:
  endpoint: ipc:///tmp/my-backend.sock
```

Frames go over a shared-memory ring, and commands and events over ZMQ. The
host behaves exactly as in-process. Start the sidecar as an app-owned process
next to the worker. `tests/fixtures/sidecar_stub.py` is the minimal one.

## Optional: a procedure-specific UI view

xr-ai-ui needs nothing for a new procedure. The picker, the tutorial, the
NowCard (title, step bar, instruction, reference frame, corrections) and the
sessions list all come from `/api/procedures` and `guidance.state`.

To show something only your backend knows, such as the RPi hole map:

1. **Publish it** in `snapshot().extra` (and `OverlayUpdate.extra`), and emit
   a `Verdict` when it changes so `guidance.state` is republished.
2. **Parse it in xr-ai-ui** in `lib/guidance/<task>.ts`: a defensive
   `parse…(extra)` and an `is…(live)` check on the procedure id or backend
   name. See `lib/guidance/rpi-hat.ts`.
3. **Draw it** in a component under `components/procedures/<task>/`. It only
   draws, and knows nothing about the card.
4. **Pass it to the card from the parent pages**,
   `components/live/live-runner.tsx` and
   `components/sessions/session-list.tsx`, as the NowCard's `aside`. It is
   shown beside the card body in place of the reference frame. If the task
   can be done out of order, also pass `doneSteps`.

## Checklist

- `worker/.venv/bin/python -m pytest -q` passes. This includes
  `tests/test_tools_and_config.py`, which lists the shipped procedures; add
  yours there.
- `uvx ruff check --isolated --select E,F,W,I --line-length 120 --target-version py311 --exclude .venv apps/sop-guidance`
  is clean.
- Detector weights are committed through LFS (`git lfs status`).
- Rebuild the worker (Docker: `docker compose -f compose.yaml -f compose.cpu.yaml build worker`).
  The startup log shows `procedure <id> backend=<name> steps=<n>` and a
  `GUIDANCE_YOLO_PREHEAT ready` line per detector.
- `curl 127.0.0.1:8093/api/procedures` lists it with the capabilities you
  expect.
- Say "Hey Helix, guide me through <alias>" and hear "Step 1 of N: …".
