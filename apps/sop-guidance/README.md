<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# SOP guidance

Spoken, camera-checked guidance through standard operating procedures, for
XR glasses and the xr-ai-ui web client. The wearer says "Hey Helix, guide me
through changing the nose pads", says yes when asked to confirm guided mode,
and the worker announces each step, watches the
input camera, speaks a correction when a step is done wrong, and advances when
the step is visibly complete. Questions asked on the way are answered with a
short reminder of the current step. Two procedures ship:
`nosepad-replacement` (Luma Ultra magnetic nose pads), graded by a
vision-language model, and `rpi-hat-assembly` (Raspberry Pi M.2 HAT+), judged
by a detector and a per-hole state machine.

The orchestrator starts DeviceIOHub, an app-owned DashScope STT/TTS shim and
the guidance worker. Hosted DashScope models do the language and vision work
by default; `yaml/models.local.json` switches to the shared self-hosted stack.

> **Adding a new guidance task?** Start with
> [`docs/adding-a-procedure.md`](docs/adding-a-procedure.md). It covers a
> VLM-graded task that is data only, adding a detector or geometry plugin,
> writing a custom backend, running one as a sidecar, and adding a
> procedure-specific view to xr-ai-ui.

## How it works

```mermaid
flowchart TD
    C[xr-ai-ui or glasses] <-->|LiveKit| H[DeviceIOHub]
    subgraph W[sop_guidance_worker]
        V[VoiceAgent: STT, voice gate, TTS]
        I[Interaction: wake word, fast paths]
        F[Foreground: idle tool loop / guidance turn]
        O[Scene observer: VLM every 2 s]
        G[Guidance host: sessions, takeover, resume]
        B[Procedure backend: vlm, rpi_hat_judge, remote, ...]
        P[Preview: detector overlay, frame cache]
        S[Speech router + playback tracker]
        R[Session recorder]
        A[App HTTP API]
    end
    H <--> V
    V --> I --> F
    I -->|stop, next, resume| G
    F -->|guidance__* tools| G
    G <--> B
    H -->|camera| P --> B
    H -->|camera| O -->|scene memory| F
    P -->|xr-hub-overlay-pid| H
    B -->|step, cue, verdict| G --> S --> V
    G --> R
    A -->|procedures, sessions, frames| C
```

- **Tools outside guidance, one JSON turn inside it.** Outside guidance the
  foreground offers `current_view`, `guidance__list_procedures`,
  `guidance__start` and `guidance__status`; the `procedure_id` argument is an
  exact enum of the enabled procedure folders, so the model cannot invent one.
  During guidance each question is one model call that returns a spoken reply
  and an action (none, advance, restep, check, exit, switch, navigate), which
  code applies. Both prompts are the old fork's.
- **Scene memory.** Outside guidance a background observer asks the VLM what
  changed on each watched camera every 2 s and condenses that into a scene
  summary every minute, so the assistant can answer "what did I just do?".
  It pauses while that camera is guided or a turn is in flight (`observer:` in
  the worker yaml; `enabled: false` turns it off).
- **Backends decide completion.** A procedure names a backend. `vlm` grades a
  fresh frame against the step's teacher reference frames (at least two passes
  in a row plus a hold), with an optional detector overlay and a geometry
  plugin that may veto a pass but never grant one. `rpi_hat_judge` is a
  deterministic judge: its own detector, per-hole votes and a step counter
  decide, voice never advances it, and it draws its own boxes on the return
  video. It is a separate package under `backends/`, found through the
  `sop_guidance.backends` entry point, so the worker only lists it as a
  dependency. `remote` runs any backend in a sidecar process.
- **Guidance never talks over itself.** Steps are not graded while their own
  instruction is still playing, corrections back off per step, and late
  results for an older step are dropped.
- **Sessions survive.** Every session is checkpointed under `run/guidance/`.
  "Stop guidance" works without the wake word; "resume" picks up at the saved
  step, also after a worker restart. A second client asking to guide is asked
  to confirm a takeover, and keeps the camera the first session was using.

## Inside each backend

These diagrams zoom into the "Procedure backend" box above. Every backend
sits behind the same seam (`guidance/sop_guidance/backends/base.py`):

- **The host drives the run.** It calls `open_run`, `start`, `on_frame` and
  `command`.
- **The run reports back only through `emit`.** Its events are
  `StepChanged`, `Cue`, `Verdict`, `OverlayUpdate` and `RunFinished`.
- **The host owns everything the wearer sees and hears.** That covers speech,
  `guidance.state`, the return video and the recorder.

Colours:

| Colour | Part |
|---|---|
| Blue | Guidance core |
| Purple | A backend |
| Amber | Procedure data |
| Green | Model services |
| Grey | Outside the worker |

Dashed arrows happen once at startup; solid arrows happen during a run.

### `vlm`: graded by a vision-language model (nose pad)

```mermaid
flowchart TD
    subgraph DATA["procedures/nosepad-replacement/ and profiles/nosepad/"]
        direction LR
        YAML["procedure.yaml<br/>backend: vlm"]
        SOP["sop.json + frames/<br/>steps, key info,<br/>teacher reference frames"]
        GEOF["profiles/nosepad/geometry.py<br/>pad-in-hand, pad-on-glasses gates"]
    end
    DETY["yaml/detectors.yaml<br/>profile nosepad-v5 + hand_yolov8n"]

    subgraph W["sop_guidance_worker process"]
        REG["app._build_backend<br/>registry.resolve_backend('vlm')"]
        HOST["Guidance host<br/>host.py GuidanceHost"]
        PV["Preview loop<br/>preview.py PreviewManager<br/>annotates every frame, caches it"]

        subgraph VIS["vision/ (shared by any backend)"]
            ANN["FrameAnnotator<br/>OpenVINO YOLO, then hand detector"]
            GEO["Geometry plugin<br/>can veto a pass, never grant one"]
        end

        subgraph VLMB["backend: vlm (backends/vlm/)"]
            MON["VlmRun._monitor_loop<br/>tick 0.25 s, one check in flight,<br/>waits out the step's own speech"]
            IMG["VlmRun._student_image<br/>newest annotated preview frame"]
            CHK["grading.check_step<br/>1 compare with teacher frames<br/>2 live frame vs key info<br/>3 diagnose the mistake"]
            APPLY["VlmRun._apply_check<br/>2 passes in a row + hold_seconds"]
            COR["VlmRun._maybe_correct<br/>5 s first gap, back-off to 60 s"]
        end
    end
    VLMS["VLM service<br/>qwen3.6-35b-a3b"]

    YAML -.-> REG -.->|"create_backend()"| HOST
    DETY -.-> ANN
    GEOF -.-> GEO
    SOP -.-> CHK

    HOST -->|"open_run, start, command"| MON
    ANN --> PV
    PV -->|"frame cache"| IMG
    MON --> IMG --> CHK
    GEO -->|"veto"| CHK
    CHK <-->|"prompt + images"| VLMS
    CHK -->|"emit Verdict"| HOST
    CHK -->|"CheckResult"| APPLY
    APPLY -->|"passed: emit StepChanged<br/>or RunFinished"| HOST
    APPLY -->|"failed"| COR
    COR -->|"emit Cue correction"| HOST

    classDef core fill:#E3F2FD,stroke:#1565C0,color:#0D47A1
    classDef backend fill:#EDE7F6,stroke:#4527A0,color:#311B92
    classDef data fill:#FFF8E1,stroke:#F9A825,color:#5D4037
    classDef model fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20
    class REG,HOST,PV,ANN,GEO core
    class MON,IMG,CHK,APPLY,COR backend
    class YAML,SOP,GEOF,DETY data
    class VLMS model
```

### `rpi_hat_judge`: judged by a detector and a state machine (Raspberry Pi HAT)

This backend is its own package in `backends/rpi-hat-judge/`. It registers
under the `sop_guidance.backends` entry point, and `worker/pyproject.toml`
lists it as a dependency. The worker imports it and runs it in-process: no
container, no port. It never calls a language model.

```mermaid
flowchart TD
    subgraph DATA["procedures/rpi-hat-assembly/"]
        direction LR
        YAML["procedure.yaml<br/>backend: rpi_hat_judge"]
        SPEC["sop_rpi_hat.json<br/>5 steps, hole per step,<br/>screen and spoken text"]
    end
    DETY["yaml/detectors.yaml<br/>profile rpi-hat-v7 at 1280 + hand_yolov8n"]
    EP["worker/pyproject.toml dependency<br/>entry point sop_guidance.backends:<br/>rpi_hat_judge = rpi_hat_judge:create_backend"]

    subgraph W["sop_guidance_worker process"]
        REG["app._build_backend<br/>registry.resolve_backend('rpi_hat_judge')"]
        HOST["Guidance host<br/>host.py GuidanceHost"]
        PUMP["GuidanceHost._pump_frames<br/>session camera at frame_hz 3"]
        PV["Preview loop<br/>paints the judge's boxes,<br/>runs no detector itself"]

        subgraph JB["backend: rpi_hat_judge (backends/rpi-hat-judge/rpi_hat_judge/)"]
            DET["RpiHatJudgeRun.on_frame<br/>FrameAnnotator.detect_array"]
            SM["smoothing.TemporalSmoother<br/>one board track, hold 0.2 s, medians"]
            OCC["occlusion.OcclusionFilter<br/>hand over board, board confidence,<br/>visible holes; six seen = accept"]
            FPC["fpc.FPCSeatState<br/>cable inside board, 2.5 s majority;<br/>only after four screws"]
            HOLES["holes.PerHoleTracker<br/>anchor match, 5-of-7 vote per hole,<br/>sentinels, step counter"]
            PROG["progress.Progress<br/>step to show, steps done,<br/>operator alerts"]
            SPEAK["RpiHatJudgeRun._speak<br/>texts.py wording"]
        end
    end

    YAML -.-> REG
    EP -.->|"import"| REG
    REG -.->|"create_backend()"| HOST
    SPEC -.-> PROG
    DETY -.-> DET

    HOST --> PUMP -->|"on_frame(frame)"| DET
    DET --> SM --> OCC -->|"verdict"| FPC --> HOLES --> PROG --> SPEAK
    SPEAK -->|"emit StepChanged with lead 'Step 1, …, done.'<br/>emit Cue correction: out of order, removed, skipped<br/>emit RunFinished after the cable"| HOST
    PROG -->|"emit OverlayUpdate: boxes + next hole"| PV
    PROG -->|"emit Verdict: hole map"| HOST
    HOST -->|"guidance.state.extra"| UI["xr-ai-ui HoleMap<br/>in the NowCard"]

    classDef core fill:#E3F2FD,stroke:#1565C0,color:#0D47A1
    classDef backend fill:#EDE7F6,stroke:#4527A0,color:#311B92
    classDef data fill:#FFF8E1,stroke:#F9A825,color:#5D4037
    classDef client fill:#ECEFF1,stroke:#546E7A,color:#263238,stroke-dasharray: 4 3
    class REG,HOST,PUMP,PV core
    class DET,SM,OCC,FPC,HOLES,PROG,SPEAK backend
    class YAML,SPEC,DETY,EP data
    class UI client
```

### `remote`: any backend in its own process

For a backend whose dependencies cannot share the worker's environment. The
backend itself is unchanged; the host cannot tell the two placements apart.

```mermaid
flowchart LR
    subgraph W["sop_guidance_worker process"]
        HOST["Guidance host"]
        RB["backends/remote.py<br/>RemoteBackend, RemoteRun"]
    end
    subgraph SC["sidecar process, its own uv project"]
        SRV["remote.serve(backend, endpoint)"]
        ANY["Any ProcedureBackend<br/>and its runs"]
    end
    YAML["procedure.yaml<br/>backend: remote<br/>endpoint: ipc://…"]

    YAML -.-> RB
    HOST <-->|"same calls and events<br/>as in-process"| RB
    RB -->|"frames: shared-memory ring"| SRV
    RB <-->|"commands, events, snapshots:<br/>ZMQ + msgpack"| SRV
    SRV <--> ANY

    classDef core fill:#E3F2FD,stroke:#1565C0,color:#0D47A1
    classDef backend fill:#EDE7F6,stroke:#4527A0,color:#311B92
    classDef data fill:#FFF8E1,stroke:#F9A825,color:#5D4037
    class HOST,RB core
    class SRV,ANY backend
    class YAML data
```

## One guidance task, end to end

What happens from the first spoken sentence to the last step, with the code
that does each part.

### Nose pad replacement (`vlm`)

```mermaid
flowchart TD
    Q1(["1 · 'Hey Helix, what am I looking at?'"])
    subgraph S1["Normal interaction"]
        direction TB
        A1["agent.py GuidanceAgent<br/>STT final transcript"]
        A2["interaction.py Interaction.on_speech<br/>shape gate, wake word, intent classifier"]
        A3["Interaction._handle<br/>fast paths first"]
        A4["foreground.py Foreground._idle_turn<br/>quick ack, then tool loop with context"]
        A5["Tools: current_view, guidance__list_procedures,<br/>guidance__start, guidance__status"]
        A6["observer.py SceneObserver<br/>VLM look every 2 s, scene summary"]
        A1 --> A2 --> A3 -->|"anything else"| A4
        A4 <-->|"tool loop"| A5
        A6 -.->|"scene memory"| A4
    end

    Q2(["2 · 'Hey Helix, guide me through the nose pad replacement'"])
    subgraph S2["Start guidance"]
        direction TB
        B0["GuidanceHost.begin → offer<br/>'I'll walk you through … Ready to start?'<br/>'yes' → confirm_start"]
        B1["GuidanceHost._enter<br/>recorder session, preview with nosepad-v5,<br/>backend.open_run"]
        B2["VlmRun.start → emit StepChanged(0)"]
        B3["GuidanceHost._announce<br/>'Step 1 of 4: …' → SpeechRouter.say → TTS"]
        B0 --> B1 --> B2 --> B3
    end

    subgraph S3["The step loop · backends/vlm/run.py"]
        direction TB
        C1["_monitor_loop<br/>waits for the announcement to end"]
        C2["_grade: annotated frame<br/>+ grading.check_step, 3 tiers"]
        C3{"2 passes<br/>+ hold?"}
        C4["_advance → StepChanged<br/>host adds a short acknowledgement"]
        C5["_maybe_correct → Cue<br/>spoken correction, NowCard turns red"]
        C6["Last step → RunFinished<br/>'You've completed all steps…'"]
        C1 --> C2 --> C3
        C3 -->|"yes"| C4
        C3 -->|"no"| C5
        C4 -->|"next step"| C1
        C5 --> C1
        C4 -->|"after the last step"| C6
    end

    Q3(["3 · While guided: 'Hey Helix, …' (wake word required)"])
    subgraph S4["Talking during guidance"]
        direction LR
        D1["'next'<br/>host.command(next) → VlmRun._advance"]
        D2["'stop guidance'<br/>host.stop, checkpoint kept for resume"]
        D3["Questions<br/>Foreground._guidance_turn: one JSON call,<br/>reply + action (advance, check, restep)"]
    end

    Q1 --> A1
    Q2 -->|"same gates"| A3
    A3 -->|"text.guidance_request<br/>+ match_procedure"| B1
    A5 -->|"guidance__start(procedure_id)"| B1
    B3 --> C1
    C6 ~~~ Q3
    Q3 --> S4

    classDef say fill:#FFF8E1,stroke:#F9A825,color:#5D4037
    classDef core fill:#E3F2FD,stroke:#1565C0,color:#0D47A1
    classDef backend fill:#EDE7F6,stroke:#4527A0,color:#311B92
    class Q1,Q2,Q3 say
    class A1,A2,A3,A4,A5,A6,B1,B3,D1,D2,D3 core
    class B2,C1,C2,C3,C4,C5,C6 backend
```

### Raspberry Pi HAT assembly (`rpi_hat_judge`)

Normal interaction and the start are the same code as for the nose pad; only
the backend differs. Voice never moves the judge: its capabilities turn off
voice advance, step jumps, on-demand checks and resume.

```mermaid
flowchart TD
    Q1(["1 · 'Hey Helix, what's on my desk?'"])
    subgraph S1["Normal interaction (same as nose pad)"]
        direction TB
        A1["GuidanceAgent → Interaction.on_speech<br/>gates, then Interaction._handle"]
        A4["Foreground._idle_turn<br/>tools + scene memory"]
        A1 -->|"anything else"| A4
    end

    Q2(["2 · 'Hey Helix, guide me through the raspberry pi assembly'"])
    subgraph S2["Start guidance"]
        direction TB
        B1["GuidanceHost.begin → _enter<br/>preview paints backend boxes,<br/>frame pump at 3 Hz"]
        B2["RpiHatJudgeRun.start → emit StepChanged(0)"]
        B3["'Step 1 of 5: Lay the board flat on the desk and<br/>drive screw 1 into the top-left hole.'"]
        B1 --> B2 --> B3
    end

    subgraph S3["Every frame · rpi_hat_judge/backend.py RpiHatJudgeRun._judge"]
        direction TB
        C1["Detect: v7 boxes + hand boxes"]
        C2["TemporalSmoother → OcclusionFilter"]
        C3{"frame<br/>trusted?"}
        C4["PerHoleTracker: vote per hole, step counter<br/>FPCSeatState once four screws are in"]
        C5["Progress: step to show = next empty hole"]
        C6["emit OverlayUpdate + Verdict<br/>boxes on the return video, hole map to the UI"]
        C7{"what<br/>changed?"}
        C8["StepChanged with lead<br/>'Step 1, install the top-left screw, done.<br/>Step 2 of 5: …'"]
        C9["Cue correction<br/>'Out of order. screw 3 went in while<br/>hole 2 … is still empty. Next, …'"]
        C10["Cable seated → RunFinished<br/>'Step 5, seat the ribbon cable, done.'"]
        C1 --> C2 --> C3
        C3 -->|"yes"| C4 --> C5 --> C6
        C3 -->|"no: hand, low confidence, no board"| C6
        C6 --> C7
        C7 -->|"nothing"| C1
        C7 -->|"next hole"| C8 --> C1
        C7 -->|"operator alert"| C9 --> C1
        C7 -->|"all done"| C10
    end

    Q3(["3 · While guided: 'Hey Helix, …' (wake word required)"])
    subgraph S4["Talking during guidance"]
        direction LR
        D1["'next'<br/>refused by capabilities:<br/>'The camera confirms each step…'"]
        D2["'repeat', 'which step'<br/>RunCommand(repeat)"]
        D3["'start over'<br/>GuidanceHost.begin again: a fresh run,<br/>fresh votes, model stays loaded"]
        D4["Questions<br/>Foreground._guidance_turn, prompt carries<br/>the judge's state and 'never say a step is done'"]
    end

    Q1 --> A1
    Q2 -->|"same gates"| A1
    A1 -->|"guidance_request matches an alias"| B1
    A4 -->|"or the LLM calls guidance__start"| B1
    B3 --> C1
    C10 ~~~ Q3
    Q3 --> S4

    classDef say fill:#FFF8E1,stroke:#F9A825,color:#5D4037
    classDef core fill:#E3F2FD,stroke:#1565C0,color:#0D47A1
    classDef backend fill:#EDE7F6,stroke:#4527A0,color:#311B92
    class Q1,Q2,Q3 say
    class A1,A4,B1,B3,D1,D2,D3,D4 core
    class B2,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10 backend
```

## Configure

| File | What it holds |
|---|---|
| `yaml/sop_guidance_worker.yaml` | Models file, wake word, foreground, preview, scene observer, voice, app API, recording, and the `guidance_defaults` every procedure inherits |
| `procedures/<id>/procedure.yaml` | One procedure: title, aliases, backend, and overrides of `guidance_defaults` |
| `procedures/<id>/sop.json`, `frames/` | Its steps and teacher reference frames (`vlm` backend) |
| `backends/<name>/` | A procedure backend installed as its own package, such as `rpi-hat-judge` |
| `yaml/detectors.yaml` | Named detector profiles (weights under `detectors/`, in Git LFS) |
| `yaml/models.dashscope.json` | Hosted LLM/VLM and the speech shim on port 8106 (default) |
| `yaml/models.local.json` | The shared self-hosted model servers |
| `yaml/voice_gate.yaml` | Voice gate; wake matching is done by the worker itself |
| `yaml/device_io_hub.yaml` | Room, ports, web client, return video |
| `services/dashscope-speech/dashscope_speech.yaml` | STT/TTS models, voice and context |

Secrets come from the environment only: `DASHSCOPE_API_KEY` for the hosted
models and speech, and optionally `SOP_GUIDANCE_API_TOKEN` to require a bearer
token on the app API.

## Run

From `apps/sop-guidance/`, with hosted DashScope models:

```bash
export DASHSCOPE_API_KEY=...
uv sync
uv run sop_guidance
```

With the self-hosted stack instead, set `models_config: models.local.json` in
the worker yaml, start the shared models and skip the speech shim:

```bash
uv run --project ../../model-server-samples/model-servers model_servers
uv run sop_guidance --no-speech
```

`--capture` also records participant video, audio and data traffic. Open the
web-client URL DeviceIOHub prints, or point xr-ai-ui at the hub, and say
"Hey Helix".

To run the same stack in Docker, see `docker/compose.yaml`:

```bash
cd docker
cp .env.example .env   # set DASHSCOPE_API_KEY
docker compose -f compose.yaml -f compose.cpu.yaml up -d --build
```

Builds run on the host network and use the mirrors set in `docker/.env`
(`APT_MIRROR_HOST`, `PIP_INDEX_URL`, `TORCH_INDEX_URL`; see `.env.example`),
which matter where PyPI and download.pytorch.org are slow. The worker image
bakes in NLTK's `punkt_tab`, which the voice pipeline would otherwise fetch
from GitHub at startup.

## Clients

Clients steer the worker with JSON on named data topics, and the worker
answers on others:

| Topic | Direction | Payload |
|---|---|---|
| `guidance.control` | client → worker | `{"action": "start" \| "stop" \| "resume" \| "confirm" \| "cancel" \| "ready", "procedure_id"?, "session_id"?, "token"?, "request_id"}` |
| `guidance.result` | worker → client | The reply to one control, with its `request_id` |
| `guidance.state` | worker → all | Session id, status, owner, input participant, step, instruction, capabilities |
| `guidance.overlay` | worker → owner | Boxes from a backend that draws its own overlay |
| `xr.main_input` | client → worker | Which participant's camera this client drives |
| `xr.voice_output` | client → worker | `default`, `selected_input` or `web_client` |
| `xr.yolo_mode`, `xr.wake_mode` | client → worker | Live overlay and wake word outside guidance |
| `agent.response`, `chat.user` | worker → all | The transcript |

Untopiced data is typed text and is handled like speech. The app API on
`127.0.0.1:8093` serves `/api/health`, `/api/procedures`, `/api/procedures/{id}`
and its files (reference frames), `/api/sessions` and `/api/live`.

## Test

```bash
uv sync --project worker
worker/.venv/bin/python -m pytest -q
```

## Add a procedure

The step-by-step guide is [`docs/adding-a-procedure.md`](docs/adding-a-procedure.md). It covers a VLM-graded task that is data only, adding a detector or a geometry plugin, writing a custom backend package, running a backend as a sidecar, and adding a procedure-specific view to xr-ai-ui. In short:

A procedure is a folder under `procedures/` with a `procedure.yaml` whose
`id` matches the folder name. For the built-in `vlm` backend, add a schema v1
`sop.json` and the reference frames it names; a detector profile and a
geometry module are optional. No Python is needed:
`tests/fixtures/procedures/filter-swap/` is a complete example. The worker
validates every folder at startup and offers each enabled id to the model as
an exact `procedure_id`.

A procedure that needs its own grader gets a backend package. Implement
`ProcedureBackend` and `ProcedureRun` from `sop_guidance.backends.base`,
register the factory under the `sop_guidance.backends` entry-point group, and
add the package to `worker/pyproject.toml`; the host, tools, speech and UI do
not change. [`backends/rpi-hat-judge`](backends/rpi-hat-judge/README.md) is
the worked example.

A backend that cannot share the worker's environment runs as a sidecar
process instead. Wrap an ordinary backend with
`sop_guidance.backends.remote.serve(backend, endpoint)` in that process, and
point the procedure at it:

```yaml
backend: remote
backend_config:
  endpoint: ipc:///tmp/my-backend.sock
```

Frames reach the sidecar over a shared-memory ring and events come back over
ZMQ. `tests/fixtures/sidecar_stub.py` is a minimal sidecar.

## Evals

`eval/` holds two live-model checks. `eval/eval.py` is a foreground routing
eval driven by `eval/cases.yaml`, in the same style as the tea sample.
`eval/replay.py` re-grades checks recorded by the old glasses worker through
the `vlm` backend and reports agreement and latency against the recorded
verdicts. Pass the recorded session folders on the command line; they are
never read from or written to the repo. Refer to [`eval/README.md`](eval/README.md).
