<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# SOP guidance evals

Two live-model evals. Both read the checked-in worker and procedure
configuration, so they test what the worker ships.

| File | What it checks |
|---|---|
| `eval.py` + `cases.yaml` | Foreground routing: for each case the idle or active system prompt and tool set the worker would build, one LLM call, then the chosen tool, its arguments and the reply text. Mirrors `agent-samples/tea-making-sample/eval/`. |
| `replay.py` | Grading parity: recorded grounded checks from old glasses-worker sessions are re-graded by the new `vlm` backend and compared with the recorded verdicts. |
| `recordings.py`, `metrics.py` | Recording parser and the agreement/latency summaries; standard library only. |
| `test_replay_helpers.py` | Unit tests for those two helpers. They need no models or recordings. |

Run everything from `apps/sop-guidance/` with the worker environment. Both
evals default to `yaml/models.dashscope.json`, which reads `DASHSCOPE_API_KEY`
from the environment. Pass `--models` to use another models file.

```bash
export DASHSCOPE_API_KEY=...            # never commit it
worker/.venv/bin/python eval/eval.py                  # fails below 80% pass rate
worker/.venv/bin/python -m pytest eval -q             # helper tests, offline
```

## Replay

The old worker, at debug level `full`, wrote one folder per session:
`<YYYYmmdd_HHMMSS>_<procedure>/` holding `events.jsonl`, `calls.jsonl` and
`step_NN/call_NNNN/in_NN.png`. Each `in_NN.png` is an image the model saw.
Each `CHECK` event is paired with the grading calls that share its student
frame. A check whose calls were not recorded is skipped and counted.

Each selected check runs through the shipping path. The backend is built from
`procedure.yaml` the same way the worker builds it. Then a `VlmRun` is opened on
the recorded step and `command("check")` runs `run.py`, then
`grading.check_step`, with the backend's overlay prompting, teacher annotation
and geometry veto. The recorded student frame goes in as the preview's
annotated frame, so the new VLM call sees the same pixels as the old one.

Recordings are never read from the repo, and no recording is ever written to
it. Pass the folders on the command line. A folder of session folders also
works:

```bash
# Copy some sessions out of the old stack's volume (read-only mount).
docker run --rm -v xr-ai-mac_xr-glasses-run:/r:ro -v "$PWD/run/replay-data:/out" alpine \
  sh -c 'cp -r /r/debug/20260918_014945_nosepad_replacement /out/ && chown -R '"$(id -u):$(id -g)"' /out'

# No API calls: recorded responses answer each tier (parser, tier and veto parity, prompt drift).
worker/.venv/bin/python eval/replay.py run/replay-data --mode offline --limit 0

# The new evaluator on the recorded inputs, plus the recorded prompts re-asked verbatim.
worker/.venv/bin/python eval/replay.py run/replay-data --baseline \
  --geometry recorded --teacher recorded --limit 48 --concurrency 4

# The whole new pipeline: v5 detector and geometry on the frame, SOP reference frames.
worker/.venv/bin/python eval/replay.py run/replay-data --limit 48 --concurrency 4
```

| Flag | Meaning |
|---|---|
| `--mode live\|offline` | `live` asks the VLM. `offline` answers each tier with its recorded response, so it makes no API calls. |
| `--baseline` | Also re-asks the recorded prompts with the recorded images. This measures how often the old evaluator agrees with its own recording on a second draw. The target is new agreement at or above that. |
| `--geometry redetect\|recorded\|off` | `redetect` (the default) inpaints the old overlay, then runs the new detector and geometry profile. The results supply the prompt's geometry prose and the veto. `recorded` reuses the recorded prose and the recorded veto. |
| `--teacher sop\|recorded` | `sop` annotates the procedure's reference frames with the new detector. `recorded` reuses the recorded annotated teacher frames. |
| `--limit`, `--per-session`, `--steps` | Bound the run. Sampling is deterministic and round-robins over (session, step). |
| `--concurrency` | 1 to 4 checks in flight. |
| `--rescore OUT` | Rebuilds `summary.json` and `report.md` from `OUT/results.jsonl`. |

Results go to `--out`, by default `run/replay/<timestamp>/`, which is
gitignored:

- `results.jsonl` has one row per check.
- `summary.json` holds agreement, kappa, confusion by step, issue-text
  similarity, spoken-rule violations and latency p50/p95.
- `report.md` renders the summary and lists every disagreement with both
  issue texts.
- `prompt_diffs/step_NN.diff` compares the recorded comparison prompt with the
  new one.

Caveats:

- The recorded frames already carry the old overlay. In `redetect` mode it is
  inpainted before detection, but a pad hidden under an old label tab can
  still be missed.
- Replay grades single frames. The live preview's 0.5 s voting window, which
  lets geometry coast across a missed box, is absent here.
- The recorded latency was measured on the old stack at recording time.
  `--baseline` gives a same-time, same-network comparison.

### Last run (2026-09-25, 5 sessions, 121 replayable checks, qwen3.6-35b-a3b)

| run | checks | agreement with recorded | notes |
|---|---|---|---|
| offline, recorded geometry | 121 | 100% | parser, tiers and veto at parity; prompts identical except the hand legend wording |
| live, recorded inputs, draw 1 / draw 2 | 48 / 48 | 100% / 93.8% | baseline (old prompts re-asked) 97.7% on draw 2 |
| live, full new pipeline | 48 | 95.8% | both misses are new `pad_in_hand` vetoes on a pad held over the glasses |

Check latency p50/p95, same time and network: new 2.36/3.49 s, baseline 2.79/3.94 s.
