# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-model routing eval for idle and active SOP guidance turns.

Mirrors ``agent-samples/tea-making-sample/eval/eval.py``: each case in
``cases.yaml`` builds the exact system prompt and tool set the worker's
foreground would use for that turn, asks the configured LLM once, and checks
the tool it chose, the tool arguments, and the reply text. Tools are only
validated, never invoked, so no session starts and nothing is spoken.

Case fields:

- ``query``: the wearer's words, wake word removed.
- ``route``: ``idle`` (default) or ``active``; ``step`` is the 1-based step an
  active case is on, ``procedure`` the procedure id (default: the first).
- ``verdict``: optional last grounded check for an active case:
  ``{completed, observation, age_s, checks: [{requirement, visible, evidence}]}``.
- ``wearer_requests`` / ``history`` (``[[wearer, assistant], ...]``): active context.
- ``expected_tool``: a tool name, ``null`` for a plain answer, or a list of
  acceptable names. ``expected_args``: a subset the first call's arguments
  must match.
- ``expected_response_pattern`` / ``forbidden_response_pattern`` (regex, case
  insensitive) and ``max_response_chars`` check the reply text.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from loguru import logger
from sop_guidance.backends.base import BackendServices, ModelHandles
from sop_guidance.backends.registry import resolve_backend
from sop_guidance.backends.vlm.grading import CheckResult
from sop_guidance.backends.vlm.run import VlmRun
from sop_guidance.host import GuidanceHost, HostSettings, LoadedProcedure
from sop_guidance.procedures import discover_procedures
from sop_guidance.recorder import SessionStore
from sop_guidance.tools import build_guidance_tools
from sop_guidance_worker.config import load_config
from sop_guidance_worker.foreground import Foreground, _merge
from xr_ai_models import ChatMessage, load_models_config, make_llm
from xr_ai_tools import ToolSet
from xr_ai_tools.tool_calling import tool_definitions

_APP = Path(__file__).resolve().parents[1]
_MIN_PASS_RATE = 0.80


class _Recorder:
    recording = False

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


class _Context:
    """Just enough RunContext for a run that is described, never monitored."""

    owner = "eval"

    def __init__(self, session_id: str, requests: tuple[str, ...]) -> None:
        self.session_id = session_id
        self._requests = requests

    @property
    def input_participant(self) -> str:
        return self.owner

    @property
    def models(self) -> ModelHandles:
        return ModelHandles()

    @property
    def recorder(self) -> _Recorder:
        return _Recorder()

    async def emit(self, event: Any) -> None:
        return None

    def speech_remaining_s(self) -> float:
        return 0.0

    def last_heard_us(self) -> int:
        return 0

    def wearer_requests(self) -> tuple[str, ...]:
        return self._requests

    def latest_frame(self) -> None:
        return None

    async def fetch_frame(self) -> None:
        return None


def _load_procedures(config: Any, artifacts: Path) -> list[LoadedProcedure]:
    loaded = []
    for entry in discover_procedures(config.procedures_dir, config.guidance_defaults):
        # No detector: routing never grades a frame, and the prompt text the
        # foreground sees does not depend on one.
        backend = resolve_backend(entry.spec.backend)(BackendServices(
            entry=entry, config=entry.spec.backend_config, artifacts_dir=artifacts,
        ))
        loaded.append(LoadedProcedure(entry=entry, backend=backend, models=ModelHandles()))
    if not loaded:
        raise SystemExit(f"no enabled procedures under {config.procedures_dir}")
    return loaded


def _verdict(spec: dict[str, Any] | None) -> CheckResult | None:
    if not spec:
        return None
    age_us = int(float(spec.get("age_s", 1.0)) * 1_000_000)
    return CheckResult(
        completed=bool(spec.get("completed", False)),
        current_observation=str(spec.get("observation", "")),
        checks=[dict(c) for c in spec.get("checks", [])],
        issue=str(spec.get("issue", "")),
        timestamp_us=time.time_ns() // 1_000 - age_us,
    )


def _active_turn(
    foreground: Foreground, procedure: LoadedProcedure, case: dict[str, Any], pid: str,
) -> tuple[str, ToolSet]:
    step = int(case.get("step", 1))
    requests = tuple(str(r) for r in case.get("wearer_requests", []))
    run = VlmRun(procedure.backend, _Context(pid, requests), start_step=step - 1)
    run._last_result = _verdict(case.get("verdict"))  # the monitor's cached check
    session = SimpleNamespace(
        procedure=procedure,
        run=run,
        wearer_requests=list(requests),
        turn_history=[tuple(pair) for pair in case.get("history", [])],
    )
    system = foreground._active_system(session, run.turn_context().prompt_block, False)
    return system, build_guidance_tools(foreground._host, active=True)


def _check(case: dict[str, Any], tools: ToolSet, calls: list[Any], content: str) -> list[str]:
    errors: list[str] = []
    for call in calls:
        tool = tools.get(call.name)
        if tool is None:
            errors.append(f"unknown tool {call.name!r}")
            continue
        try:
            tool.request_model.model_validate_json(call.arguments or "{}")
        except ValueError as exc:
            errors.append(f"invalid {call.name!r} arguments: {exc}")
    expected = case.get("expected_tool")
    options = expected if isinstance(expected, list) else [expected]
    actual = [call.name for call in calls]
    if not any((actual == [] if o is None else actual == [o]) for o in options):
        errors.append(f"tools {actual!r}, expected {expected!r}")
    expected_args = case.get("expected_args") or {}
    if expected_args and calls:
        try:
            arguments = json.loads(calls[0].arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
        for key, value in expected_args.items():
            if arguments.get(key) != value:
                errors.append(f"argument {key}={arguments.get(key)!r}, expected {value!r}")
    text = " ".join(content.split())
    pattern = case.get("expected_response_pattern")
    if pattern and not calls and re.search(str(pattern), text, re.I) is None:
        errors.append(f"response did not match {pattern!r}")
    forbidden = case.get("forbidden_response_pattern")
    if forbidden and re.search(str(forbidden), text, re.I) is not None:
        errors.append(f"response matched forbidden {forbidden!r}")
    limit = case.get("max_response_chars")
    if limit is not None and len(text) > int(limit):
        errors.append(f"response is {len(text)} characters, over {limit}")
    return errors


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--cases", type=Path, default=_APP / "eval" / "cases.yaml")
    parser.add_argument("--worker-config", type=Path,
                        default=_APP / "yaml" / "sop_guidance_worker.yaml")
    parser.add_argument("--models", type=Path, default=None,
                        help="models JSON; default: the worker config's models_config")
    parser.add_argument("--only", default="", help="regex over case names")
    args = parser.parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    cases = yaml.safe_load(args.cases.read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if re.search(args.only, c["name"])]
    config = load_config(args.worker_config)
    llm = make_llm(load_models_config(args.models or config.models_config),
                   config.foreground.llm_role)
    passed = 0
    with tempfile.TemporaryDirectory(prefix="sop-eval-") as scratch:
        procedures = _load_procedures(config, Path(scratch) / "artifacts")
        by_id = {p.id: p for p in procedures}
        host = GuidanceHost(
            procedures=procedures,
            store=SessionStore(Path(scratch) / "sessions", level="off"),
            ports=SimpleNamespace(),
            settings=HostSettings(),
        )
        foreground = Foreground(host=host, llm=llm, vlm=None, config=config, speech=None,
                                frames=None, input_of=lambda pid: pid)
        try:
            for index, case in enumerate(cases):
                pid = f"sop-eval-{index}"
                if case.get("route", "idle") == "active":
                    procedure = by_id[case.get("procedure", procedures[0].id)]
                    system, tools = _active_turn(foreground, procedure, case, pid)
                else:
                    system = config.prompt("idle")
                    tools = _merge(
                        build_guidance_tools(host, active=False),
                        ToolSet({"current_view": foreground._current_view_tool(pid)}),
                    )
                response = await llm.chat(
                    (ChatMessage(role="system", content=system),
                     ChatMessage(role="user", content=str(case["query"]))),
                    tools=tool_definitions(tools),
                    max_tokens=config.foreground.max_tokens,
                    temperature=0.0,
                )
                calls = list(response.tool_calls or [])
                content = response.content or ""
                errors = _check(case, tools, calls, content)
                label = "PASS" if not errors else "MISS"
                shown = [f"{c.name}({c.arguments})" for c in calls]
                print(f"{label} {case['name']}: tools={shown} content={content!r}")
                for error in errors:
                    print(f"     - {error}")
                passed += not errors
        finally:
            await llm.close()
    print(f"RESULT {passed}/{len(cases)} cases passed")
    if cases and passed / len(cases) < _MIN_PASS_RATE:
        raise SystemExit(f"overall pass rate {passed / len(cases):.1%} is below {_MIN_PASS_RATE:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
