# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guidance tools, procedure discovery and configuration layering."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import HostHarness, write_procedure
from pydantic import ValidationError
from sop_guidance.procedures import (
    GuidanceDefaults,
    ProcedureConfigError,
    discover_procedures,
    load_procedure,
)
from sop_guidance.tools import TurnScope, build_guidance_tools, current_turn
from sop_guidance_worker.config import load_config
from xr_ai_tools.tool_calling import tool_definitions

APP = Path(__file__).resolve().parents[1]


def _schema(tools, name: str) -> dict:
    for definition in tool_definitions(tools):
        if definition.name == name:
            return definition.parameters
    raise KeyError(name)


def test_procedure_id_is_an_exact_enum(harness: HostHarness) -> None:
    tools = build_guidance_tools(harness.host, active=False)
    schema = _schema(tools, "guidance__start")

    assert schema["properties"]["procedure_id"]["enum"] == ["lid-demo"]
    assert "the lid" in schema["properties"]["procedure_id"]["description"]
    with pytest.raises(ValidationError):
        tools.get("guidance__start").request_model.model_validate({"procedure_id": "lid"})


def test_idle_and_active_tool_sets(harness: HostHarness) -> None:
    idle = {name for name, _ in build_guidance_tools(harness.host, active=False).items()}
    active = {name for name, _ in build_guidance_tools(harness.host, active=True).items()}

    assert idle == {"guidance__list_procedures", "guidance__start", "guidance__status"}
    assert active - idle == {
        "guidance__stop", "guidance__advance", "guidance__repeat_step",
        "guidance__check_now", "guidance__switch_procedure",
    }


async def test_start_tool_acts_for_the_turn_participant(harness: HostHarness) -> None:
    tools = build_guidance_tools(harness.host, active=False)
    tool = tools.get("guidance__start")
    token = current_turn.set(TurnScope("alice", "guide me through the lid"))
    try:
        result = await tool.handler(tool.request_model(procedure_id="lid-demo"))
    finally:
        current_turn.reset(token)

    assert result.say == ""  # the host already announced step 1
    assert harness.host.session_of("alice") is not None


def test_defaults_merge_under_the_procedure(tmp_path: Path) -> None:
    folder = write_procedure(tmp_path, backend_config="{monitor: {tick_s: 0.5}}")
    defaults = GuidanceDefaults(
        foreground={"reminder_interval_s": 30},
        backend_config={"scripted": {"monitor": {"tick_s": 0.25, "check_interval_s": 2.0}}},
    )

    spec = load_procedure(folder, defaults).spec

    assert spec.foreground.reminder_interval_s == 30
    assert spec.backend_config == {"monitor": {"tick_s": 0.5, "check_interval_s": 2.0}}


def test_folder_name_must_match_id(tmp_path: Path) -> None:
    folder = write_procedure(tmp_path, "lid-demo")
    folder.rename(tmp_path / "other")

    with pytest.raises(ProcedureConfigError, match="must match its folder name"):
        load_procedure(tmp_path / "other", GuidanceDefaults())


def test_unknown_procedure_field_names_the_file(tmp_path: Path) -> None:
    write_procedure(tmp_path, colour="blue")

    with pytest.raises(ProcedureConfigError, match=r"procedure\.yaml: colour"):
        discover_procedures(tmp_path, GuidanceDefaults())


def test_disabled_procedures_are_validated_but_hidden(tmp_path: Path) -> None:
    write_procedure(tmp_path, "shown")
    write_procedure(tmp_path, "hidden", enabled="false")

    assert [e.id for e in discover_procedures(tmp_path, GuidanceDefaults())] == ["shown"]


def test_shipped_configuration_loads() -> None:
    config = load_config(APP / "yaml" / "sop_guidance_worker.yaml")
    entries = discover_procedures(config.procedures_dir, config.guidance_defaults)

    assert [e.id for e in entries] == ["nosepad-replacement"]
    spec = entries[0].spec
    assert spec.backend == "vlm"
    assert spec.backend_config["detector"]["profile"] == "nosepad-v5"
    assert spec.backend_config["monitor"]["check_interval_s"] == 1.0
    assert entries[0].active_prompt()
    assert config.prompt("active") and config.prompt("idle") and config.prompt("current_view")


def test_shipped_procedure_builds_its_backend() -> None:
    from sop_guidance.backends.registry import resolve_backend
    from sop_guidance.backends.vlm import VlmBackendConfig

    config = load_config(APP / "yaml" / "sop_guidance_worker.yaml")
    entry = discover_procedures(config.procedures_dir, config.guidance_defaults)[0]
    VlmBackendConfig.model_validate(entry.spec.backend_config)
    assert callable(resolve_backend("vlm"))


def test_spoken_step_jumps_are_entry_requests() -> None:
    from sop_guidance.text import guidance_request

    assert guidance_request("Skip ahead to step four.") == ("step", 4, "")
    assert guidance_request("take me to step 2") == ("step", 2, "")
    assert guidance_request("move on to step three") == ("step", 3, "")
    # Only with a step number: these are questions or reports, not jumps.
    assert guidance_request("move on to the next part") is None
    assert guidance_request("go to the sink") is None
    assert guidance_request("resume guidance") == ("resume", 0, "")
