# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The app HTTP API: procedures, sessions and the recorded session tree.

The UI's server reads it; browsers never call it directly. When the token
environment variable is set every request needs ``Authorization: Bearer
<token>``. Session control stays on the room's data channel, so this API is
read-only.
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sop_guidance.host import GuidanceHost, LoadedProcedure
from sop_guidance.recorder import SessionStore

from .config import ApiConfig


def _frame_url(procedure: LoadedProcedure, path: str) -> str:
    if not path:
        return ""
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(procedure.entry.directory)
        # A replaced frame keeps its name; the version gets it past the cache.
        version = resolved.stat().st_mtime_ns
    except (ValueError, OSError):
        return ""
    return f"/api/procedures/{procedure.id}/files/{relative.as_posix()}?v={version}"


def _summary(procedure: LoadedProcedure) -> dict[str, Any]:
    spec = procedure.entry.spec
    return {
        "id": procedure.id,
        "title": spec.title,
        "aliases": list(spec.aliases),
        "description": spec.description,
        "backend": procedure.backend.name,
        "steps": procedure.total_steps,
        "capabilities": procedure.backend.capabilities.model_dump(mode="json"),
        "ui": _ui(procedure),
    }


def _ui(procedure: LoadedProcedure) -> dict[str, Any]:
    ui = dict(procedure.entry.spec.ui)
    # Written relative to the procedure folder; sent as a versioned file URL
    # like the step images, so a replaced thumbnail is not served from cache.
    thumbnail = str(ui.get("thumbnail") or "")
    if thumbnail:
        ui["thumbnail"] = _frame_url(procedure, str(procedure.entry.directory / thumbnail))
    return ui


def create_api(*, host: GuidanceHost, store: SessionStore, config: ApiConfig) -> FastAPI:
    token = os.environ.get(config.token_env, "").strip()
    token_bytes = token.encode("utf-8")

    app = FastAPI(title="SOP guidance")

    # Middleware rather than a dependency: mounted apps (the /debug tree)
    # bypass router dependencies.
    @app.middleware("http")
    async def authorize(request: Request, call_next: Any) -> Any:
        if token:
            scheme, _, value = request.headers.get("authorization", "").partition(" ")
            # Bytes, not str: compare_digest raises on non-ASCII text, and a
            # header can carry any latin-1 byte.
            if scheme.lower() != "bearer" or not hmac.compare_digest(
                    value.strip().encode("latin-1", "replace"), token_bytes):
                return JSONResponse({"detail": "missing or invalid bearer token"},
                                    status_code=401)
        return await call_next(request)

    def procedure_or_404(procedure_id: str) -> LoadedProcedure:
        procedure = host.procedure(procedure_id)
        if procedure is None:
            raise HTTPException(status_code=404, detail="unknown procedure")
        return procedure

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "procedures": len(host.procedures()),
                "sessions": len(host.sessions())}

    @app.get("/api/procedures")
    async def procedures() -> dict[str, Any]:
        return {"procedures": [_summary(p) for p in host.procedures()]}

    @app.get("/api/procedures/{procedure_id}")
    async def procedure(procedure_id: str) -> dict[str, Any]:
        found = procedure_or_404(procedure_id)
        detail = _summary(found)
        detail["parts"] = list(found.backend.parts())
        detail["step_list"] = [
            {
                "number": step.number,
                "instruction": step.instruction,
                "title": step.title,
                "gradeable": step.gradeable,
                "reference_images": [
                    url for url in (_frame_url(found, p) for p in step.reference_images) if url
                ],
                "before_image": _frame_url(found, step.before_image),
                "requirements": list(step.requirements),
                "done_when": step.done_when,
            }
            for step in found.backend.steps()
        ]
        return detail

    @app.get("/api/procedures/{procedure_id}/files/{path:path}")
    async def procedure_file(procedure_id: str, path: str) -> FileResponse:
        found = procedure_or_404(procedure_id)
        root = found.entry.directory
        target = (root / path).resolve()
        # Only images under the procedure folder; never its config or prompts.
        if (not target.is_relative_to(root) or not target.is_file()
                or target.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}):
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(target)

    @app.get("/api/sessions")
    async def sessions() -> dict[str, Any]:
        return {"sessions": store.list_sessions(), "level": store.level}

    @app.get("/api/live")
    async def live() -> dict[str, Any]:
        return {"sessions": [host.state_of(s) for s in host.sessions()]}

    # The recorded tree, laid out as the UI's session pages read it.
    app.mount("/debug", StaticFiles(directory=store.root, check_dir=False), name="debug")
    return app


__all__ = ["create_api"]
