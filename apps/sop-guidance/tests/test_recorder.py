# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session store and debug recorder: on-disk contract read by the dashboard."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from sop_guidance import recorder
from sop_guidance.recorder import SessionHandle, SessionStore

# Explicit so the module runs under pytest-asyncio's strict mode as well as auto.
pytestmark = pytest.mark.asyncio


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _open(store: SessionStore, *, owner: str = "alice", procedure_id: str = "bench") -> SessionHandle:
    return store.open_session(
        procedure_id=procedure_id,
        procedure_name="Bench setup",
        total_steps=3,
        owner=owner,
        config={"backend": "vlm", "api_key": "sk-secret"},
    )


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    s = SessionStore(tmp_path / "debug", level="frames")
    await s.start()
    yield s
    await s.aclose()


async def test_open_and_end_write_meta_and_index(store: SessionStore) -> None:
    handle = _open(store)
    await store.flush()
    directory = store.session_dir(handle.session_id)
    meta = _json(directory / "meta.json")
    assert meta["outcome"] == "running" and meta["ended_us"] is None
    assert meta["session_id"] == handle.session_id
    assert meta["demo"] == "Bench setup" and meta["procedure_id"] == "bench" and meta["steps"] == 3
    assert meta["managed"] is True and meta["resumable"] is False and meta["level"] == "frames"
    assert meta["owner"] == "alice" and meta["step"] == 1
    assert isinstance(meta["updated_us"], int) and isinstance(meta["started_us"], int)
    index = _json(store.root / "index.json")
    assert [s["session_id"] for s in index["sessions"]] == [handle.session_id]
    assert _json(directory / "config.json") == {"backend": "vlm", "api_key": "<redacted>"}

    before = meta["updated_us"]
    handle.set_step(1, "Place the insert")
    handle.heartbeat()
    await store.flush()
    meta = _json(directory / "meta.json")
    assert meta["step"] == 2 and meta["instruction"] == "Place the insert"
    assert meta["updated_us"] >= before

    handle.end("completed")
    await store.flush()
    meta = _json(directory / "meta.json")
    assert meta["outcome"] == "completed" and isinstance(meta["ended_us"], int) and meta["resumable"] is False
    assert _json(store.root / "index.json")["sessions"][0]["outcome"] == "completed"
    events = [e["event"] for e in _jsonl(directory / "events.jsonl")]
    assert events == ["SESSION_START", "STEP", "SESSION_END"]
    assert not handle.recording
    handle.chat("user", "ignored after end", "alice")
    assert store.list_sessions()[0]["outcome"] == "completed"


async def test_level_off_still_persists_session_state(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, level="off")
    await store.start()
    try:
        handle = _open(store)
        assert not handle.recording
        image = tmp_path / "frame.png"
        image.write_bytes(b"png")
        handle.chat("user", "guide me", "alice")
        handle.save_checkpoint(step=0, procedure="Bench setup", history=[("q", "a")])
        handle.record_call(kind="vlm", name="check", request={}, response="yes", latency_ms=1.0, images=[str(image)])
        handle.capture_clip("correction")
        handle.end("stopped", reason="session_control")
        await store.flush()
        directory = store.session_dir(handle.session_id)
        meta = _json(directory / "meta.json")
        assert meta["level"] == "off" and meta["outcome"] == "stopped" and meta["resumable"] is True
        assert meta["calls"] == 0 and meta["reason"] == "session_control"
        assert _json(directory / "checkpoint.json")["history"] == [["q", "a"]]
        conversation = _json(directory / "conversation.json")
        assert conversation[0]["role"] == "user" and conversation[0]["text"] == "guide me"
        assert conversation[0]["participant"] == "alice" and isinstance(conversation[0]["at"], int)
        assert not (directory / "calls.jsonl").exists()
        assert not list(directory.glob("step_*"))
        assert (directory / "events.jsonl").exists()
    finally:
        await store.aclose()


async def test_record_call_hardlinks_images_and_scrubs(store: SessionStore, tmp_path: Path) -> None:
    handle = _open(store)
    assert handle.recording
    src = tmp_path / "live.png"
    src.write_bytes(b"original")
    handle.record_call(
        kind="llm",
        name="guidance_turn",
        request={"messages": ["hi", "x" * 30_000], "api_key": "sk-1", "extra": {"Authorization": "Bearer t"}},
        response={"reply": "ok"},
        latency_ms=12.345,
        error="",
        images=[str(src)],
    )
    await store.flush()
    directory = store.session_dir(handle.session_id)
    (record,) = _jsonl(directory / "calls.jsonl")
    assert record["seq"] == 1 and record["kind"] == "llm" and record["step"] == 1
    assert record["latency_ms"] == 12.3 and record["error"] is None
    assert record["request"]["api_key"] == "<redacted>"
    assert record["request"]["extra"]["Authorization"] == "<redacted>"
    assert record["request"]["messages"][1].endswith("<truncated>")
    assert record["artifacts"] == ["step_01/call_0001/in_00.png"]
    pinned = directory / record["artifacts"][0]
    assert pinned.stat().st_ino == src.stat().st_ino
    # The source being replaced must not change what the model was shown.
    replacement = tmp_path / "next.png"
    replacement.write_bytes(b"newer")
    replacement.replace(src)
    assert pinned.read_bytes() == b"original"
    handle.heartbeat()
    await store.flush()
    assert _json(directory / "meta.json")["calls"] == 1


async def test_a_frame_reused_before_the_writer_runs_keeps_what_was_recorded(
    store: SessionStore, tmp_path: Path,
) -> None:
    # Backends keep a small ring of frame names; a writer that is behind must
    # still pin the frame each call saw, not the one that later took its name.
    handle = _open(store)
    ring = tmp_path / "judged_0.jpg"
    for frame in (b"first", b"second"):
        fresh = tmp_path / "fresh.jpg"
        fresh.write_bytes(frame)
        fresh.replace(ring)
        handle.record_call(kind="judge", name="judge_frame", request={}, response={},
                           latency_ms=1.0, images=[str(ring)])
    await store.flush()
    directory = store.session_dir(handle.session_id)
    assert [(directory / r["artifacts"][0]).read_bytes()
            for r in _jsonl(directory / "calls.jsonl")] == [b"first", b"second"]


async def test_a_dropped_call_reports_no_artifacts(
    store: SessionStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _open(store)
    src = tmp_path / "live.jpg"
    src.write_bytes(b"x")
    monkeypatch.setattr(recorder, "_MAX_QUEUED_EVIDENCE", 0)
    assert handle.record_call(kind="judge", name="judge_frame", request={}, response={},
                              latency_ms=1.0, images=[str(src)]) == []
    await store.flush()
    assert not (store.session_dir(handle.session_id) / "step_01").exists()


async def test_full_level_clip_ring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recorder, "_ffmpeg_exe", lambda: "")
    store = SessionStore(tmp_path, level="full", clip_seconds=2, clip_fps=1)
    await store.start()
    try:
        handle = _open(store)
        assert handle.preview_due() and not handle.preview_due()
        for i in range(3):
            handle.push_preview_frame(f"jpeg{i}".encode(), i)
        handle.capture_clip("correction")
        handle.end("completed")
        await store.flush()
        directory = store.session_dir(handle.session_id)
        clips = [r for r in _jsonl(directory / "calls.jsonl") if r["kind"] == "clip"]
        assert [c["name"] for c in clips] == ["correction", "session-end"]
        first = clips[0]
        assert first["response"] == {"frames": 2} and first["video"].endswith("/clip.mp4")
        assert [(directory / a).read_bytes() for a in first["artifacts"]] == [b"jpeg1", b"jpeg2"]
        assert _json(directory / "meta.json")["clips"] == 2
    finally:
        await store.aclose()


async def test_frames_level_has_no_preview(store: SessionStore) -> None:
    handle = _open(store)
    assert not handle.preview_due()
    handle.push_preview_frame(b"jpeg", 1)
    handle.capture_clip("correction")
    await store.flush()
    assert not (store.session_dir(handle.session_id) / "calls.jsonl").exists()


async def test_latest_resumable_and_resume(store: SessionStore) -> None:
    done = _open(store)
    done.save_checkpoint(step=0)
    done.end("completed")
    handle = _open(store)
    sid = handle.session_id
    handle.set_step(1, "Second")
    handle.chat("user", "next step", "alice")
    handle.save_checkpoint(step=1, procedure="Bench setup", procedure_id="bench", wearer_requests=["small insert"])
    with pytest.raises(ValueError, match="Only stopped"):
        store.load_session(sid)
    handle.end("superseded", reason="participant_takeover")
    # Served before the writer has caught up.
    meta, checkpoint = store.load_session(sid)
    assert meta["outcome"] == "superseded" and checkpoint["step"] == 1
    assert store.latest_resumable("alice", "bench") == sid
    assert store.latest_resumable("bob", "bench") == ""
    assert store.latest_resumable("alice", "other") == ""

    resumed, restored = store.resume_session(sid, "bob")
    assert resumed.session_id == sid and restored["wearer_requests"] == ["small insert"]
    resumed.chat("agent", "Step 2 of 3", "bob")
    await store.flush()
    directory = store.session_dir(sid)
    meta = _json(directory / "meta.json")
    assert meta["outcome"] == "running" and meta["owner"] == "bob" and meta["step"] == 2
    assert meta["ended_us"] is None and meta["resumable"] is False
    assert [m["text"] for m in _json(directory / "conversation.json")] == ["next step", "Step 2 of 3"]
    assert "SESSION_RESUME" in [e["event"] for e in _jsonl(directory / "events.jsonl")]
    assert store.latest_resumable("alice", "bench") == ""
    with pytest.raises(ValueError):
        store.resume_session(sid, "carol")


async def test_load_session_rejects_bad_ids(store: SessionStore) -> None:
    legacy = store.root / "legacy"
    legacy.mkdir()
    (legacy / "meta.json").write_text(json.dumps({"session_id": "legacy", "outcome": "stopped"}))
    for bad in ("", ".", "..", "../legacy", "a/b", "missing"):
        with pytest.raises(ValueError, match="Unknown session"):
            store.load_session(bad)
    with pytest.raises(ValueError, match="no saved checkpoint"):
        store.load_session("legacy")
    with pytest.raises(ValueError):
        store.session_dir("../debug")
    assert store.session_dir("legacy") == legacy


async def test_mark_running_interrupted(tmp_path: Path) -> None:
    root = tmp_path / "debug"
    for name, checkpoint in (("crashed", True), ("bare", False)):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "meta.json").write_text(
            json.dumps({"session_id": name, "outcome": "running", "owner": "alice", "started_us": 1})
        )
        if checkpoint:
            (directory / "checkpoint.json").write_text(json.dumps({"step": 1}))
    store = SessionStore(root)
    assert store.mark_running_interrupted() == 2
    await store.start()
    try:
        live = _open(store)
        await store.flush()
        assert store.mark_running_interrupted("again") == 0
        crashed = _json(root / "crashed" / "meta.json")
        assert crashed["outcome"] == "interrupted" and crashed["reason"] == "worker_restarted"
        assert crashed["resumable"] is True and isinstance(crashed["ended_us"], int)
        assert _json(root / "bare" / "meta.json")["resumable"] is False
        await store.flush()
        index = {s["session_id"]: s for s in _json(root / "index.json")["sessions"]}
        assert index["crashed"]["outcome"] == "interrupted"
        assert index[live.session_id]["outcome"] == "running"
        assert store.load_session("crashed")[1] == {"step": 1}
    finally:
        await store.aclose()
    # aclose ends what is still open, so the next start has nothing to repair.
    assert _json(root / live.session_id / "meta.json")["outcome"] == "interrupted"
    assert SessionStore(root).mark_running_interrupted() == 0


async def test_prune_evicts_oldest_and_skips_open(tmp_path: Path) -> None:
    root = tmp_path / "debug"
    for i, name in enumerate(("old1", "old2")):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "meta.json").write_text(json.dumps({"session_id": name, "started_us": i + 1}))
        (directory / "blob").write_bytes(b"x" * 100_000)
    store = SessionStore(root, max_bytes=250_000)
    await store.start()
    try:
        oldest_open = _open(store)
        await store.flush()
        open_dir = store.session_dir(oldest_open.session_id)
        (open_dir / "blob").write_bytes(b"x" * 100_000)
        os.utime(open_dir, (1_000, 1_000))
        os.utime(root / "old1", (2_000, 2_000))
        os.utime(root / "old2", (3_000, 3_000))

        newest = _open(store)
        await store.flush()
        assert open_dir.is_dir(), "an open session is never evicted, even when oldest"
        assert not (root / "old1").exists(), "oldest closed session goes first"
        assert (root / "old2").is_dir(), "eviction stops once under budget"
        assert store.session_dir(newest.session_id).is_dir()
        ids = {s["session_id"] for s in _json(root / "index.json")["sessions"]}
        assert "old1" not in ids and {"old2", oldest_open.session_id, newest.session_id} <= ids
        assert "old1" not in {s["session_id"] for s in store.list_sessions()}
    finally:
        await store.aclose()


async def test_two_concurrent_handles(store: SessionStore, tmp_path: Path) -> None:
    image = tmp_path / "f.png"
    image.write_bytes(b"png")
    a = _open(store, owner="alice", procedure_id="bench")
    b = _open(store, owner="bob", procedure_id="bench")
    assert a.session_id != b.session_id
    a.set_step(2, "Third")
    for handle in (a, b):
        handle.record_call(kind="vlm", name="check", request={}, response="no", latency_ms=1, images=[str(image)])
        handle.chat("user", f"hi from {handle.session_id}", "p")
        handle.save_checkpoint(step=0)
    b.record_call(kind="vlm", name="check", request={}, response="yes", latency_ms=1)
    a.end("stopped")
    b.end("completed")
    await store.flush()
    da, db = store.session_dir(a.session_id), store.session_dir(b.session_id)
    assert [r["seq"] for r in _jsonl(da / "calls.jsonl")] == [1]
    assert [r["seq"] for r in _jsonl(db / "calls.jsonl")] == [1, 2]
    assert (da / "step_03" / "call_0001" / "in_00.png").exists()
    assert (db / "step_01" / "call_0001" / "in_00.png").exists()
    ma, mb = _json(da / "meta.json"), _json(db / "meta.json")
    assert (ma["owner"], ma["outcome"], ma["resumable"], ma["calls"]) == ("alice", "stopped", True, 1)
    assert (mb["owner"], mb["outcome"], mb["resumable"], mb["calls"]) == ("bob", "completed", False, 2)
    assert _json(da / "conversation.json")[0]["text"] == f"hi from {a.session_id}"
    index = _json(store.root / "index.json")["sessions"]
    assert {s["session_id"] for s in index} == {a.session_id, b.session_id}
    assert store.latest_resumable("alice", "bench") == a.session_id
    assert store.latest_resumable("bob", "bench") == ""


async def test_checkpoint_is_snapshotted(store: SessionStore) -> None:
    handle = _open(store)
    history = [["q", "a"]]
    handle.save_checkpoint(step=0, history=history)
    history.append(["later", "mutation"])
    handle.end("stopped")
    await store.flush()
    assert _json(store.session_dir(handle.session_id) / "checkpoint.json")["history"] == [["q", "a"]]


async def test_open_requires_start(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        _open(SessionStore(tmp_path))
