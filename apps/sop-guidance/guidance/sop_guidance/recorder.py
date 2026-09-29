# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guidance session store and debug recorder.

Every guided run is persisted as a session directory: its metadata, the
checkpoint it can be resumed from, and the user-facing conversation. That part
is unconditional. On top of it the capture level adds evidence: at ``frames``
every model call made while the session is open is recorded with its exact
request and reply, and the image files the model was shown are hardlinked in;
at ``full`` a rolling strip of the wearer's view (``clip_seconds`` at
``clip_fps``) is kept in memory and frozen to disk on demand, so a wrong
verdict can be replayed against what the camera saw around it.

Three properties hold the design together:

* **Disk writes are queued.** Every write goes onto one queue drained by one
  writer task, in order. The guidance monitor is timing-sensitive (settle
  windows, speech-lead, static-frame skips), and a synchronous 40 MB write
  inside a check would change the behaviour it exists to measure. Reads that
  must see not-yet-written state (``load_session`` right after ``end``) are
  served from memory.
* **Frames are hardlinked, not copied.** Callers pass paths to images that
  already exist; ``os.link`` pins the inode so the debug copy survives the
  source being overwritten, at no I/O cost and with no re-encode, which
  guarantees the bytes on disk are the bytes the model saw. The link is made
  when the call is recorded, not when the writer reaches it: callers reuse a
  small ring of file names, and a writer that is behind would otherwise link
  whatever frame had since replaced the one the model saw.
* **Secrets never enter.** Callers pass request *bodies*, never headers, and
  ``_scrub`` drops anything that looks like a credential regardless.

Layout under ``root``::

    index.json                       every session, newest first
    <session_id>/
      meta.json                      owner, step, timing, outcome, heartbeat
      config.json                    engine config the run was opened with
      checkpoint.json                saved step, context, wearer preferences
      conversation.json              complete user-facing guidance conversation
      events.jsonl                   SESSION_START / STEP / CHECK / ... / SESSION_END
      calls.jsonl                    one record per model call   (frames, full)
      step_01/
        call_0003/in_00.png          frames handed to the model  (frames, full)
        clip_0004/clip.mp4           rolling clip                (full)
        clip_0004/{t00.jpg,...}      the same frames, individually

The dashboard (xr-ai-ui) reads this tree through a plain file server, so every
file name and field name here is part of its contract.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from loguru import logger

LEVEL_OFF = "off"
LEVEL_FRAMES = "frames"
LEVEL_FULL = "full"
LEVELS = (LEVEL_OFF, LEVEL_FRAMES, LEVEL_FULL)

OUTCOME_RUNNING = "running"
OUTCOME_COMPLETED = "completed"
OUTCOME_STOPPED = "stopped"
OUTCOME_INTERRUPTED = "interrupted"
OUTCOME_SUPERSEDED = "superseded"
OUTCOMES = (OUTCOME_COMPLETED, OUTCOME_STOPPED, OUTCOME_INTERRUPTED, OUTCOME_SUPERSEDED)
RESUMABLE_OUTCOMES = frozenset({OUTCOME_STOPPED, OUTCOME_INTERRUPTED, OUTCOME_SUPERSEDED})

# Long prompts (whole narration timelines) are recorded truncated. The point is
# to see which prompt ran, not to archive it. The UI detects the suffix.
_MAX_STR = 20_000
_SECRET_HINTS = ("authorization", "api_key", "apikey", "token", "secret", "password")

# Evidence (calls, events, clips) beyond this many queued writes is dropped:
# losing a debug record is always better than growing memory without bound.
# State writes (meta, checkpoint, conversation, index) are never dropped.
_MAX_QUEUED_EVIDENCE = 4096
_MAX_BATCH = 256

JsonDict = dict[str, Any]


def _now_us() -> int:
    return int(time.time() * 1_000_000)


@cache
def _ffmpeg_exe() -> str:
    """Path to a usable ffmpeg, or "" when none is installed.

    ``imageio-ffmpeg`` ships a static build with libx264, which is what makes
    the clip a file a browser will actually play: OpenCV's Linux wheels carry no
    H.264 encoder at all, and its one working mp4 fourcc (``mp4v``, MPEG-4
    Part 2) produces a video Chrome refuses. A missing encoder degrades to the
    JPEGs beside the clip rather than failing.
    """
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        path = shutil.which("ffmpeg") or ""
        if not path:
            logger.warning("no ffmpeg available (imageio-ffmpeg not installed); clips are written as JPEG frames only")
        return path


def _encode_clip(frames: Sequence[bytes], dest: Path, fps: float) -> bool:
    """Mux JPEGs into an H.264 mp4. Returns False if unencodable."""
    exe = _ffmpeg_exe()
    if not exe or not frames:
        return False
    cmd = [
        exe, "-y", "-loglevel", "error",
        # The input codec is named because image2pipe's probe can fail to
        # recognise small or uniform JPEGs, which then yields "no streams".
        "-f", "image2pipe", "-framerate", f"{fps:g}", "-c:v", "mjpeg", "-i", "-",
        # yuv420p is what every browser decoder expects; the scale filter rounds
        # odd dimensions down, since yuv420p cannot represent them and ffmpeg
        # would fail the whole encode over a single odd row.
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-pix_fmt", "yuv420p",
        # Without faststart the moov atom lands at the end of the file, and a
        # browser streaming it over HTTP cannot start playback until the whole
        # clip has downloaded.
        "-movflags", "+faststart",
        str(dest),
    ]  # fmt: skip
    try:
        proc = subprocess.run(cmd, input=b"".join(frames), capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("guidance clip encode failed: {}", exc)
        return False
    if proc.returncode != 0:
        logger.warning("guidance clip encode failed: {}", proc.stderr.decode("utf-8", "replace").strip()[:200])
        return False
    return True


def _scrub(value: Any, depth: int = 0) -> Any:
    """Drop credential-shaped keys and cap long strings, recursively.

    An allowlist would be safer still, but request bodies are open-ended
    (extra bodies are operator-supplied), so a key-name denylist plus the rule
    that callers never hand over headers is what is actually enforceable.
    """
    if depth > 8:
        return "<nested>"
    if isinstance(value, dict):
        out: JsonDict = {}
        for k, v in value.items():
            if any(h in str(k).lower() for h in _SECRET_HINTS):
                out[str(k)] = "<redacted>"
            else:
                out[str(k)] = _scrub(v, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [_scrub(v, depth + 1) for v in value[:64]]
    if isinstance(value, str):
        return value if len(value) <= _MAX_STR else value[:_MAX_STR] + "…<truncated>"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _scrub(str(value), depth + 1)


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _pin(src: str, dest: Path) -> bool:
    """Hardlink ``src`` at ``dest``; True when ``dest`` now holds it."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dest)
    except FileExistsError:
        return True
    except OSError:
        return False
    return True


def _atomic_write(path: Path, text: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def _valid_id(session_id: str) -> bool:
    return (
        isinstance(session_id, str)
        and bool(session_id)
        and session_id not in {".", ".."}
        and Path(session_id).name == session_id
        and "\\" not in session_id
    )


@dataclass(frozen=True, slots=True)
class _Write:
    """One queued disk operation.

    The target path travels with the item rather than being resolved in the
    writer: by the time the writer reaches it the handle may have ended, and
    resolving late would drop the tail of the run.
    """

    kind: str  # "json" | "append" | "clip" | "prune" | "index"
    session_id: str = ""
    path: Path | None = None
    text: str = ""
    record: JsonDict | None = None
    artifacts: tuple[tuple[str, Path], ...] = ()
    frames: tuple[bytes, ...] = ()
    evidence: bool = False


class SessionStore:
    """Owns the session tree: index, retention budget, repair, and the writer."""

    def __init__(
        self,
        root: Path,
        *,
        level: str = LEVEL_FRAMES,
        max_bytes: int = 2_000_000_000,
        clip_seconds: float = 10.0,
        clip_fps: float = 1.0,
    ) -> None:
        normalized = (level or LEVEL_OFF).strip().lower()
        if normalized not in LEVELS:
            logger.warning("unknown guidance capture level {!r}; capturing session state only", level)
            normalized = LEVEL_OFF
        self._root = Path(root)
        self._level = normalized
        self._max_bytes = max(0, int(max_bytes))
        self._clip_seconds = max(1.0, float(clip_seconds))
        self._clip_fps = max(0.1, float(clip_fps))
        # Authoritative in-process view of every session's meta, keyed by
        # directory name. Disk lags it by whatever is still queued.
        self._metas: dict[str, JsonDict] = {}
        self._loaded = False
        # JSON documents queued but maybe not yet on disk, per session, so reads
        # right after a write see it. Dropped once the session has nothing queued.
        self._unwritten: dict[str, dict[str, str]] = {}
        self._open: dict[str, SessionHandle] = {}
        self._pending: dict[str, int] = {}
        # Guards _open and _pending against the pruner, which runs on the
        # writer thread and must never evict a live or still-draining session.
        self._lock = threading.Lock()
        self._queue: asyncio.Queue[_Write] | None = None
        self._writer: asyncio.Task[None] | None = None
        self._index_queued = False
        self._queued_evidence = 0
        self._overflowing = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def level(self) -> str:
        return self._level

    @property
    def root(self) -> Path:
        return self._root

    async def start(self) -> None:
        """Create the tree, load existing sessions, and start the writer."""
        if self._writer is not None and not self._writer.done():
            return
        await asyncio.to_thread(self._root.mkdir, parents=True, exist_ok=True)
        if not self._loaded:
            metas = await asyncio.to_thread(self._scan_metas)
            self._metas = {**metas, **self._metas}
            self._loaded = True
        self._queue = asyncio.Queue()
        self._index_queued = False
        self._queued_evidence = 0
        self._writer = asyncio.create_task(self._drain(), name="guidance-session-writer")
        self._request_index()
        logger.info(
            "guidance sessions: level={} dir={} budget={:.1f} GB clip={:g}s@{:g}fps",
            self._level, self._root, self._max_bytes / 1e9, self._clip_seconds, self._clip_fps,
        )  # fmt: skip

    async def aclose(self) -> None:
        """End any still-open session as interrupted, flush, stop the writer."""
        for handle in list(self._open.values()):
            handle.end(OUTCOME_INTERRUPTED, "worker_stopped")
        await self.flush()
        writer, self._writer, self._queue = self._writer, None, None
        if writer is not None:
            writer.cancel()
            await asyncio.gather(writer, return_exceptions=True)

    async def flush(self) -> None:
        """Wait until everything queued so far is on disk."""
        if self._queue is not None and self._running():
            await self._queue.join()

    def mark_running_interrupted(self, reason: str = "worker_restarted") -> int:
        """Close out sessions a previous process left marked running.

        A new worker cannot still own a previous process's live run, so any
        ``running`` session not open in this store is ended as interrupted.
        Runs synchronously: it is a startup step and its count is the result.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        self._ensure_loaded()
        repaired = 0
        now = _now_us()
        for sid, meta in self._scan_metas().items():
            if meta.get("outcome") != OUTCOME_RUNNING or sid in self._open:
                continue
            known = self._metas.get(sid)
            if known is not None and known.get("outcome") != OUTCOME_RUNNING:
                continue  # its terminal meta is still queued
            meta.update(
                outcome=OUTCOME_INTERRUPTED,
                reason=reason,
                ended_us=now,
                resumable=(self._root / sid / "checkpoint.json").exists(),
            )
            try:
                _atomic_write(self._root / sid / "meta.json", _dumps(meta))
            except OSError as exc:
                logger.warning("guidance session {} repair failed: {}", sid, exc)
                continue
            self._metas[sid] = meta
            repaired += 1
        if repaired:
            logger.info("guidance sessions: marked {} stale running session(s) interrupted ({})", repaired, reason)
        if self._running():
            self._request_index()
        else:
            self._write_index_now()
        return repaired

    # ── sessions ─────────────────────────────────────────────────────────────

    def open_session(
        self,
        *,
        procedure_id: str,
        procedure_name: str,
        total_steps: int,
        owner: str,
        config: JsonDict,
    ) -> SessionHandle:
        self._require_started()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in procedure_name)[:48]
        base = f"{stamp}_{safe}" if safe else stamp
        # mkdir and registration happen under the lock so the pruner, which
        # lists directories before snapshotting the protected set, can never
        # see this directory without also seeing it protected.
        with self._lock:
            session_id = base
            suffix = 1
            while True:
                directory = self._root / session_id
                try:
                    directory.mkdir(parents=True)
                    break
                except FileExistsError:
                    session_id = f"{base}-{suffix}"
                    suffix += 1
            handle = SessionHandle(
                self,
                session_id=session_id,
                directory=directory,
                procedure_id=procedure_id,
                procedure_name=procedure_name,
                total_steps=total_steps,
                owner=owner,
                started_us=_now_us(),
            )
            self._open[session_id] = handle
        handle._write_meta()
        self._queue_json(session_id, directory / "config.json", _dumps(_scrub(config)), keep=False)
        handle.note(
            "SESSION_START", demo=procedure_name, procedure_id=procedure_id, steps=total_steps, level=self._level,
        )  # fmt: skip
        self._put(_Write("prune"))
        # Indexed at START, not only at END. The dashboard is a static page over
        # a plain file server, so index.json is its only way to discover a
        # session; without this a run in progress, or one that ended by the
        # worker being killed, is on disk in full and invisible.
        self._request_index()
        logger.info("guidance session {} opened (level={}, owner={})", session_id, self._level, owner)
        return handle

    def resume_session(self, session_id: str, owner: str) -> tuple[SessionHandle, JsonDict]:
        """Reopen a stopped/interrupted/superseded session under ``owner``."""
        self._require_started()
        meta, checkpoint = self.load_session(session_id)
        directory = self._root / session_id
        conversation_text = self._read_text(session_id, "conversation.json")
        try:
            conversation = json.loads(conversation_text) if conversation_text else []
        except ValueError:
            conversation = []
        with self._lock:
            if session_id in self._open:
                raise ValueError("That session is already running.")
            handle = SessionHandle(
                self,
                session_id=session_id,
                directory=directory,
                procedure_id=str(meta.get("procedure_id", "")),
                procedure_name=str(meta.get("demo", "")),
                total_steps=int(meta.get("steps", 0) or 0),
                owner=owner,
                started_us=int(meta.get("started_us") or _now_us()),
                step_index=max(0, int(meta.get("step", 1) or 1) - 1),
                instruction=str(meta.get("instruction", "")),
                calls=int(meta.get("calls", 0) or 0),
                clips=int(meta.get("clips", 0) or 0),
                checkpoint_saved=True,
                conversation=conversation if isinstance(conversation, list) else [],
                # Unique artifact names across resumes, even when the previous
                # process died with part of its queue unwritten.
                seq=_now_us(),
            )
            self._open[session_id] = handle
        handle._write_meta()
        self._request_index()
        handle.note("SESSION_RESUME", owner=owner)
        logger.info("guidance session {} resumed by {}", session_id, owner)
        return handle, checkpoint

    def load_session(self, session_id: str) -> tuple[JsonDict, JsonDict]:
        """Return ``(meta, checkpoint)`` of a resumable session.

        Raises ``ValueError`` with a message fit to show the user.
        """
        if not _valid_id(session_id):
            raise ValueError("Unknown session.")
        self._ensure_loaded()
        meta = self._metas.get(session_id)
        if meta is None:
            meta = self._read_disk_meta(session_id)
            if meta is None:
                raise ValueError("Unknown session.")
        text = self._read_text(session_id, "checkpoint.json")
        try:
            checkpoint = json.loads(text) if text else None
        except ValueError:
            checkpoint = None
        if not isinstance(checkpoint, dict):
            raise ValueError("This recording has no saved checkpoint and cannot be resumed.")
        if meta.get("outcome") not in RESUMABLE_OUTCOMES:
            raise ValueError("Only stopped, interrupted, or replaced sessions can be resumed.")
        return dict(meta), checkpoint

    def latest_resumable(self, owner: str, procedure_id: str = "") -> str:
        """Most recently ended resumable session of ``owner``, or "".

        With ``procedure_id`` only that procedure's sessions count.
        """
        self._ensure_loaded()
        matches = [
            meta
            for meta in self._metas.values()
            if meta.get("owner") == owner
            and (not procedure_id or meta.get("procedure_id") == procedure_id)
            and meta.get("resumable")
        ]
        if not matches:
            return ""
        return str(max(matches, key=lambda m: m.get("ended_us") or 0)["session_id"])

    def list_sessions(self) -> list[JsonDict]:
        """Index entries, newest first."""
        self._ensure_loaded()
        return self._index_snapshot()

    def session_dir(self, session_id: str) -> Path:
        """Directory of an existing session; rejects anything but a plain name."""
        if not _valid_id(session_id):
            raise ValueError("Unknown session.")
        directory = self._root / session_id
        if not directory.is_dir():
            raise ValueError("Unknown session.")
        return directory

    # ── internals shared with handles ────────────────────────────────────────

    def _running(self) -> bool:
        return self._queue is not None and self._writer is not None and not self._writer.done()

    def _require_started(self) -> None:
        if not self._running():
            raise RuntimeError("SessionStore.start() must be awaited before sessions are opened")

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._metas = {**self._scan_metas(), **self._metas}
            self._loaded = True

    def _set_meta(self, session_id: str, meta: JsonDict) -> None:
        self._metas[session_id] = meta
        self._queue_json(session_id, self._root / session_id / "meta.json", _dumps(meta), keep=False)

    def _closed(self, handle: SessionHandle) -> None:
        with self._lock:
            if self._open.get(handle.session_id) is handle:
                del self._open[handle.session_id]

    def _queue_json(self, session_id: str, path: Path, text: str, *, keep: bool) -> None:
        if keep:
            self._unwritten.setdefault(session_id, {})[path.name] = text
        self._put(_Write("json", session_id=session_id, path=path, text=text))

    def _queue_evidence(self, item: _Write) -> bool:
        """Queue one debug record; False when it was dropped for backlog."""
        if self._queued_evidence >= _MAX_QUEUED_EVIDENCE:
            # Say so once per overflow so a truncated bundle is never mistaken
            # for a run that made fewer calls than it did.
            if not self._overflowing:
                logger.warning("guidance session writer is behind; dropping debug records")
                self._overflowing = True
            return False
        self._put(item)
        return True

    def _put(self, item: _Write) -> None:
        queue = self._queue
        if queue is None:
            return
        if item.evidence:
            self._queued_evidence += 1
        if item.session_id:
            with self._lock:
                self._pending[item.session_id] = self._pending.get(item.session_id, 0) + 1
        queue.put_nowait(item)

    def _request_index(self) -> None:
        # Coalesced: the snapshot is taken when the writer reaches the item, so
        # one queued rebuild already covers every change made before then.
        if not self._index_queued:
            self._index_queued = True
            self._put(_Write("index"))

    def _index_snapshot(self) -> list[JsonDict]:
        sessions = [dict(meta) for meta in self._metas.values()]
        sessions.sort(key=lambda s: s.get("started_us") or 0, reverse=True)
        return sessions

    def _read_text(self, session_id: str, name: str) -> str:
        text = self._unwritten.get(session_id, {}).get(name)
        if text is not None:
            return text
        try:
            return (self._root / session_id / name).read_text(encoding="utf-8")
        except OSError:
            return ""

    def _read_disk_meta(self, session_id: str) -> JsonDict | None:
        try:
            meta = json.loads((self._root / session_id / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return meta if isinstance(meta, dict) else None

    def _scan_metas(self) -> dict[str, JsonDict]:
        metas: dict[str, JsonDict] = {}
        try:
            paths = sorted(self._root.glob("*/meta.json"))
        except OSError:
            return metas
        for path in paths:
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(meta, dict):
                metas[path.parent.name] = meta
        return metas

    def _write_index_now(self) -> None:
        try:
            _atomic_write(self._root / "index.json", _dumps({"sessions": self._index_snapshot()}))
        except OSError as exc:
            logger.debug("guidance index write failed: {}", exc)

    # ── writer ───────────────────────────────────────────────────────────────

    async def _drain(self) -> None:
        queue = self._queue
        assert queue is not None
        while True:
            batch = [await queue.get()]
            while len(batch) < _MAX_BATCH:
                try:
                    batch.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            index: list[JsonDict] | None = None
            if any(item.kind == "index" for item in batch):
                self._index_queued = False
                index = self._index_snapshot()
            gone: list[str] = []
            try:
                gone = await asyncio.to_thread(self._write_batch, batch, index)
            except Exception:
                logger.exception("guidance session write failed")
            finally:
                self._settle(batch, gone)
                for _ in batch:
                    queue.task_done()

    def _settle(self, batch: Iterable[_Write], gone: Iterable[str]) -> None:
        for item in batch:
            if item.evidence:
                self._queued_evidence -= 1
            if item.session_id:
                with self._lock:
                    left = self._pending.get(item.session_id, 1) - 1
                    if left > 0:
                        self._pending[item.session_id] = left
                    else:
                        self._pending.pop(item.session_id, None)
                if left <= 0:
                    self._unwritten.pop(item.session_id, None)
        if self._queued_evidence < _MAX_QUEUED_EVIDENCE // 2:
            self._overflowing = False
        dropped = [sid for sid in gone if sid not in self._open and sid in self._metas]
        for sid in dropped:
            self._metas.pop(sid, None)
            self._unwritten.pop(sid, None)
        if dropped:
            self._request_index()

    def _write_batch(self, batch: Sequence[_Write], index: list[JsonDict] | None) -> list[str]:
        """Run on the writer thread. Returns session ids no longer on disk."""
        gone: list[str] = []
        for item in batch:
            try:
                if item.kind == "json" and item.path is not None:
                    _atomic_write(item.path, item.text)
                elif item.kind == "append" and item.path is not None:
                    self._write_append(item)
                elif item.kind == "clip" and item.path is not None:
                    self._write_clip(item)
                elif item.kind == "prune":
                    gone.extend(self._prune())
            except Exception:
                logger.exception("guidance session write failed: {} {}", item.kind, item.path)
        if index is not None:
            # Filtered against the disk so a crashed run, a manual deletion, or
            # the pruner can never leave the dashboard pointing at a session
            # that is no longer there.
            present: list[JsonDict] = []
            for meta in index:
                sid = str(meta.get("session_id", ""))
                if sid and sid not in gone and (self._root / sid).is_dir():
                    present.append(meta)
                elif sid not in gone:
                    gone.append(sid)
            try:
                _atomic_write(self._root / "index.json", _dumps({"sessions": present}))
            except OSError as exc:
                logger.debug("guidance index write failed: {}", exc)
        return gone

    @staticmethod
    def _write_append(item: _Write) -> None:
        assert item.path is not None
        for src, dest in item.artifacts:
            if _pin(src, dest):
                continue
            # Cross-device (the source may live in /tmp on another mount) or
            # already gone. A copy still captures the evidence; only the
            # zero-cost property is lost.
            try:
                shutil.copy2(src, dest)
            except OSError as exc:
                logger.debug("guidance artifact {} unavailable: {}", src, exc)
        with open(item.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(item.record, ensure_ascii=False, default=str) + "\n")

    def _write_clip(self, item: _Write) -> None:
        assert item.path is not None
        item.path.mkdir(parents=True, exist_ok=True)
        # The individual frames are written whether or not the encode works:
        # they are already in memory, they let the dashboard step one frame at
        # a time, and they are what it falls back to on a host with no ffmpeg.
        for i, jpeg in enumerate(item.frames):
            (item.path / f"t{i:02d}.jpg").write_bytes(jpeg)
        _encode_clip(item.frames, item.path / "clip.mp4", self._clip_fps)

    def _prune(self) -> list[str]:
        """Evict oldest sessions until the tree fits ``max_bytes``.

        Retention is "keep everything" up to a ceiling, not "keep N": the run
        directory is a volume shared with other logs and recordings, and filling
        it takes the worker down with an error that points nowhere near here.
        Open sessions, and closed ones with writes still queued, are never
        evicted.
        """
        if self._max_bytes <= 0:
            return []
        entries: list[tuple[float, Path, int]] = []
        total = 0
        for child in self._root.iterdir():
            if not child.is_dir():
                continue
            size = 0
            for path in child.rglob("*"):
                try:
                    if path.is_file():
                        size += path.stat().st_size
                except OSError:
                    continue
            total += size
            try:
                entries.append((child.stat().st_mtime, child, size))
            except OSError:
                continue
        if total <= self._max_bytes:
            return []
        # Snapshot after listing: see open_session for why this order matters.
        with self._lock:
            protected = set(self._open) | set(self._pending)
        evicted: list[str] = []
        for _mtime, path, size in sorted(entries):
            if total <= self._max_bytes:
                break
            if path.name in protected:
                continue
            shutil.rmtree(path, ignore_errors=True)
            total -= size
            evicted.append(path.name)
            logger.info("guidance sessions: evicted {} ({:.1f} MB)", path.name, size / 1e6)
        return evicted


class SessionHandle:
    """One open guided run. Obtained from :class:`SessionStore`; never built directly.

    Every method is a cheap in-memory update plus queued writes, safe to call
    from timing-sensitive code, and a no-op once the handle has ended.
    """

    def __init__(
        self,
        store: SessionStore,
        *,
        session_id: str,
        directory: Path,
        procedure_id: str,
        procedure_name: str,
        total_steps: int,
        owner: str,
        started_us: int,
        step_index: int = 0,
        instruction: str = "",
        calls: int = 0,
        clips: int = 0,
        checkpoint_saved: bool = False,
        conversation: list[JsonDict] | None = None,
        seq: int = 0,
    ) -> None:
        self._store = store
        self._session_id = session_id
        self._directory = directory
        self._procedure_id = procedure_id
        self._procedure_name = procedure_name
        self._total_steps = total_steps
        self._owner = owner
        self._started_us = started_us
        self._step_index = step_index
        self._instruction = instruction
        self._calls = calls
        self._clips = clips
        self._checkpoint_saved = checkpoint_saved
        self._conversation: list[JsonDict] = list(conversation or [])
        self._seq = seq
        self._outcome = OUTCOME_RUNNING
        self._reason = ""
        self._ended_us = 0
        # (timestamp_us, jpeg bytes) sampled at clip_fps, oldest first.
        self._ring: deque[tuple[int, bytes]] = deque(maxlen=max(1, int(store._clip_seconds * store._clip_fps)))
        self._last_ring_push_us = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def recording(self) -> bool:
        """True while open and capturing model calls (level is not ``off``)."""
        return self._active and self._store.level != LEVEL_OFF

    @property
    def _active(self) -> bool:
        return self._outcome == OUTCOME_RUNNING

    def set_step(self, index: int, instruction: str) -> None:
        if not self._active:
            return
        self._step_index = index
        self._instruction = instruction
        self.note("STEP", step=index + 1, instruction=instruction)
        # Refreshed per step so a live run's call/clip counts advance in the
        # dashboard instead of sitting at zero until guidance ends.
        self._write_meta()
        self._store._request_index()

    def save_checkpoint(self, **state: Any) -> None:
        """Persist the resumable state. Snapshotted now; later mutation of the
        caller's objects does not leak into the file."""
        if not self._active:
            return
        text = _dumps(state)
        self._checkpoint_saved = bool(state)
        self._store._queue_json(self._session_id, self._directory / "checkpoint.json", text, keep=True)
        self._write_meta()

    def chat(self, role: str, text: str, pid: str) -> None:
        if not self._active or not text:
            return
        self._conversation.append({"role": role, "text": text, "at": _now_us() // 1000, "participant": pid})
        self._store._queue_json(
            self._session_id, self._directory / "conversation.json", _dumps(self._conversation), keep=True,
        )  # fmt: skip

    def note(self, event: str, **fields: Any) -> None:
        """Append one lifecycle event. Written at every level."""
        if not self._active:
            return
        record = {"ts_us": _now_us(), "event": event, "step": self._step_index + 1, **_scrub(fields)}
        self._store._queue_evidence(
            _Write("append", self._session_id, self._directory / "events.jsonl", record=record, evidence=True)
        )

    def record_call(
        self,
        *,
        kind: str,
        name: str,
        request: Any,
        response: Any,
        latency_ms: float,
        error: str = "",
        images: Sequence[str] = (),
    ) -> list[str]:
        """Record one model call and pin the images it was shown.

        ``request`` must be a body, never a header map. ``images`` are paths
        that already exist; the writer hardlinks them into the call directory.
        Returns where they land, relative to the session directory, so an
        event can point at the same frame; empty when not recording.
        """
        if not self.recording:
            return []
        self._seq += 1
        self._calls += 1
        call_rel = f"step_{self._step_index + 1:02d}/call_{self._seq:04d}"
        artifacts: list[tuple[str, str]] = []
        for i, src in enumerate(images):
            if src:
                artifacts.append((str(src), f"{call_rel}/in_{i:02d}{Path(src).suffix or '.png'}"))
        record = {
            "ts_us": _now_us(),
            "seq": self._seq,
            "kind": kind,
            "name": name,
            "step": self._step_index + 1,
            "instruction": self._instruction,
            "latency_ms": round(latency_ms, 1),
            "error": error or None,
            "request": _scrub(request),
            "response": _scrub(response),
            "artifacts": [rel for _, rel in artifacts],
        }
        pinned = tuple((src, self._directory / rel) for src, rel in artifacts)
        queued = self._store._queue_evidence(
            _Write(
                "append",
                self._session_id,
                self._directory / "calls.jsonl",
                record=record,
                artifacts=pinned,
                evidence=True,
            )
        )
        if not queued:
            return []
        # Linked now, while each source still holds the frame the call saw; the
        # writer finds them in place, and copies only what could not be linked.
        # A link is a metadata operation, cheap enough for the event loop.
        for src, dest in pinned:
            _pin(src, dest)
        return [rel for _, rel in artifacts]

    def preview_due(self) -> bool:
        """Rate gate for the preview loop, checked before encoding anything.

        The preview loop runs far faster than the strip's fps, so almost every
        tick must cost nothing at all, including the JPEG encode, which is the
        expensive part. Only ``full`` level wants frames.
        """
        if not self._active or self._store.level != LEVEL_FULL:
            return False
        now_us = _now_us()
        if now_us - self._last_ring_push_us < int(1_000_000 / self._store._clip_fps):
            return False
        self._last_ring_push_us = now_us
        return True

    def push_preview_frame(self, jpeg: bytes, timestamp_us: int) -> None:
        if not self._active or not jpeg or self._store.level != LEVEL_FULL:
            return
        self._ring.append((timestamp_us, jpeg))

    def capture_clip(self, reason: str) -> None:
        """Freeze the rolling strip to disk. ``full`` level only; the strip is
        not cleared, so back-to-back captures may share frames."""
        if not self._active or self._store.level != LEVEL_FULL or not self._ring:
            return
        self._clips += 1
        self._seq += 1
        rel = f"step_{self._step_index + 1:02d}/clip_{self._seq:04d}"
        frames = tuple(jpeg for _ts, jpeg in self._ring)
        manifest = {
            "ts_us": _now_us(),
            "seq": self._seq,
            "kind": "clip",
            "name": reason,
            "step": self._step_index + 1,
            "instruction": self._instruction,
            "latency_ms": 0.0,
            "error": None,
            "request": {"reason": reason, "fps": self._store._clip_fps, "seconds": self._store._clip_seconds},
            "response": {"frames": len(frames)},
            "video": f"{rel}/clip.mp4",
            "artifacts": [f"{rel}/t{i:02d}.jpg" for i in range(len(frames))],
        }
        store = self._store
        store._queue_evidence(
            _Write("append", self._session_id, self._directory / "calls.jsonl", record=manifest, evidence=True)
        )
        store._queue_evidence(_Write("clip", self._session_id, self._directory / rel, frames=frames, evidence=True))

    def heartbeat(self) -> None:
        """Refresh ``updated_us``. The dashboard treats a running session whose
        heartbeat is older than 20 s as unverified, so owners call this every
        few seconds."""
        if not self._active:
            return
        self._write_meta()
        self._store._request_index()

    def end(self, outcome: str, reason: str = "") -> None:
        if not self._active:
            return
        if outcome not in OUTCOMES:
            logger.warning("guidance session {} ended with non-standard outcome {!r}", self._session_id, outcome)
        # The last thing the wearer did before guidance ended is what you want
        # when a run finished on the wrong verdict or was stopped in
        # frustration, and no other trigger covers it.
        self.capture_clip("session-end")
        self.note("SESSION_END", outcome=outcome)
        self._outcome = outcome or OUTCOME_STOPPED
        self._reason = reason
        self._ended_us = _now_us()
        self._write_meta()
        self._store._request_index()
        self._store._closed(self)
        logger.info(
            "guidance session {} ended: outcome={} calls={} clips={}",
            self._session_id, outcome, self._calls, self._clips,
        )  # fmt: skip

    def _write_meta(self) -> None:
        self._store._set_meta(
            self._session_id,
            {
                "session_id": self._session_id,
                "demo": self._procedure_name,
                "procedure_id": self._procedure_id,
                "steps": self._total_steps,
                "started_us": self._started_us,
                "ended_us": self._ended_us or None,
                "outcome": self._outcome,
                "level": self._store.level,
                "calls": self._calls,
                "clips": self._clips,
                "owner": self._owner,
                "step": self._step_index + 1,
                "instruction": self._instruction,
                "reason": self._reason,
                "resumable": self._checkpoint_saved and self._outcome in RESUMABLE_OUTCOMES,
                "managed": True,
                "updated_us": _now_us(),
            },
        )
