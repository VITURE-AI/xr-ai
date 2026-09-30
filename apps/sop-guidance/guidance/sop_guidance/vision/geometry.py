# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load task-specific detector-geometry reasoning from a module on disk.

The detector itself is generic: it finds boxes and draws them. What those boxes
*mean* is not. "A pad inside a hand box is being held even when it overlaps the
glasses" is a fact about replacing nose pads, and the thresholds behind it were
measured on one SOP's teacher frames. So the vision package owns no task
knowledge at all: a procedure profile ships a plugin file (for example
``profiles/nosepad/geometry.py``) and the caller loads it by path and hands the
instance to whatever needs it. There is no process-wide "active" plugin.

Contract -- every hook is OPTIONAL, and a plugin declares only what it has:

    GATES: tuple[str, ...] = ()
        Gate names procedure steps may name as their geometry gate.

    def analyze(detections: list[Detection]) -> object
    def analyze(detections: list[Detection], stream: str = "") -> object
        Classify the boxes into whatever this task needs. The return value is
        OPAQUE to the caller: it is handed back to ``describe`` and ``veto`` and
        is never inspected, so a plugin picks its own shape.

        Either signature is accepted. ``stream`` names the continuous video the
        frame came from -- the participant id -- and is what lets a plugin
        reason across frames rather than one at a time. It is passed only to a
        plugin that declares it, so the one-argument form keeps working; a
        plugin that wants it must default it, because a one-off annotation has
        no stream to give and an unkeyed call must stay stateless. The key is
        per participant on purpose: one preview loop runs per wearer, and a
        single shared history would interleave two of them.

    def describe(geometry: object) -> str
        Prose appended to the VLM prompt. '' to say nothing.

    def veto(geometry: object, gate: str) -> str
        Why *gate* rejects this frame, or '' when it does not object.

    def request_veto(geometry: object, gate: str, requests: Sequence[str]) -> str
        Why this frame contradicts what the wearer ASKED FOR, or ''. *requests*
        are the wearer's spoken requirements, oldest first; the caller only
        asks when there is at least one and the step names a gate. Same
        one-directional contract as ``veto``. It exists because the VLM, told
        the wearer asked for size one, has passed a step while writing in its
        own evidence that the detector saw size zero in the hand.

    def overlay_guide() -> str
        Extra clause for the box-legend prompt block.

    def spoken_example() -> str
        A concrete example correction, for the prompt rule that shapes spoken
        output. '' to keep the neutral default.

    def contradiction_example() -> str
        A concrete example of the observation-contradicts-verdict failure, for
        the same rule. '' to keep the neutral default. Separate from
        ``spoken_example`` because they illustrate different mistakes and a
        model given one generic and one concrete example follows the concrete
        one.

``veto`` returns a REASON, not a verdict, and that is load-bearing rather than
stylistic. Geometry is veto-only by design: it may turn a VLM "yes" into a "no",
never the reverse, because letting detection force a pass means a missed box
could complete a step that never happened. With a string return there is no
value a plugin can produce that a caller would read as a pass, so the invariant
survives a buggy -- or malicious -- plugin.

Splitting ``analyze`` from ``describe`` keeps prose and vetoes independently
optional: a plugin that only vetoes should not be building prose nothing reads.

Incoherent combinations are rejected at load, not at first frame, because each
is a config mistake rather than a runtime condition: ``veto`` could never
receive a token without ``analyze``; ``GATES`` naming gates that no ``veto``
implements would let a step opt into nothing; and a plugin with no hooks at all
is a path typo, not a no-op -- the way to have no geometry is ``NullGeometry``.

State: every ``load_geometry_plugin`` call executes the file into a FRESH module
object, so module-level state in a plugin (the nose-pad plugin's vote windows)
belongs to that one loaded instance, not to the process. Two loads are two
independent plugins.

SECURITY: this executes Python from disk, so the plugin file is part of the
application's trusted code, not data. Only ever load plugins from the app's own
profile directories.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from loguru import logger

if TYPE_CHECKING:
    from .overlay import Detection

# Hooks looked up on a loaded module. Order is the order they are reported in.
_HOOKS = (
    "analyze", "describe", "veto", "request_veto",
    "overlay_guide", "spoken_example", "contradiction_example",
)


@runtime_checkable
class GeometryPlugin(Protocol):
    """Structural type for a geometry plugin. See the module docstring."""

    GATES: tuple[str, ...]

    def analyze(self, detections: list[Detection], stream: str = "") -> object: ...
    def describe(self, geometry: object) -> str: ...
    def veto(self, geometry: object, gate: str) -> str: ...
    def request_veto(self, geometry: object, gate: str, requests: Sequence[str]) -> str: ...
    def overlay_guide(self) -> str: ...
    def spoken_example(self) -> str: ...
    def contradiction_example(self) -> str: ...


def _accepts_stream(hook: Any) -> bool:
    """Whether *hook* declares a second positional parameter for the stream key.

    Sniffed rather than required, so a one-argument ``analyze(detections)``
    keeps working. An unreadable signature -- a builtin, a C callable, an
    exotic ``__call__`` -- answers False: calling with one argument is what the
    contract always guaranteed, so the safe reading of "cannot tell" is that.
    """
    if hook is None:
        return False
    try:
        params = inspect.signature(hook).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values()):
        return True
    return len([
        p for p in params.values()
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]) >= 2


class NullGeometry:
    """The no-plugin case: no gates, no token, no prose, no veto.

    Behaviourally identical to running with no geometry reasoning -- boxes are
    still drawn, nothing reasons about them. A real object rather than a
    ``None`` every call site has to remember to check.
    """

    GATES: tuple[str, ...] = ()
    name = "none"

    def analyze(self, detections: list[Detection], stream: str = "") -> object:
        return None

    def describe(self, geometry: object) -> str:
        return ""

    def veto(self, geometry: object, gate: str) -> str:
        return ""

    def request_veto(self, geometry: object, gate: str, requests: Sequence[str]) -> str:
        return ""

    def overlay_guide(self) -> str:
        return ""

    def spoken_example(self) -> str:
        return ""

    def contradiction_example(self) -> str:
        return ""


class LoadedGeometry:
    """A plugin module behind the full hook surface, absent hooks defaulted.

    Wrapping rather than using the module directly means call sites never probe
    for a hook: an optional one a plugin left out answers with the same empty
    string the null plugin gives.
    """

    def __init__(self, module: ModuleType, path: Path) -> None:
        self._module = module
        self.path = path
        self.name: str = getattr(module, "__name__", path.stem)
        gates = getattr(module, "GATES", ())
        if isinstance(gates, str):
            raise ValueError(
                f"{path}: GATES must be a tuple/list of names, not a string"
            )
        self.GATES: tuple[str, ...] = tuple(
            str(g).strip().lower() for g in gates if str(g).strip()
        )
        # Resolved once at load, not per frame: this runs at video rate, and a
        # plugin cannot change its own signature between frames.
        self._analyze_takes_stream = _accepts_stream(getattr(module, "analyze", None))

    @property
    def module(self) -> ModuleType:
        """The executed plugin module (for tests and diagnostics)."""
        return self._module

    def analyze(self, detections: list[Detection], stream: str = "") -> object:
        hook = getattr(self._module, "analyze", None)
        if hook is None:
            return None
        if self._analyze_takes_stream:
            return hook(detections, stream)
        return hook(detections)

    def describe(self, geometry: object) -> str:
        hook = getattr(self._module, "describe", None)
        if hook is None or geometry is None:
            return ""
        return str(hook(geometry) or "")

    def veto(self, geometry: object, gate: str) -> str:
        hook = getattr(self._module, "veto", None)
        if hook is None or geometry is None or not gate:
            return ""
        return str(hook(geometry, gate) or "")

    def request_veto(self, geometry: object, gate: str, requests: Sequence[str]) -> str:
        hook = getattr(self._module, "request_veto", None)
        said = tuple(r for r in requests if r and str(r).strip())
        if hook is None or geometry is None or not gate or not said:
            return ""
        return str(hook(geometry, gate, said) or "")

    def overlay_guide(self) -> str:
        hook = getattr(self._module, "overlay_guide", None)
        return "" if hook is None else str(hook() or "")

    def spoken_example(self) -> str:
        hook = getattr(self._module, "spoken_example", None)
        return "" if hook is None else str(hook() or "")

    def contradiction_example(self) -> str:
        hook = getattr(self._module, "contradiction_example", None)
        return "" if hook is None else str(hook() or "")

    def __repr__(self) -> str:
        return f"LoadedGeometry(path={str(self.path)!r}, gates={self.GATES!r})"


def _validate(module: ModuleType, path: Path, gates: tuple[str, ...]) -> None:
    """Reject hook combinations that cannot do anything useful."""
    present = {h for h in _HOOKS if getattr(module, h, None) is not None}
    if not present and not gates:
        raise ValueError(
            f"{path}: defines none of {list(_HOOKS)} — nothing to load. To run "
            "without geometry reasoning, use NullGeometry instead."
        )
    if "analyze" not in present:
        for hook in ("veto", "request_veto"):
            if hook in present:
                raise ValueError(
                    f"{path}: defines {hook}() but no analyze() — {hook} would "
                    "never receive anything to judge."
                )
        if "describe" in present:
            raise ValueError(
                f"{path}: defines describe() but no analyze() — describe would "
                "never receive anything to describe."
            )
    if "request_veto" in present and not gates:
        raise ValueError(
            f"{path}: defines request_veto() but declares no GATES — it only "
            "runs on a step that names a gate, so it could never run."
        )
    if gates and "veto" not in present:
        raise ValueError(
            f"{path}: declares GATES {list(gates)} but no veto() — a step "
            "naming one of those gates would silently opt into nothing."
        )
    if "veto" in present and not gates:
        raise ValueError(
            f"{path}: defines veto() but declares no GATES — every gate name "
            "would be rejected, so veto could never run."
        )


def load_geometry_plugin(
    path: str | Path, *, base_dir: Path | None = None,
) -> GeometryPlugin:
    """Import the plugin file at *path* and wrap it as a ``LoadedGeometry``.

    An empty *path* answers ``NullGeometry``. A relative *path* resolves
    against *base_dir* when given, else the current directory.

    Imported by file path rather than by entry point on purpose: plugins live
    beside the procedure profile they serve, not in an installed package.

    Raises on any problem -- a missing file, a syntax error, an incoherent hook
    set. A named-but-unusable plugin is a configuration error, so it surfaces
    once at load rather than as silently absent vetoes discovered mid-session.
    """
    if not str(path).strip():
        logger.info("geometry plugin: none configured — no geometry reasoning")
        return NullGeometry()

    resolved = Path(path).expanduser()
    if not resolved.is_absolute() and base_dir is not None:
        resolved = base_dir / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"geometry plugin not found: {resolved}")

    # Namespaced, and keyed by the resolved path: every profile's plugin is
    # called `geometry.py`, so the stem alone would make two profiles collide
    # in sys.modules. Identical across reloads of one file, so repeated loads
    # replace rather than accumulate.
    digest = hashlib.sha1(str(resolved).encode()).hexdigest()[:10]
    module_name = f"sop_guidance_geometry_{resolved.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load geometry plugin: {resolved}")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so dataclasses (and a plugin split across files)
    # can resolve the module, and dropped again on failure so a broken load
    # leaves nothing behind.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    plugin = LoadedGeometry(module, resolved)
    _validate(module, resolved, plugin.GATES)
    logger.info(
        "geometry plugin: {}  hooks={}  gates={}",
        resolved,
        ",".join(h for h in _HOOKS if getattr(module, h, None) is not None) or "none",
        ",".join(plugin.GATES) or "none",
    )
    return plugin
