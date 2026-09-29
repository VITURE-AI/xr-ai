# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YOLO annotations for guidance frames.

Deliberately task-agnostic: this module finds boxes, draws them, and caches the
result. What the boxes MEAN belongs to a geometry plugin the caller hands in --
see ``geometry`` for the contract and ``profiles/nosepad/geometry.py`` for the
reference implementation. The box primitives here (``Detection``,
``union_coverage``) are the public helper API those plugins build on.

Everything is instance-based: a ``FrameAnnotator`` owns one detector profile,
its loaded models and its geometry plugin. Callers that want to share one
detector (and one device allocation) share the instance.

ultralytics, cv2 and OpenVINO are imported lazily inside methods so importing
this module stays cheap.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from .geometry import GeometryPlugin, NullGeometry

if TYPE_CHECKING:
    import numpy as np


class OverlayStyle(BaseModel):
    """How boxes and their label tabs are drawn."""

    model_config = ConfigDict(extra="forbid")

    boxes: bool = True
    line_width_px: int = 1
    line_type: str = "anti_aliased"
    labels: bool = True
    confidence_text: bool = False
    font: str = "cv2.FONT_HERSHEY_SIMPLEX"
    font_scale: float = 0.42
    font_thickness: int = 1
    label_padding_px: int = 3
    # Checkpoint class name -> label drawn and handed to the geometry plugin.
    # Empty means "every class, under the checkpoint's own name".
    class_labels: dict[str, str] = Field(default_factory=dict)
    # Checkpoint class name -> "#RRGGBB". Unlisted classes get a stable colour
    # derived from the class name.
    class_colors_rgb: dict[str, str] = Field(default_factory=dict)


class HandDetectorConfig(BaseModel):
    """A second, hand-only detector run after the task checkpoint.

    The task checkpoint's own `hand` class is trained on the SOP's footage and
    misses the pose that dominates it -- a hand wrapped around the glasses --
    and the geometry plugin's central rule is "a pad inside a hand box is being
    held", so a missed hand is a wrong reading rather than a missing box. This
    detector runs on the same frame and REPLACES the class named by
    ``replaces_class``, which is dropped from the results while this is enabled:
    these boxes are not a second opinion beside it, they are the hands.
    ``label`` and ``color_rgb`` therefore default to what that class used, so
    nothing downstream -- the geometry plugin's label matching, the VLM legend, a
    human reading the overlay -- can tell the boxes changed source.

    Measured on 40 live preview frames: this detector found a hand in 28, the
    task checkpoint's own `hand` class in 24 (its extra hits included a bare
    forearm at the frame edge), MediaPipe's hand landmarker in 17.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # A `.pt` file or an exported model directory -- the same two shapes as
    # `DetectorProfile.model`, and a relative path resolves the same way.
    model: str = ""
    conf: float = 0.40
    iou: float = 0.70
    # This model's own inference size (it was trained at 640), not the task
    # checkpoint's.
    imgsz: int = 640
    # Its own device, deliberately not inherited from the task detector. That one
    # runs an OpenVINO IR and so takes `intel:cpu`; a `.pt` here is torch and
    # takes `cpu` or a CUDA ordinal. Handing torch an `intel:*` string fails at
    # predict, so the two cannot share a field.
    device: int | str = "cpu"
    # Which class to read out of this detector. A dedicated hand model has
    # exactly one, but naming it keeps a multi-class checkpoint usable here.
    # Empty takes every box the model returns.
    source_class: str = "hand"
    # Deliberately what the superseded class used: the geometry plugin matches on
    # labels, and these boxes ARE the hands. Keep in step with the entries
    # `replaces_class` names in `overlay.class_labels` / `class_colors_rgb`.
    label: str = "hand"
    color_rgb: str = "#F4A6B5"
    # The task-checkpoint class this detector supersedes, suppressed from that
    # pass while `enabled`. It stays in `overlay.class_labels` because that
    # mapping is checked against the checkpoint's own classes at load: the class
    # still exists in the model, it is just not what the frame is annotated
    # with. Empty keeps those boxes, and one hand then carries two.
    replaces_class: str = "hand"


class DetectorProfile(BaseModel):
    """One detector configuration: checkpoint, inference settings and style."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # Whether a failed annotation must fail the caller's check (a guidance
    # step) rather than fall back to the raw frame. The annotator only reports
    # this; enforcing it is the caller's job.
    required: bool = True
    # Let ``warmup`` load the checkpoint and run one throwaway inference, so
    # the first real frame does not pay for the device allocation and kernel
    # autotune. Ignored when `enabled` is false -- there is nothing to warm.
    preheat: bool = False
    # A torch checkpoint (`.pt` file) or an exported model directory — an
    # OpenVINO IR export is a `*_openvino_model/` directory, not a file.
    model: str = ""
    imgsz: int = 768
    # CUDA ordinal (`0`), `cpu`, or an OpenVINO target (`intel:cpu`, `intel:gpu`,
    # `intel:npu`) when `model` points at an IR export. The intel targets fail
    # soft by design in ultralytics — an absent device compiles for CPU with only
    # a warning — so the warmup line reports what OpenVINO actually chose rather
    # than what was asked for. See `_execution_devices`.
    device: int | str = 0
    conf: float = 0.25
    iou: float = 0.70
    overlay: OverlayStyle = Field(default_factory=OverlayStyle)
    hands: HandDetectorConfig = Field(default_factory=HandDetectorConfig)

    def resolve_paths(self, base_dir: Path) -> DetectorProfile:
        """A copy with relative ``model`` / ``hands.model`` made absolute under *base_dir*."""
        hands = self.hands.model_copy(update={"model": _resolve_path(self.hands.model, base_dir)})
        return self.model_copy(update={"model": _resolve_path(self.model, base_dir), "hands": hands})


def _resolve_path(raw: str, base_dir: Path) -> str:
    if not raw:
        return raw
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def load_detector_profiles(path: Path) -> dict[str, DetectorProfile]:
    """Parse a detectors YAML into named profiles.

    The file holds a top-level ``profiles:`` mapping of name -> profile.
    Relative model paths are resolved against the YAML file's own directory, so
    the file and the weights it names move together.
    """
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("profiles"), dict):
        raise ValueError(f"{path}: expected a top-level `profiles:` mapping")
    profiles: dict[str, DetectorProfile] = {}
    for name, body in raw["profiles"].items():
        try:
            profile = DetectorProfile.model_validate(body or {})
        except Exception as exc:
            raise ValueError(f"{path}: detector profile {name!r} is invalid: {exc}") from exc
        profiles[str(name)] = profile.resolve_paths(path.parent)
    return profiles


_LFS_POINTER_PREFIX = b"version https://git-lfs"


def weight_problems(profile: DetectorProfile) -> list[str]:
    """Why *profile*'s weights cannot load: missing, or still git-lfs pointer files.

    For a backend's ``validate``: a pointer file only fails deep in the loader,
    as an unsupported format, on the first guided frame.
    """
    paths = [Path(profile.model)] if profile.model else []
    if profile.hands.enabled and profile.hands.model:
        paths.append(Path(profile.hands.model))
    problems: list[str] = []
    for path in paths:
        candidates = ([path] if path.is_file()
                      else sorted(path.glob("*.bin")) if path.is_dir() else [])
        if not candidates:
            problems.append(f"detector weights {path} do not exist")
            continue
        for candidate in candidates:
            with candidate.open("rb") as stream:
                if stream.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX:
                    problems.append(
                        f"detector weights {candidate} are a git-lfs pointer; run git lfs pull")
                    break
    return problems


@dataclass(frozen=True)
class Detection:
    """One detector box, in source-image pixel coordinates."""

    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    def overlap_fraction(self, other: Detection) -> float:
        """How much of *self* lies inside *other*, 0..1."""
        if self.area <= 0.0:
            return 0.0
        ix = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        return (ix * iy) / self.area

    def intersects(self, other: Detection) -> bool:
        return (
            self.x1 < other.x2 and other.x1 < self.x2
            and self.y1 < other.y2 and other.y1 < self.y2
        )

    def gap_to(self, other: Detection) -> float:
        """Edge-to-edge pixel distance; 0 when the boxes overlap."""
        dx = max(0.0, max(self.x1 - other.x2, other.x1 - self.x2))
        dy = max(0.0, max(self.y1 - other.y2, other.y1 - self.y2))
        return (dx * dx + dy * dy) ** 0.5


@dataclass
class AnnotatedFrame:
    """An in-memory frame with its boxes drawn on, plus what they mean."""

    # The annotated BGR array. The same array the caller passed in: annotation
    # draws in place.
    image: np.ndarray
    width: int
    height: int
    # Drawn labels, in detection order.
    detections: tuple[str, ...]
    boxes: tuple[Detection, ...]
    # Geometry prose for the VLM prompt; '' unless the annotator was built with
    # ``spatial_context=True``.
    spatial_context: str
    # Whatever the geometry plugin's `analyze` returned; opaque here, handed
    # back to the same plugin's `describe` / `veto`.
    geometry: object


@dataclass(frozen=True)
class OverlayArtifact:
    """An annotated frame written to disk, for callers that need a path."""

    path: str
    applied: bool
    detections: tuple[str, ...] = ()
    error: str = ""
    spatial_context: str = ""
    # Whatever the geometry plugin's `analyze` returned. Opaque here by design:
    # it travels back to the same plugin's `describe`/`veto` and nothing in
    # this module reads a field on it.
    geometry: object | None = None
    boxes: tuple[Detection, ...] = ()


def union_coverage(pad: Detection, boxes: list[Detection]) -> float:
    """Fraction of *pad* covered by the union of *boxes*, 0..1.

    Public because every geometry plugin wants it and the reasoning behind it is
    not obvious enough to expect each one to rediscover.

    Not max-over-boxes: when the detector splits one object into adjacent boxes,
    a box straddling the seam lands ~0.5 in each and would never clear a 0.6 bar
    against either alone while lying wholly inside the union. Not a bounding box
    over all boxes either — that spans everything between distant detections and
    would call a far-away object "inside".

    Exact, via coordinate compression: split the pad into the grid its
    intersections induce and add the cells covered by at least one box.
    """
    if pad.area <= 0.0 or not boxes:
        return 0.0
    xs = {pad.x1, pad.x2}
    ys = {pad.y1, pad.y2}
    for box in boxes:
        if box.x1 < pad.x2 and box.x2 > pad.x1:
            xs.update((max(box.x1, pad.x1), min(box.x2, pad.x2)))
        if box.y1 < pad.y2 and box.y2 > pad.y1:
            ys.update((max(box.y1, pad.y1), min(box.y2, pad.y2)))
    xs_sorted = sorted(xs)
    ys_sorted = sorted(ys)
    covered = 0.0
    for left, right in zip(xs_sorted, xs_sorted[1:], strict=False):
        for bottom, top in zip(ys_sorted, ys_sorted[1:], strict=False):
            cx = (left + right) / 2
            cy = (bottom + top) / 2
            if any(b.x1 <= cx <= b.x2 and b.y1 <= cy <= b.y2 for b in boxes):
                covered += (right - left) * (top - bottom)
    return min(1.0, covered / pad.area)


def _hex_to_bgr(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"#?([0-9a-fA-F]{6})", value.strip())
    if match is None:
        raise ValueError(f"invalid RGB color {value!r}; expected #RRGGBB")
    rgb = match.group(1)
    return int(rgb[4:6], 16), int(rgb[2:4], 16), int(rgb[0:2], 16)


def _auto_color(name: str) -> tuple[int, int, int]:
    """A stable BGR colour for a class the profile does not assign one to.

    Only reached by a detector whose classes are not enumerated in the profile
    -- a general-purpose checkpoint with dozens of them, where hand-picking
    colours is neither practical nor useful. Derived from the class name rather
    than its index so a class keeps its colour across checkpoints that order
    their classes differently, and kept away from the dark end so the box stays
    visible against the unlit scenes this runs on.
    """
    digest = hashlib.sha1(name.encode()).digest()
    # 128..255 per channel: bright enough to read on a dark frame, and spread
    # widely enough across the cube that neighbouring classes stay tellable.
    return (128 + digest[0] % 128, 128 + digest[1] % 128, 128 + digest[2] % 128)


def _tensor_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


# Every colour a class can be configured with needs a name in the prompt
# legend. An unmapped hex falls through to the hex itself, and the legend then
# tells the VLM to look for a "#F2C94C box" -- which is not a thing it can see.
_COLOR_NAMES = {
    "#2F80ED": "blue",
    "#F4A6B5": "light pink",
    "#27AE60": "green",
    "#F2C94C": "yellow",
}

ModelFactory = Callable[[str], Any]


class FrameAnnotator:
    """Lazily load one detector profile and render stable, VLM-oriented overlays."""

    def __init__(
        self,
        profile: DetectorProfile,
        *,
        geometry: GeometryPlugin | None = None,
        spatial_context: bool = False,
        artifacts_dir: Path,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self._profile = profile
        # Injected rather than loaded here so a bad plugin surfaces once at
        # startup, and so tests can pass a synthetic plugin.
        self.geometry: GeometryPlugin = geometry if geometry is not None else NullGeometry()
        # Whether annotate_* return the plugin's prose. A plugin's prose is
        # written for its own task, so a caller running a different procedure on
        # the same detector turns this off. Geometry is still analysed (vetoes
        # do not depend on it); only the prompt text is withheld.
        self.spatial_context = spatial_context
        self._artifacts_dir = Path(artifacts_dir).expanduser().resolve()
        self._model_path = Path(profile.model).expanduser().resolve() if profile.model else None
        # Resolved and checked here, at construction, rather than on the first
        # frame that wants it. `required` turns a per-frame annotation failure
        # into a failed step, so a missing file discovered lazily would read as
        # the wearer never completing a step rather than as a missing download.
        self._hands_model_path = (
            Path(profile.hands.model).expanduser().resolve() if profile.hands.model else None
        )
        if profile.enabled and profile.hands.enabled:
            if self._hands_model_path is None:
                raise ValueError(
                    "detector profile `hands` is enabled but its `model` is empty; "
                    "point it at a hand detector checkpoint",
                )
            # `exists`, not `is_file`: an exported model is a directory, exactly
            # as for the task checkpoint.
            if not self._hands_model_path.exists():
                raise FileNotFoundError(
                    f"hand detector checkpoint not found: {self._hands_model_path}",
                )
        self._model_factory = model_factory
        self._model: Any | None = None
        self._cv2: Any | None = None
        self._hands: Any | None = None
        # One lock for both models: an ultralytics model is not safe to predict
        # on from two threads.
        self._lock = threading.Lock()

    @property
    def profile(self) -> DetectorProfile:
        return self._profile

    @property
    def artifacts_dir(self) -> Path:
        return self._artifacts_dir

    # ── public API ────────────────────────────────────────────────────────────

    async def annotate_array(self, image: np.ndarray, *, stream: str = "") -> AnnotatedFrame:
        """Annotate an in-memory BGR frame; detection runs off the event loop.

        Video-rate callers must never touch the disk: PNG encoding a 720p frame
        costs more than the inference. Callers that later need a file encode one
        lazily via ``write_png``.

        Draws on and returns *image* itself -- the caller owns the array.

        *stream* is handed to the geometry plugin as the identity of the video
        this frame belongs to, which is what lets a plugin reason across frames.
        Pass the participant id: there is one preview loop per wearer, and a
        plugin keeping per-stream history would otherwise interleave them.
        Empty means the plugin gets no temporal context, which is correct for a
        one-off frame.

        A disabled profile draws nothing and returns the frame untouched.
        """
        if not self._profile.enabled:
            height, width = image.shape[:2]
            return AnnotatedFrame(
                image=image, width=width, height=height, detections=(), boxes=(),
                spatial_context="", geometry=None,
            )
        return await asyncio.to_thread(self._annotate_array_sync, image, stream)

    async def annotate_file(
        self, path: str, *, role: str = "student", stream: str = "",
    ) -> OverlayArtifact:
        """Annotate an image file and write the result as a PNG under ``artifacts_dir``.

        Never raises: a failure answers ``applied=False`` with ``error`` set, and
        the caller decides (from ``profile.required``) whether that fails a step.

        Only ``role="student"`` frames are analysed by the geometry plugin.
        Teacher frames are stills from the SOP, not moments in the wearer's
        video, and their overlays are cached by source + model identity.
        """
        if not self._profile.enabled:
            return OverlayArtifact(path=path, applied=False)
        try:
            return await asyncio.to_thread(self._annotate_file_sync, path, role, stream)
        except Exception as exc:
            logger.exception("GUIDANCE_YOLO  role={}  source={}  error={}", role, path, exc)
            return OverlayArtifact(path=path, applied=False, error=str(exc))

    async def detect_array(self, image: np.ndarray) -> list[Detection]:
        """Every box in an in-memory BGR frame, drawing nothing; runs off the event loop.

        For a backend that judges the boxes itself and has them drawn elsewhere.
        No geometry plugin sees them. A disabled profile finds nothing.
        """
        if not self._profile.enabled:
            return []
        return await asyncio.to_thread(self._detect_array_sync, image)

    def write_png(self, image: np.ndarray, stem: str) -> str:
        """Encode one already-annotated frame into ``artifacts_dir``; returns its path."""
        import cv2

        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        output = self._artifacts_dir / f"{stem}.png"
        temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.png")
        if not cv2.imwrite(str(temporary), image):
            raise OSError(f"OpenCV could not write YOLO overlay: {temporary}")
        os.replace(temporary, output)
        return str(output)

    def prompt_block(self, image_names: list[str]) -> str:
        """The box-legend block that tells the VLM how to read the overlay."""
        style = self._profile.overlay
        aliases = style.class_labels
        hands = self._profile.hands
        superseded = hands.replaces_class if hands.enabled else ""
        legend: list[str] = []
        for class_name, label in aliases.items():
            # The superseded class draws nothing, so a line for it would send the
            # VLM looking for a box that is never on the frame.
            if class_name and class_name == superseded:
                continue
            rgb = style.class_colors_rgb.get(class_name, "")
            color = _COLOR_NAMES.get(rgb.upper(), rgb or "colored")
            legend.append(f"- {color} `{label}` box: detector class `{class_name}`")
        if hands.enabled:
            color = _COLOR_NAMES.get(hands.color_rgb.upper(), hands.color_rgb or "colored")
            legend.append(
                f"- {color} `{hands.label}` box: dedicated hand detector",
            )
        images = ", ".join(image_names)
        # The legend is generic -- it is just the configured classes and colors.
        # Only the "what to look at" clause is task-specific, so the plugin
        # supplies that and a plugin-less run simply omits it.
        guide = self.geometry.overlay_guide()
        focus = f", {guide}" if guide else ""
        return (
            f"YOLO OVERLAY GUIDE ({images}):\n"
            + "\n".join(legend)
            + "\nUse the thin boxes and labels to locate these task objects and reason about "
            f"their spatial relationships{focus}. Treat every "
            "box as a detector hint, not ground truth: verify the object from the pixels, do not "
            "turn an apparent false positive into evidence, and do not treat a missing box as "
            "proof that the object is absent. Confidence values are intentionally hidden.\n\n"
        )

    def draw(self, image: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw externally supplied boxes in this profile's style; returns *image*.

        For boxes that did not come from this annotator's own detector (a
        backend that returns its own detections). Draws in place, like
        ``annotate_array``. Colours are looked up by drawn label: the hand
        detector's label, then any configured class whose label (or raw name)
        matches, else the same stable auto colour an unconfigured class gets.
        """
        import cv2

        for detection in detections:
            self._draw_detection(cv2, image, detection, self._color_for_label(detection.label))
        return image

    def warmup(self) -> bool:
        """Pay the detector's first-inference cost now. Blocking; returns whether it ran.

        Loading alone is not enough: ultralytics builds the model on the host
        and only moves it to ``profile.device`` on the first ``predict``, so the
        device allocation and the kernel autotune for this ``imgsz`` would still
        land on the first real frame. Hence a throwaway inference on a blank
        frame. It runs the whole combined pass, so it also loads and warms the
        hand detector when ``hands`` is on.

        Does nothing unless both ``enabled`` and ``preheat`` are set -- the
        single decision point, so no caller can preheat a detector the rest of
        the pipeline will never consult.

        Never raises. A bad checkpoint here is the same bad checkpoint the lazy
        path would hit, and that path already reports it per frame (see
        ``annotate_file`` and ``profile.required``); crashing at startup would
        take everything else down over a feature that only annotates frames.
        """
        profile = self._profile
        if not (profile.enabled and profile.preheat):
            return False
        started = time.perf_counter()
        try:
            import numpy as np

            with self._lock:
                model, cv2 = self._load()
                # Square blank frame at the configured inference size: no
                # detections to draw, and no letterboxing to make the autotune
                # miss the shape real frames will use.
                blank = np.zeros((profile.imgsz, profile.imgsz, 3), dtype=np.uint8)
                self._detect(cv2, model, blank)
                execution = self._execution_devices(model)
        except Exception as exc:
            logger.exception(
                "GUIDANCE_YOLO_PREHEAT  failed  model={}  device={}  error={}",
                self._model_path, profile.device, exc,
            )
            return False
        logger.info(
            "GUIDANCE_YOLO_PREHEAT  ready  model={}  device={}  execution={}  imgsz={}  took={:.2f}s",
            self._model_path,
            profile.device,
            execution or "n/a",
            profile.imgsz,
            time.perf_counter() - started,
        )
        return True

    # ── annotation ────────────────────────────────────────────────────────────

    def _annotate_array_sync(self, image: np.ndarray, stream: str = "") -> AnnotatedFrame:
        with self._lock:
            model, cv2 = self._load()
            detections = self._detect(cv2, model, image)
        geometry = self.geometry.analyze(detections, stream)
        height, width = image.shape[:2]
        return AnnotatedFrame(
            image=image,
            width=width,
            height=height,
            detections=tuple(d.label for d in detections),
            boxes=tuple(detections),
            spatial_context=self.geometry.describe(geometry) if self.spatial_context else "",
            geometry=geometry,
        )

    def _detect_array_sync(self, image: np.ndarray) -> list[Detection]:
        with self._lock:
            model, _cv2 = self._load()
            drawn = self._detect_yolo(model, image) + self._detect_hands(image)
        return [detection for detection, _ in drawn]

    def _annotate_file_sync(self, image_path: str, role: str, stream: str = "") -> OverlayArtifact:
        with self._lock:
            source = Path(image_path).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"guidance frame not found: {source}")
            output = self._artifact_path(source)
            is_student = role == "student"
            # Only teacher frames are served from the cache: a cached artifact
            # carries no boxes, and a student frame must reach the geometry
            # plugin every time or its stream window misses the sample.
            if not is_student and output.is_file():
                return OverlayArtifact(path=str(output), applied=True)

            model, cv2 = self._load()
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"OpenCV could not decode guidance frame: {source}")

            detections = self._detect(cv2, model, image)

            self._artifacts_dir.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.png")
            if not cv2.imwrite(str(temporary), image):
                raise OSError(f"OpenCV could not write YOLO overlay: {temporary}")
            os.replace(temporary, output)
        labels = [d.label for d in detections]
        logger.info(
            "GUIDANCE_YOLO  role={}  source={}  overlay={}  detections={}",
            role, source, output, ",".join(labels) or "none",
        )
        # Teacher frames get no stream even when one is available: folding a
        # reference image into the live window would vote with a frame that is
        # not part of the stream at all.
        geometry = self.geometry.analyze(detections, stream) if is_student else None
        return OverlayArtifact(
            path=str(output),
            applied=True,
            detections=tuple(labels),
            # Only the student frame is judged, so teacher geometry would be
            # noise in a prompt that already carries the teacher image.
            spatial_context=self.geometry.describe(geometry) if self.spatial_context else "",
            geometry=geometry,
            boxes=tuple(detections),
        )

    def _detect(self, cv2: Any, model: Any, image: np.ndarray) -> list[Detection]:
        """Run every detector over *image*, then draw the combined boxes once.

        The passes are sequential and drawing is deferred to the end on purpose:
        both detectors read pixels, so a box painted by the first would be part
        of the second one's input.
        """
        drawn = self._detect_yolo(model, image)
        drawn += self._detect_hands(image)
        for detection, color in drawn:
            self._draw_detection(cv2, image, detection, color)
        return [detection for detection, _ in drawn]

    def _detect_yolo(
        self, model: Any, image: np.ndarray,
    ) -> list[tuple[Detection, tuple[int, int, int]]]:
        """Boxes for every configured class, paired with the colour to draw in."""
        profile = self._profile
        results = model.predict(
            source=image,
            imgsz=profile.imgsz,
            device=profile.device,
            conf=profile.conf,
            iou=profile.iou,
            save=False,
            verbose=False,
        )
        result = results[0]
        boxes = getattr(result, "boxes", None)
        detections: list[tuple[Detection, tuple[int, int, int]]] = []
        if boxes is None:
            return detections
        xyxy = _tensor_numpy(boxes.xyxy)
        classes = _tensor_numpy(boxes.cls)
        confidences = _tensor_numpy(boxes.conf)
        names = getattr(result, "names", getattr(model, "names", {}))
        superseded = profile.hands.replaces_class if profile.hands.enabled else ""
        # An empty map means "every class the checkpoint has, under its own
        # name" -- the same reading `_load` gives it when deciding whether the
        # class list can have drifted. A general-purpose detector has 80 COCO
        # classes it does not rename.
        configured = profile.overlay.class_labels
        for coords, class_id, confidence in zip(xyxy, classes, confidences, strict=True):
            raw_name = str(names[int(class_id)])
            if configured and raw_name not in configured:
                continue
            # Dropped before it becomes a Detection, not just left undrawn: the
            # geometry plugin reads the returned list, and a hand it can see but
            # the wearer cannot is worse than either detector alone.
            if raw_name and raw_name == superseded:
                continue
            x1, y1, x2, y2 = (float(v) for v in coords)
            configured_color = profile.overlay.class_colors_rgb.get(raw_name)
            detections.append((
                Detection(
                    label=configured.get(raw_name, raw_name),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=float(confidence),
                ),
                _hex_to_bgr(configured_color) if configured_color else _auto_color(raw_name),
            ))
        return detections

    def _detect_hands(
        self, image: np.ndarray,
    ) -> list[tuple[Detection, tuple[int, int, int]]]:
        """Hand boxes from the dedicated detector, or nothing when it is off."""
        settings = self._profile.hands
        if not settings.enabled:
            return []
        model = self._load_hands()
        results = model.predict(
            source=image,
            imgsz=settings.imgsz,
            device=settings.device,
            conf=settings.conf,
            iou=settings.iou,
            save=False,
            verbose=False,
        )
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        names = getattr(result, "names", getattr(model, "names", {}))
        wanted = settings.source_class.strip().lower()
        color = _hex_to_bgr(settings.color_rgb)
        xyxy = _tensor_numpy(boxes.xyxy)
        classes = _tensor_numpy(boxes.cls)
        confidences = _tensor_numpy(boxes.conf)
        detections: list[tuple[Detection, tuple[int, int, int]]] = []
        for coords, class_id, confidence in zip(xyxy, classes, confidences, strict=True):
            if wanted and str(names[int(class_id)]).lower() != wanted:
                continue
            x1, y1, x2, y2 = (float(v) for v in coords)
            detections.append((
                Detection(
                    label=settings.label,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=float(confidence),
                ),
                color,
            ))
        return detections

    # ── model loading ─────────────────────────────────────────────────────────

    def _new_model(self, path: Path) -> Any:
        factory = self._model_factory
        if factory is None:
            from ultralytics import YOLO

            factory = YOLO
        return factory(str(path))

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._cv2 is not None:
            return self._model, self._cv2
        if self._model_path is None:
            raise ValueError("YOLO overlay model path is empty")
        # `exists`, not `is_file`: an OpenVINO export is a directory holding the
        # .xml/.bin pair, and ultralytics takes that directory directly.
        if not self._model_path.exists():
            raise FileNotFoundError(f"YOLO checkpoint not found: {self._model_path}")
        if self._model_path.is_dir():
            if not any(self._model_path.glob("*.xml")):
                raise FileNotFoundError(
                    f"YOLO model directory holds no OpenVINO IR (*.xml): {self._model_path}"
                )
            # Ultralytics infers the format from the directory NAME, not its
            # contents, so a correct IR under any other name is rejected deep in
            # its loader as "not a supported model format" with no mention of the
            # name. Fail here instead, where the reason is sayable.
            if not self._model_path.name.endswith("_openvino_model"):
                raise ValueError(
                    "OpenVINO model directory must be named '*_openvino_model' for "
                    f"ultralytics to recognise the format: {self._model_path}"
                )

        import cv2

        model = self._new_model(self._model_path)
        actual = self._class_names(model)
        expected = set(self._profile.overlay.class_labels)
        # An empty map means "draw the checkpoint's own class names", and there
        # is then nothing that could have drifted from them. The check exists to
        # catch a RENAMING profile that no longer matches the checkpoint it
        # renames -- swap the model and a stale `class_labels` silently
        # mislabels every box -- so it only applies once renames are configured.
        if expected and actual != expected:
            raise ValueError(
                "YOLO checkpoint classes do not match overlay config: "
                f"model={sorted(actual)}, configured={sorted(expected)}"
            )
        self._model = model
        self._cv2 = cv2
        return model, cv2

    def _load_hands(self) -> Any:
        """The hand detector, built once and reused (under ``self._lock``)."""
        if self._hands is not None:
            return self._hands
        if self._hands_model_path is None:
            raise ValueError("hand detector model path is empty")
        self._hands = self._new_model(self._hands_model_path)
        return self._hands

    def _class_names(self, model: Any) -> set[str]:
        """Class names the loaded model was trained with.

        Exported formats do not populate ``model.names`` until their backend
        loads, which ultralytics defers to the first ``predict`` — so an OpenVINO
        directory reports nothing here. Fall back to the ``metadata.yaml`` the
        export writes beside the IR rather than accepting the empty set, which
        the caller's mismatch check would read as "no classes configured" and
        wave through.

        A newer ultralytics runtime reports ``class0``, ``class1``, … instead of
        nothing, which passes that check while naming no real class, so an
        all-placeholder set takes the same fallback.
        """
        names = getattr(model, "names", None) or {}
        runtime_names = (
            {str(name) for name in names.values()}
            if isinstance(names, dict)
            else {str(name) for name in names}
            if isinstance(names, list)
            else set()
        )
        placeholders = bool(runtime_names) and all(
            name.startswith("class") and name[5:].isdigit()
            for name in runtime_names
        )
        if (
            (not runtime_names or placeholders)
            and self._model_path is not None
            and self._model_path.is_dir()
        ):
            metadata = self._model_path / "metadata.yaml"
            if not metadata.is_file():
                raise FileNotFoundError(
                    f"OpenVINO export is missing metadata.yaml: {self._model_path}"
                )
            names = (yaml.safe_load(metadata.read_text()) or {}).get("names") or {}
        if isinstance(names, list):
            names = dict(enumerate(names))
        return {str(name) for name in names.values()}

    def _model_fingerprint(self) -> str:
        """Content identity of the model, for the overlay cache key.

        A ``.pt`` is one file, but an OpenVINO export is a directory, and a
        directory's own mtime and size do not change when a weight file inside it
        is rewritten in place. Stat-ing the path itself would therefore let a
        re-export silently reuse overlays drawn by the previous model, so fold in
        every file the directory holds instead.
        """
        if self._model_path is None:
            raise ValueError("YOLO overlay model path is empty")
        if not self._model_path.is_dir():
            stat = self._model_path.stat()
            return f"{stat.st_mtime_ns}:{stat.st_size}"
        parts: list[str] = []
        for child in sorted(p for p in self._model_path.rglob("*") if p.is_file()):
            stat = child.stat()
            parts.append(
                f"{child.relative_to(self._model_path)}:{stat.st_mtime_ns}:{stat.st_size}",
            )
        return "|".join(parts)

    def _execution_devices(self, model: Any) -> str:
        """Hardware OpenVINO actually compiled for; "" on every other backend.

        Worth reporting because an ``intel:*`` request fails soft: ultralytics
        logs a warning and compiles for AUTO/CPU when the named device is absent,
        so a profile configured for the NPU on a host without one would otherwise
        look from the log like it was using it.
        """
        backend = getattr(getattr(model, "predictor", None), "model", None)
        compiled = getattr(backend, "ov_compiled_model", None)
        if compiled is None:
            return ""
        try:
            devices = compiled.get_property("EXECUTION_DEVICES")
        except Exception:
            return ""
        # The property is a list for CPU and GPU but a bare string for NPU, and
        # joining a string yields "N,P,U".
        if isinstance(devices, str):
            return devices
        return ",".join(str(d) for d in devices)

    def _artifact_path(self, source: Path) -> Path:
        if self._model_path is None:
            raise ValueError("YOLO overlay model path is empty")
        source_stat = source.stat()
        identity = "|".join((
            str(source.resolve()),
            str(source_stat.st_mtime_ns),
            str(source_stat.st_size),
            str(self._model_path),
            self._model_fingerprint(),
            self._profile.model_dump_json(),
        ))
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        return self._artifacts_dir / f"{source.stem}_yolo_{digest}.png"

    # ── drawing ───────────────────────────────────────────────────────────────

    def _color_for_label(self, label: str) -> tuple[int, int, int]:
        profile = self._profile
        if profile.hands.enabled and label == profile.hands.label:
            return _hex_to_bgr(profile.hands.color_rgb)
        colors = profile.overlay.class_colors_rgb
        for raw_name, drawn in profile.overlay.class_labels.items():
            if drawn == label and raw_name in colors:
                return _hex_to_bgr(colors[raw_name])
        if label in colors:
            return _hex_to_bgr(colors[label])
        return _auto_color(label)

    def _draw_detection(
        self,
        cv2: Any,
        image: np.ndarray,
        detection: Detection,
        color: tuple[int, int, int],
    ) -> None:
        style = self._profile.overlay
        height, width = image.shape[:2]
        x1, y1, x2, y2 = (
            int(round(value))
            for value in (detection.x1, detection.y1, detection.x2, detection.y2)
        )
        x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
        y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
        line_type = cv2.LINE_AA if style.line_type == "anti_aliased" else cv2.LINE_8

        if style.boxes:
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                thickness=style.line_width_px,
                lineType=line_type,
            )
        if not style.labels:
            return
        if style.font != "cv2.FONT_HERSHEY_SIMPLEX":
            raise ValueError(f"unsupported YOLO overlay font: {style.font}")
        label = detection.label
        if style.confidence_text:
            label = f"{label} {detection.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            style.font_scale,
            style.font_thickness,
        )
        padding = style.label_padding_px
        box_width = text_width + 2 * padding
        box_height = text_height + baseline + 2 * padding
        label_x1 = min(x1, max(0, width - box_width))
        label_y1 = y1 - box_height if y1 >= box_height else y1
        label_x2 = min(width - 1, label_x1 + box_width)
        label_y2 = min(height - 1, label_y1 + box_height)
        cv2.rectangle(image, (label_x1, label_y1), (label_x2, label_y2), color, thickness=-1)
        luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        text_color = (0, 0, 0) if luminance > 160 else (255, 255, 255)
        cv2.putText(
            image,
            label,
            (label_x1 + padding, label_y1 + padding + text_height),
            font,
            style.font_scale,
            text_color,
            style.font_thickness,
            line_type,
        )
