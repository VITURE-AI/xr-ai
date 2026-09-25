# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sop_guidance.vision and the nose-pad geometry plugin.

Pure CPU and model-free except ``test_real_nosepad_ir_on_reference_frame``,
which runs only when ultralytics and openvino are importable. Detectors are
stubs that return fixed boxes, so what is under test is the overlay's own
behaviour: class filtering, the hand-detector merge, drawing, the prompt
legend, and the geometry plugin's gates, voting and coasting.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest
from sop_guidance.vision import (
    AnnotatedFrame,
    Detection,
    DetectorProfile,
    FrameAnnotator,
    GeometryPlugin,
    NullGeometry,
    OverlayArtifact,
    bgr_to_i420,
    bgr_to_jpeg,
    bgr_to_png,
    bgr_to_rgb24,
    frame_to_bgr,
    load_bgr,
    load_detector_profiles,
    load_geometry_plugin,
    union_coverage,
)
from sop_guidance.vision.overlay import _hex_to_bgr

APP_DIR = Path(__file__).resolve().parents[1]
NOSEPAD_PLUGIN = APP_DIR / "profiles" / "nosepad" / "geometry.py"
DETECTORS_YAML = APP_DIR / "yaml" / "detectors.yaml"
REFERENCE_FRAME = Path(__file__).resolve().parent / "fixtures" / "step_04_b.jpg"


# ── helpers ───────────────────────────────────────────────────────────────────


def _box(label: str, x1: float, y1: float, x2: float, y2: float, conf: float = 0.95) -> Detection:
    return Detection(label=label, x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)


def _nosepad():
    """A fresh nose-pad plugin: its own vote windows, isolated from other tests."""
    return load_geometry_plugin(NOSEPAD_PLUGIN)


def _load_plugin_source(tmp_path: Path, source: str):
    path = tmp_path / "probe_plugin.py"
    path.write_text(source)
    return load_geometry_plugin(path)


class _StubBoxes:
    """The three parallel arrays ultralytics hands back on `result.boxes`."""

    def __init__(self, rows: list) -> None:
        self.xyxy = np.array([row[1:5] for row in rows], dtype=float).reshape(-1, 4)
        self.cls = np.array([row[0] for row in rows], dtype=int)
        self.conf = np.array([row[5] for row in rows], dtype=float)


class _StubYolo:
    """A detector that records the pixels it was given and returns fixed boxes."""

    def __init__(self, rows: list, names: dict) -> None:
        self.rows = rows
        self.names = names
        self.seen: np.ndarray | None = None
        self.calls = 0

    def predict(self, source, **_kwargs):
        self.calls += 1
        self.seen = source.copy()
        boxes = _StubBoxes(self.rows)
        return [type("_Result", (), {"boxes": boxes, "names": self.names})()]


_TASK_CLASSES = {0: "glasses_lens_area", 1: "hand"}
_HAND_CLASSES = {0: "hand"}


def _annotator(
    tmp_path: Path,
    yolo_rows: list,
    hands_rows: list | None = None,
    *,
    hands_enabled: bool | None = None,
    replaces_class: str = "hand",
    geometry: GeometryPlugin | None = None,
    spatial_context: bool = False,
    task_names: dict | None = None,
    hand_names: dict | None = None,
    **profile_fields,
) -> tuple[FrameAnnotator, _StubYolo, _StubYolo]:
    """An annotator wired to a stub task detector and a stub hand detector.

    One factory serves both models, told apart by the path it is handed --
    which is how the real one works too.
    """
    task_path = tmp_path / "task.pt"
    hand_path = tmp_path / "hand.pt"
    task_path.write_bytes(b"stub")
    hand_path.write_bytes(b"stub")
    task = _StubYolo(yolo_rows, task_names or _TASK_CLASSES)
    hands = _StubYolo(hands_rows or [], hand_names or _HAND_CLASSES)
    enabled = bool(hands_rows) if hands_enabled is None else hands_enabled
    profile = DetectorProfile.model_validate({
        "enabled": True,
        "model": str(task_path),
        "overlay": {
            "class_labels": {"glasses_lens_area": "glasses", "hand": "hand"},
            "class_colors_rgb": {"glasses_lens_area": "#2F80ED", "hand": "#F4A6B5"},
        },
        "hands": {"enabled": enabled, "model": str(hand_path), "replaces_class": replaces_class},
        **profile_fields,
    })
    by_path = {str(task_path.resolve()): task, str(hand_path.resolve()): hands}
    annotator = FrameAnnotator(
        profile,
        geometry=geometry,
        spatial_context=spatial_context,
        artifacts_dir=tmp_path / "artifacts",
        model_factory=lambda path: by_path[path],
    )
    return annotator, task, hands


def _annotate(annotator: FrameAnnotator, image: np.ndarray, stream: str = "") -> AnnotatedFrame:
    return asyncio.run(annotator.annotate_array(image, stream=stream))


def _blank(size: int = 400) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.uint8)


# ── box primitives ────────────────────────────────────────────────────────────


def test_detection_relations():
    a = _box("a", 0, 0, 10, 10)
    b = _box("b", 5, 0, 15, 10)
    far = _box("c", 20, 0, 30, 10)
    assert a.area == 100 and a.width == 10
    assert a.overlap_fraction(b) == pytest.approx(0.5)
    assert a.intersects(b) and not a.intersects(far)
    assert a.gap_to(b) == 0.0
    assert a.gap_to(far) == pytest.approx(10.0)
    assert _box("z", 0, 0, 0, 0).overlap_fraction(a) == 0.0


def test_union_coverage_counts_a_seam_as_covered():
    """A pad straddling two adjacent boxes is wholly inside their union."""
    pad = _box("nosepad", 90, 10, 110, 30)
    left = _box("glasses", 0, 0, 100, 100)
    right = _box("glasses", 100, 0, 200, 100)
    assert pad.overlap_fraction(left) == pytest.approx(0.5)
    assert union_coverage(pad, [left, right]) == pytest.approx(1.0)
    # Not a bounding box over everything: two distant boxes do not cover the gap.
    gap_pad = _box("nosepad", 140, 10, 160, 30)
    assert union_coverage(gap_pad, [left, _box("g", 300, 0, 400, 100)]) == 0.0
    assert union_coverage(pad, []) == 0.0


# ── frames ────────────────────────────────────────────────────────────────────


def _pattern(height: int = 48, width: int = 64) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_frames_rgb_family_round_trip_exactly():
    from xr_ai_hub import PixelFormat

    image = _pattern()
    h, w = image.shape[:2]
    assert np.array_equal(frame_to_bgr(bgr_to_rgb24(image), w, h, PixelFormat.RGB24), image)
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA).tobytes()
    assert np.array_equal(frame_to_bgr(rgba, w, h, PixelFormat.RGBA), image)
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA).tobytes()
    assert np.array_equal(frame_to_bgr(bgra, w, h, PixelFormat.BGRA), image)


def test_frames_yuv_round_trip_approximately():
    from xr_ai_hub import PixelFormat

    # Flat colour: I420 subsamples chroma, so only a smooth image survives it.
    image = np.full((48, 64, 3), (40, 120, 200), dtype=np.uint8)
    i420 = bgr_to_i420(image)
    assert len(i420) == 64 * 48 * 3 // 2
    back = frame_to_bgr(i420, 64, 48, PixelFormat.I420)
    assert back.shape == image.shape
    assert np.abs(back.astype(int) - image.astype(int)).max() <= 3

    # NV12 is I420 with the chroma planes interleaved.
    y = np.frombuffer(i420, dtype=np.uint8)[: 64 * 48]
    u = np.frombuffer(i420, dtype=np.uint8)[64 * 48: 64 * 48 + 32 * 24]
    v = np.frombuffer(i420, dtype=np.uint8)[64 * 48 + 32 * 24:]
    nv12 = np.concatenate([y, np.stack([u, v], axis=1).reshape(-1)]).tobytes()
    back_nv12 = frame_to_bgr(nv12, 64, 48, PixelFormat.NV12)
    assert np.abs(back_nv12.astype(int) - image.astype(int)).max() <= 3


def test_frames_unsupported_format_raises():
    with pytest.raises(ValueError, match="unsupported"):
        frame_to_bgr(b"\0" * 12, 2, 2, 99)


def test_png_is_lossless_and_load_bgr_reads_it(tmp_path):
    image = _pattern()
    path = tmp_path / "frame.png"
    path.write_bytes(bgr_to_png(image))
    assert np.array_equal(load_bgr(path), image)
    assert np.array_equal(load_bgr(str(path)), image)


def test_load_bgr_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        load_bgr(tmp_path / "missing.png")
    garbage = tmp_path / "garbage.png"
    garbage.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        load_bgr(garbage)


def test_jpeg_downscales_and_never_raises():
    image = _pattern(600, 1920)
    decoded = cv2.imdecode(np.frombuffer(bgr_to_jpeg(image), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (300, 960, 3)
    small = cv2.imdecode(
        np.frombuffer(bgr_to_jpeg(_pattern(), max_width=960), np.uint8), cv2.IMREAD_COLOR,
    )
    assert small.shape == (48, 64, 3)
    assert bgr_to_jpeg(object()) == b""


# ── drawing ───────────────────────────────────────────────────────────────────


def test_drawing_draws_box_and_label_tab_in_configured_colour(tmp_path):
    annotator, _, _ = _annotator(tmp_path, [(0, 100.0, 100.0, 300.0, 200.0, 0.9)])
    image = _blank()
    frame = _annotate(annotator, image)

    assert frame.image is image, "annotation draws in place on the caller's array"
    assert (frame.width, frame.height) == (400, 400)
    assert frame.detections == ("glasses",)
    assert frame.boxes == (_box("glasses", 100.0, 100.0, 300.0, 200.0, 0.9),)
    assert image.any(), "the box must change pixels"

    blue = _hex_to_bgr("#2F80ED")
    assert blue == (0xED, 0x80, 0x2F)
    # The label tab is a filled rectangle ABOVE the box when there is room; its
    # corner is padding, not text, so it carries the exact class colour.
    assert tuple(int(v) for v in image[92, 101]) == blue
    # Inside the tab, the text is drawn in a contrasting colour: the blue has
    # luminance < 160, so white text.
    tab = image[80:100, 100:160]
    assert (tab == 255).all(axis=2).any(), "label text must be drawn on the tab"
    # The box interior stays untouched (outline only).
    assert not image[140:160, 180:220].any()


def test_label_tab_moves_inside_when_box_touches_the_top(tmp_path):
    annotator, _, _ = _annotator(tmp_path, [(1, 50.0, 0.0, 150.0, 80.0, 0.9)], hands_enabled=False)
    image = _blank()
    _annotate(annotator, image)
    pink = _hex_to_bgr("#F4A6B5")
    assert tuple(int(v) for v in image[2, 52]) == pink
    # Pink is bright (luminance > 160): the text goes black, so the tab holds
    # both the pink fill and dark text pixels.
    tab = image[0:18, 50:90]
    assert (tab == 0).all(axis=2).any()


def test_labels_off_draws_no_tab(tmp_path):
    annotator, _, _ = _annotator(
        tmp_path, [(0, 100.0, 100.0, 300.0, 200.0, 0.9)],
        overlay={
            "class_labels": {"glasses_lens_area": "glasses", "hand": "hand"},
            "class_colors_rgb": {"glasses_lens_area": "#2F80ED"},
            "labels": False,
        },
    )
    image = _blank()
    _annotate(annotator, image)
    assert image.any()
    assert not image[80:99, 100:200].any(), "no label tab when labels are off"


def test_draw_external_boxes_uses_profile_style(tmp_path):
    annotator, task, _ = _annotator(
        tmp_path, [], [(0, 0, 0, 1, 1, 0.5)],
    )
    image = _blank()
    out = annotator.draw(image, [
        _box("glasses", 100, 100, 300, 200),
        _box("hand", 100, 300, 200, 380),
        _box("cup", 250, 250, 350, 350),
    ])
    assert out is image
    assert task.calls == 0, "draw must not run the detector"
    assert tuple(int(v) for v in image[92, 101]) == _hex_to_bgr("#2F80ED")
    assert tuple(int(v) for v in image[292, 101]) == _hex_to_bgr("#F4A6B5")
    # Unconfigured label: a stable bright auto colour, every channel >= 128.
    assert (image[242, 251] >= 128).all()


def test_unconfigured_classes_draw_under_their_own_names(tmp_path):
    """An empty class_labels map draws every class (the live COCO profile)."""
    annotator, _, _ = _annotator(
        tmp_path, [(0, 10.0, 10.0, 50.0, 50.0, 0.9), (1, 60.0, 60.0, 90.0, 90.0, 0.8)],
        task_names={0: "cup", 1: "person"},
        overlay={"class_labels": {}, "class_colors_rgb": {}},
    )
    frame = _annotate(annotator, _blank())
    assert frame.detections == ("cup", "person")


def test_class_label_mismatch_refuses_to_load(tmp_path):
    annotator, _, _ = _annotator(
        tmp_path, [(0, 10.0, 10.0, 50.0, 50.0, 0.9)], task_names={0: "glasses_lens_area"},
    )
    with pytest.raises(ValueError, match="do not match"):
        _annotate(annotator, _blank())


# ── hand detector merge ───────────────────────────────────────────────────────


def test_hand_boxes_carry_the_label_geometry_matches(tmp_path):
    annotator, _, _ = _annotator(
        tmp_path,
        yolo_rows=[(0, 10.0, 10.0, 110.0, 90.0, 0.9)],
        hands_rows=[(0, 88.0, 88.0, 212.0, 212.0, 0.88)],
    )
    frame = _annotate(annotator, _blank())
    assert sorted(frame.detections) == ["glasses", "hand"]
    hand = next(d for d in frame.boxes if d.label == "hand")
    assert (round(hand.x1), round(hand.y1), round(hand.x2), round(hand.y2)) == (88, 88, 212, 212)
    assert hand.confidence == pytest.approx(0.88)


def test_neither_detector_sees_the_other_s_boxes(tmp_path):
    """Drawing waits for both passes: a painted box would be detector input."""
    annotator, task, hands = _annotator(
        tmp_path,
        yolo_rows=[(1, 10.0, 10.0, 110.0, 90.0, 0.9)],
        hands_rows=[(0, 200.0, 200.0, 300.0, 300.0, 0.8)],
    )
    image = _blank()
    _annotate(annotator, image)
    assert not task.seen.any(), "the task detector must run on the untouched frame"
    assert not hands.seen.any(), "the hand detector must not see the first pass's boxes"
    assert image.any(), "the combined boxes must still be drawn onto the frame"


def test_disabled_hand_detector_is_never_loaded(tmp_path):
    annotator, _, hands = _annotator(
        tmp_path, yolo_rows=[(1, 10.0, 10.0, 110.0, 90.0, 0.9)], hands_rows=[],
    )
    frame = _annotate(annotator, _blank())
    assert hands.calls == 0
    assert frame.detections == ("hand",), "the task detector's hands survive when disabled"


def test_missing_hand_checkpoint_refuses_construction(tmp_path):
    profile = DetectorProfile.model_validate({
        "enabled": True,
        "model": str(tmp_path / "task.pt"),
        "hands": {"enabled": True, "model": str(tmp_path / "hand_yolov8n.pt")},
    })
    with pytest.raises(FileNotFoundError, match="hand_yolov8n.pt"):
        FrameAnnotator(profile, artifacts_dir=tmp_path)
    empty = profile.model_copy(update={"hands": profile.hands.model_copy(update={"model": ""})})
    with pytest.raises(ValueError, match="empty"):
        FrameAnnotator(empty, artifacts_dir=tmp_path)


def test_the_task_hand_class_is_superseded(tmp_path):
    """Dropped from the returned detections, not merely left undrawn."""
    annotator, _, _ = _annotator(
        tmp_path,
        yolo_rows=[(0, 10.0, 10.0, 110.0, 90.0, 0.9), (1, 200.0, 200.0, 300.0, 300.0, 0.8)],
        hands_rows=[(0, 88.0, 88.0, 212.0, 212.0, 0.8)],
    )
    frame = _annotate(annotator, _blank())
    hands = [d for d in frame.boxes if d.label == "hand"]
    assert len(hands) == 1 and round(hands[0].x1) == 88
    assert [d.label for d in frame.boxes if d.label != "hand"] == ["glasses"]
    legend = annotator.prompt_block(["student.png"])
    assert legend.count("`hand` box") == 1


def test_empty_replaces_class_keeps_both(tmp_path):
    annotator, _, _ = _annotator(
        tmp_path,
        yolo_rows=[(1, 200.0, 200.0, 300.0, 300.0, 0.8)],
        hands_rows=[(0, 88.0, 88.0, 212.0, 212.0, 0.8)],
        replaces_class="",
    )
    frame = _annotate(annotator, _blank())
    assert frame.detections.count("hand") == 2


def test_only_the_named_source_class_is_read(tmp_path):
    annotator, _, _ = _annotator(
        tmp_path,
        yolo_rows=[(0, 10.0, 10.0, 110.0, 90.0, 0.9)],
        hands_rows=[(0, 88.0, 88.0, 212.0, 212.0, 0.8), (1, 300.0, 300.0, 380.0, 380.0, 0.9)],
        hand_names={0: "hand", 1: "face"},
    )
    frame = _annotate(annotator, _blank())
    assert frame.detections.count("hand") == 1


# ── prompt block ──────────────────────────────────────────────────────────────


def test_prompt_block_nosepad_profile_verbatim(tmp_path):
    profile = load_detector_profiles(DETECTORS_YAML)["nosepad-v5"]
    annotator = FrameAnnotator(profile, geometry=_nosepad(), artifacts_dir=tmp_path)
    block = annotator.prompt_block(["teacher.png", "student.png"])
    assert block == (
        "YOLO OVERLAY GUIDE (teacher.png, student.png):\n"
        "- blue `glasses` box: detector class `glasses_lens_area`\n"
        "- green `nosepad_0` box: detector class `nosepad_0`\n"
        "- yellow `nosepad_1` box: detector class `nosepad_1`\n"
        "- light pink `hand` box: dedicated hand detector\n"
        "Use the thin boxes and labels to locate these task objects and reason about their "
        "spatial relationships, including whether the nosepad is aligned with or seated in the "
        "glasses area and how the hand is interacting with them. Treat every box as a detector "
        "hint, not ground truth: verify the object from the pixels, do not turn an apparent "
        "false positive into evidence, and do not treat a missing box as proof that the object "
        "is absent. Confidence values are intentionally hidden.\n\n"
    )


def test_prompt_block_without_plugin_omits_focus_clause(tmp_path):
    annotator, _, _ = _annotator(tmp_path, [], hands_enabled=False)
    block = annotator.prompt_block(["a.png"])
    assert block.startswith("YOLO OVERLAY GUIDE (a.png):\n")
    assert "- blue `glasses` box: detector class `glasses_lens_area`" in block
    assert "- light pink `hand` box: detector class `hand`" in block
    assert "dedicated hand detector" not in block
    assert "their spatial relationships. Treat every box" in block


# ── annotator: spatial context, files, warmup ─────────────────────────────────


_SEATED_ROWS = [(0, 100.0, 100.0, 300.0, 200.0, 0.95), (2, 180.0, 130.0, 200.0, 150.0, 0.9)]
_SEATED_NAMES = {0: "glasses_lens_area", 1: "hand", 2: "nosepad_0", 3: "nosepad_1"}
_NOSEPAD_OVERLAY = {
    "class_labels": {
        "glasses_lens_area": "glasses", "hand": "hand",
        "nosepad_0": "nosepad_0", "nosepad_1": "nosepad_1",
    },
    "class_colors_rgb": {
        "glasses_lens_area": "#2F80ED", "hand": "#F4A6B5",
        "nosepad_0": "#27AE60", "nosepad_1": "#F2C94C",
    },
}


def test_spatial_context_flag_withholds_prose_not_geometry(tmp_path):
    for flag in (False, True):
        annotator, _, _ = _annotator(
            tmp_path, _SEATED_ROWS, hands_enabled=False, task_names=_SEATED_NAMES,
            overlay=_NOSEPAD_OVERLAY, geometry=_nosepad(), spatial_context=flag,
        )
        frame = _annotate(annotator, _blank())
        assert frame.geometry.seated == 1, "geometry is analysed either way (vetoes need it)"
        assert annotator.geometry.veto(frame.geometry, "pad_on_glasses") == ""
        if flag:
            assert "DETECTOR GEOMETRY" in frame.spatial_context
            assert "nosepad_0" in frame.spatial_context
        else:
            assert frame.spatial_context == ""


def test_annotate_array_passes_stream_to_the_plugin(tmp_path):
    seen: list[str] = []

    class _Probe(NullGeometry):
        def analyze(self, detections, stream=""):
            seen.append(stream)
            return len(detections)

    annotator, _, _ = _annotator(
        tmp_path, [(0, 1.0, 1.0, 5.0, 5.0, 0.9)], hands_enabled=False, geometry=_Probe(),
    )
    frame = _annotate(annotator, _blank(), stream="pid-1")
    assert seen == ["pid-1"] and frame.geometry == 1


def test_disabled_profile_is_a_no_op(tmp_path):
    annotator, task, _ = _annotator(tmp_path, [(0, 1.0, 1.0, 5.0, 5.0, 0.9)], enabled=False)
    image = _blank()
    frame = _annotate(annotator, image)
    assert frame.detections == () and not image.any() and task.calls == 0
    art = asyncio.run(annotator.annotate_file(str(tmp_path / "x.png")))
    assert art == OverlayArtifact(path=str(tmp_path / "x.png"), applied=False)
    assert annotator.warmup() is False


def test_annotate_file_writes_png_and_caches_teacher_frames(tmp_path):
    annotator, task, _ = _annotator(
        tmp_path, _SEATED_ROWS, hands_enabled=False, task_names=_SEATED_NAMES,
        overlay=_NOSEPAD_OVERLAY, geometry=_nosepad(), spatial_context=True,
    )
    source = tmp_path / "frame.png"
    source.write_bytes(bgr_to_png(_blank()))

    student = asyncio.run(annotator.annotate_file(str(source), role="student"))
    assert student.applied and student.error == ""
    assert Path(student.path).parent == annotator.artifacts_dir
    assert Path(student.path).name.startswith("frame_yolo_")
    assert load_bgr(student.path).any(), "the written PNG carries the drawn boxes"
    assert student.detections == ("glasses", "nosepad_0")
    assert student.geometry.seated == 1 and "DETECTOR GEOMETRY" in student.spatial_context

    teacher = asyncio.run(annotator.annotate_file(str(source), role="teacher"))
    assert teacher.path == student.path
    assert task.calls == 1, "a cached teacher overlay must not rerun the detector"
    assert teacher.geometry is None and teacher.spatial_context == ""

    # Student frames always rerun, so the plugin's stream window sees them.
    asyncio.run(annotator.annotate_file(str(source), role="student"))
    assert task.calls == 2


def test_annotate_file_reports_errors_instead_of_raising(tmp_path):
    annotator, _, _ = _annotator(tmp_path, [], hands_enabled=False)
    missing = asyncio.run(annotator.annotate_file(str(tmp_path / "nope.png")))
    assert not missing.applied and "not found" in missing.error


def test_write_png(tmp_path):
    annotator, _, _ = _annotator(tmp_path, [], hands_enabled=False)
    path = annotator.write_png(_pattern(), "student_123")
    assert path == str(annotator.artifacts_dir / "student_123.png")
    assert np.array_equal(load_bgr(path), _pattern())


def test_warmup_runs_one_blank_inference_when_preheat(tmp_path):
    annotator, task, hands = _annotator(
        tmp_path, [], [(0, 0.0, 0.0, 1.0, 1.0, 0.1)], preheat=True, imgsz=64,
    )
    assert annotator.warmup() is True
    assert task.seen.shape == (64, 64, 3) and hands.calls == 1
    cold, cold_task, _ = _annotator(tmp_path, [], hands_enabled=False)
    assert cold.warmup() is False and cold_task.calls == 0


def test_warmup_never_raises(tmp_path):
    profile = DetectorProfile(enabled=True, preheat=True, model=str(tmp_path / "missing.pt"))
    assert FrameAnnotator(profile, artifacts_dir=tmp_path).warmup() is False


def test_openvino_directory_must_be_named_for_ultralytics(tmp_path):
    bad = tmp_path / "nosepad-ir"
    bad.mkdir()
    (bad / "best.xml").write_text("<net/>")
    annotator = FrameAnnotator(
        DetectorProfile(enabled=True, model=str(bad)), artifacts_dir=tmp_path,
        model_factory=lambda _p: pytest.fail("must refuse before loading"),
    )
    with pytest.raises(ValueError, match="_openvino_model"):
        _annotate(annotator, _blank())


def test_openvino_class_names_fall_back_to_metadata(tmp_path):
    ir = tmp_path / "x_openvino_model"
    ir.mkdir()
    (ir / "best.xml").write_text("<net/>")
    (ir / "metadata.yaml").write_text("names:\n  0: glasses_lens_area\n  1: hand\n")
    profile = DetectorProfile.model_validate({
        "enabled": True, "model": str(ir),
        "overlay": {"class_labels": {"glasses_lens_area": "glasses", "hand": "hand"}},
    })
    # An exported backend reports nothing before its first predict, and a newer
    # runtime reports placeholders; both must read the export's metadata.yaml.
    for runtime_names in ({}, {0: "class0", 1: "class1"}):
        stub = _StubYolo([(0, 10.0, 10.0, 50.0, 50.0, 0.9)], runtime_names)
        annotator = FrameAnnotator(profile, artifacts_dir=tmp_path, model_factory=lambda _p, s=stub: s)
        assert annotator._class_names(stub) == {"glasses_lens_area", "hand"}
        stub.names = {0: "glasses_lens_area", 1: "hand"}
        frame = _annotate(annotator, _blank())
        assert frame.detections == ("glasses",)


# ── detector profiles YAML ────────────────────────────────────────────────────


def test_load_detector_profiles_resolves_paths():
    profiles = load_detector_profiles(DETECTORS_YAML)
    assert {"nosepad-v5", "live-coco"} <= set(profiles)
    detectors = (APP_DIR / "detectors").resolve()

    nosepad = profiles["nosepad-v5"]
    assert Path(nosepad.model) == detectors / "nosepad-v5-int8_openvino_model"
    assert Path(nosepad.hands.model) == detectors / "hand" / "hand_yolov8n.pt"
    assert (nosepad.imgsz, nosepad.device, nosepad.conf, nosepad.iou) == (768, "intel:cpu", 0.30, 0.70)
    assert nosepad.enabled and nosepad.required and nosepad.preheat
    assert nosepad.hands.enabled and nosepad.hands.conf == 0.40 and nosepad.hands.device == "cpu"
    assert nosepad.hands.replaces_class == "hand" and nosepad.hands.color_rgb == "#F4A6B5"
    assert nosepad.overlay.class_colors_rgb == _NOSEPAD_OVERLAY["class_colors_rgb"]
    assert nosepad.overlay.class_labels == _NOSEPAD_OVERLAY["class_labels"]

    live = profiles["live-coco"]
    assert Path(live.model) == detectors / "yolo26s-int8_openvino_model"
    assert (live.imgsz, live.conf, live.required, live.hands.enabled) == (640, 0.40, False, False)
    assert live.overlay.confidence_text and live.overlay.class_labels == {}

    for profile in profiles.values():
        assert Path(profile.model).is_dir(), f"missing weights: {profile.model}"
        assert Path(profile.model).name.endswith("_openvino_model")
        assert any(Path(profile.model).glob("*.xml")) and any(Path(profile.model).glob("*.bin"))
        assert (Path(profile.model) / "metadata.yaml").is_file()
    assert Path(nosepad.hands.model).is_file()


def test_load_detector_profiles_rejects_typos(tmp_path):
    path = tmp_path / "detectors.yaml"
    path.write_text("profiles:\n  x:\n    model: a.pt\n    confidence: 0.3\n")
    with pytest.raises(ValueError, match="'x' is invalid"):
        load_detector_profiles(path)
    path.write_text("x:\n  model: a.pt\n")
    with pytest.raises(ValueError, match="profiles"):
        load_detector_profiles(path)
    path.write_text("profiles:\n  x:\n    model: sub/a.pt\n")
    assert load_detector_profiles(path)["x"].model == str((tmp_path / "sub" / "a.pt").resolve())


# ── geometry plugin loader ────────────────────────────────────────────────────


def test_null_geometry_is_silent():
    for plugin in (NullGeometry(), load_geometry_plugin("")):
        assert isinstance(plugin, GeometryPlugin)
        assert plugin.GATES == ()
        geometry = plugin.analyze([_box("nosepad", 0, 0, 10, 10)])
        assert plugin.describe(geometry) == ""
        assert plugin.veto(geometry, "pad_on_glasses") == ""
        assert plugin.overlay_guide() == "" and plugin.spoken_example() == ""
        assert plugin.contradiction_example() == ""


def test_missing_plugin_file_raises():
    with pytest.raises(FileNotFoundError):
        load_geometry_plugin("/nonexistent/nope.py")


@pytest.mark.parametrize("source", [
    'GATES = ("g",)\ndef veto(g, gate): return "no"\n',          # veto without analyze
    "def describe(g): return 'x'\n",                              # describe without analyze
    'GATES = ("g",)\ndef analyze(d): return 1\n',                # gates without veto
    'def analyze(d): return 1\ndef veto(g, gate): return "no"\n',  # veto without gates
    "X = 1\n",                                                     # no hooks at all
    'GATES = "g"\ndef analyze(d): return 1\ndef veto(g, gate): return ""\n',  # GATES as str
])
def test_incoherent_hook_sets_rejected(tmp_path, source):
    with pytest.raises(ValueError):
        _load_plugin_source(tmp_path, source)


def test_optional_hooks_default_empty(tmp_path):
    plugin = _load_plugin_source(tmp_path, "def analyze(detections): return len(detections)\n")
    assert isinstance(plugin, GeometryPlugin)
    assert plugin.GATES == ()
    geometry = plugin.analyze([_box("nosepad", 0, 0, 10, 10)])
    assert geometry == 1
    assert plugin.describe(geometry) == ""
    assert plugin.veto(geometry, "anything") == ""
    assert plugin.overlay_guide() == ""


def test_veto_cannot_grant_a_pass(tmp_path):
    """The contract is a reason string, so no return value can mean 'pass'."""
    plugin = _load_plugin_source(
        tmp_path,
        'GATES = ("g",)\n'
        "def analyze(d): return object()\n"
        "def veto(g, gate): return True\n",
    )
    out = plugin.veto(plugin.analyze([]), "g")
    assert isinstance(out, str) and out != ""


def test_one_arg_plugin_gets_no_stream(tmp_path):
    plugin = _load_plugin_source(
        tmp_path,
        'GATES = ("g",)\n'
        "def analyze(detections): return len(detections)\n"
        'def veto(geometry, gate): return "no" if geometry else ""\n',
    )
    assert plugin.analyze([_box("nosepad", 0, 0, 10, 10)], "some-pid") == 1


def test_two_loads_are_independent_instances():
    """No process-global plugin state: each load has its own vote windows."""
    a, b = _nosepad(), _nosepad()
    assert a.module is not b.module
    glasses = _box("glasses", 100, 100, 300, 200)
    pad = _box("nosepad", 180, 130, 200, 150)
    for _ in range(5):
        a.analyze([glasses, pad], "pid")
    assert "pid" in a.module._history and "pid" not in b.module._history


# ── nose-pad geometry: classification and gates ───────────────────────────────


def test_shipped_nosepad_plugin_loads():
    plugin = _nosepad()
    assert set(plugin.GATES) == {"pad_on_glasses", "no_pad_on_glasses", "pad_in_hand"}
    assert "nosepad" in plugin.overlay_guide()
    assert "nose pad" in plugin.spoken_example()


def test_nosepad_spoken_examples_keep_tuned_wording():
    plugin = _nosepad()
    assert plugin.spoken_example() == (
        "the nose pad is still attached to the bridge; pull it straight back off"
    )
    assert plugin.contradiction_example() == (
        "Do not describe a pad as seated and then report that it is not seated"
    )


def test_nosepad_classifies_held_seated_loose():
    """Hand beats glasses, and a distant pad is neither."""
    plugin = _nosepad()
    glasses = _box("glasses", 100, 100, 300, 200)

    seated = plugin.analyze([glasses, _box("nosepad", 180, 130, 200, 150)])
    assert (seated.seated, seated.held) == (1, 0)

    held = plugin.analyze([glasses, _box("nosepad", 180, 130, 200, 150), _box("hand", 150, 110, 260, 190)])
    assert (held.seated, held.held) == (0, 1)
    assert (held.held_at_bridge, held.held_away) == (1, 0)

    # In a hand, off the glasses, but only just: still at the bridge.
    near = plugin.analyze([
        glasses, _box("nosepad", 310, 150, 330, 170), _box("hand", 300, 140, 380, 190),
    ])
    assert (near.held_at_bridge, near.held_away) == (1, 0)

    # In a hand, far away: a spare the hand happens to be in front of.
    away = plugin.analyze([
        glasses, _box("nosepad", 500, 400, 520, 420), _box("hand", 470, 370, 560, 460),
    ])
    assert (away.held_at_bridge, away.held_away) == (0, 1)

    loose = plugin.analyze([glasses, _box("nosepad", 600, 600, 620, 620)])
    assert (loose.seated, loose.held, loose.loose) == (0, 0, 1)


def test_nosepad_gates_read_opposite_counts():
    """The decoy case: holding a second pad must not clear the bridge."""
    plugin = _nosepad()
    glasses = _box("glasses", 100, 100, 300, 200)
    decoy = plugin.analyze([
        glasses,
        _box("nosepad", 180, 130, 200, 150),   # still on the bridge
        _box("nosepad", 500, 400, 520, 420),   # second pad, in a hand
        _box("hand", 470, 370, 560, 460),
    ])
    assert decoy.seated == 1 and decoy.held == 1
    assert plugin.veto(decoy, "no_pad_on_glasses") == (
        "There is still a nose pad on the glasses. Holding a different "
        "pad does not count — take the one on the bridge off first."
    )
    assert (decoy.held_at_bridge, decoy.held_away) == (0, 1)
    assert decoy.pads_on_glasses == 1

    at_bridge = plugin.analyze([
        glasses,
        _box("nosepad", 180, 130, 200, 150),   # still on the bridge
        _box("nosepad", 310, 150, 330, 170),   # replacement, 5% away
        _box("hand", 300, 140, 380, 190),
    ])
    assert (at_bridge.held_at_bridge, at_bridge.held_away) == (1, 0)
    assert at_bridge.pads_on_glasses == 0
    assert plugin.veto(at_bridge, "pad_on_glasses") == (
        "The pad still looks like it is in your hand rather than in the "
        "bridge. Press it into the slot and let go of it."
    )
    assert plugin.veto(at_bridge, "no_pad_on_glasses") != ""

    only_seated = plugin.analyze([glasses, _box("nosepad", 180, 130, 200, 150)])
    assert plugin.veto(only_seated, "no_pad_on_glasses") == (
        "The nose pad still looks attached to the bridge. Pull it straight "
        "back off the frame, not just hold it."
    )

    clear = plugin.analyze([glasses])
    assert plugin.veto(clear, "no_pad_on_glasses") == ""
    assert plugin.veto(clear, "pad_on_glasses") == (
        "I do not see a nose pad resting in the bridge yet. Push it into the "
        "slot until it stays there on its own."
    )


def test_pad_in_hand_gate_reads_both_halves_of_held():
    plugin = _nosepad()
    glasses = _box("glasses", 100, 100, 300, 200)

    on_table = plugin.analyze([
        glasses,
        _box("nosepad", 500, 400, 520, 420),
        _box("nosepad", 540, 400, 560, 420),
        _box("hand", 600, 380, 700, 470),
    ])
    assert on_table.held == 0
    assert plugin.veto(on_table, "pad_in_hand") == (
        "You have not picked up a replacement nose pad yet. Lift one off "
        "the table and hold it."
    )

    lifted = plugin.analyze([
        glasses, _box("nosepad", 500, 400, 520, 420), _box("hand", 470, 370, 560, 460),
    ])
    assert (lifted.held_at_bridge, lifted.held_away) == (0, 1)
    assert plugin.veto(lifted, "pad_in_hand") == ""

    at_bridge = plugin.analyze([
        glasses, _box("nosepad", 180, 130, 200, 150), _box("hand", 150, 110, 260, 190),
    ])
    assert plugin.veto(at_bridge, "pad_in_hand") == ""
    assert plugin.veto(plugin.analyze([]), "pad_in_hand") == ""


def test_prose_names_the_held_pads_detector_class():
    plugin = _nosepad()
    glasses = _box("glasses", 100, 100, 300, 200)

    held = plugin.analyze([
        glasses, _box("nosepad_0", 500, 400, 520, 420), _box("hand", 470, 370, 560, 460),
    ])
    assert "nosepad_0" in held.prose
    assert "weigh it heavily" in held.prose

    seated = plugin.analyze([glasses, _box("nosepad_1", 180, 130, 200, 150)])
    assert "nosepad_1" in seated.prose

    plain = plugin.analyze([
        glasses, _box("nosepad", 500, 400, 520, 420), _box("hand", 470, 370, 560, 460),
    ])
    assert "the detector reads it as" not in plain.prose
    assert "weigh it heavily" not in plain.prose
    assert (held.held_away, plain.held_away) == (1, 1)


def test_seated_prose_is_verbatim():
    plugin = _nosepad()
    seated = plugin.analyze([_box("glasses", 100, 100, 300, 200), _box("nosepad", 180, 130, 200, 150)])
    assert plugin.describe(seated) == (
        "DETECTOR GEOMETRY for the live student image (advisory, computed from "
        "the same boxes):\n"
        "- 1 nose pad box overlaps the glasses and is NOT inside a hand box — consistent with "
        "a pad already fitted to the glasses."
        "\nThese relations are box arithmetic, not ground truth: the detector "
        "misses pads, invents them, and draws boxes larger than the object. Use "
        "them to decide WHICH pad to look at, then confirm the actual state from "
        "the pixels. Do not let a relation here override what you can plainly "
        "see, and do not report a pad as fitted on this basis alone.\n\n"
    )


def test_spare_behind_a_hand_does_not_unseat_a_fitted_pad():
    plugin = _nosepad()
    glasses = _box("glasses", 100, 100, 300, 200)
    frame = plugin.analyze([
        glasses,
        _box("nosepad", 180, 130, 200, 150),   # correctly seated
        _box("nosepad", 480, 320, 500, 340),   # spare on the desk...
        _box("hand", 440, 280, 560, 400),      # ...with a hand in front of it
    ])
    assert frame.seated == 1
    assert (frame.held_at_bridge, frame.held_away) == (0, 1)
    assert frame.pads_on_glasses == 1
    assert plugin.veto(frame, "pad_on_glasses") == ""
    assert "says nothing about whether a pad is already on the glasses" in frame.prose
    assert "still in progress" not in frame.prose


@pytest.mark.parametrize(("gap", "expect_away"), [(48, False), (52, True)])
def test_held_away_boundary_stays_conservative(gap, expect_away):
    # Glasses width 200, so 1px of gap is 0.5%. 48px -> 24%, 52px -> 26%.
    plugin = _nosepad()
    glasses = _box("glasses", 100, 100, 300, 200)
    x1 = 300 + gap
    frame = plugin.analyze([
        glasses,
        _box("nosepad", 180, 130, 200, 150),
        _box("nosepad", x1, 150, x1 + 20, 170),
        _box("hand", x1 - 5, 140, x1 + 60, 190),
    ])
    assert frame.held_away == (1 if expect_away else 0)
    assert frame.pads_on_glasses == (1 if expect_away else 0)


def test_veto_silent_without_evidence():
    plugin = _nosepad()
    faint = plugin.analyze([
        _box("glasses", 100, 100, 300, 200, conf=0.20), _box("nosepad", 180, 130, 200, 150),
    ])
    assert plugin.veto(faint, "pad_on_glasses") == ""
    assert plugin.veto(plugin.analyze([]), "pad_on_glasses") == ""
    seated = plugin.analyze([_box("glasses", 100, 100, 300, 200), _box("nosepad", 180, 130, 200, 150)])
    assert plugin.veto(seated, "unknown_gate") == ""
    assert plugin.veto(seated, "") == ""


# ── nose-pad geometry: temporal voting and coasting ───────────────────────────


_GLASSES = _box("glasses", 100, 100, 300, 200)
_PAD = _box("nosepad", 180, 130, 200, 150)


def test_unkeyed_analyze_stays_stateless():
    plugin = _nosepad()
    for _ in range(6):
        assert plugin.analyze([_GLASSES, _PAD]).seated == 1
        assert plugin.analyze([_GLASSES]).seated == 0
    assert plugin.module._history == {}


def test_one_frame_of_held_still_blocks_the_pass():
    """The discount reads raw as well as voted: the frame that saw the hand wins."""
    plugin = _nosepad()
    hand = _box("hand", 170, 120, 215, 165)
    for _ in range(4):
        plugin.analyze([_GLASSES, _PAD], "held-vote-pid")
    held = plugin.analyze([_GLASSES, _PAD, hand], "held-vote-pid")
    assert held.raw_held_at_bridge == 1
    assert held.held_at_bridge == 0, "the vote must still smooth it away, or this proves nothing"
    assert held.pads_on_glasses == 0
    assert plugin.veto(held, "pad_on_glasses")


def test_flicker_voted_out_persistence_kept():
    plugin = _nosepad()
    for _ in range(4):
        plugin.analyze([_GLASSES], "flicker-pid")
    flicker = plugin.analyze([_GLASSES, _PAD], "flicker-pid")
    assert (flicker.raw_seated, flicker.seated) == (1, 0)
    assert "Not every box above held still" in flicker.prose

    steady = [plugin.analyze([_GLASSES, _PAD], "steady-pid").seated for _ in range(5)]
    assert steady[-1] == 1

    for _ in range(4):
        plugin.analyze([_GLASSES, _PAD], "dropout-pid")
    dropout = plugin.analyze([_GLASSES], "dropout-pid")
    assert (dropout.raw_seated, dropout.seated) == (0, 1)


def test_coast_holds_a_pad_the_median_loses():
    plugin = _nosepad()
    for _ in range(4):
        plugin.analyze([_GLASSES, _PAD], "coast-pid")
    for _ in range(4):
        plugin.analyze([_GLASSES], "coast-pid")
    coasting = plugin.analyze([_GLASSES], "coast-pid")
    assert coasting.raw_seated == 0
    assert coasting.seated == 1
    assert coasting.coasted == ("seated",)
    assert plugin.veto(coasting, "pad_on_glasses") == ""
    assert "no box in THIS frame" in plugin.describe(coasting)


def test_coast_expires_so_a_removal_lands():
    plugin = _nosepad()
    module = plugin.module
    for _ in range(4):
        plugin.analyze([_GLASSES, _PAD], "expire-pid")
    for _ in range(4):
        plugin.analyze([_GLASSES], "expire-pid")
    assert plugin.analyze([_GLASSES], "expire-pid").coasted == ("seated",)

    module._confirmed["expire-pid"] = tuple(
        (at - module._COAST_S - 1.0, count) for at, count in module._confirmed["expire-pid"]
    )
    expired = plugin.analyze([_GLASSES], "expire-pid")
    assert expired.seated == 0
    assert expired.coasted == ()
    assert plugin.veto(expired, "no_pad_on_glasses") == ""


def test_coast_never_invents_a_pad():
    plugin = _nosepad()
    for _ in range(4):
        plugin.analyze([_GLASSES], "invent-pid")
    flicker = plugin.analyze([_GLASSES, _PAD], "invent-pid")
    assert (flicker.raw_seated, flicker.seated) == (1, 0)
    assert flicker.coasted == ()
    for _ in range(3):
        plugin.analyze([_GLASSES], "invent-pid")
        blip = plugin.analyze([_GLASSES, _PAD], "invent-pid")
        assert blip.seated == 0 and blip.coasted == ()


def test_coast_is_per_stream():
    plugin = _nosepad()
    for _ in range(4):
        plugin.analyze([_GLASSES, _PAD], "coast-A")
    for _ in range(4):
        plugin.analyze([_GLASSES], "coast-B")
    intruder = plugin.analyze([_GLASSES], "coast-B")
    assert intruder.seated == 0
    assert intruder.coasted == ()


def test_obscured_bridge_is_reported_as_unknown():
    plugin = _nosepad()
    # Covering the centre band (x 166..234 at _BRIDGE_BAND_FRAC 0.34) and no pad.
    hand = _box("hand", 150, 90, 250, 210)
    for _ in range(4):
        plugin.analyze([_GLASSES, hand], "obscured-pid")
    blocked = plugin.analyze([_GLASSES, hand], "obscured-pid")
    assert blocked.bridge_obscured
    assert "CANNOT say whether a pad is seated" in plugin.describe(blocked)


def test_no_gate_acts_on_an_obscured_bridge():
    plugin = _nosepad()
    hand = _box("hand", 150, 90, 250, 210)
    for _ in range(4):
        plugin.analyze([_GLASSES, hand], "obscured-clear-pid")
    blocked = plugin.analyze([_GLASSES, hand], "obscured-clear-pid")
    assert blocked.bridge_obscured and blocked.seated == 0
    assert plugin.veto(blocked, "no_pad_on_glasses") == ""
    veto = plugin.veto(blocked, "pad_on_glasses")
    assert "Push it into the slot" in veto
    assert "Move your hand away" not in veto


def test_a_pad_box_at_the_bridge_is_not_obscured():
    plugin = _nosepad()
    hand = _box("hand", 150, 90, 250, 210)
    for _ in range(4):
        plugin.analyze([_GLASSES, _PAD, hand], "answered-pid")
    answered = plugin.analyze([_GLASSES, _PAD, hand], "answered-pid")
    assert not answered.bridge_obscured
    assert "still looks like it is in your hand" in plugin.veto(answered, "pad_on_glasses")

    elsewhere = _box("hand", 10, 10, 60, 60)
    for _ in range(4):
        plugin.analyze([_GLASSES, elsewhere], "far-hand-pid")
    assert not plugin.analyze([_GLASSES, elsewhere], "far-hand-pid").bridge_obscured


def test_obscured_is_voted_and_never_coasted():
    plugin = _nosepad()
    hand = _box("hand", 150, 90, 250, 210)
    for _ in range(4):
        plugin.analyze([_GLASSES], "sweep-pid")
    assert not plugin.analyze([_GLASSES, hand], "sweep-pid").bridge_obscured

    for _ in range(5):
        plugin.analyze([_GLASSES, hand], "release-pid")
    assert plugin.analyze([_GLASSES, hand], "release-pid").bridge_obscured
    for _ in range(6):
        plugin.analyze([_GLASSES], "release-pid")
    released = plugin.analyze([_GLASSES], "release-pid")
    assert not released.bridge_obscured
    assert "CANNOT say whether a pad is seated" not in plugin.describe(released)


def test_coasted_survives_the_prose_exit():
    plugin = _nosepad()
    spare = _box("nosepad", 20, 20, 40, 40)
    for _ in range(4):
        plugin.analyze([_GLASSES, _PAD, spare], "prose-coast-pid")
    for _ in range(4):
        plugin.analyze([_GLASSES, spare], "prose-coast-pid")
    coasting = plugin.analyze([_GLASSES, spare], "prose-coast-pid")
    assert coasting.prose
    assert coasting.raw_seated == 0
    assert coasting.seated == 1
    assert "seated" in coasting.coasted


def test_cold_window_falls_back_to_raw():
    plugin = _nosepad()
    first = plugin.analyze([_GLASSES, _PAD], "cold-pid")
    assert first.seated == 1
    assert plugin.veto(first, "pad_on_glasses") == ""


def test_streams_do_not_interleave():
    plugin = _nosepad()
    for _ in range(5):
        with_pad = plugin.analyze([_GLASSES, _PAD], "wearer-a")
        plugin.analyze([_GLASSES], "wearer-b")
    assert with_pad.seated == 1
    intruder = plugin.analyze([_GLASSES, _PAD], "wearer-b")
    assert (intruder.raw_seated, intruder.seated) == (1, 0)


def test_voting_through_the_annotator_uses_the_stream(tmp_path):
    """End to end: annotate_array's stream reaches the plugin's vote window."""
    annotator, task, _ = _annotator(
        tmp_path, [(0, 100.0, 100.0, 300.0, 200.0, 0.95)], hands_enabled=False,
        task_names=_SEATED_NAMES, overlay=_NOSEPAD_OVERLAY, geometry=_nosepad(),
    )
    for _ in range(4):
        _annotate(annotator, _blank(), stream="pid")
    task.rows = _SEATED_ROWS  # a pad appears for one frame only
    flicker = _annotate(annotator, _blank(), stream="pid").geometry
    assert (flicker.raw_seated, flicker.seated) == (1, 0)
    one_off = _annotate(annotator, _blank()).geometry
    assert one_off.seated == 1, "an unkeyed frame is judged alone"


# ── real model (optional) ─────────────────────────────────────────────────────


@pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is None or importlib.util.find_spec("openvino") is None,
    reason="needs ultralytics and openvino",
)
def test_real_nosepad_ir_on_reference_frame(tmp_path):
    profile = load_detector_profiles(DETECTORS_YAML)["nosepad-v5"]
    annotator = FrameAnnotator(
        profile, geometry=_nosepad(), spatial_context=True, artifacts_dir=tmp_path,
    )
    image = load_bgr(REFERENCE_FRAME)
    original = image.copy()
    frame = _annotate(annotator, image)
    assert "glasses" in frame.detections, frame.detections
    assert not np.array_equal(frame.image, original)
    assert frame.geometry.glasses_seen
    artifact = asyncio.run(annotator.annotate_file(str(REFERENCE_FRAME), role="student"))
    assert artifact.applied, artifact.error
    assert Path(artifact.path).is_file()
