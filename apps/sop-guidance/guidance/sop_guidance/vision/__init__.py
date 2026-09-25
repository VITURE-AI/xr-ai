# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detector overlays, geometry plugins and frame conversion for SOP guidance.

Importing this package is cheap: cv2, ultralytics and OpenVINO load lazily on
first use.
"""

from .frames import (
    bgr_to_i420,
    bgr_to_jpeg,
    bgr_to_png,
    bgr_to_rgb24,
    frame_to_bgr,
    load_bgr,
)
from .geometry import GeometryPlugin, LoadedGeometry, NullGeometry, load_geometry_plugin
from .overlay import (
    AnnotatedFrame,
    Detection,
    DetectorProfile,
    FrameAnnotator,
    HandDetectorConfig,
    OverlayArtifact,
    OverlayStyle,
    load_detector_profiles,
    union_coverage,
)

__all__ = [
    "AnnotatedFrame",
    "Detection",
    "DetectorProfile",
    "FrameAnnotator",
    "GeometryPlugin",
    "HandDetectorConfig",
    "LoadedGeometry",
    "NullGeometry",
    "OverlayArtifact",
    "OverlayStyle",
    "bgr_to_i420",
    "bgr_to_jpeg",
    "bgr_to_png",
    "bgr_to_rgb24",
    "frame_to_bgr",
    "load_bgr",
    "load_detector_profiles",
    "load_geometry_plugin",
    "union_coverage",
]
