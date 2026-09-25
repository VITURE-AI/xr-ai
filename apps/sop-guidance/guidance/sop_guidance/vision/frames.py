# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pixel conversion between hub frame payloads, BGR arrays and encoded images.

cv2 does these in SIMD; a numpy float path is far too slow to run at video
rate. cv2 is imported inside each function so importing this module stays
cheap for code that never touches pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from xr_ai_hub import PixelFormat


def frame_to_bgr(data: bytes, width: int, height: int, fmt: PixelFormat) -> np.ndarray:
    """Convert a hub ``FrameData`` payload into an HxWx3 uint8 BGR array."""
    import cv2
    import numpy as np
    from xr_ai_hub import PixelFormat

    arr = np.frombuffer(data, dtype=np.uint8)
    if fmt == PixelFormat.I420:
        return cv2.cvtColor(arr.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_I420)
    if fmt == PixelFormat.NV12:
        return cv2.cvtColor(arr.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV12)
    if fmt == PixelFormat.RGB24:
        return cv2.cvtColor(arr.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    if fmt == PixelFormat.RGBA:
        return cv2.cvtColor(arr.reshape(height, width, 4), cv2.COLOR_RGBA2BGR)
    if fmt == PixelFormat.BGRA:
        return cv2.cvtColor(arr.reshape(height, width, 4), cv2.COLOR_BGRA2BGR)
    raise ValueError(f"unsupported live frame pixel format: {fmt!r}")


def bgr_to_jpeg(image: np.ndarray, *, max_width: int = 960, quality: int = 80) -> bytes:
    """Encode a BGR array as JPEG, downscaled to *max_width*.

    Meant for frames that exist to be looked at (debug clips, previews), not
    inferred on, so full camera resolution buys nothing. Returns b"" rather
    than raising: a debug frame is never worth taking a video loop down for.
    """
    import cv2

    try:
        height, width = image.shape[:2]
        if width > max_width:
            scale = max_width / float(width)
            image = cv2.resize(
                image, (max_width, max(2, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else b""
    except Exception:
        return b""


def bgr_to_png(image: np.ndarray) -> bytes:
    """Encode a BGR array as a lossless PNG. Raises ``ValueError`` on failure."""
    import cv2

    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("OpenCV could not encode the frame as PNG")
    return buf.tobytes()


def bgr_to_i420(image: np.ndarray) -> bytes:
    """Pack a BGR array as planar I420.

    I420 carries 1.5 bytes per pixel against RGB24's 3, and the hub hands the
    buffer straight to LiveKit, which wants YUV anyway -- so this halves the IPC
    payload and skips a conversion rather than costing one. Width and height
    must be even.
    """
    import cv2

    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).tobytes()


def bgr_to_rgb24(image: np.ndarray) -> bytes:
    """Pack a BGR array for LiveKit's RGB24 input buffer."""
    import cv2

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes()


def load_bgr(path: str | Path) -> np.ndarray:
    """Read an image file into an HxWx3 uint8 BGR array.

    Raises ``ValueError`` when the file is missing or cannot be decoded; cv2
    itself answers ``None`` for both, which is too easy to pass along.
    """
    import cv2

    image = cv2.imread(str(Path(path).expanduser()), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read an image from {path}")
    return image
