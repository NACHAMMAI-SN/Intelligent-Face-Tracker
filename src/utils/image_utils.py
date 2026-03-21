"""Minimal image helpers used across the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def clip_bbox_to_frame(
    bbox: list[int] | tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Clip bbox coordinates to image boundaries."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width))
    y2 = max(0, min(y2, frame_height))
    return x1, y1, x2, y2


def crop_face_safe(
    frame: np.ndarray,
    bbox: list[int] | tuple[int, int, int, int],
) -> np.ndarray | None:
    """Safely crop face ROI from frame; return None when invalid."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None
    if frame.ndim != 3 or frame.shape[2] != 3:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clip_bbox_to_frame(bbox, w, h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def estimate_blur_score(image: np.ndarray) -> float:
    """Estimate sharpness using Laplacian variance (higher is sharper)."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_low_quality_face(
    face_crop: np.ndarray,
    min_size: int = 32,
    blur_threshold: float = 15.0,
) -> bool:
    """Return True if crop is too small or too blurry for reliable embedding."""
    if face_crop is None or not isinstance(face_crop, np.ndarray) or face_crop.size == 0:
        return True
    h, w = face_crop.shape[:2]
    if h < min_size or w < min_size:
        return True
    return estimate_blur_score(face_crop) < blur_threshold


def save_image(path: str | Path, image: np.ndarray) -> dict[str, Any]:
    """Save image to disk and return simple metadata."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(output), image)
    if not ok:
        raise RuntimeError(f"Failed to save image: {output}")

    h, w = image.shape[:2]
    return {"path": str(output.as_posix()), "width": int(w), "height": int(h)}
