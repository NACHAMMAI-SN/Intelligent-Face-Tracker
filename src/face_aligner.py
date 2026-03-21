"""Lightweight face crop preprocessing for embedding extraction.

This module intentionally avoids heavy landmark alignment for hackathon speed.
It provides a clean, optional preprocessing step:
- safe bbox clipping
- optional padding around face box
- square resize
- pixel normalization

The output is designed to be consumed directly by embedder.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


class FaceAlignerError(ValueError):
    """Raised when face preprocessing inputs are invalid."""


@dataclass
class FaceAlignerConfig:
    """Configuration for lightweight face preprocessing."""

    target_size: int = 112
    padding_ratio: float = 0.15
    normalize: bool = True
    normalization_mode: str = "zero_one"  # "zero_one" or "minus_one_one"


class FaceAligner:
    """Minimal face preprocessor to stabilize embeddings.

    Notes:
        - This is not geometric landmark alignment.
        - It is intentionally lightweight and robust for hackathon constraints.
    """

    def __init__(self, config: FaceAlignerConfig | None = None) -> None:
        self.config = config or FaceAlignerConfig()
        self._validate_config(self.config)

    @classmethod
    def from_config(cls, app_config: dict[str, Any]) -> "FaceAligner":
        """Build aligner from app config with practical defaults.

        Optional config keys:
            - face_align_target_size
            - face_align_padding_ratio
            - face_align_normalize
            - face_align_normalization_mode
        """
        cfg = FaceAlignerConfig(
            target_size=int(app_config.get("face_align_target_size", 112)),
            padding_ratio=float(app_config.get("face_align_padding_ratio", 0.15)),
            normalize=bool(app_config.get("face_align_normalize", True)),
            normalization_mode=str(app_config.get("face_align_normalization_mode", "zero_one")),
        )
        return cls(cfg)

    def preprocess(self, frame: np.ndarray, bbox: list[int] | tuple[int, int, int, int]) -> np.ndarray:
        """Extract and preprocess a face crop.

        Args:
            frame: OpenCV BGR frame (H, W, 3).
            bbox: Face bbox in [x1, y1, x2, y2] format.

        Returns:
            Preprocessed face as numpy array shaped (target_size, target_size, 3).
            - dtype float32 when normalize=True
            - dtype uint8 when normalize=False

        Raises:
            FaceAlignerError: For invalid inputs or empty crop.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise FaceAlignerError("frame must be a valid numpy array.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise FaceAlignerError("frame must have shape (H, W, 3).")

        x1, y1, x2, y2 = self._parse_bbox(bbox)
        x1, y1, x2, y2 = self._expand_and_clip_bbox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            padding_ratio=self.config.padding_ratio,
        )

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            raise FaceAlignerError("Face crop is empty after bbox clipping.")

        # Mild denoise/smoothing can help unstable webcam streams.
        # Keep it very light to avoid latency impact.
        crop = cv2.GaussianBlur(crop, (3, 3), 0)

        # Resize to square target expected by most face embedders.
        resized = cv2.resize(
            crop,
            (self.config.target_size, self.config.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        if not self.config.normalize:
            return resized

        return self._normalize(resized, mode=self.config.normalization_mode)

    def preprocess_bgr_to_rgb(
        self, frame: np.ndarray, bbox: list[int] | tuple[int, int, int, int]
    ) -> np.ndarray:
        """Convenience helper: preprocess then convert BGR output to RGB."""
        face = self.preprocess(frame=frame, bbox=bbox)
        if face.dtype != np.uint8:
            # For normalized arrays, channel swap still applies.
            return face[..., ::-1]
        return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _parse_bbox(bbox: list[int] | tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise FaceAlignerError("bbox must be [x1, y1, x2, y2].")
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            raise FaceAlignerError("bbox is invalid: x2 must be > x1 and y2 must be > y1.")
        return x1, y1, x2, y2

    @staticmethod
    def _expand_and_clip_bbox(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        frame_width: int,
        frame_height: int,
        padding_ratio: float,
    ) -> tuple[int, int, int, int]:
        """Add context around face and clip coordinates to frame bounds."""
        width = x2 - x1
        height = y2 - y1
        pad_x = int(width * padding_ratio)
        pad_y = int(height * padding_ratio)

        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(frame_width, x2 + pad_x)
        ny2 = min(frame_height, y2 + pad_y)

        # Final guard to avoid zero-area crops due to extreme clipping.
        if nx2 <= nx1:
            nx2 = min(frame_width, nx1 + 1)
        if ny2 <= ny1:
            ny2 = min(frame_height, ny1 + 1)
        return nx1, ny1, nx2, ny2

    @staticmethod
    def _normalize(image_bgr: np.ndarray, mode: str) -> np.ndarray:
        """Normalize pixel values for embedding models."""
        image = image_bgr.astype(np.float32)

        normalized_mode = mode.strip().lower()
        if normalized_mode == "zero_one":
            return image / 255.0
        if normalized_mode == "minus_one_one":
            return (image / 127.5) - 1.0
        raise FaceAlignerError(
            "normalization_mode must be 'zero_one' or 'minus_one_one'."
        )

    @staticmethod
    def _validate_config(config: FaceAlignerConfig) -> None:
        if config.target_size <= 0:
            raise FaceAlignerError("target_size must be > 0.")
        if not (0.0 <= config.padding_ratio <= 1.0):
            raise FaceAlignerError("padding_ratio must be between 0.0 and 1.0.")
        mode = config.normalization_mode.strip().lower()
        if mode not in {"zero_one", "minus_one_one"}:
            raise FaceAlignerError(
                "normalization_mode must be 'zero_one' or 'minus_one_one'."
            )
        config.normalization_mode = mode
