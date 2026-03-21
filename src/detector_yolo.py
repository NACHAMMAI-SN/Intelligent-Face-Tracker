"""YOLO-based face detection module.

This wrapper is intentionally face-focused and practical for hackathon use:
- model is initialized once
- detect(frame) returns normalized, tracker-friendly face detections
- confidence threshold and model path are configurable
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


class FaceDetectorError(RuntimeError):
    """Raised when detector setup or inference fails."""


class YOLOFaceDetector:
    """Face detector backed by a YOLO model.

    Expected config keys:
        - detector_model_path: str
        - detection_confidence: float

    Notes:
        - The model is expected to be a face detector.
        - We still guard class filtering with an optional face class id set
          in case the provided model has multiple classes.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.45,
        face_class_ids: set[int] | None = None,
    ) -> None:
        if not model_path or not str(model_path).strip():
            raise FaceDetectorError("model_path cannot be empty.")
        if not (0.0 <= float(confidence_threshold) <= 1.0):
            raise FaceDetectorError("confidence_threshold must be between 0.0 and 1.0.")

        self.model_path = str(model_path).strip()
        self.confidence_threshold = float(confidence_threshold)

        # Optional class guard: for pure face models this is usually {0}.
        self.face_class_ids = face_class_ids if face_class_ids is not None else {0}

        self._model = self._load_model(self.model_path)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "YOLOFaceDetector":
        """Construct detector from app config dictionary."""
        model_path = str(config.get("detector_model_path", "")).strip()
        confidence_threshold = float(config.get("detection_confidence", 0.45))
        return cls(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            face_class_ids={0},
        )

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run face detection on a frame.

        Args:
            frame: BGR image as numpy array (OpenCV format).

        Returns:
            A list of face detections. Each item is:
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "class_id": int,
                "label": "face",
                "area": int,
                "center": [cx, cy]
            }

        Raises:
            FaceDetectorError: If inference fails or frame is invalid.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise FaceDetectorError("detect() expects a valid numpy frame.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise FaceDetectorError("detect() expects a color frame with shape (H, W, 3).")

        try:
            results = self._model.predict(
                source=frame,
                conf=self.confidence_threshold,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - runtime/model errors
            raise FaceDetectorError(f"YOLO inference failed: {exc}") from exc

        detections: list[dict[str, Any]] = []
        frame_h, frame_w = frame.shape[:2]

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                class_id = int(box.cls[0].item()) if box.cls is not None else 0

                # Keep this module explicitly face-oriented.
                if self.face_class_ids and class_id not in self.face_class_ids:
                    continue
                if conf < self.confidence_threshold:
                    continue

                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = self._clip_bbox(xyxy, frame_w, frame_h)

                if x2 <= x1 or y2 <= y1:
                    continue

                width = x2 - x1
                height = y2 - y1
                area = width * height
                center_x = x1 + width // 2
                center_y = y1 + height // 2

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(conf, 4),
                        "class_id": class_id,
                        "label": "face",
                        "area": area,
                        "center": [center_x, center_y],
                    }
                )

        # Highest confidence first is often easier for downstream logic.
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    @staticmethod
    def _clip_bbox(xyxy: list[float], frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Clip floating-point bbox coordinates to valid integer image bounds."""
        x1 = max(0, min(int(round(xyxy[0])), frame_w - 1))
        y1 = max(0, min(int(round(xyxy[1])), frame_h - 1))
        x2 = max(0, min(int(round(xyxy[2])), frame_w - 1))
        y2 = max(0, min(int(round(xyxy[3])), frame_h - 1))
        return x1, y1, x2, y2

    @staticmethod
    def _load_model(model_path: str) -> YOLO:
        """Load YOLO model once and validate path shape early."""
        path = Path(model_path)
        if not path.exists():
            raise FaceDetectorError(f"YOLO model file not found: {model_path}")
        if not path.is_file():
            raise FaceDetectorError(f"YOLO model path is not a file: {model_path}")

        try:
            return YOLO(model_path)
        except Exception as exc:  # pragma: no cover - runtime/model errors
            raise FaceDetectorError(f"Failed to initialize YOLO model: {exc}") from exc
