"""Face crop saving utilities for ENTRY and EXIT events.

Features:
- event-based + date-based folder structure
- unique, non-overwriting filenames
- automatic directory creation
- structured metadata return for logging/database use
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np


class CropSaverError(RuntimeError):
    """Raised when crop saving fails."""


@dataclass
class CropSaveResult:
    """Result payload after saving one crop."""

    event_type: str
    person_id: str
    saved_path: str
    filename: str
    date_folder: str
    timestamp_iso: str
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to dictionary."""
        return {
            "event_type": self.event_type,
            "person_id": self.person_id,
            "saved_path": self.saved_path,
            "filename": self.filename,
            "date_folder": self.date_folder,
            "timestamp_iso": self.timestamp_iso,
            "width": self.width,
            "height": self.height,
        }


class CropSaver:
    """Save face crops to structured folders for ENTRY/EXIT events."""

    def __init__(self, entry_crop_dir: str, exit_crop_dir: str, image_ext: str = ".jpg") -> None:
        self.entry_crop_dir = Path(entry_crop_dir)
        self.exit_crop_dir = Path(exit_crop_dir)
        self.image_ext = image_ext if image_ext.startswith(".") else f".{image_ext}"

        # Create base directories once.
        self.entry_crop_dir.mkdir(parents=True, exist_ok=True)
        self.exit_crop_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CropSaver":
        """Construct CropSaver from app config."""
        return cls(
            entry_crop_dir=str(config.get("entry_crop_dir", "data/faces/entry")),
            exit_crop_dir=str(config.get("exit_crop_dir", "data/faces/exit")),
            image_ext=str(config.get("crop_image_ext", ".jpg")),
        )

    def save_entry_crop(
        self,
        face_crop: np.ndarray,
        person_id: str,
        event_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Save ENTRY crop and return metadata."""
        result = self._save(face_crop=face_crop, person_id=person_id, event_type="ENTRY", event_time=event_time)
        return result.to_dict()

    def save_exit_crop(
        self,
        face_crop: np.ndarray,
        person_id: str,
        event_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Save EXIT crop and return metadata."""
        result = self._save(face_crop=face_crop, person_id=person_id, event_type="EXIT", event_time=event_time)
        return result.to_dict()

    def _save(
        self,
        *,
        face_crop: np.ndarray,
        person_id: str,
        event_type: str,
        event_time: datetime | None,
    ) -> CropSaveResult:
        """Internal save method shared by ENTRY and EXIT."""
        if face_crop is None or not isinstance(face_crop, np.ndarray) or face_crop.size == 0:
            raise CropSaverError("face_crop must be a non-empty numpy array.")
        if face_crop.ndim not in (2, 3):
            raise CropSaverError("face_crop must be 2D or 3D image array.")

        pid = self._sanitize_person_id(person_id)
        if not pid:
            raise CropSaverError("person_id cannot be empty.")

        normalized_event = event_type.strip().upper()
        if normalized_event not in {"ENTRY", "EXIT"}:
            raise CropSaverError("event_type must be 'ENTRY' or 'EXIT'.")

        ts = (event_time or datetime.now(timezone.utc)).astimezone(timezone.utc)
        date_folder = ts.strftime("%Y-%m-%d")
        timestamp_part = ts.strftime("%Y%m%dT%H%M%S%fZ")

        base_dir = self.entry_crop_dir if normalized_event == "ENTRY" else self.exit_crop_dir
        target_dir = base_dir / date_folder
        target_dir.mkdir(parents=True, exist_ok=True)

        # Add random suffix to guarantee uniqueness and avoid overwrites.
        unique_suffix = uuid4().hex[:8]
        filename = f"{timestamp_part}_{pid}_{normalized_event.lower()}_{unique_suffix}{self.image_ext}"
        file_path = target_dir / filename

        # Ensure BGR/gray writable image format.
        image_to_write = self._prepare_image(face_crop)
        ok = cv2.imwrite(str(file_path), image_to_write)
        if not ok:
            raise CropSaverError(f"Failed to write crop image: {file_path}")

        h, w = image_to_write.shape[:2]
        return CropSaveResult(
            event_type=normalized_event,
            person_id=pid,
            saved_path=str(file_path.as_posix()),
            filename=filename,
            date_folder=date_folder,
            timestamp_iso=ts.isoformat(),
            width=int(w),
            height=int(h),
        )

    @staticmethod
    def _sanitize_person_id(person_id: str) -> str:
        """Keep filename-safe person id while preserving readability."""
        text = str(person_id).strip()
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
        return safe.strip("_")

    @staticmethod
    def _prepare_image(face_crop: np.ndarray) -> np.ndarray:
        """Convert crop to a valid uint8 image for OpenCV write."""
        img = face_crop
        if img.dtype != np.uint8:
            # Assume normalized float image ranges [0,1] or [-1,1] when not uint8.
            min_v = float(np.min(img))
            max_v = float(np.max(img))
            if min_v >= 0.0 and max_v <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = (((img + 1.0) * 127.5).clip(0, 255)).astype(np.uint8)
        return img
