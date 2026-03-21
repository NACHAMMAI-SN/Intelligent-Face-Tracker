"""Robust video input handler for local file and RTSP modes."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np


class VideoSourceError(RuntimeError):
    """Raised when the video source cannot be opened or configured correctly."""


class VideoSource:
    """OpenCV-backed source supporting local video and RTSP.

    Expected config fields:
        - input_mode: "video" | "rtsp"
        - video_path: str
        - rtsp_url: str
        - reconnect_enabled: bool
        - reconnect_delay_seconds: int | float
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.input_mode = str(config.get("input_mode", "video")).strip().lower()
        self.video_path = str(config.get("video_path", "")).strip()
        self.rtsp_url = str(config.get("rtsp_url", "")).strip()
        self.reconnect_enabled = bool(config.get("reconnect_enabled", True))
        self.reconnect_delay_seconds = max(0.0, float(config.get("reconnect_delay_seconds", 5)))

        # Kept as public attributes for easier debugging.
        self.cap: cv2.VideoCapture | None = None
        self.frame_index: int = 0

        self._opened_once = False
        self._fps: float = 0.0
        self._next_reconnect_at: float = 0.0

    def open(self) -> None:
        """Open the configured source mode."""
        if self.input_mode not in {"video", "rtsp"}:
            raise VideoSourceError("input_mode must be either 'video' or 'rtsp'.")

        if self.input_mode == "video":
            if not self.video_path:
                raise VideoSourceError("video_path is required when input_mode='video'.")
            print(f"Opening video file... {self.video_path}")
            self._open_capture(self.video_path)
            self._fps = float(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap is not None else 0.0
            self.frame_index = 0
            self._opened_once = True
            return

        if self.input_mode == "rtsp":
            if not self.rtsp_url:
                raise VideoSourceError("rtsp_url is required when input_mode='rtsp'.")
            print("Opening RTSP stream...")
            self._open_capture(self.rtsp_url)
            self._fps = 0.0  # RTSP FPS is often unreliable; use wall clock for timestamps.
            self.frame_index = 0
            self._opened_once = True
            self._next_reconnect_at = 0.0
            return

    def is_opened(self) -> bool:
        """Return True when capture is currently opened."""
        return self.cap is not None and self.cap.isOpened()

    def read(self) -> tuple[np.ndarray | None, int | None, float | None]:
        """Read one frame and return (frame, frame_index, timestamp_seconds)."""
        if not self.is_opened():
            if self.input_mode == "rtsp" and self.reconnect_enabled and self._opened_once:
                if not self._attempt_reconnect():
                    return None, None, None
            else:
                return None, None, None

        assert self.cap is not None  # guarded by is_opened() above
        ok, frame = self.cap.read()

        if ok and frame is not None:
            current_index = self.frame_index
            self.frame_index += 1
            return frame, current_index, self._compute_timestamp_seconds(current_index)

        if self.input_mode == "video":
            # Clean EOF for local file mode.
            return None, None, None

        if self.input_mode == "rtsp":
            if self.reconnect_enabled:
                self._attempt_reconnect()
                return None, None, None
            else:
                return None, None, None

        return None, None, None

    def release(self) -> None:
        """Safely release capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _open_capture(self, source: str) -> None:
        """Create and validate a VideoCapture."""
        self.release()

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise VideoSourceError(f"Failed to open source: {source}")
        self.cap = cap

    def _attempt_reconnect(self) -> bool:
        """Attempt one RTSP reconnect without blocking pipeline polling."""
        if self.input_mode != "rtsp":
            return False

        # Throttle reconnect attempts using a monotonic clock.
        now = time.monotonic()
        if now < self._next_reconnect_at:
            return False

        print("RTSP reconnecting...")
        self._next_reconnect_at = now + self.reconnect_delay_seconds

        try:
            self._open_capture(self.rtsp_url)
            # Do not consume a frame here; let read() fetch the next frame.
            print("RTSP reconnected successfully")
            return True
        except VideoSourceError:
            return False

    def _compute_timestamp_seconds(self, frame_index: int) -> float:
        """Compute frame timestamp in seconds."""
        if self.input_mode == "video" and self._fps > 0:
            return frame_index / self._fps
        return time.time()
