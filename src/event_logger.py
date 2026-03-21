"""Structured event logging for mandatory and operational logs.

This module writes:
- append-only JSONL events to logs/events.log (mandatory)
- operational logs to logs/app.log (optional but useful)

It can also mirror event records into the database repository.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.repositories import Repository


SUPPORTED_EVENT_TYPES = {"ENTRY", "EXIT", "REGISTERED", "RECOGNIZED"}


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


class EventLoggerError(RuntimeError):
    """Raised when event logger configuration or write fails."""


class EventLogger:
    """Write mandatory event logs and optional app logs.

    Args:
        logs_dir: Base logs directory.
        events_log_filename: File name for mandatory event stream.
        app_log_filename: File name for operational logs.
        source_type: Source marker (e.g., "video" or "rtsp").
        repository: Optional repository used to mirror events into DB.
    """

    def __init__(
        self,
        logs_dir: str = "logs",
        events_log_filename: str = "events.log",
        app_log_filename: str = "app.log",
        source_type: str = "video",
        repository: Repository | None = None,
    ) -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.events_log_path = self.logs_dir / events_log_filename
        self.app_log_path = self.logs_dir / app_log_filename
        self.source_type = str(source_type).strip().lower() or "video"
        self.repository = repository

        # Ensure files exist for predictable hackathon/demo behavior.
        self.events_log_path.touch(exist_ok=True)
        self.app_log_path.touch(exist_ok=True)

        # Configure operational logger once.
        self._logger = logging.getLogger("face_tracker_app")
        self._logger.setLevel(logging.INFO)
        if not any(
            isinstance(handler, logging.FileHandler)
            and Path(getattr(handler, "baseFilename", "")) == self.app_log_path
            for handler in self._logger.handlers
        ):
            handler = logging.FileHandler(self.app_log_path, encoding="utf-8")
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        repository: Repository | None = None,
    ) -> "EventLogger":
        """Build logger from app config."""
        logs_dir = str(config.get("logs_dir", "logs"))
        source_type = str(config.get("input_mode", "video"))
        return cls(
            logs_dir=logs_dir,
            events_log_filename="events.log",
            app_log_filename="app.log",
            source_type=source_type,
            repository=repository,
        )

    def log_event(
        self,
        *,
        event_type: str,
        person_id: str | None = None,
        visit_id: int | None = None,
        track_id: int | None = None,
        frame_index: int | None = None,
        crop_path: str | None = None,
        timestamp: str | None = None,
        source_type: str | None = None,
        meta: dict[str, Any] | None = None,
        mirror_to_db: bool = True,
    ) -> dict[str, Any]:
        """Write one event line to mandatory events.log (JSONL).

        Returns the normalized event payload that was written.
        """
        normalized_type = str(event_type).strip().upper()
        if normalized_type not in SUPPORTED_EVENT_TYPES:
            raise EventLoggerError(
                f"Unsupported event_type '{event_type}'. "
                f"Allowed: {sorted(SUPPORTED_EVENT_TYPES)}"
            )

        ts = timestamp or utc_now_iso()
        payload: dict[str, Any] = {
            "timestamp": ts,
            "event_type": normalized_type,
            "person_id": person_id,
            "visit_id": visit_id,
            "track_id": track_id,
            "frame_index": frame_index,
            "crop_path": crop_path,
            "source_type": (source_type or self.source_type),
            "meta": meta or {},
        }

        # Append-only JSONL write.
        try:
            with self.events_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except OSError as exc:
            raise EventLoggerError(f"Failed writing events log: {exc}") from exc

        # Optional DB mirror keeps file and database aligned.
        if mirror_to_db and self.repository is not None:
            try:
                self.repository.write_event(
                    event_type=normalized_type,
                    person_id=person_id,
                    visit_id=visit_id,
                    track_id=track_id,
                    frame_index=frame_index,
                    event_time=ts,
                    meta={
                        **(meta or {}),
                        "crop_path": crop_path,
                        "source_type": payload["source_type"],
                    },
                )
            except Exception as exc:  # noqa: BLE001
                # Keep pipeline alive; failure is captured in app.log.
                self._logger.exception("Failed to mirror event to DB: %s", exc)

        self._logger.info(
            "event=%s person_id=%s visit_id=%s track_id=%s frame_index=%s crop_path=%s",
            normalized_type,
            person_id,
            visit_id,
            track_id,
            frame_index,
            crop_path,
        )
        return payload

    def log_entry(
        self,
        *,
        person_id: str,
        visit_id: int | None = None,
        track_id: int | None = None,
        frame_index: int | None = None,
        crop_path: str | None = None,
        timestamp: str | None = None,
        meta: dict[str, Any] | None = None,
        mirror_to_db: bool = True,
    ) -> dict[str, Any]:
        """Convenience wrapper for ENTRY event."""
        return self.log_event(
            event_type="ENTRY",
            person_id=person_id,
            visit_id=visit_id,
            track_id=track_id,
            frame_index=frame_index,
            crop_path=crop_path,
            timestamp=timestamp,
            meta=meta,
            mirror_to_db=mirror_to_db,
        )

    def log_exit(
        self,
        *,
        person_id: str,
        visit_id: int | None = None,
        track_id: int | None = None,
        frame_index: int | None = None,
        crop_path: str | None = None,
        timestamp: str | None = None,
        meta: dict[str, Any] | None = None,
        mirror_to_db: bool = True,
    ) -> dict[str, Any]:
        """Convenience wrapper for EXIT event."""
        return self.log_event(
            event_type="EXIT",
            person_id=person_id,
            visit_id=visit_id,
            track_id=track_id,
            frame_index=frame_index,
            crop_path=crop_path,
            timestamp=timestamp,
            meta=meta,
            mirror_to_db=mirror_to_db,
        )

    def log_registered(
        self,
        *,
        person_id: str,
        track_id: int | None = None,
        frame_index: int | None = None,
        timestamp: str | None = None,
        meta: dict[str, Any] | None = None,
        mirror_to_db: bool = True,
    ) -> dict[str, Any]:
        """Convenience wrapper for REGISTERED event."""
        return self.log_event(
            event_type="REGISTERED",
            person_id=person_id,
            visit_id=None,
            track_id=track_id,
            frame_index=frame_index,
            crop_path=None,
            timestamp=timestamp,
            meta=meta,
            mirror_to_db=mirror_to_db,
        )

    def log_recognized(
        self,
        *,
        person_id: str,
        visit_id: int | None = None,
        track_id: int | None = None,
        frame_index: int | None = None,
        similarity: float | None = None,
        timestamp: str | None = None,
        meta: dict[str, Any] | None = None,
        mirror_to_db: bool = True,
    ) -> dict[str, Any]:
        """Convenience wrapper for RECOGNIZED event."""
        payload_meta = dict(meta or {})
        if similarity is not None:
            payload_meta["similarity"] = float(similarity)
        return self.log_event(
            event_type="RECOGNIZED",
            person_id=person_id,
            visit_id=visit_id,
            track_id=track_id,
            frame_index=frame_index,
            crop_path=None,
            timestamp=timestamp,
            meta=payload_meta,
            mirror_to_db=mirror_to_db,
        )

    def log_app_info(self, message: str, **extra: Any) -> None:
        """Write informational operational logs to app.log."""
        if extra:
            self._logger.info("%s | %s", message, json.dumps(extra, ensure_ascii=True))
        else:
            self._logger.info("%s", message)

    def log_app_warning(self, message: str, **extra: Any) -> None:
        """Write warning operational logs to app.log."""
        if extra:
            self._logger.warning("%s | %s", message, json.dumps(extra, ensure_ascii=True))
        else:
            self._logger.warning("%s", message)

    def log_app_error(self, message: str, **extra: Any) -> None:
        """Write error operational logs to app.log."""
        if extra:
            self._logger.error("%s | %s", message, json.dumps(extra, ensure_ascii=True))
        else:
            self._logger.error("%s", message)
