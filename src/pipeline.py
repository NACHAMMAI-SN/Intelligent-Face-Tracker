"""End-to-end face tracking pipeline orchestration.

This pipeline coordinates all modules in a tracker-first design:
- source -> detect -> track -> embed/recognize/register -> visit -> logging

It supports both local video and RTSP modes via the same VideoSource interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.auto_register import AutoRegistrar
from src.config_loader import load_config
from src.crop_saver import CropSaver
from src.db import init_database
from src.detector_yolo import YOLOFaceDetector
from src.embedder import FaceEmbedder
from src.event_logger import EventLogger
from src.recognizer import FaceRecognizer
from src.repositories import Repository
from src.resource_snapshot import snapshot_for_app_log
from src.tracker import FaceTracker
from src.utils.image_utils import crop_face_safe
from src.video_source import VideoSource
from src.visit_manager import VisitManager


@dataclass
class PipelineStats:
    """Runtime counters useful for debugging/demo visibility."""

    frames_seen: int = 0
    frames_processed: int = 0
    detections_total: int = 0
    recognized_total: int = 0
    registered_total: int = 0
    identity_reuse_bindings_total: int = 0
    entries_total: int = 0
    exits_total: int = 0
    unique_visitors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_seen": self.frames_seen,
            "frames_processed": self.frames_processed,
            "detections_total": self.detections_total,
            "recognized_total": self.recognized_total,
            "registered_total": self.registered_total,
            "identity_reuse_bindings_total": self.identity_reuse_bindings_total,
            "entries_total": self.entries_total,
            "exits_total": self.exits_total,
            "unique_visitors": self.unique_visitors,
        }


class Pipeline:
    """Main orchestrator for detection, tracking, recognition, visits, and logging."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        video_source: VideoSource,
        detector: YOLOFaceDetector,
        tracker: FaceTracker,
        embedder: FaceEmbedder | None,
        recognizer: FaceRecognizer | None,
        auto_register: AutoRegistrar | None,
        visit_manager: VisitManager | None,
        crop_saver: CropSaver,
        repository: Repository,
        event_logger: EventLogger,
    ) -> None:
        self.config = config
        self.video_source = video_source
        self.detector = detector
        self.tracker = tracker
        self.embedder = embedder
        self.recognizer = recognizer
        self.auto_register = auto_register
        self.visit_manager = visit_manager
        self.crop_saver = crop_saver
        self.repository = repository
        self.event_logger = event_logger
        self.recognition_enabled = bool(config.get("recognition_enabled", True))

        self.detection_frame_skip = int(config.get("detection_frame_skip", 1))
        self.recognition_interval = int(config.get("recognition_interval", 5))

        # Required runtime maps (small, in-memory, hackathon-friendly).
        self.track_to_person: dict[int, str] = {}
        self.track_last_embedding_frame: dict[int, int] = {}
        self.track_last_face_crop: dict[int, np.ndarray] = {}
        self.person_last_face_crop: dict[str, np.ndarray] = {}
        self.track_last_reason_log_frame: dict[int, int] = {}

        self.stats = PipelineStats(unique_visitors=self.repository.get_unique_visitor_count())

    @classmethod
    def from_config(cls, config_path: str = "config.json") -> "Pipeline":
        """Build full pipeline graph from config file."""
        config = load_config(config_path)

        # DB schema initialization before repositories are used.
        init_database(config["db_path"])
        repository = Repository(config["db_path"])

        # Component construction.
        video_source = VideoSource(config)
        detector = YOLOFaceDetector.from_config(config)
        tracker = FaceTracker.from_config(config)
        recognition_enabled = bool(config.get("recognition_enabled", True))
        embedder: FaceEmbedder | None = None
        recognizer: FaceRecognizer | None = None
        auto_register: AutoRegistrar | None = None
        visit_manager: VisitManager | None = None
        if recognition_enabled:
            embedder = FaceEmbedder.from_config(config)
            recognizer = FaceRecognizer.from_config(config)
            auto_register = AutoRegistrar.from_config(config, repository=repository)
            visit_manager = VisitManager.from_config(config, repository=repository)
        else:
            print("Running in detection/tracking-only mode (recognition disabled).")
        crop_saver = CropSaver.from_config(config)
        event_logger = EventLogger.from_config(config, repository=repository)

        # Load known identities into recognizer gallery at startup.
        if recognizer is not None:
            known = repository.get_all_known_embeddings()
            for person_id, embeddings in known.items():
                for embedding in embeddings:
                    try:
                        recognizer.add_identity(person_id, embedding)
                    except ValueError:
                        continue

        return cls(
            config=config,
            video_source=video_source,
            detector=detector,
            tracker=tracker,
            embedder=embedder,
            recognizer=recognizer,
            auto_register=auto_register,
            visit_manager=visit_manager,
            crop_saver=crop_saver,
            repository=repository,
            event_logger=event_logger,
        )

    def run(self) -> dict[str, Any]:
        """Run pipeline until end-of-stream or interruption."""
        self.video_source.open()
        self.event_logger.log_app_info("Pipeline started", input_mode=self.config.get("input_mode"))
        self.event_logger.log_app_info("Resource usage", **snapshot_for_app_log())
        input_mode = str(self.config.get("input_mode", "video")).lower()
        empty_read_streak = 0

        # Baseline persisted ENTRY/EXIT counts so final stats reflect this run only.
        stats_baseline_entry = self.repository.get_event_count_by_type("ENTRY")
        stats_baseline_exit = self.repository.get_event_count_by_type("EXIT")

        try:
            while True:
                frame, frame_index, _ = self.video_source.read()
                if frame is None or frame_index is None:
                    # EOF for local video, or temporary RTSP failure.
                    # For RTSP, VideoSource already attempts reconnect internally.
                    if input_mode == "video":
                        break
                    empty_read_streak += 1
                    # Lightweight visibility for RTSP reconnect/empty reads.
                    if empty_read_streak in {1, 30} or empty_read_streak % 120 == 0:
                        self.event_logger.log_app_warning(
                            "RTSP source returned empty frame",
                            empty_read_streak=empty_read_streak,
                            reconnect_enabled=bool(self.config.get("reconnect_enabled", True)),
                            reconnect_delay_seconds=float(self.config.get("reconnect_delay_seconds", 5)),
                        )
                    continue

                if empty_read_streak > 0 and input_mode == "rtsp":
                    self.event_logger.log_app_info(
                        "RTSP stream recovered",
                        empty_read_streak=empty_read_streak,
                    )
                empty_read_streak = 0
                self.stats.frames_seen += 1

                # Apply frame skip for expensive stages.
                if frame_index % self.detection_frame_skip != 0:
                    continue
                self.stats.frames_processed += 1

                detections = self.detector.detect(frame)
                self.stats.detections_total += len(detections)

                track_result = self.tracker.update(detections=detections, frame_index=frame_index)
                active_tracks = track_result.get("active_tracks", [])
                dead_tracks = track_result.get("dead_tracks", [])

                currently_seen_person_ids: set[str] = set()

                for track in active_tracks:
                    track_id = int(track["track_id"])
                    bbox = track["bbox"]
                    det_conf = float(track.get("confidence", 0.0))
                    track_age_frames = int(frame_index - int(track.get("created_frame", frame_index)) + 1)
                    track_hit_count = int(track.get("hit_count", 1))

                    face_crop = crop_face_safe(frame, bbox)
                    if face_crop is None:
                        self._maybe_log_track_reason(
                            track_id=track_id,
                            frame_index=frame_index,
                            reason="invalid_face_crop",
                            detail={
                                "bbox": bbox,
                                "detection_confidence": det_conf,
                            },
                        )
                        continue
                    crop_shape = [int(v) for v in face_crop.shape]
                    self.track_last_face_crop[track_id] = face_crop
                    if not self.recognition_enabled:
                        continue

                    known_person = self.track_to_person.get(track_id)
                    if known_person:
                        currently_seen_person_ids.add(known_person)
                        self.person_last_face_crop[known_person] = face_crop
                        self.visit_manager.handle_confirmed_sighting(
                            person_id=known_person,
                            seen_at=datetime.now(timezone.utc),
                            track_id=track_id,
                            frame_index=frame_index,
                        )

                    # Recognition runs at configured intervals, not every frame.
                    should_run_recognition = self._should_run_recognition(track_id, frame_index)
                    if not should_run_recognition:
                        # Keep unknown-track stability updated even on non-recognition frames.
                        if (
                            self.auto_register is not None
                            and track_id not in self.track_to_person
                        ):
                            reg_result = self.auto_register.process_observation(
                                track_id=track_id,
                                frame_index=frame_index,
                                bbox=bbox,
                                detection_confidence=det_conf,
                                recognition_result={"is_unknown": True, "is_confirmed": False},
                                embedding=None,
                                face_crop=face_crop,
                                track_hit_count=track_hit_count,
                            )
                            self._maybe_log_track_reason(
                                track_id=track_id,
                                frame_index=frame_index,
                                reason=str(reg_result.get("reason", "")),
                                detail={
                                    "bbox": bbox,
                                    "crop_shape": crop_shape,
                                    "detection_confidence": det_conf,
                                    "unknown_hits": reg_result.get("unknown_hits"),
                                    "track_age_frames": reg_result.get("track_age_frames"),
                                    "track_hit_count": reg_result.get("track_hit_count"),
                                    "best_detection_confidence": reg_result.get(
                                        "best_detection_confidence",
                                    ),
                                    "best_quality_score": reg_result.get("best_quality_score"),
                                },
                            )
                        continue

                    self.track_last_embedding_frame[track_id] = frame_index
                    embedding = self.embedder.embed(face_crop)
                    if embedding is None:
                        embed_debug = dict(getattr(self.embedder, "last_debug_info", {}) or {})
                        self._maybe_log_track_reason(
                            track_id=track_id,
                            frame_index=frame_index,
                            reason="embedding_unavailable",
                            detail={
                                "bbox": bbox,
                                "crop_shape": crop_shape,
                                "track_age_frames": track_age_frames,
                                "track_hit_count": track_hit_count,
                                "detection_confidence": det_conf,
                                **embed_debug,
                            },
                        )
                        continue

                    recognition = self.recognizer.recognize(embedding, track_id=track_id)
                    self._maybe_log_track_reason(
                        track_id=track_id,
                        frame_index=frame_index,
                        reason=str(recognition.get("reason", "")),
                        detail={
                            "bbox": bbox,
                            "crop_shape": crop_shape,
                            "track_age_frames": track_age_frames,
                            "track_hit_count": track_hit_count,
                            "similarity": recognition.get("similarity"),
                            "is_unknown": recognition.get("is_unknown"),
                            "is_confirmed": recognition.get("is_confirmed"),
                        },
                    )
                    if recognition.get("is_confirmed") and recognition.get("person_id"):
                        person_id = str(recognition["person_id"])
                        self.track_to_person[track_id] = person_id
                        self.person_last_face_crop[person_id] = face_crop
                        currently_seen_person_ids.add(person_id)
                        self.stats.recognized_total += 1
                        self.event_logger.log_app_info(
                            "Track recognized",
                            track_id=track_id,
                            person_id=person_id,
                            similarity=recognition.get("similarity"),
                        )

                        # Save entry crop only when person does not already have an open visit.
                        entry_crop_path = None
                        if not self.repository.has_open_visit(person_id):
                            entry_crop_path = self._save_entry_crop_safe(person_id, face_crop)

                        visit_result = self.visit_manager.handle_confirmed_sighting(
                            person_id=person_id,
                            seen_at=datetime.now(timezone.utc),
                            track_id=track_id,
                            frame_index=frame_index,
                            entry_crop_path=entry_crop_path,
                        )

                        # RECOGNIZED is intentionally written via event_logger only.
                        # event_logger mirrors it to DB, so we do not write it elsewhere.
                        self.event_logger.log_recognized(
                            person_id=person_id,
                            visit_id=visit_result.get("visit_id"),
                            track_id=track_id,
                            frame_index=frame_index,
                            similarity=recognition.get("similarity"),
                            meta={"reason": recognition.get("reason")},
                        )

                        # ENTRY is already in DB via visit_manager. Write mandatory file event.
                        if visit_result.get("entry_emitted"):
                            self.stats.entries_total += 1
                            self.event_logger.log_app_info(
                                "Entry opened",
                                track_id=track_id,
                                person_id=person_id,
                                visit_id=visit_result.get("visit_id"),
                            )
                            self.event_logger.log_entry(
                                person_id=person_id,
                                visit_id=visit_result.get("visit_id"),
                                track_id=track_id,
                                frame_index=frame_index,
                                crop_path=entry_crop_path,
                                meta={"reason": "new_visit_opened"},
                                mirror_to_db=False,
                            )
                        continue

                    # Unknown flow: try stable auto-registration.
                    reg_result = self.auto_register.process_observation(
                        track_id=track_id,
                        frame_index=frame_index,
                        bbox=bbox,
                        detection_confidence=det_conf,
                        recognition_result=recognition,
                        embedding=embedding,
                        face_crop=face_crop,
                        track_hit_count=track_hit_count,
                    )
                    self._maybe_log_track_reason(
                        track_id=track_id,
                        frame_index=frame_index,
                        reason=str(reg_result.get("reason", "")),
                        detail={
                            "bbox": bbox,
                            "crop_shape": crop_shape,
                            "detection_confidence": det_conf,
                            "unknown_hits": reg_result.get("unknown_hits"),
                            "track_age_frames": reg_result.get("track_age_frames"),
                            "track_hit_count": reg_result.get("track_hit_count"),
                            "quality_score": reg_result.get("quality_score"),
                            "best_detection_confidence": reg_result.get(
                                "best_detection_confidence",
                            ),
                            "best_quality_score": reg_result.get("best_quality_score"),
                            "matched_similarity": reg_result.get("matched_similarity"),
                        },
                    )

                    if reg_result.get("registered") and reg_result.get("person_id"):
                        person_id = str(reg_result["person_id"])
                        reg_reason = str(reg_result.get("reason", ""))
                        self.track_to_person[track_id] = person_id
                        self.person_last_face_crop[person_id] = face_crop
                        currently_seen_person_ids.add(person_id)

                        if reg_reason == "auto_registered_new_identity":
                            self.stats.registered_total += 1
                            self.event_logger.log_app_info(
                                "New identity registered",
                                track_id=track_id,
                                person_id=person_id,
                                reason="auto_registered_new_identity",
                                new_person_created=True,
                                effective_reuse_threshold=reg_result.get("effective_reuse_threshold"),
                                diagnostic_gallery_best_person_id=reg_result.get(
                                    "diagnostic_gallery_best_person_id",
                                ),
                                diagnostic_gallery_best_similarity=reg_result.get(
                                    "diagnostic_gallery_best_similarity",
                                ),
                            )
                        elif reg_reason == "duplicate_prevented_existing_identity":
                            self.stats.identity_reuse_bindings_total += 1
                            self.event_logger.log_app_info(
                                "Existing identity reused for track",
                                track_id=track_id,
                                person_id=person_id,
                                reason="duplicate_prevented_existing_identity",
                                duplicate_prevented=True,
                                no_new_person_created=True,
                                matched_similarity=reg_result.get("matched_similarity"),
                                effective_reuse_threshold=reg_result.get("effective_reuse_threshold"),
                            )
                        else:
                            self.event_logger.log_app_info(
                                "Track bound after registration",
                                track_id=track_id,
                                person_id=person_id,
                                reason=reg_reason,
                                matched_similarity=reg_result.get("matched_similarity"),
                            )

                        # Gallery: new sample for this person (new or reused).
                        self.recognizer.add_identity(person_id, embedding)
                        self.stats.unique_visitors = self.repository.get_unique_visitor_count()

                        # New identity: DB REGISTERED via auto_register.write_event;
                        # events.log REGISTERED here (mirror_to_db=False avoids duplicate DB row).
                        if reg_reason == "auto_registered_new_identity":
                            self.event_logger.log_registered(
                                person_id=person_id,
                                track_id=track_id,
                                frame_index=frame_index,
                                meta={"reason": "auto_registered_new_identity"},
                                mirror_to_db=False,
                            )

                        entry_crop_path = None
                        if not self.repository.has_open_visit(person_id):
                            entry_crop_path = self._save_entry_crop_safe(person_id, face_crop)

                        visit_result = self.visit_manager.handle_confirmed_sighting(
                            person_id=person_id,
                            seen_at=datetime.now(timezone.utc),
                            track_id=track_id,
                            frame_index=frame_index,
                            entry_crop_path=entry_crop_path,
                        )
                        if visit_result.get("entry_emitted"):
                            self.stats.entries_total += 1
                            self.event_logger.log_app_info(
                                "Entry opened",
                                track_id=track_id,
                                person_id=person_id,
                                visit_id=visit_result.get("visit_id"),
                            )
                            self.event_logger.log_entry(
                                person_id=person_id,
                                visit_id=visit_result.get("visit_id"),
                                track_id=track_id,
                                frame_index=frame_index,
                                crop_path=entry_crop_path,
                                meta={"reason": "new_visit_opened"},
                                mirror_to_db=False,
                            )

                # Handle dead tracks by cleaning memory; timeout path controls EXIT emission.
                self._handle_dead_tracks(dead_tracks=dead_tracks)

                # Timeout-based missing handling for open visits.
                if self.recognition_enabled:
                    self._handle_missing_open_visits(
                        seen_person_ids=currently_seen_person_ids,
                        frame_index=frame_index,
                    )

        except KeyboardInterrupt:
            self.event_logger.log_app_warning("Pipeline interrupted by user")
        finally:
            self._finalize_pending_exits()
            self.video_source.release()
            self.stats.entries_total = (
                self.repository.get_event_count_by_type("ENTRY") - stats_baseline_entry
            )
            self.stats.exits_total = (
                self.repository.get_event_count_by_type("EXIT") - stats_baseline_exit
            )
            self.stats.unique_visitors = self.repository.get_unique_visitor_count()
            self.event_logger.log_app_info("Resource usage", **snapshot_for_app_log())
            self.event_logger.log_app_info("Pipeline stopped", stats=self.stats.to_dict())

        return self.stats.to_dict()

    def _should_run_recognition(self, track_id: int, frame_index: int) -> bool:
        """Recognition scheduling gate based on configurable interval."""
        last_frame = self.track_last_embedding_frame.get(track_id)
        if last_frame is None:
            return True
        return (frame_index - last_frame) >= self.recognition_interval

    def _handle_dead_tracks(self, dead_tracks: list[dict[str, Any]]) -> None:
        """Process dead track transitions without forcing EXIT events.

        EXIT decisions are timeout-driven in VisitManager and finalized on shutdown.
        """
        for track in dead_tracks:
            track_id = int(track["track_id"])
            self.event_logger.log_app_info(
                "Track dead",
                track_id=track_id,
                created_frame=track.get("created_frame"),
                last_seen_frame=track.get("last_seen_frame"),
                hit_count=track.get("hit_count"),
                lost_count=track.get("lost_count"),
                status=track.get("status"),
            )

            # Let auto-registration module clear its state.
            if self.auto_register is not None:
                self.auto_register.handle_track_dead(track_id)
            if self.recognizer is not None:
                self.recognizer.clear_track_history(track_id)

            # Cleanup track memory and identity binding; missing timeout handles exit.
            self.track_to_person.pop(track_id, None)
            self.track_last_face_crop.pop(track_id, None)
            self.track_last_embedding_frame.pop(track_id, None)

    def _handle_missing_open_visits(self, seen_person_ids: set[str], frame_index: int) -> None:
        """Timeout-based EXIT management for persons not seen this cycle."""
        if self.visit_manager is None:
            return
        open_person_ids = self.visit_manager.get_open_person_ids()
        missing_person_ids = [pid for pid in open_person_ids if pid not in seen_person_ids]

        for person_id in missing_person_ids:
            exit_crop_path = self._save_exit_crop_for_person(person_id)
            result = self.visit_manager.handle_missing_person(
                person_id=person_id,
                current_time=datetime.now(timezone.utc),
                frame_index=frame_index,
                exit_crop_path=exit_crop_path,
                force_exit=False,
            )
            if result.get("exit_emitted"):
                self.stats.exits_total += 1
                self.event_logger.log_app_info(
                    "Exit emitted",
                    person_id=person_id,
                    visit_id=result.get("visit_id"),
                    reason="visit_timeout_exit",
                )
                self.event_logger.log_exit(
                    person_id=person_id,
                    visit_id=result.get("visit_id"),
                    track_id=None,
                    frame_index=frame_index,
                    crop_path=exit_crop_path,
                    meta={"reason": "visit_timeout_exit"},
                    mirror_to_db=False,
                )
                self.person_last_face_crop.pop(person_id, None)

    def _save_entry_crop_safe(self, person_id: str, face_crop: np.ndarray | None) -> str | None:
        """Save entry crop safely and return path when possible."""
        if face_crop is None:
            return None
        try:
            data = self.crop_saver.save_entry_crop(face_crop=face_crop, person_id=person_id)
            return str(data.get("saved_path"))
        except Exception as exc:  # noqa: BLE001
            self.event_logger.log_app_warning("Failed to save entry crop", error=str(exc), person_id=person_id)
            return None

    def _save_exit_crop_for_person(self, person_id: str) -> str | None:
        """Save exit crop using last known face crop for person, if available."""
        face_crop = self.person_last_face_crop.get(person_id)
        if face_crop is None:
            return None
        try:
            data = self.crop_saver.save_exit_crop(face_crop=face_crop, person_id=person_id)
            return str(data.get("saved_path"))
        except Exception as exc:  # noqa: BLE001
            self.event_logger.log_app_warning("Failed to save exit crop", error=str(exc), person_id=person_id)
            return None

    def _finalize_pending_exits(self) -> None:
        """Close any remaining OPEN visits on end-of-stream/shutdown."""
        if self.visit_manager is None:
            return
        final_time = datetime.now(timezone.utc)
        open_person_ids = list(self.visit_manager.get_open_person_ids())
        for person_id in open_person_ids:
            exit_crop_path = self._save_exit_crop_for_person(person_id)
            result = self.visit_manager.handle_missing_person(
                person_id=person_id,
                current_time=final_time,
                frame_index=None,
                exit_crop_path=exit_crop_path,
                force_exit=True,
            )
            if result.get("exit_emitted"):
                self.stats.exits_total += 1
                self.event_logger.log_app_info(
                    "Exit emitted",
                    person_id=person_id,
                    visit_id=result.get("visit_id"),
                    reason="shutdown_finalize",
                )
                self.event_logger.log_exit(
                    person_id=person_id,
                    visit_id=result.get("visit_id"),
                    track_id=None,
                    frame_index=None,
                    crop_path=exit_crop_path,
                    meta={"reason": "shutdown_finalize"},
                    mirror_to_db=False,
                )
                self.person_last_face_crop.pop(person_id, None)

    def _maybe_log_track_reason(
        self,
        *,
        track_id: int,
        frame_index: int,
        reason: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Emit concise per-track debug reason logs with rate limiting."""
        if not reason:
            return

        last_logged = self.track_last_reason_log_frame.get(track_id, -10_000)
        if frame_index - last_logged < max(1, self.recognition_interval):
            return

        self.track_last_reason_log_frame[track_id] = frame_index

        safe_detail = dict(detail or {})
        if "reason" in safe_detail:
            safe_detail["detail_reason"] = safe_detail.pop("reason")

        self.event_logger.log_app_info(
            "Track decision",
            track_id=track_id,
            frame_index=frame_index,
            reason=reason,
            **safe_detail,
        )
