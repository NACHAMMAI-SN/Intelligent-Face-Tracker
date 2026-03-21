"""Automatic registration of previously unknown, stable face tracks.

This module works in a tracker-first pipeline:
1) tracker provides stable track_id and lifecycle metadata
2) recognizer indicates whether observation is known or unknown
3) auto-registrar decides when unknown track is stable enough to register
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import numpy as np

from src.repositories import Repository


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrackRegistrationState:
    """In-memory state used to decide if a track should be auto-registered."""

    first_seen_frame: int
    last_seen_frame: int
    unknown_hits: int = 0
    best_detection_confidence: float = 0.0
    best_quality_score: float = 0.0
    is_registered: bool = False
    registered_person_id: str | None = None
    last_registration_frame: int | None = None
    # Keep last few quality estimates for debugging/tuning.
    quality_history: list[float] = field(default_factory=list)


class AutoRegistrar:
    """Decide when and how to auto-register unknown tracks.

    Practical registration criteria (all required):
    - track has been visible for minimum duration in frames
    - detection confidence passes threshold
    - estimated face quality passes threshold
    - unknown classification repeated enough times
    """

    def __init__(
        self,
        repository: Repository,
        min_track_duration_frames: int = 8,
        min_track_hits: int = 8,
        min_detection_confidence: float = 0.45,
        min_unknown_hits: int = 3,
        min_quality_score: float = 20.0,
        registration_cooldown_frames: int = 120,
        identity_reuse_similarity_threshold: float = 0.58,
        recognition_threshold: float = 0.45,
        fragmentation_merge_slack: float = 0.10,
    ) -> None:
        if min_track_duration_frames < 1:
            raise ValueError("min_track_duration_frames must be >= 1.")
        if min_track_hits < 1:
            raise ValueError("min_track_hits must be >= 1.")
        if min_unknown_hits < 1:
            raise ValueError("min_unknown_hits must be >= 1.")
        if registration_cooldown_frames < 1:
            raise ValueError("registration_cooldown_frames must be >= 1.")
        if not (0.0 <= min_detection_confidence <= 1.0):
            raise ValueError("min_detection_confidence must be between 0.0 and 1.0.")
        if not (0.0 <= identity_reuse_similarity_threshold <= 1.0):
            raise ValueError("identity_reuse_similarity_threshold must be between 0.0 and 1.0.")
        if not (0.0 <= recognition_threshold <= 1.0):
            raise ValueError("recognition_threshold must be between 0.0 and 1.0.")
        if not (0.0 <= fragmentation_merge_slack <= 0.5):
            raise ValueError("fragmentation_merge_slack must be between 0.0 and 0.5.")

        self.repository = repository
        self.min_track_duration_frames = int(min_track_duration_frames)
        self.min_track_hits = int(min_track_hits)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_unknown_hits = int(min_unknown_hits)
        self.min_quality_score = float(min_quality_score)
        self.registration_cooldown_frames = int(registration_cooldown_frames)
        self.identity_reuse_similarity_threshold = float(identity_reuse_similarity_threshold)
        self.recognition_threshold = float(recognition_threshold)
        self.fragmentation_merge_slack = float(fragmentation_merge_slack)

        # track_id -> state
        self._state_by_track: dict[int, TrackRegistrationState] = {}

    @classmethod
    def from_config(cls, config: dict[str, Any], repository: Repository) -> "AutoRegistrar":
        """Build auto-registrar using global config with safe defaults."""
        return cls(
            repository=repository,
            min_track_duration_frames=int(config.get("auto_register_min_track_duration_frames", 4)),
            min_track_hits=int(config.get("auto_register_min_track_hits", 8)),
            min_detection_confidence=float(config.get("auto_register_min_detection_confidence", 0.45)),
            min_unknown_hits=int(config.get("auto_register_min_unknown_hits", 3)),
            min_quality_score=float(config.get("auto_register_min_quality_score", 8.0)),
            registration_cooldown_frames=int(config.get("auto_register_cooldown_frames", 120)),
            identity_reuse_similarity_threshold=float(config.get("auto_register_identity_reuse_threshold", 0.58)),
            recognition_threshold=float(config.get("recognition_threshold", 0.45)),
            fragmentation_merge_slack=float(config.get("auto_register_fragmentation_merge_slack", 0.10)),
        )

    def process_observation(
        self,
        *,
        track_id: int,
        frame_index: int,
        bbox: list[int] | tuple[int, int, int, int],
        detection_confidence: float,
        recognition_result: dict[str, Any] | None,
        embedding: np.ndarray | None,
        face_crop: np.ndarray | None = None,
        track_hit_count: int | None = None,
    ) -> dict[str, Any]:
        """Process one tracked observation and auto-register if eligible.

        Args:
            track_id: Stable tracker id.
            frame_index: Current frame number.
            bbox: Face bounding box [x1, y1, x2, y2].
            detection_confidence: Detection confidence from detector.
            recognition_result: Recognizer output dict. Expected keys include:
                - is_unknown (bool)
                - is_confirmed (bool)
                - person_id (optional)
            embedding: Face embedding vector for this observation.
            face_crop: Optional face crop for quality estimation.

        Returns:
            {
              "registered": bool,
              "person_id": str | None,
              "reason": str,
              "track_id": int,
              "frame_index": int,
              "quality_score": float,
              "unknown_hits": int,
              "track_age_frames": int
            }
        """
        state = self._state_by_track.get(track_id)
        if state is None:
            state = TrackRegistrationState(
                first_seen_frame=frame_index,
                last_seen_frame=frame_index,
            )
            self._state_by_track[track_id] = state
        else:
            state.last_seen_frame = frame_index

        # Compute quality estimate from crop if provided, else fallback to confidence.
        quality_score = self._estimate_quality(face_crop, detection_confidence)
        state.quality_history.append(quality_score)
        if len(state.quality_history) > 20:
            state.quality_history = state.quality_history[-20:]

        state.best_detection_confidence = max(state.best_detection_confidence, float(detection_confidence))
        state.best_quality_score = max(state.best_quality_score, quality_score)

        track_age = frame_index - state.first_seen_frame + 1
        effective_track_hits = int(track_hit_count) if track_hit_count is not None else track_age

        # If recognizer already confirms a known person, never register new identity.
        if self._is_known_match(recognition_result):
            self._reset_unknown_state(track_id, frame_index)
            return self._result(
                registered=False,
                person_id=recognition_result.get("person_id") if recognition_result else None,
                reason="known_person_match",
                track_id=track_id,
                frame_index=frame_index,
                quality_score=quality_score,
                unknown_hits=state.unknown_hits,
                track_age_frames=track_age,
                track_hit_count=effective_track_hits,
                best_detection_confidence=state.best_detection_confidence,
                best_quality_score=state.best_quality_score,
            )

        # Count repeated unknowns only when recognizer says unknown/unconfirmed.
        if self._is_unknown(recognition_result):
            state.unknown_hits += 1

        if state.is_registered:
            return self._result(
                registered=False,
                person_id=state.registered_person_id,
                reason="already_registered_for_track",
                track_id=track_id,
                frame_index=frame_index,
                quality_score=quality_score,
                unknown_hits=state.unknown_hits,
                track_age_frames=track_age,
                track_hit_count=effective_track_hits,
                best_detection_confidence=state.best_detection_confidence,
                best_quality_score=state.best_quality_score,
            )

        # Keep accumulating stability even when embedding is not computed this frame.
        if embedding is None:
            return self._result(
                registered=False,
                person_id=None,
                reason="pending_embedding",
                track_id=track_id,
                frame_index=frame_index,
                quality_score=quality_score,
                unknown_hits=state.unknown_hits,
                track_age_frames=track_age,
                track_hit_count=effective_track_hits,
                best_detection_confidence=state.best_detection_confidence,
                best_quality_score=state.best_quality_score,
            )

        if not self._passes_registration_gate(
            state=state,
            track_age_frames=track_age,
            track_hit_count=effective_track_hits,
        ):
            return self._result(
                registered=False,
                person_id=None,
                reason="not_stable_enough",
                track_id=track_id,
                frame_index=frame_index,
                quality_score=quality_score,
                unknown_hits=state.unknown_hits,
                track_age_frames=track_age,
                track_hit_count=effective_track_hits,
                best_detection_confidence=state.best_detection_confidence,
                best_quality_score=state.best_quality_score,
            )

        # Cooldown guard: do not repeatedly register the same continuous track.
        if (
            state.last_registration_frame is not None
            and frame_index - state.last_registration_frame < self.registration_cooldown_frames
        ):
            return self._result(
                registered=False,
                person_id=state.registered_person_id,
                reason="registration_cooldown_active",
                track_id=track_id,
                frame_index=frame_index,
                quality_score=quality_score,
                unknown_hits=state.unknown_hits,
                track_age_frames=track_age,
                track_hit_count=effective_track_hits,
                best_detection_confidence=state.best_detection_confidence,
                best_quality_score=state.best_quality_score,
            )

        now = utc_now_iso()
        query = self._normalize_embedding(embedding)
        reuse_pid, reuse_sim = self._find_best_gallery_match(query)
        eff_reuse = self._effective_identity_reuse_threshold()

        if reuse_pid is not None and reuse_sim >= eff_reuse:
            person_id = reuse_pid
            self.repository.update_person_last_seen(person_id, seen_at=now)
            self.repository.store_embedding(
                person_id=person_id,
                embedding=np.asarray(embedding, dtype=np.float32),
                quality_score=quality_score,
                created_at=now,
            )
            # No REGISTERED event: that type means a new person row was enrolled.
            # Reuse binds an unknown track to an existing person_id (see app.log / pipeline).

            state.is_registered = True
            state.registered_person_id = person_id
            state.last_registration_frame = frame_index

            return self._result(
                registered=True,
                person_id=person_id,
                reason="duplicate_prevented_existing_identity",
                track_id=track_id,
                frame_index=frame_index,
                quality_score=quality_score,
                unknown_hits=state.unknown_hits,
                track_age_frames=track_age,
                track_hit_count=effective_track_hits,
                best_detection_confidence=state.best_detection_confidence,
                best_quality_score=state.best_quality_score,
                matched_similarity=float(reuse_sim),
                effective_reuse_threshold=float(eff_reuse),
            )

        person_id = self._new_person_id()

        # Persist person + first embedding + event.
        self.repository.create_person(
            person_id=person_id,
            created_at=now,
            first_seen_at=now,
            last_seen_at=now,
            is_active=True,
        )
        self.repository.store_embedding(
            person_id=person_id,
            embedding=np.asarray(embedding, dtype=np.float32),
            quality_score=quality_score,
            created_at=now,
        )
        self.repository.write_event(
            event_type="REGISTERED",
            person_id=person_id,
            visit_id=None,
            track_id=track_id,
            frame_index=frame_index,
            event_time=now,
            meta={
                "auto_registered": True,
                "registration_reason": "auto_registered_new_identity",
                "bbox": [int(v) for v in bbox],
                "track_age_frames": track_age,
                "unknown_hits": state.unknown_hits,
                "detection_confidence": float(detection_confidence),
                "quality_score": float(quality_score),
            },
        )

        state.is_registered = True
        state.registered_person_id = person_id
        state.last_registration_frame = frame_index

        return self._result(
            registered=True,
            person_id=person_id,
            reason="auto_registered_new_identity",
            track_id=track_id,
            frame_index=frame_index,
            quality_score=quality_score,
            unknown_hits=state.unknown_hits,
            track_age_frames=track_age,
            track_hit_count=effective_track_hits,
            best_detection_confidence=state.best_detection_confidence,
            best_quality_score=state.best_quality_score,
            matched_similarity=None,
            effective_reuse_threshold=float(eff_reuse),
            diagnostic_gallery_best_person_id=reuse_pid,
            diagnostic_gallery_best_similarity=float(reuse_sim) if reuse_pid is not None else None,
        )

    def handle_track_dead(self, track_id: int) -> None:
        """Clear state when tracker marks a track dead."""
        self._state_by_track.pop(track_id, None)

    def _passes_registration_gate(
        self,
        state: TrackRegistrationState,
        track_age_frames: int,
        track_hit_count: int,
    ) -> bool:
        """Return True when all practical registration conditions are satisfied."""
        if track_age_frames < self.min_track_duration_frames:
            return False
        if track_hit_count < self.min_track_hits:
            return False
        if state.unknown_hits < self.min_unknown_hits:
            return False
        if state.best_detection_confidence < self.min_detection_confidence:
            return False
        if state.best_quality_score < self.min_quality_score:
            return False
        return True

    @staticmethod
    def _is_unknown(recognition_result: dict[str, Any] | None) -> bool:
        """Interpret recognizer result as unknown/unconfirmed."""
        if not recognition_result:
            return True
        if recognition_result.get("is_confirmed") is True:
            return False
        if recognition_result.get("person_id"):
            # Person id exists but not confirmed strongly enough.
            return True
        return bool(recognition_result.get("is_unknown", True))

    @staticmethod
    def _is_known_match(recognition_result: dict[str, Any] | None) -> bool:
        """True only for confirmed known identity."""
        if not recognition_result:
            return False
        return bool(recognition_result.get("is_confirmed")) and bool(recognition_result.get("person_id"))

    @staticmethod
    def _estimate_quality(face_crop: np.ndarray | None, detection_confidence: float) -> float:
        """Estimate crop quality with a lightweight blur + size heuristic.

        Returns a score where larger is better. Typical useful range: 0-200+.
        """
        if face_crop is None or not isinstance(face_crop, np.ndarray) or face_crop.size == 0:
            return max(0.0, float(detection_confidence) * 100.0)

        h, w = face_crop.shape[:2]
        if h < 4 or w < 4:
            return 0.0

        # Use Laplacian variance as a practical sharpness proxy.
        gray = (
            face_crop if face_crop.ndim == 2 else np.dot(face_crop[..., :3], np.array([0.114, 0.587, 0.299]))
        ).astype(np.float32)
        gy, gx = np.gradient(gray)
        sharpness = float(np.var(gx) + np.var(gy))
        size_factor = min(1.0, (h * w) / float(128 * 128))
        conf_factor = max(0.0, min(1.0, float(detection_confidence)))
        return sharpness * (0.5 + 0.5 * size_factor) * (0.7 + 0.3 * conf_factor)

    def _reset_unknown_state(self, track_id: int, frame_index: int) -> None:
        """Reset unknown hit accumulation when track is recognized as known."""
        state = self._state_by_track.get(track_id)
        if state is None:
            return
        state.unknown_hits = 0
        state.last_seen_frame = frame_index

    @staticmethod
    def _new_person_id() -> str:
        """Generate stable, readable person id."""
        return f"person_{uuid4().hex[:12]}"

    def _effective_identity_reuse_threshold(self) -> float:
        """Similarity bar for binding an unknown track to an existing person.

        Recognizer sends observations to auto-register only when it does *not* emit a
        confirmed match. With ``recognition_confirmation_min_hits <= 1``, the common case
        is ``below_threshold``: best gallery cosine is **strictly below**
        ``recognition_threshold``. A reuse bar **above** that value (e.g. 0.58 > 0.45)
        can never be satisfied on that path, so identity reuse stays at zero.

        We combine the configured reuse cap with a merge floor derived from recognition
        so returning faces on new tracks can still match the same stored embedding.
        """
        merge_floor = max(0.0, self.recognition_threshold - self.fragmentation_merge_slack)
        return min(self.identity_reuse_similarity_threshold, merge_floor)

    def _find_best_gallery_match(self, query: np.ndarray | None) -> tuple[str | None, float]:
        """Return best (person_id, similarity) against all stored embeddings.

        Uses the same cosine convention as FaceRecognizer (normalized vectors, score in [0,1]).
        """
        if query is None:
            return None, 0.0

        gallery = self.repository.get_all_known_embeddings()
        best_person_id: str | None = None
        best_similarity = -1.0

        for person_id, samples in gallery.items():
            person_best = -1.0
            for sample in samples:
                s = self._normalize_embedding(sample)
                if s is None:
                    continue
                sim = self._cosine_similarity(query, s)
                if sim > person_best:
                    person_best = sim
            if person_best > best_similarity:
                best_similarity = person_best
                best_person_id = person_id

        if best_person_id is None or best_similarity < 0:
            return None, 0.0
        return best_person_id, float(best_similarity)

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray | Any) -> np.ndarray | None:
        """Normalize vector to unit length for cosine similarity (aligned with FaceRecognizer)."""
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            return None
        norm = np.linalg.norm(vec)
        if norm <= 1e-8:
            return None
        return vec / norm

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity for already normalized vectors; clamp to [0, 1]."""
        score = float(np.dot(a, b))
        return max(0.0, min(1.0, score))

    @staticmethod
    def _result(
        *,
        registered: bool,
        person_id: str | None,
        reason: str,
        track_id: int,
        frame_index: int,
        quality_score: float,
        unknown_hits: int,
        track_age_frames: int,
        track_hit_count: int,
        best_detection_confidence: float,
        best_quality_score: float,
        matched_similarity: float | None = None,
        effective_reuse_threshold: float | None = None,
        diagnostic_gallery_best_person_id: str | None = None,
        diagnostic_gallery_best_similarity: float | None = None,
    ) -> dict[str, Any]:
        """Build consistent result payload."""
        return {
            "registered": bool(registered),
            "person_id": person_id,
            "reason": reason,
            "track_id": int(track_id),
            "frame_index": int(frame_index),
            "quality_score": float(quality_score),
            "unknown_hits": int(unknown_hits),
            "track_age_frames": int(track_age_frames),
            "track_hit_count": int(track_hit_count),
            "best_detection_confidence": float(best_detection_confidence),
            "best_quality_score": float(best_quality_score),
            "matched_similarity": matched_similarity,
            "effective_reuse_threshold": effective_reuse_threshold,
            "diagnostic_gallery_best_person_id": diagnostic_gallery_best_person_id,
            "diagnostic_gallery_best_similarity": diagnostic_gallery_best_similarity,
        }
