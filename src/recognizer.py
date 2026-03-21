"""Face recognition using cosine similarity over stored embeddings.

This module is designed for tracker-first pipelines:
- tracker provides stable track_id
- recognizer compares embeddings against registered identities
- optional per-track confirmation history reduces identity flicker
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MatchResult:
    """Recognition output for one embedding observation."""

    person_id: str | None
    similarity: float
    is_match: bool
    is_confirmed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to pipeline-friendly dictionary."""
        return {
            "person_id": self.person_id,
            "similarity": float(self.similarity),
            "is_match": bool(self.is_match),
            "is_confirmed": bool(self.is_confirmed),
            "is_unknown": not bool(self.is_confirmed),
            "reason": self.reason,
        }


class FaceRecognizer:
    """Cosine-similarity face recognizer with optional match confirmation.

    Args:
        match_threshold: Minimum cosine similarity to consider a candidate match.
        confirmation_window: Number of recent observations to keep per track.
        confirmation_min_hits: Required repeated hits in window to confirm identity.
            Set to 1 for immediate acceptance.
    """

    def __init__(
        self,
        match_threshold: float = 0.45,
        confirmation_window: int = 5,
        confirmation_min_hits: int = 2,
    ) -> None:
        if not (0.0 <= match_threshold <= 1.0):
            raise ValueError("match_threshold must be between 0.0 and 1.0.")
        if confirmation_window < 1:
            raise ValueError("confirmation_window must be >= 1.")
        if confirmation_min_hits < 1:
            raise ValueError("confirmation_min_hits must be >= 1.")
        if confirmation_min_hits > confirmation_window:
            raise ValueError("confirmation_min_hits cannot be greater than confirmation_window.")

        self.match_threshold = float(match_threshold)
        self.confirmation_window = int(confirmation_window)
        self.confirmation_min_hits = int(confirmation_min_hits)

        # Gallery: person_id -> list of normalized embeddings.
        self._gallery: dict[str, list[np.ndarray]] = {}

        # Track-level memory for repeated match confirmation.
        self._track_history: dict[int, deque[str | None]] = {}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FaceRecognizer":
        """Build recognizer from app config."""
        return cls(
            match_threshold=float(config.get("recognition_threshold", 0.45)),
            confirmation_window=int(config.get("recognition_confirmation_window", 3)),
            confirmation_min_hits=int(config.get("recognition_confirmation_min_hits", 1)),
        )

    def add_identity(self, person_id: str, embedding: np.ndarray) -> None:
        """Add one embedding sample to identity gallery."""
        pid = str(person_id).strip()
        if not pid:
            raise ValueError("person_id cannot be empty.")

        vec = self._normalize_embedding(embedding)
        if vec is None:
            raise ValueError("embedding is invalid or empty.")

        self._gallery.setdefault(pid, []).append(vec)

    def set_identity_embeddings(self, person_id: str, embeddings: list[np.ndarray]) -> None:
        """Replace identity embeddings with a fresh list."""
        pid = str(person_id).strip()
        if not pid:
            raise ValueError("person_id cannot be empty.")

        normalized: list[np.ndarray] = []
        for emb in embeddings:
            vec = self._normalize_embedding(emb)
            if vec is not None:
                normalized.append(vec)

        if not normalized:
            raise ValueError("No valid embeddings provided for identity.")
        self._gallery[pid] = normalized

    def remove_identity(self, person_id: str) -> None:
        """Remove identity from gallery if present."""
        self._gallery.pop(str(person_id), None)

    def clear_gallery(self) -> None:
        """Remove all registered identities."""
        self._gallery.clear()

    def clear_track_history(self, track_id: int | None = None) -> None:
        """Clear match confirmation history for one track or all tracks."""
        if track_id is None:
            self._track_history.clear()
            return
        self._track_history.pop(int(track_id), None)

    def recognize(self, embedding: np.ndarray, track_id: int | None = None) -> dict[str, Any]:
        """Recognize identity for one incoming face embedding.

        Args:
            embedding: Incoming face embedding vector.
            track_id: Optional track id from tracker to support repeated confirmation.

        Returns:
            {
              "person_id": str | None,
              "similarity": float,
              "is_match": bool,
              "is_confirmed": bool,
              "is_unknown": bool,
              "reason": str
            }
        """
        query = self._normalize_embedding(embedding)
        if query is None:
            return MatchResult(
                person_id=None,
                similarity=0.0,
                is_match=False,
                is_confirmed=False,
                reason="invalid_embedding",
            ).to_dict()

        if not self._gallery:
            return MatchResult(
                person_id=None,
                similarity=0.0,
                is_match=False,
                is_confirmed=False,
                reason="empty_gallery",
            ).to_dict()

        best_person_id, best_similarity = self._find_best_match(query)
        if best_person_id is None:
            return MatchResult(
                person_id=None,
                similarity=0.0,
                is_match=False,
                is_confirmed=False,
                reason="no_candidates",
            ).to_dict()

        is_match = best_similarity >= self.match_threshold
        if not is_match:
            # Track unknown history for stability if track id is provided.
            self._push_history(track_id, None)
            return MatchResult(
                person_id=None,
                similarity=best_similarity,
                is_match=False,
                is_confirmed=False,
                reason="below_threshold",
            ).to_dict()

        # Raw match exists; optionally require repeated confirmation.
        self._push_history(track_id, best_person_id)
        is_confirmed = self._is_confirmed(track_id, best_person_id)

        return MatchResult(
            person_id=best_person_id if is_confirmed else None,
            similarity=best_similarity,
            is_match=True,
            is_confirmed=is_confirmed,
            reason="confirmed_match" if is_confirmed else "pending_confirmation",
        ).to_dict()

    def get_gallery_size(self) -> int:
        """Return number of registered identities."""
        return len(self._gallery)

    def get_identity_count(self, person_id: str) -> int:
        """Return number of embeddings stored for given identity."""
        return len(self._gallery.get(person_id, []))

    def _find_best_match(self, query_embedding: np.ndarray) -> tuple[str | None, float]:
        """Find top person by max cosine similarity against gallery samples."""
        best_person_id: str | None = None
        best_similarity = -1.0

        for person_id, samples in self._gallery.items():
            if not samples:
                continue
            # Compare against all samples of identity; take strongest score.
            scores = [self._cosine_similarity(query_embedding, sample) for sample in samples]
            person_best = max(scores) if scores else -1.0
            if person_best > best_similarity:
                best_similarity = person_best
                best_person_id = person_id

        if best_similarity < 0:
            return None, 0.0
        return best_person_id, float(best_similarity)

    def _push_history(self, track_id: int | None, person_id: str | None) -> None:
        """Update per-track recognition history."""
        if track_id is None:
            return
        tid = int(track_id)
        if tid not in self._track_history:
            self._track_history[tid] = deque(maxlen=self.confirmation_window)
        self._track_history[tid].append(person_id)

    def _is_confirmed(self, track_id: int | None, person_id: str) -> bool:
        """Decide whether a match is confirmed using track history."""
        if self.confirmation_min_hits <= 1 or track_id is None:
            return True

        history = self._track_history.get(int(track_id))
        if not history:
            return False
        hits = sum(1 for pid in history if pid == person_id)
        return hits >= self.confirmation_min_hits

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray | Any) -> np.ndarray | None:
        """Normalize vector to unit length for cosine similarity."""
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            return None
        norm = np.linalg.norm(vec)
        if norm <= 1e-8:
            return None
        return vec / norm

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity for already normalized vectors."""
        # Clamp to [0, 1] for cleaner confidence interpretation.
        score = float(np.dot(a, b))
        return max(0.0, min(1.0, score))
