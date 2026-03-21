"""Simple multi-face tracker with lifecycle states.

Design goals:
- Keep track continuity with stable track_id values.
- Track faces only (detections are expected from face detector).
- Support lifecycle states: active -> lost -> dead.
- Retain lost tracks briefly before declaring them dead.
- Expose transitions and active tracks in a pipeline-friendly format.

This implementation uses greedy IoU matching for simplicity and explainability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


TRACK_STATUS_ACTIVE = "active"
TRACK_STATUS_LOST = "lost"
TRACK_STATUS_DEAD = "dead"


@dataclass
class Track:
    """State container for a single tracked face."""

    track_id: int
    bbox: list[int]  # [x1, y1, x2, y2]
    confidence: float
    created_frame: int
    last_seen_frame: int
    hit_count: int = 1
    lost_count: int = 0
    status: str = TRACK_STATUS_ACTIVE

    # Optional recognition memory used later by recognizer/auto-register modules.
    recognition_memory: dict[str, Any] = field(
        default_factory=lambda: {
            "person_id": None,
            "best_score": None,
            "label_history": [],
            "embedding_history": [],
        }
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable track dictionary for pipeline use."""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "created_frame": self.created_frame,
            "last_seen_frame": self.last_seen_frame,
            "hit_count": self.hit_count,
            "lost_count": self.lost_count,
            "status": self.status,
            "recognition_memory": self.recognition_memory,
        }


class FaceTracker:
    """Practical IoU-based tracker for multi-face detection streams.

    Args:
        max_lost_frames: Frames a track can remain unmatched before becoming dead.
        iou_threshold: Minimum IoU for matching detection to existing track.

    Expected detection input format:
        [
          {"bbox": [x1, y1, x2, y2], "confidence": 0.9, ...},
          ...
        ]
    """

    def __init__(self, max_lost_frames: int = 30, iou_threshold: float = 0.3) -> None:
        if max_lost_frames < 1:
            raise ValueError("max_lost_frames must be >= 1.")
        if not (0.0 <= iou_threshold <= 1.0):
            raise ValueError("iou_threshold must be between 0.0 and 1.0.")

        self.max_lost_frames = int(max_lost_frames)
        self.iou_threshold = float(iou_threshold)

        self._tracks: dict[int, Track] = {}
        self._next_track_id: int = 1

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FaceTracker":
        """Build tracker from global app config."""
        return cls(
            max_lost_frames=int(config.get("tracker_max_lost_frames", 30)),
            iou_threshold=float(config.get("tracker_iou_threshold", 0.3)),
        )

    def update(self, detections: list[dict[str, Any]], frame_index: int) -> dict[str, Any]:
        """Update tracker with detections from one frame.

        This function performs:
        1) match detections to active/lost tracks by IoU
        2) update matched tracks
        3) create new tracks for unmatched detections
        4) mark unmatched tracks as lost/dead

        Args:
            detections: Face detection list for current frame.
            frame_index: Current frame index.

        Returns:
            {
              "active_tracks": [...],
              "lost_tracks": [...],
              "dead_tracks": [...],   # tracks transitioned to dead this frame
              "transitions": [...],   # state transition events
            }
        """
        self._validate_detections(detections)
        if frame_index < 0:
            raise ValueError("frame_index must be >= 0.")

        transitions: list[dict[str, Any]] = []
        dead_this_frame: list[Track] = []

        candidate_track_ids = [
            tid
            for tid, tr in self._tracks.items()
            if tr.status in (TRACK_STATUS_ACTIVE, TRACK_STATUS_LOST)
        ]

        matches, unmatched_track_ids, unmatched_detection_indices = self._match_detections(
            detections=detections,
            candidate_track_ids=candidate_track_ids,
        )

        # 1) Apply matched updates.
        for track_id, det_idx in matches:
            track = self._tracks[track_id]
            detection = detections[det_idx]

            prev_status = track.status
            track.bbox = [int(v) for v in detection["bbox"]]
            track.confidence = float(detection.get("confidence", track.confidence))
            track.last_seen_frame = frame_index
            track.hit_count += 1
            track.lost_count = 0
            track.status = TRACK_STATUS_ACTIVE

            if prev_status == TRACK_STATUS_LOST:
                transitions.append(
                    {
                        "track_id": track.track_id,
                        "from": TRACK_STATUS_LOST,
                        "to": TRACK_STATUS_ACTIVE,
                        "frame_index": frame_index,
                        "reason": "reacquired",
                    }
                )

        # 2) Create tracks for unmatched detections.
        for det_idx in unmatched_detection_indices:
            detection = detections[det_idx]
            track = Track(
                track_id=self._next_track_id,
                bbox=[int(v) for v in detection["bbox"]],
                confidence=float(detection.get("confidence", 0.0)),
                created_frame=frame_index,
                last_seen_frame=frame_index,
            )
            self._tracks[track.track_id] = track
            self._next_track_id += 1

            transitions.append(
                {
                    "track_id": track.track_id,
                    "from": None,
                    "to": TRACK_STATUS_ACTIVE,
                    "frame_index": frame_index,
                    "reason": "created",
                }
            )

        # 3) Mark unmatched tracks as lost/dead.
        for track_id in unmatched_track_ids:
            track = self._tracks[track_id]
            if track.status == TRACK_STATUS_DEAD:
                continue

            track.lost_count += 1
            previous = track.status

            if track.lost_count > self.max_lost_frames:
                track.status = TRACK_STATUS_DEAD
                dead_this_frame.append(track)
                transitions.append(
                    {
                        "track_id": track.track_id,
                        "from": previous,
                        "to": TRACK_STATUS_DEAD,
                        "frame_index": frame_index,
                        "reason": "lost_timeout",
                    }
                )
            else:
                if previous != TRACK_STATUS_LOST:
                    track.status = TRACK_STATUS_LOST
                    transitions.append(
                        {
                            "track_id": track.track_id,
                            "from": previous,
                            "to": TRACK_STATUS_LOST,
                            "frame_index": frame_index,
                            "reason": "missing_detection",
                        }
                    )

        active_tracks = [t.to_dict() for t in self._tracks.values() if t.status == TRACK_STATUS_ACTIVE]
        lost_tracks = [t.to_dict() for t in self._tracks.values() if t.status == TRACK_STATUS_LOST]
        dead_tracks = [t.to_dict() for t in dead_this_frame]

        return {
            "active_tracks": active_tracks,
            "lost_tracks": lost_tracks,
            "dead_tracks": dead_tracks,
            "transitions": transitions,
        }

    def get_active_tracks(self) -> list[dict[str, Any]]:
        """Return only currently active tracks."""
        return [track.to_dict() for track in self._tracks.values() if track.status == TRACK_STATUS_ACTIVE]

    def get_tracks(self, include_dead: bool = False) -> list[dict[str, Any]]:
        """Return track snapshots for debugging or pipeline inspection."""
        if include_dead:
            return [track.to_dict() for track in self._tracks.values()]
        return [track.to_dict() for track in self._tracks.values() if track.status != TRACK_STATUS_DEAD]

    def clear_dead_tracks(self) -> int:
        """Delete dead tracks from memory and return number removed."""
        dead_ids = [tid for tid, tr in self._tracks.items() if tr.status == TRACK_STATUS_DEAD]
        for tid in dead_ids:
            del self._tracks[tid]
        return len(dead_ids)

    def _match_detections(
        self,
        detections: list[dict[str, Any]],
        candidate_track_ids: list[int],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Greedy IoU matching between current detections and candidate tracks."""
        matches: list[tuple[int, int]] = []
        used_track_ids: set[int] = set()
        used_detection_indices: set[int] = set()

        # Build all possible pairs above threshold.
        scored_pairs: list[tuple[float, int, int]] = []
        for track_id in candidate_track_ids:
            track = self._tracks[track_id]
            for det_idx, detection in enumerate(detections):
                iou = _iou(track.bbox, detection["bbox"])
                if iou >= self.iou_threshold:
                    scored_pairs.append((iou, track_id, det_idx))

        # Highest IoU first (greedy bipartite assignment).
        scored_pairs.sort(key=lambda x: x[0], reverse=True)

        for _, track_id, det_idx in scored_pairs:
            if track_id in used_track_ids or det_idx in used_detection_indices:
                continue
            used_track_ids.add(track_id)
            used_detection_indices.add(det_idx)
            matches.append((track_id, det_idx))

        unmatched_track_ids = [tid for tid in candidate_track_ids if tid not in used_track_ids]
        unmatched_detection_indices = [i for i in range(len(detections)) if i not in used_detection_indices]
        return matches, unmatched_track_ids, unmatched_detection_indices

    @staticmethod
    def _validate_detections(detections: list[dict[str, Any]]) -> None:
        """Basic validation for detector output compatibility."""
        if not isinstance(detections, list):
            raise ValueError("detections must be a list.")
        for idx, detection in enumerate(detections):
            if not isinstance(detection, dict):
                raise ValueError(f"detections[{idx}] must be a dict.")
            if "bbox" not in detection:
                raise ValueError(f"detections[{idx}] missing 'bbox'.")
            bbox = detection["bbox"]
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError(f"detections[{idx}]['bbox'] must be [x1, y1, x2, y2].")


def _iou(box_a: list[int] | tuple[int, int, int, int], box_b: list[int] | tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes."""
    ax1, ay1, ax2, ay2 = [int(v) for v in box_a]
    bx1, by1, bx2, by2 = [int(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter_area

    if denom <= 0:
        return 0.0
    return inter_area / denom
