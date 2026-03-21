"""Repository layer for database operations.

This module wraps SQL queries into reusable methods for pipeline and visit logic.
It uses helpers from src.db for consistent transaction handling.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.db import fetch_all, fetch_one, transaction


def utc_now_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class Repository:
    """High-level data access helper for app entities."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    # -------------------------------------------------------------------------
    # Person and embedding methods
    # -------------------------------------------------------------------------
    def create_person(
        self,
        person_id: str,
        created_at: str | None = None,
        first_seen_at: str | None = None,
        last_seen_at: str | None = None,
        is_active: bool = True,
    ) -> bool:
        """Create a person record.

        Returns:
            True if inserted, False if person already exists.
        """
        ts = created_at or utc_now_iso()
        first_seen = first_seen_at or ts
        last_seen = last_seen_at or ts

        with transaction(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO persons
                    (person_id, created_at, first_seen_at, last_seen_at, is_active)
                VALUES (?, ?, ?, ?, ?)
                """,
                (person_id, ts, first_seen, last_seen, 1 if is_active else 0),
            )
            return cursor.rowcount > 0

    def person_exists(self, person_id: str) -> bool:
        """Return True if person is registered."""
        row = fetch_one(
            self.db_path,
            "SELECT 1 FROM persons WHERE person_id = ? LIMIT 1",
            (person_id,),
        )
        return row is not None

    def get_person(self, person_id: str) -> dict[str, Any] | None:
        """Fetch one person by id."""
        row = fetch_one(
            self.db_path,
            """
            SELECT person_id, created_at, first_seen_at, last_seen_at, is_active
            FROM persons
            WHERE person_id = ?
            """,
            (person_id,),
        )
        return self._row_to_dict(row)

    def update_person_last_seen(self, person_id: str, seen_at: str | None = None) -> bool:
        """Update last_seen_at; returns True if person row was updated."""
        ts = seen_at or utc_now_iso()
        with transaction(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE persons SET last_seen_at = ? WHERE person_id = ?",
                (ts, person_id),
            )
            return cursor.rowcount > 0

    def store_embedding(
        self,
        person_id: str,
        embedding: np.ndarray,
        quality_score: float | None = None,
        created_at: str | None = None,
    ) -> int:
        """Store one normalized embedding for a person.

        Returns:
            New embedding_id.
        """
        if not self.person_exists(person_id):
            raise ValueError(f"Person not found: {person_id}")

        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError("embedding cannot be empty")

        blob = vector.tobytes()
        ts = created_at or utc_now_iso()

        with transaction(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO face_embeddings
                    (person_id, vector_blob, quality_score, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (person_id, blob, quality_score, ts),
            )
            return int(cursor.lastrowid)

    def get_person_embeddings(self, person_id: str) -> list[np.ndarray]:
        """Return all embeddings for a single person."""
        rows = fetch_all(
            self.db_path,
            """
            SELECT vector_blob
            FROM face_embeddings
            WHERE person_id = ?
            ORDER BY embedding_id ASC
            """,
            (person_id,),
        )
        return [np.frombuffer(row["vector_blob"], dtype=np.float32) for row in rows]

    def get_all_known_embeddings(self) -> dict[str, list[np.ndarray]]:
        """Return embeddings grouped by person_id for recognition gallery."""
        rows = fetch_all(
            self.db_path,
            """
            SELECT person_id, vector_blob
            FROM face_embeddings
            ORDER BY embedding_id ASC
            """
        )
        grouped: dict[str, list[np.ndarray]] = {}
        for row in rows:
            pid = row["person_id"]
            grouped.setdefault(pid, []).append(np.frombuffer(row["vector_blob"], dtype=np.float32))
        return grouped

    def list_persons(self) -> list[dict[str, Any]]:
        """Return all registered persons."""
        rows = fetch_all(
            self.db_path,
            """
            SELECT person_id, created_at, first_seen_at, last_seen_at, is_active
            FROM persons
            ORDER BY created_at ASC
            """
        )
        return [dict(row) for row in rows]

    # -------------------------------------------------------------------------
    # Visit methods
    # -------------------------------------------------------------------------
    def has_open_visit(self, person_id: str) -> bool:
        """Return True if person currently has an OPEN visit."""
        row = fetch_one(
            self.db_path,
            """
            SELECT 1
            FROM visits
            WHERE person_id = ? AND status = 'OPEN'
            LIMIT 1
            """,
            (person_id,),
        )
        return row is not None

    def get_open_visit(self, person_id: str) -> dict[str, Any] | None:
        """Fetch current OPEN visit for person (if any)."""
        row = fetch_one(
            self.db_path,
            """
            SELECT visit_id, person_id, entry_time, exit_time, entry_crop_path, exit_crop_path, status
            FROM visits
            WHERE person_id = ? AND status = 'OPEN'
            ORDER BY visit_id DESC
            LIMIT 1
            """,
            (person_id,),
        )
        return self._row_to_dict(row)

    def open_visit(
        self,
        person_id: str,
        entry_time: str | None = None,
        entry_crop_path: str | None = None,
    ) -> int:
        """Open visit for person if none open; returns visit_id.

        If an OPEN visit already exists, returns that existing visit_id.
        """
        existing = self.get_open_visit(person_id)
        if existing:
            return int(existing["visit_id"])

        ts = entry_time or utc_now_iso()
        with transaction(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO visits
                    (person_id, entry_time, exit_time, entry_crop_path, exit_crop_path, status)
                VALUES (?, ?, NULL, ?, NULL, 'OPEN')
                """,
                (person_id, ts, entry_crop_path),
            )
            return int(cursor.lastrowid)

    def close_visit(
        self,
        person_id: str,
        exit_time: str | None = None,
        exit_crop_path: str | None = None,
    ) -> int | None:
        """Close latest OPEN visit for person.

        Returns:
            Closed visit_id, or None if no open visit exists.
        """
        open_visit = self.get_open_visit(person_id)
        if not open_visit:
            return None

        visit_id = int(open_visit["visit_id"])
        ts = exit_time or utc_now_iso()
        with transaction(self.db_path) as conn:
            conn.execute(
                """
                UPDATE visits
                SET exit_time = ?, exit_crop_path = ?, status = 'CLOSED'
                WHERE visit_id = ? AND status = 'OPEN'
                """,
                (ts, exit_crop_path, visit_id),
            )
        return visit_id

    # -------------------------------------------------------------------------
    # Event methods
    # -------------------------------------------------------------------------
    def write_event(
        self,
        event_type: str,
        event_time: str | None = None,
        person_id: str | None = None,
        visit_id: int | None = None,
        track_id: int | None = None,
        frame_index: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> int:
        """Insert one event record and return event_id."""
        ts = event_time or utc_now_iso()
        meta_json = json.dumps(meta or {}, ensure_ascii=True)

        with transaction(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO events
                    (person_id, visit_id, event_type, event_time, track_id, frame_index, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (person_id, visit_id, event_type, ts, track_id, frame_index, meta_json),
            )
            return int(cursor.lastrowid)

    def get_event_count_by_type(self, event_type: str) -> int:
        """Return total rows in events for the given event_type (read-only)."""
        row = fetch_one(
            self.db_path,
            "SELECT COUNT(*) AS total FROM events WHERE event_type = ?",
            (event_type,),
        )
        return int(row["total"]) if row else 0

    def get_event_history(
        self,
        person_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return event history ordered from newest to oldest."""
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if offset < 0:
            raise ValueError("offset must be >= 0")

        if person_id:
            rows = fetch_all(
                self.db_path,
                """
                SELECT event_id, person_id, visit_id, event_type, event_time, track_id, frame_index, meta_json
                FROM events
                WHERE person_id = ?
                ORDER BY event_time DESC, event_id DESC
                LIMIT ? OFFSET ?
                """,
                (person_id, limit, offset),
            )
        else:
            rows = fetch_all(
                self.db_path,
                """
                SELECT event_id, person_id, visit_id, event_type, event_time, track_id, frame_index, meta_json
                FROM events
                ORDER BY event_time DESC, event_id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )

        output: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["meta"] = json.loads(item.pop("meta_json") or "{}")
            except json.JSONDecodeError:
                item["meta"] = {}
                item.pop("meta_json", None)
            output.append(item)
        return output

    # -------------------------------------------------------------------------
    # Counters and metrics
    # -------------------------------------------------------------------------
    def get_unique_visitor_count(self) -> int:
        """Count distinct registered persons (not tracks)."""
        row = fetch_one(
            self.db_path,
            "SELECT COUNT(DISTINCT person_id) AS total FROM persons",
        )
        return int(row["total"]) if row else 0

    def set_counter(self, key: str, value: int) -> None:
        """Upsert a value into system_counters."""
        with transaction(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO system_counters(key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, int(value)),
            )

    def get_counter(self, key: str) -> int | None:
        """Get integer counter by key."""
        row = fetch_one(
            self.db_path,
            "SELECT value FROM system_counters WHERE key = ? LIMIT 1",
            (key,),
        )
        return int(row["value"]) if row else None

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return dict(row)
