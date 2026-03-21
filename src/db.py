"""SQLite database initialization and connection helpers.

This module provides:
- schema creation for core app tables
- safe connection helpers
- transaction context manager with rollback on failure
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS persons (
    person_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    first_seen_at TEXT,
    last_seen_at TEXT,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1))
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    vector_blob BLOB NOT NULL,
    quality_score REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS visits (
    visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    entry_crop_path TEXT,
    exit_crop_path TEXT,
    status TEXT NOT NULL CHECK (status IN ('OPEN', 'CLOSED')),
    FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT,
    visit_id INTEGER,
    event_type TEXT NOT NULL CHECK (
        event_type IN ('ENTRY', 'EXIT', 'REGISTERED', 'RECOGNIZED')
    ),
    event_time TEXT NOT NULL,
    track_id INTEGER,
    frame_index INTEGER,
    meta_json TEXT,
    FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE SET NULL,
    FOREIGN KEY (visit_id) REFERENCES visits(visit_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS system_counters (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_person_id
    ON face_embeddings(person_id);

CREATE INDEX IF NOT EXISTS idx_visits_person_id
    ON visits(person_id);

CREATE INDEX IF NOT EXISTS idx_visits_status
    ON visits(status);

CREATE INDEX IF NOT EXISTS idx_events_person_id
    ON events(person_id);

CREATE INDEX IF NOT EXISTS idx_events_type_time
    ON events(event_type, event_time);
"""


class DatabaseError(RuntimeError):
    """Raised for database initialization or transaction failures."""


def init_database(db_path: str) -> None:
    """Create SQLite database file and initialize schema.

    Args:
        db_path: Path to SQLite database file.
    """
    path = Path(db_path)
    _ensure_parent_dir(path)

    with get_connection(path) as conn:
        try:
            conn.executescript(SCHEMA_SQL)
            # Seed a practical default counter entry.
            conn.execute(
                """
                INSERT INTO system_counters(key, value)
                VALUES(?, ?)
                ON CONFLICT(key) DO NOTHING
                """,
                ("unique_visitor_count", 0),
            )
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise DatabaseError(f"Failed to initialize database schema: {exc}") from exc


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Return a configured SQLite connection.

    Notes:
        - row_factory is sqlite3.Row for dict-like row access.
        - foreign_keys pragma is enabled per connection.
    """
    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn
    except sqlite3.Error as exc:
        raise DatabaseError(f"Failed to connect to SQLite database: {exc}") from exc


@contextmanager
def transaction(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Context manager for safe DB transactions.

    Example:
        with transaction("data/face_tracker.db") as conn:
            conn.execute(...)
            conn.execute(...)

    Commits on success, rolls back on any exception.
    """
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN;")
        yield conn
        conn.commit()
    except Exception as exc:  # noqa: BLE001 - pass through original exception after rollback
        conn.rollback()
        raise DatabaseError(f"Transaction failed and was rolled back: {exc}") from exc
    finally:
        conn.close()


def fetch_one(db_path: str | Path, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
    """Execute SELECT query and return one row."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(sql, params)
        return cursor.fetchone()


def fetch_all(db_path: str | Path, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    """Execute SELECT query and return all rows."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(sql, params)
        return cursor.fetchall()


def execute(db_path: str | Path, sql: str, params: tuple[Any, ...] = ()) -> int:
    """Execute write query and return last inserted row id when available."""
    with transaction(db_path) as conn:
        cursor = conn.execute(sql, params)
        return int(cursor.lastrowid or 0)


def _ensure_parent_dir(path: Path) -> None:
    """Create parent directory for DB file if missing."""
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
