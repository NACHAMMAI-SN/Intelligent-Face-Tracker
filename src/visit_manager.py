"""Visit state manager keyed by person_id.

Core guarantees:
- one ENTRY per visit
- one EXIT per visit
- at most one OPEN visit per person

This module is intentionally explicit and easy to debug for hackathon usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from src.repositories import Repository


def utc_now() -> datetime:
    """Current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def to_iso(dt: datetime) -> str:
    """Convert datetime to ISO-8601 string."""
    return dt.astimezone(timezone.utc).isoformat()


@dataclass
class PersonVisitState:
    """In-memory state for one person's current visit window."""

    visit_id: int
    is_open: bool
    entry_emitted: bool
    exit_emitted: bool
    last_seen_at: datetime
    entry_time: datetime
    last_track_id: int | None = None
    entry_crop_path: str | None = None
    exit_crop_path: str | None = None


class VisitManager:
    """Manages visit lifecycle and event emission correctness per person.

    Args:
        repository: Data access helper.
        absence_timeout_seconds: Exit timeout after person becomes missing.
    """

    def __init__(self, repository: Repository, absence_timeout_seconds: int = 5) -> None:
        if absence_timeout_seconds < 1:
            raise ValueError("absence_timeout_seconds must be >= 1")

        self.repository = repository
        self.absence_timeout_seconds = int(absence_timeout_seconds)
        self._states: dict[str, PersonVisitState] = {}

    @classmethod
    def from_config(cls, config: dict[str, Any], repository: Repository) -> "VisitManager":
        """Build VisitManager using app config defaults."""
        return cls(
            repository=repository,
            absence_timeout_seconds=int(config.get("entry_exit_absence_timeout_seconds", 5)),
        )

    def handle_confirmed_sighting(
        self,
        *,
        person_id: str,
        seen_at: datetime | None = None,
        track_id: int | None = None,
        frame_index: int | None = None,
        entry_crop_path: str | None = None,
    ) -> dict[str, Any]:
        """Handle a confirmed sighting of a known person.

        Behavior:
        - If person has no open visit: open visit + emit ENTRY once.
        - If person already has open visit: update last_seen only.
        """
        now = seen_at or utc_now()
        pid = str(person_id).strip()
        if not pid:
            raise ValueError("person_id cannot be empty")

        state = self._states.get(pid)

        # Recovery path: check DB in case process restarted.
        if state is None:
            db_open = self.repository.get_open_visit(pid)
            if db_open:
                state = PersonVisitState(
                    visit_id=int(db_open["visit_id"]),
                    is_open=True,
                    entry_emitted=True,
                    exit_emitted=False,
                    last_seen_at=now,
                    entry_time=now,
                    last_track_id=track_id,
                    entry_crop_path=db_open.get("entry_crop_path"),
                    exit_crop_path=None,
                )
                self._states[pid] = state

        # Create new visit if not currently open.
        if state is None or not state.is_open:
            visit_id = self.repository.open_visit(
                person_id=pid,
                entry_time=to_iso(now),
                entry_crop_path=entry_crop_path,
            )
            self.repository.write_event(
                event_type="ENTRY",
                person_id=pid,
                visit_id=visit_id,
                track_id=track_id,
                frame_index=frame_index,
                event_time=to_iso(now),
                meta={
                    "source": "visit_manager",
                    "entry_crop_path": entry_crop_path,
                },
            )

            self.repository.update_person_last_seen(pid, seen_at=to_iso(now))
            self._states[pid] = PersonVisitState(
                visit_id=visit_id,
                is_open=True,
                entry_emitted=True,
                exit_emitted=False,
                last_seen_at=now,
                entry_time=now,
                last_track_id=track_id,
                entry_crop_path=entry_crop_path,
                exit_crop_path=None,
            )
            return {
                "person_id": pid,
                "visit_id": visit_id,
                "entry_emitted": True,
                "exit_emitted": False,
                "state": "OPEN",
                "reason": "new_visit_opened",
            }

        # Existing open visit: refresh heartbeat only.
        state.last_seen_at = now
        state.last_track_id = track_id
        self.repository.update_person_last_seen(pid, seen_at=to_iso(now))
        return {
            "person_id": pid,
            "visit_id": state.visit_id,
            "entry_emitted": False,
            "exit_emitted": False,
            "state": "OPEN",
            "reason": "visit_refreshed",
        }

    def handle_missing_person(
        self,
        *,
        person_id: str,
        current_time: datetime | None = None,
        frame_index: int | None = None,
        exit_crop_path: str | None = None,
        force_exit: bool = False,
    ) -> dict[str, Any]:
        """Handle a potentially missing person and close visit if timeout reached.

        Args:
            person_id: Known person to evaluate for exit.
            current_time: Time used for timeout comparison.
            frame_index: Optional frame index for event metadata.
            exit_crop_path: Optional path of saved exit crop.
            force_exit: If True, close immediately (e.g., confirmed track death).
        """
        now = current_time or utc_now()
        pid = str(person_id).strip()
        if not pid:
            raise ValueError("person_id cannot be empty")

        state = self._states.get(pid)
        if state is None:
            # If we have no memory state, still protect correctness using DB open visit.
            db_open = self.repository.get_open_visit(pid)
            if not db_open:
                return {
                    "person_id": pid,
                    "visit_id": None,
                    "entry_emitted": False,
                    "exit_emitted": False,
                    "state": "NONE",
                    "reason": "no_open_visit",
                }
            state = PersonVisitState(
                visit_id=int(db_open["visit_id"]),
                is_open=True,
                entry_emitted=True,
                exit_emitted=False,
                last_seen_at=now,
                entry_time=now,
                last_track_id=None,
                entry_crop_path=db_open.get("entry_crop_path"),
                exit_crop_path=None,
            )
            self._states[pid] = state

        if not state.is_open or state.exit_emitted:
            return {
                "person_id": pid,
                "visit_id": state.visit_id,
                "entry_emitted": False,
                "exit_emitted": False,
                "state": "CLOSED",
                "reason": "already_closed",
            }

        timeout_due = state.last_seen_at + timedelta(seconds=self.absence_timeout_seconds)
        should_exit = force_exit or now >= timeout_due

        if not should_exit:
            return {
                "person_id": pid,
                "visit_id": state.visit_id,
                "entry_emitted": False,
                "exit_emitted": False,
                "state": "OPEN",
                "reason": "waiting_absence_timeout",
            }

        closed_visit_id = self.repository.close_visit(
            person_id=pid,
            exit_time=to_iso(now),
            exit_crop_path=exit_crop_path,
        )

        # If DB reports no open visit, avoid duplicate EXIT emission.
        if closed_visit_id is None:
            state.is_open = False
            state.exit_emitted = True
            return {
                "person_id": pid,
                "visit_id": state.visit_id,
                "entry_emitted": False,
                "exit_emitted": False,
                "state": "CLOSED",
                "reason": "db_already_closed",
            }

        exit_meta: dict[str, Any] = {
            "source": "visit_manager",
            "exit_crop_path": exit_crop_path,
            "force_exit": bool(force_exit),
        }
        if force_exit:
            exit_meta["reason"] = "visit_closed_forced"
        else:
            exit_meta["reason"] = "visit_timeout_exit"

        self.repository.write_event(
            event_type="EXIT",
            person_id=pid,
            visit_id=closed_visit_id,
            track_id=state.last_track_id,
            frame_index=frame_index,
            event_time=to_iso(now),
            meta=exit_meta,
        )

        state.is_open = False
        state.exit_emitted = True
        state.exit_crop_path = exit_crop_path
        close_reason = "visit_closed_forced" if force_exit else "visit_timeout_exit"
        return {
            "person_id": pid,
            "visit_id": closed_visit_id,
            "entry_emitted": False,
            "exit_emitted": True,
            "state": "CLOSED",
            "reason": close_reason,
        }

    def handle_missing_batch(
        self,
        *,
        currently_seen_person_ids: set[str] | list[str],
        current_time: datetime | None = None,
        frame_index: int | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate all open visits and close those absent beyond timeout."""
        now = current_time or utc_now()
        seen = {str(pid) for pid in currently_seen_person_ids}

        results: list[dict[str, Any]] = []
        open_person_ids = [pid for pid, state in self._states.items() if state.is_open]
        for pid in open_person_ids:
            if pid in seen:
                continue
            result = self.handle_missing_person(
                person_id=pid,
                current_time=now,
                frame_index=frame_index,
                exit_crop_path=None,
                force_exit=False,
            )
            results.append(result)
        return results

    def finalize_exits_on_shutdown(
        self,
        *,
        final_time: datetime | None = None,
        frame_index: int | None = None,
    ) -> list[dict[str, Any]]:
        """Force close all open visits at end-of-stream/shutdown."""
        now = final_time or utc_now()
        results: list[dict[str, Any]] = []
        open_person_ids = [pid for pid, state in self._states.items() if state.is_open]
        for pid in open_person_ids:
            result = self.handle_missing_person(
                person_id=pid,
                current_time=now,
                frame_index=frame_index,
                exit_crop_path=None,
                force_exit=True,
            )
            results.append(result)
        return results

    def get_open_person_ids(self) -> set[str]:
        """Return currently open visit person_ids from in-memory state."""
        return {pid for pid, state in self._states.items() if state.is_open}

    def get_state_snapshot(self) -> dict[str, dict[str, Any]]:
        """Return debug-friendly copy of internal state."""
        snapshot: dict[str, dict[str, Any]] = {}
        for pid, state in self._states.items():
            snapshot[pid] = {
                "visit_id": state.visit_id,
                "is_open": state.is_open,
                "entry_emitted": state.entry_emitted,
                "exit_emitted": state.exit_emitted,
                "last_seen_at": to_iso(state.last_seen_at),
                "entry_time": to_iso(state.entry_time),
                "last_track_id": state.last_track_id,
                "entry_crop_path": state.entry_crop_path,
                "exit_crop_path": state.exit_crop_path,
            }
        return snapshot
