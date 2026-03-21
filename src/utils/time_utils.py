"""Minimal time and timestamp utility helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def now_utc() -> datetime:
    """Return current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def now_local() -> datetime:
    """Return current timezone-aware local datetime."""
    return datetime.now().astimezone()


def iso_utc(dt: datetime | None = None) -> str:
    """Return ISO-8601 string in UTC.

    Args:
        dt: Optional datetime. If None, current UTC time is used.
    """
    value = dt or now_utc()
    return value.astimezone(timezone.utc).isoformat()


def iso_local(dt: datetime | None = None) -> str:
    """Return ISO-8601 string in local timezone.

    Args:
        dt: Optional datetime. If None, current local time is used.
    """
    value = dt or now_local()
    return value.astimezone().isoformat()


def filename_safe_timestamp(dt: datetime | None = None, use_utc: bool = True) -> str:
    """Return compact timestamp safe for filenames.

    Example:
        20260320T113012123456Z
    """
    if dt is None:
        dt = now_utc() if use_utc else now_local()
    if use_utc:
        dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y%m%dT%H%M%S%fZ")
    return dt.astimezone().strftime("%Y%m%dT%H%M%S%f")
