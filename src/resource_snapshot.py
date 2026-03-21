"""Lightweight system resource snapshot for hackathon app.log (pipeline start/end only)."""

from __future__ import annotations

from typing import Any

import psutil


def snapshot_for_app_log() -> dict[str, Any]:
    """CPU %, RAM %, and used RAM (MB) for EventLogger operational lines."""
    try:
        cpu = float(psutil.cpu_percent(interval=0.05))
        vm = psutil.virtual_memory()
        return {
            "cpu_percent": round(cpu, 1),
            "ram_percent": round(vm.percent, 1),
            "ram_used_mb": round(vm.used / (1024 * 1024), 1),
        }
    except Exception:  # noqa: BLE001 — logging must not break shutdown
        return {
            "cpu_percent": None,
            "ram_percent": None,
            "ram_used_mb": None,
        }
