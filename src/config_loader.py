"""Configuration loading and validation helpers.

This module keeps configuration handling simple and explicit:
- reads JSON safely
- applies practical defaults
- validates expected fields and value ranges
- returns a clean, app-ready dictionary
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


def load_config(config_path: str = "config.json") -> dict[str, Any]:
    """Load, validate, and normalize application configuration.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        A validated configuration dictionary with defaults applied.

    Raises:
        ConfigError: If the file cannot be read or validation fails.
    """
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    if not path.is_file():
        raise ConfigError(f"Config path is not a file: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in config file '{path}': {exc.msg}") from exc
    except OSError as exc:
        raise ConfigError(f"Failed to read config file '{path}': {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a JSON object.")

    cfg = _apply_defaults(raw)
    _validate_config(cfg, config_file=path)
    return cfg


def _apply_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply practical defaults while preserving user-provided values."""
    # Keep defaults aligned with the generated scaffold config.
    defaults: dict[str, Any] = {
        "input_mode": "video",
        "video_path": "data/test.mp4",
        "rtsp_url": "",
        "reconnect_enabled": True,
        "reconnect_delay_seconds": 5,
        "recognition_enabled": True,
        "detection_frame_skip": 2,
        "recognition_interval": 3,
        "detection_confidence": 0.45,
        "recognition_threshold": 0.45,
        "recognition_confirmation_window": 3,
        "recognition_confirmation_min_hits": 1,
        "tracker_max_lost_frames": 30,
        "auto_register_min_track_duration_frames": 4,
        "auto_register_min_track_hits": 8,
        "auto_register_min_detection_confidence": 0.45,
        "auto_register_min_unknown_hits": 3,
        "auto_register_min_quality_score": 8.0,
        "auto_register_identity_reuse_threshold": 0.58,
        "auto_register_fragmentation_merge_slack": 0.10,
        "entry_exit_absence_timeout_seconds": 5,
        "db_path": "data/face_tracker.db",
        "logs_dir": "logs",
        "entry_crop_dir": "data/faces/entry",
        "exit_crop_dir": "data/faces/exit",
        "detector_model_path": "models/yolov8n-face.pt",
        "embedder_model": {
            "provider": "insightface",
            "model_name": "buffalo_l",
            "execution_provider": "CPUExecutionProvider",
        },
    }

    cfg = {**defaults, **raw}
    if "embedder_model" in raw and isinstance(raw["embedder_model"], dict):
        cfg["embedder_model"] = {**defaults["embedder_model"], **raw["embedder_model"]}
    return cfg


def _validate_config(cfg: dict[str, Any], config_file: Path) -> None:
    """Validate field presence, types, ranges, and mode-specific rules."""
    required_top_level = [
        "input_mode",
        "video_path",
        "rtsp_url",
        "reconnect_enabled",
        "reconnect_delay_seconds",
        "detection_frame_skip",
        "recognition_interval",
        "detection_confidence",
        "recognition_threshold",
        "tracker_max_lost_frames",
        "entry_exit_absence_timeout_seconds",
        "db_path",
        "logs_dir",
        "entry_crop_dir",
        "exit_crop_dir",
        "detector_model_path",
        "embedder_model",
    ]

    for key in required_top_level:
        if key not in cfg:
            raise ConfigError(f"Missing required config field: '{key}'")

    input_mode = _as_str(cfg["input_mode"], "input_mode").lower().strip()
    if input_mode not in {"video", "rtsp"}:
        raise ConfigError("input_mode must be either 'video' or 'rtsp'.")
    cfg["input_mode"] = input_mode

    # Source validation by mode.
    video_path = _as_str(cfg["video_path"], "video_path").strip()
    rtsp_url = _as_str(cfg["rtsp_url"], "rtsp_url").strip()
    if input_mode == "video":
        if not video_path:
            raise ConfigError("video_path cannot be empty when input_mode='video'.")
    else:  # rtsp mode
        if not rtsp_url:
            raise ConfigError("rtsp_url cannot be empty when input_mode='rtsp'.")
        if not rtsp_url.lower().startswith("rtsp://"):
            raise ConfigError("rtsp_url must start with 'rtsp://'.")

    # Numeric validation (practical ranges for hackathon use).
    _as_bool(cfg["reconnect_enabled"], "reconnect_enabled")
    _as_bool(cfg["recognition_enabled"], "recognition_enabled")
    _as_int(cfg["reconnect_delay_seconds"], "reconnect_delay_seconds", minimum=1, maximum=120)
    _as_int(cfg["detection_frame_skip"], "detection_frame_skip", minimum=1, maximum=30)
    _as_int(cfg["recognition_interval"], "recognition_interval", minimum=1, maximum=300)
    _as_int(cfg["recognition_confirmation_window"], "recognition_confirmation_window", minimum=1, maximum=20)
    _as_int(cfg["recognition_confirmation_min_hits"], "recognition_confirmation_min_hits", minimum=1, maximum=20)
    _as_int(cfg["tracker_max_lost_frames"], "tracker_max_lost_frames", minimum=1, maximum=300)
    _as_int(
        cfg["auto_register_min_track_duration_frames"],
        "auto_register_min_track_duration_frames",
        minimum=1,
        maximum=300,
    )
    _as_int(cfg["auto_register_min_track_hits"], "auto_register_min_track_hits", minimum=1, maximum=300)
    _as_int(cfg["auto_register_min_unknown_hits"], "auto_register_min_unknown_hits", minimum=1, maximum=100)
    _as_int(
        cfg["entry_exit_absence_timeout_seconds"],
        "entry_exit_absence_timeout_seconds",
        minimum=1,
        maximum=3600,
    )

    _as_float(cfg["detection_confidence"], "detection_confidence", minimum=0.0, maximum=1.0)
    _as_float(cfg["recognition_threshold"], "recognition_threshold", minimum=0.0, maximum=1.0)
    _as_float(
        cfg["auto_register_min_detection_confidence"],
        "auto_register_min_detection_confidence",
        minimum=0.0,
        maximum=1.0,
    )
    _as_float(cfg["auto_register_min_quality_score"], "auto_register_min_quality_score", minimum=0.0, maximum=10000.0)
    _as_float(
        cfg["auto_register_identity_reuse_threshold"],
        "auto_register_identity_reuse_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    _as_float(
        cfg["auto_register_fragmentation_merge_slack"],
        "auto_register_fragmentation_merge_slack",
        minimum=0.0,
        maximum=0.5,
    )

    if cfg["recognition_confirmation_min_hits"] > cfg["recognition_confirmation_window"]:
        raise ConfigError("recognition_confirmation_min_hits cannot exceed recognition_confirmation_window.")

    # Path-like settings should be non-empty strings.
    path_fields = [
        "db_path",
        "logs_dir",
        "entry_crop_dir",
        "exit_crop_dir",
        "detector_model_path",
    ]
    for field in path_fields:
        value = _as_str(cfg[field], field).strip()
        if not value:
            raise ConfigError(f"{field} cannot be empty.")

    # We do not force all paths to exist now; many are created at runtime.
    # But validate parent directory of config file exists (sanity check).
    if not config_file.parent.exists():
        raise ConfigError(f"Config directory does not exist: {config_file.parent}")

    # Embedder settings
    embedder_model = cfg["embedder_model"]
    if not isinstance(embedder_model, dict):
        raise ConfigError("embedder_model must be an object.")

    for key in ("provider", "model_name", "execution_provider"):
        if key not in embedder_model:
            raise ConfigError(f"embedder_model missing required field: '{key}'")
        text = _as_str(embedder_model[key], f"embedder_model.{key}").strip()
        if not text:
            raise ConfigError(f"embedder_model.{key} cannot be empty.")

    provider = embedder_model["provider"].strip().lower()
    if provider not in {"insightface", "arcface"}:
        raise ConfigError("embedder_model.provider must be 'insightface' or 'arcface'.")
    embedder_model["provider"] = provider


def _as_str(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{field} must be a string.")
    return value


def _as_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be a boolean.")
    return value


def _as_int(value: Any, field: str, minimum: int | None = None, maximum: int | None = None) -> int:
    # bool is a subclass of int, so reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer.")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{field} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{field} must be <= {maximum}.")
    return value


def _as_float(
    value: Any, field: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a number.")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise ConfigError(f"{field} must be >= {minimum}.")
    if maximum is not None and numeric > maximum:
        raise ConfigError(f"{field} must be <= {maximum}.")
    return numeric
