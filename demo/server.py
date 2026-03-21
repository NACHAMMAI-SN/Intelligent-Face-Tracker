"""Flask demo dashboard: read-only views + optional in-browser pipeline runs.

Runs the real Pipeline.from_config / run flow using a temporary config file per request.
Does not modify the root config.json.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

_DEMO_DIR = Path(__file__).resolve().parent
_ROOT = _DEMO_DIR.parent
_UPLOAD_DIR = _DEMO_DIR / "uploads"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_config_raw() -> dict[str, Any]:
    cfg_path = _ROOT / "config.json"
    if not cfg_path.is_file():
        return {}
    with cfg_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _resolve_paths() -> tuple[Path, Path]:
    cfg = _load_config_raw()
    db_path = _ROOT / str(cfg.get("db_path", "data/face_tracker.db"))
    logs_dir = _ROOT / str(cfg.get("logs_dir", "logs"))
    return db_path, logs_dir / "app.log"


def _parse_last_pipeline_stats(app_log: Path) -> dict[str, Any] | None:
    if not app_log.is_file():
        return None
    marker = "Pipeline stopped | "
    last_stats: dict[str, Any] | None = None
    with app_log.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if marker not in line:
                continue
            idx = line.find(marker)
            try:
                payload = json.loads(line[idx + len(marker) :].strip())
            except json.JSONDecodeError:
                continue
            stats = payload.get("stats")
            if isinstance(stats, dict):
                last_stats = stats
    return last_stats


def _last_run_app_log_lines(app_log: Path) -> list[str]:
    if not app_log.is_file():
        return []
    lines = app_log.read_text(encoding="utf-8", errors="replace").splitlines()
    last_start = -1
    for i, line in enumerate(lines):
        if "Pipeline started |" in line:
            last_start = i
    if last_start < 0:
        return lines
    return lines[last_start:]


def _db_snapshot(db_path: Path) -> dict[str, Any] | None:
    if not db_path.is_file():
        return None
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS c FROM persons")
        persons_count = int(cur.fetchone()["c"])
        cur.execute("SELECT event_type, COUNT(*) AS c FROM events GROUP BY event_type")
        by_type = {str(row["event_type"]): int(row["c"]) for row in cur.fetchall()}
    finally:
        conn.close()
    return {"persons_count": persons_count, "events_by_type": by_type}


def _proof_counts(db_path: Path) -> dict[str, Any]:
    snap = _db_snapshot(db_path)
    if snap is None:
        return {
            "persons": None,
            "registered_events": None,
            "entries": None,
            "exits": None,
        }
    by = snap["events_by_type"]
    return {
        "persons": snap["persons_count"],
        "registered_events": by.get("REGISTERED"),
        "entries": by.get("ENTRY"),
        "exits": by.get("EXIT"),
    }


def _db_path_from_runtime_cfg(cfg: dict[str, Any]) -> Path:
    raw = cfg.get("db_path", "data/face_tracker.db")
    p = Path(str(raw))
    return p.resolve() if p.is_absolute() else (_ROOT / p).resolve()


def _redact_rtsp_url(url: str) -> str:
    u = (url or "").strip()
    if not u.startswith("rtsp://"):
        return u
    if "@" not in u:
        return u
    try:
        rest = u.split("://", 1)[1]
        if "@" in rest:
            hostpart = rest.split("@", 1)[1]
            return f"rtsp://***@{hostpart}"
    except (ValueError, IndexError):
        pass
    return "rtsp://***"


def _config_summary_for_api() -> dict[str, Any]:
    try:
        from src.config_loader import load_config

        cfg = load_config(str(_ROOT / "config.json"))
    except Exception as exc:  # noqa: BLE001
        raw = _load_config_raw()
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "input_mode": raw.get("input_mode"),
            "video_path": raw.get("video_path"),
            "rtsp_url_redacted": _redact_rtsp_url(str(raw.get("rtsp_url", ""))),
            "db_path": raw.get("db_path"),
            "logs_dir": raw.get("logs_dir"),
            "recognition_enabled": raw.get("recognition_enabled"),
        }
    return {
        "ok": True,
        "input_mode": cfg.get("input_mode"),
        "video_path": cfg.get("video_path"),
        "rtsp_url_redacted": _redact_rtsp_url(str(cfg.get("rtsp_url", ""))),
        "db_path": cfg.get("db_path"),
        "logs_dir": cfg.get("logs_dir"),
        "recognition_enabled": cfg.get("recognition_enabled"),
        "detection_frame_skip": cfg.get("detection_frame_skip"),
        "recognition_interval": cfg.get("recognition_interval"),
    }


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(_ROOT))
    except ValueError:
        return str(p)


# --- Single-flight pipeline runs (synchronous request; no Celery / websockets) ---
_run_lock = threading.Lock()
_run_state: dict[str, Any] = {
    "status": "idle",
    "last_error": None,
    "last_stats": None,
    "last_proof": None,
    "last_source": None,
}


def _execute_pipeline_run(
    *,
    cfg_overrides: dict[str, Any],
    source_info: dict[str, Any],
) -> dict[str, Any]:
    """Merge overrides into validated base config, run pipeline from project root."""
    from src.config_loader import load_config
    from src.pipeline import Pipeline

    _run_state["status"] = "running"
    _run_state["last_error"] = None
    _run_state["last_source"] = source_info

    tmp_path: Path | None = None
    old_cwd = os.getcwd()
    try:
        base = deepcopy(load_config(str(_ROOT / "config.json")))
        base.update(cfg_overrides)

        fd, tmp_name = tempfile.mkstemp(prefix="katomaran_run_", suffix=".json", text=True)
        os.close(fd)
        tmp_path = Path(tmp_name)
        tmp_path.write_text(json.dumps(base, indent=2), encoding="utf-8")
        load_config(str(tmp_path))

        os.chdir(_ROOT)
        try:
            pipeline = Pipeline.from_config(str(tmp_path))
            stats = pipeline.run()
        finally:
            os.chdir(old_cwd)

        db_path = _db_path_from_runtime_cfg(base)
        proof = _proof_counts(db_path)

        _run_state["status"] = "done"
        _run_state["last_stats"] = stats
        _run_state["last_proof"] = proof

        return {
            "ok": True,
            "stats": stats,
            "proof": proof,
            "source": source_info,
            "db_path": _rel(db_path),
        }
    except Exception as exc:  # noqa: BLE001
        _run_state["status"] = "failed"
        _run_state["last_error"] = f"{type(exc).__name__}: {exc}"
        _run_state["last_stats"] = None
        _run_state["last_proof"] = None
        return {"ok": False, "error": _run_state["last_error"], "source": source_info}
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        if _run_state["status"] == "running":
            _run_state["status"] = "failed"
            _run_state["last_error"] = _run_state.get("last_error") or "Run ended unexpectedly."


app = Flask(
    __name__,
    template_folder=str(_DEMO_DIR / "templates"),
    static_folder=str(_DEMO_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


@app.get("/")
def index() -> str:
    return render_template("dashboard.html")


@app.get("/api/config")
def api_config() -> Any:
    return jsonify(_config_summary_for_api())


@app.get("/api/run/status")
def api_run_status() -> Any:
    return jsonify(_run_state)


@app.post("/api/run/video")
def api_run_video() -> Any:
    if not _run_lock.acquire(blocking=False):
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "A pipeline run is already in progress. Wait for it to finish.",
                }
            ),
            409,
        )
    try:
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "Missing file field (use input name 'file')."}), 400
        upload = request.files["file"]
        if not upload or not upload.filename:
            return jsonify({"ok": False, "error": "No file selected."}), 400

        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        safe = secure_filename(upload.filename) or "upload.bin"
        dest = _UPLOAD_DIR / safe
        upload.save(str(dest))
        abs_video = str(dest.resolve())

        source_info = {
            "input_mode": "video",
            "uploaded_path_relative": _rel(dest),
            "uploaded_path_absolute": abs_video,
        }
        result = _execute_pipeline_run(
            cfg_overrides={"input_mode": "video", "video_path": abs_video},
            source_info=source_info,
        )
        status = 200 if result.get("ok") else 500
        return jsonify(result), status
    finally:
        _run_lock.release()


@app.post("/api/run/rtsp")
def api_run_rtsp() -> Any:
    if not _run_lock.acquire(blocking=False):
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "A pipeline run is already in progress. Wait for it to finish.",
                }
            ),
            409,
        )
    try:
        payload = request.get_json(silent=True) or {}
        url = str(payload.get("rtsp_url", "")).strip()
        if not url:
            return jsonify({"ok": False, "error": "rtsp_url is required (JSON body)."}), 400
        if not url.lower().startswith("rtsp://"):
            return jsonify({"ok": False, "error": "rtsp_url must start with 'rtsp://'."}), 400

        source_info = {"input_mode": "rtsp", "rtsp_url": url}
        result = _execute_pipeline_run(
            cfg_overrides={"input_mode": "rtsp", "rtsp_url": url},
            source_info=source_info,
        )
        status = 200 if result.get("ok") else 500
        return jsonify(result), status
    finally:
        _run_lock.release()


@app.get("/api/stats")
def api_stats() -> Any:
    db_path, app_log = _resolve_paths()
    snap = _db_snapshot(db_path)
    stats = _parse_last_pipeline_stats(app_log)
    segment = _last_run_app_log_lines(app_log)
    reuse_in_last_run = sum(
        1 for line in segment if "Existing identity reused for track" in line
    )
    return jsonify(
        {
            "pipeline_stats_last_run": stats,
            "db": snap,
            "last_run_identity_reuse_log_lines": reuse_in_last_run,
            "paths": {"db_path": _rel(db_path), "app_log": _rel(app_log)},
        }
    )


@app.get("/api/events-summary")
def api_events_summary() -> Any:
    db_path, _ = _resolve_paths()
    snap = _db_snapshot(db_path)
    if snap is None:
        return jsonify({"events_by_type": {}, "persons_count": None, "error": "database not found"})
    return jsonify(
        {
            "events_by_type": snap["events_by_type"],
            "persons_count": snap["persons_count"],
            "paths": {"db_path": _rel(db_path)},
        }
    )


@app.get("/api/proof")
def api_proof() -> Any:
    db_path, app_log = _resolve_paths()
    snap = _db_snapshot(db_path)
    stats = _parse_last_pipeline_stats(app_log)
    segment = _last_run_app_log_lines(app_log)
    reuse_lines = sum(1 for line in segment if "Existing identity reused for track" in line)

    persons_count = snap["persons_count"] if snap else None
    uv_log = stats.get("unique_visitors") if stats else None
    aligned: bool | None = None
    if persons_count is not None and uv_log is not None:
        aligned = int(persons_count) == int(uv_log)

    return jsonify(
        {
            "persons_table_count": persons_count,
            "unique_visitors_last_pipeline_stats": uv_log,
            "unique_visitors_matches_persons_table": aligned,
            "registered_total_last_run": stats.get("registered_total") if stats else None,
            "identity_reuse_bindings_last_run": stats.get("identity_reuse_bindings_total")
            if stats
            else None,
            "identity_reuse_log_lines_last_run": reuse_lines,
            "events_registered_total_in_db": snap["events_by_type"].get("REGISTERED")
            if snap
            else None,
            "events_entry_total_in_db": snap["events_by_type"].get("ENTRY") if snap else None,
            "events_exit_total_in_db": snap["events_by_type"].get("EXIT") if snap else None,
            "notes": [
                "Reused tracks bind to an existing person_id; they do not add a row to persons.",
                "REGISTERED (DB + events.log) is for brand-new identities; reuse bindings increment "
                "Identity Reuse Bindings without a new persons row or REGISTERED for that binding.",
                "Unique Visitors in terminal stats uses COUNT(distinct person_id) on persons, "
                "aligned with the persons table after a full pipeline run.",
                "REGISTERED row count in SQLite may exceed a single run's Registered Total if the DB "
                "accumulated events across runs.",
            ],
        }
    )


def main() -> None:
    app.run(host="127.0.0.1", port=5050, debug=False)


if __name__ == "__main__":
    main()
