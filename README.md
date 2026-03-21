# Intelligent Face Tracker with Auto-Registration and Unique Visitor Counting

End-to-end **tracker-first** pipeline: stable `track_id` from the tracker drives face crops, embeddings, recognition, optional auto-registration, and visit lifecycle. Supports **local video** and **RTSP** via `src/video_source.py`.

## Problem statement addressed

- Multi-face **detection** and **tracking**
- **Recognition** of known identities (cosine similarity + optional per-track confirmation)
- **Auto-registration** of unknown but stable faces into SQLite
- **Duplicate prevention / identity reuse**: fragmented tracks can bind to an **existing** `person_id` when gallery similarity passes an **effective** threshold (see `docs/assumptions.md`)
- **Unique visitor** count = distinct rows in `persons` (not track count)
- **ENTRY** / **EXIT** visit events with SQLite + mandatory JSONL `events.log`

## Objective

Demo-ready judging: clear modules, reproducible commands, observable **DB**, **`logs/`**, and **terminal stats**.

## Tracker-first pipeline

Per processed frame:

1. **Track** — `FaceTracker` maintains `track_id`, hit counts, and lifecycle (active / lost / dead).
2. **Per-track crop** — `crop_face_safe()` uses the tracker box for the identity path.
3. **Embed** — `FaceEmbedder` (InsightFace) on a schedule (`recognition_interval`).
4. **Recognize** — `FaceRecognizer` matches against an in-memory gallery (loaded from DB and updated at runtime).
5. **Auto-register** — If not a confirmed match, `AutoRegistrar` applies stability gates, then **reuses** an existing person (DB gallery) or **creates** a new `person_id`.
6. **Visits** — `VisitManager` opens/closes visits; **ENTRY** / **EXIT** written to DB and logs as configured.
7. **Persistence** — `Repository` + `EventLogger` for `persons`, `face_embeddings`, `visits`, `events`.

## Recognition, auto-registration, and duplicate prevention

- **Known face** — Confirmed recognition binds `track_id` → `person_id`, increments **Recognized Total**, emits **RECOGNIZED**, refreshes visits.
- **Unknown face** — After stability gates, the registrar compares the embedding to `get_all_known_embeddings()` in SQLite. If similarity ≥ **effective reuse threshold** (`min(auto_register_identity_reuse_threshold, recognition_threshold - auto_register_fragmentation_merge_slack)`), the track binds to that person: **Identity Reuse Bindings Total** increases; `app.log` shows `Existing identity reused for track`, `duplicate_prevented_existing_identity`, `no_new_person_created`; **no** new `persons` row and **no** `REGISTERED` event for that binding.
- **Brand-new identity** — Otherwise `create_person`, **REGISTERED** (DB + `events.log` via logger), **Registered Total** increases.

## Unique visitors and event logging

| Output | Role |
| --- | --- |
| **`persons`** | One row per enrolled identity; **Unique Visitors** = `COUNT(DISTINCT person_id)`. |
| **`events` (SQLite)** | `ENTRY`, `EXIT`, `REGISTERED`, `RECOGNIZED` per schema in `src/db.py`. |
| **`logs/events.log`** | Mandatory JSONL stream (`EventLogger`). |
| **`logs/app.log`** | Operational logs, including identity reuse proof lines. |
| **Crops** | `data/faces/entry/`, `data/faces/exit/` (optional; may be gitignored). |

## Tech stack

Python 3.11+, OpenCV, NumPy, Ultralytics YOLO, InsightFace, ONNX Runtime, SQLite (stdlib).

## Architecture (module flow)

`VideoSource` → `YOLOFaceDetector` → `FaceTracker` → crop → `FaceEmbedder` → `FaceRecognizer` → (`AutoRegistrar` if unknown) → `VisitManager` → `EventLogger` / `Repository` → terminal stats from `Pipeline`.

### Diagram (Mermaid)

- Source: [architecture_diagram.mmd](docs/architecture_diagram.mmd)
- Rendered for preview: [architecture_diagram.md](docs/architecture_diagram.md)

## Project structure

```text
Katomaran/
├── main.py
├── config.json
├── requirements.txt
├── README.md
├── docs/
│   ├── architecture_diagram.mmd
│   └── architecture_diagram.md
├── demo/
├── scripts/
│   └── init_db.py
├── src/
├── logs/
├── data/
└── models/
```

(Other scripts and modules omitted for brevity.)

## Requirement → file mapping

| Area | Module |
| --- | --- |
| Detection | `src/detector_yolo.py` |
| Tracking | `src/tracker.py` |
| Embeddings | `src/embedder.py`, `src/face_aligner.py` |
| Recognition | `src/recognizer.py` |
| Auto-register + reuse | `src/auto_register.py` |
| Visits | `src/visit_manager.py` |
| Events + DB | `src/event_logger.py`, `src/repositories.py`, `src/db.py` |
| Orchestration | `src/pipeline.py`, `main.py` |

## Installation

```powershell
cd E:\Katomaran
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place detector weights per `config.json` (`detector_model_path`). InsightFace downloads its model on first use as configured.

## How to demo

From the project root:

```powershell
cd E:\Katomaran
python .\scripts\init_db.py
python .\main.py
```

With a venv:

```powershell
.\.venv\Scripts\python.exe .\scripts\init_db.py
.\.venv\Scripts\python.exe .\main.py
```

**For judges:** final terminal stats, `logs\events.log`, `logs\app.log`, and `data\face_tracker.db`.

## Demo dashboard (read-only views + optional browser runs)

Lightweight **Flask** UI and JSON APIs under `demo/`. **Read-only** sections read `config.json` for `db_path` / `logs_dir`, parse the last **`Pipeline stopped`** line from `logs/app.log`, and query SQLite.

**Optional:** start the real pipeline from the browser instead of (or after) `main.py`. Each browser run saves uploads under `demo/uploads/` and uses a **temporary JSON config** merged from root `config.json` plus that run's `input_mode` / `video_path` or `rtsp_url`. Root **`config.json` is not overwritten**. Only **one** run at a time; a second request returns HTTP **409**.

With dependencies installed (`pip install -r requirements.txt`), from the project root:

```powershell
.\.venv\Scripts\python.exe .\scripts\init_db.py
.\.venv\Scripts\python.exe .\main.py
.\.venv\Scripts\python.exe .\demo\server.py
```

Use **`main.py`** for a CLI-only pipeline run. For the dashboard (and optional browser-triggered runs), **`init_db.py`** then **`demo\server.py`** is enough; you do not need **`main.py`** first. Open **http://127.0.0.1:5050** in a browser.

### APIs

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/config` | Summary of validated `config.json` (RTSP URL redacted in `rtsp_url_redacted`). |
| GET | `/api/stats` | Last pipeline stats from `app.log` + DB snapshot. |
| GET | `/api/proof` | Proof facts + cumulative DB event counts (including ENTRY/EXIT). |
| GET | `/api/events-summary` | SQLite `events` counts + persons. |
| GET | `/api/run/status` | `idle` / `running` / `done` / `failed` and last run metadata. |
| POST | `/api/run/video` | Multipart field `file`: save upload, run `Pipeline`, return JSON `stats` + DB proof counts. |
| POST | `/api/run/rtsp` | JSON body with `rtsp_url` (`rtsp://…`): run pipeline on stream; returns JSON `stats` + proof. |

### Caveats

`POST /api/run/*` is **synchronous**: the request blocks until the pipeline finishes. Long videos or RTSP streams may run for a long time; browsers or proxies can time out.

## Expected output (verified — clean run, demo clip)

Example terminal stats from a **latest clean run** (identity reuse active; duplicate inflation reduced):

| Stat | Example value |
| --- | --- |
| **Registered Total** | **12** (new `person_id` enrollments this run only) |
| **Identity Reuse Bindings Total** | **17** (fragmented tracks bound to existing persons) |
| **Unique Visitors** | **12** (matches `persons` / distinct identities) |
| **Entries Total** | **23** |
| **Exits Total** | **23** |

### Logs

| Log | Notes |
| --- | --- |
| `logs/app.log` | Many lines **Existing identity reused for track** with `duplicate_prevented_existing_identity`, `no_new_person_created=true`, `effective_reuse_threshold` (e.g. **0.35**). |
| `logs/events.log` | **`REGISTERED`** only for true new identities (not for reuse bindings). |

Exact numbers may vary with `config.json`, video, and DB state; the **relationships** above should hold when reuse is working.

## Demo proof

- **Reused tracks** bind to an existing `person_id`; they do **not** add a row to **`persons`**.
- **`REGISTERED`** (DB + `events.log`) is emitted **only** for **brand-new** identities (`auto_registered_new_identity`), not for duplicate-prevented reuse.
- **Unique Visitors** uses `Repository.get_unique_visitor_count()` → **`COUNT(DISTINCT person_id)` on `persons`**, so it stays aligned with the persons table when reuse binds tracks without new enrollments.

## SQLite and log outputs

| Path | Content |
| --- | --- |
| `data/face_tracker.db` | `persons`, `face_embeddings`, `visits`, `events` |
| `logs/events.log` | JSONL: `ENTRY`, `EXIT`, `REGISTERED`, `RECOGNIZED` |
| `logs/app.log` | Pipeline and track decisions; identity reuse lines |

## Terminal metric semantics

| Stat | Meaning |
| --- | --- |
| **Unique Visitors** | `COUNT(DISTINCT person_id)` from `persons`. |
| **Registered Total** | New enrollments this run (`auto_registered_new_identity` only). |
| **Identity Reuse Bindings Total** | Tracks bound via duplicate prevention; no new person; no `REGISTERED` for that binding. |
| **Recognized Total** | Confirmed gallery matches this run. |
| **Entries / Exits** | Visit opens and closes from `VisitManager`. |

## entry_exit_absence_timeout_seconds

- **Larger** value → longer grace before EXIT when the face is missing → fewer ENTRY/EXIT cycles on the same clip.
- **Smaller** value → EXIT sooner → more cycles than with a longer timeout.
- Default in config is **5** s (looser than **3** s; stricter than **8** s in relative terms).

## Limitations and future work

- Detector and tracker quality dominate ID switches; dense crowds and motion still fragment tracks.
- Reuse uses a merge floor vs. recognition threshold; tune `auto_register_fragmentation_merge_slack` if needed.
- Single process; no multi-camera fusion or production HA.
- Tests, CI, and exportable judge reports are out of scope for this hackathon build. The Flask dashboard under `demo/` provides read-only views and optional in-browser pipeline runs as described above.

## Hackathon footer

This project is part of a hackathon run by <https://katomaran.com>.