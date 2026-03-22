# Intelligent Face Tracker with Auto-Registration and Unique Visitor Counting

An end-to-end **tracker-first** face analytics pipeline for hackathon evaluation. The system detects and tracks multiple faces, recognizes known identities, auto-registers stable unknown faces, prevents duplicate identity creation through identity reuse, records `ENTRY` / `EXIT` visit events, and persists outputs to **SQLite** and structured logs.

It supports both:

- **Local video input**
- **RTSP stream input**

The project also includes a lightweight **Flask demo dashboard** for viewing results and optionally triggering runs from the browser.

---

## Demo Video

YouTube Demo Link: [Watch the project demo](https://youtu.be/VDnZWiXQpRQ)

## Problem statement coverage

This project addresses the required real-time intelligent video analytics workflow:

- Multi-face **detection**
- Multi-face **tracking**
- Face **recognition**
- **Auto-registration** of unknown stable faces
- **Duplicate prevention / identity reuse**
- **Unique visitor counting**
- `ENTRY` / `EXIT` **visit lifecycle events**
- **SQLite persistence**
- Structured **application logs**
- Structured **event logs**
- Demo-ready **frontend dashboard**
- Supporting **documentation**, **planning**, **architecture**, and **compute estimate**

---

## Objective

The goal of this submission is to build a modular, demo-ready AI video processing application that is easy to run, easy to verify, and easy to explain during hackathon judging.

The application is designed so judges can verify outputs through:

- Terminal run statistics
- SQLite database records
- Application logs
- Event logs
- Saved entry and exit crop images
- Dashboard views
- Documentation in the `docs/` folder

---

## Core features

### 1. Face detection

The system detects faces from video frames using a YOLO-based face detector.

**Main file:** `src/detector_yolo.py`

### 2. Face tracking

A tracker-first design is used so each active face is associated with a stable `track_id`. Recognition, visit handling, and auto-registration all happen around this tracked identity flow.

**Main file:** `src/tracker.py`

### 3. Face embedding

For tracked face crops, embeddings are generated using InsightFace.

**Main files:**

- `src/embedder.py`
- `src/face_aligner.py`

### 4. Face recognition

Known identities are matched using embedding similarity against an in-memory gallery loaded from SQLite.

**Main file:** `src/recognizer.py`

### 5. Auto-registration

Unknown but stable faces can be automatically enrolled as new identities after passing stability and quality checks.

**Main file:** `src/auto_register.py`

### 6. Duplicate prevention / identity reuse

Before creating a new identity, the pipeline checks whether the face is actually similar enough to an existing person in the database. If so, it reuses the existing `person_id` instead of creating a duplicate.

This is one of the most important parts of the project because it keeps **Unique Visitors** aligned with real identities instead of raw fragmented tracks.

### 7. Visit lifecycle

The system creates and closes visits and emits:

- `ENTRY`
- `EXIT`

**Main file:** `src/visit_manager.py`

### 8. Persistence and logging

The pipeline writes:

- Persons
- Embeddings
- Visits
- Events

to SQLite, and also writes:

- `logs/app.log`
- `logs/events.log`

**Main files:**

- `src/db.py`
- `src/repositories.py`
- `src/event_logger.py`

### 9. Demo dashboard

A lightweight Flask dashboard is included for:

- Viewing latest stats
- Viewing proof summaries
- Viewing DB-backed counts
- Optionally running the pipeline from uploaded video or RTSP

**Main file:** `demo/server.py`

---

## Tracker-first pipeline

For each processed frame, the high-level flow is:

1. **Video input** is read from local file or RTSP.
2. **Face detection** identifies face bounding boxes.
3. **Tracking** maintains `track_id` across frames.
4. **Per-track crop extraction** creates stable crops for identity work.
5. **Embedding generation** converts face crops into vectors.
6. **Recognition** attempts to match against known identities.
7. **Auto-registration** handles stable unknown faces.
8. **Identity reuse logic** prevents duplicate person creation.
9. **Visit manager** emits `ENTRY` and `EXIT`.
10. **Repository and logger** persist outputs to DB and logs.
11. **Pipeline stats** are shown in terminal and written to logs.

---

## Recognition, auto-registration, and duplicate prevention

### Known face flow

If a face matches a known gallery identity with sufficient confidence:

- The track binds to the existing `person_id`
- A `RECOGNIZED` event may be emitted
- The visit is refreshed
- `Recognized Total` increases

### Unknown face flow

If a face is not confirmed as known:

- Stability and quality checks are applied
- The embedding is compared to stored known embeddings
- If similarity crosses the effective reuse threshold, the pipeline:
  - Reuses the existing `person_id`
  - Increases `Identity Reuse Bindings Total`
  - Avoids creating a duplicate person
- Otherwise:
  - A new `person_id` is created
  - A `REGISTERED` event is emitted
  - `Registered Total` increases

This behavior helps ensure:

- Fewer duplicate identities
- More realistic visitor counting
- Better alignment between DB records and run metrics

---

## Unique visitors

**Unique Visitors** is based on the number of distinct people stored in the `persons` table, not the number of tracks.

That means:

- One real person should ideally map to one `person_id`
- Fragmented tracks should not inflate the visitor count if identity reuse works properly

---

## Outputs generated by the system

### SQLite database

**File:** `data/face_tracker.db`

**Main tables:**

- `persons`
- `face_embeddings`
- `visits`
- `events`

### Application log

**File:** `logs/app.log`

**Contains:**

- Pipeline start/stop
- Operational decisions
- Identity reuse lines
- Resource usage snapshots
- Final stats

### Event log

**File:** `logs/events.log`

Contains machine-readable JSONL event records such as:

- `REGISTERED`
- `RECOGNIZED`
- `ENTRY`
- `EXIT`

### Saved crop images

**Folders:** `data/faces/entry/`, `data/faces/exit/`

These store cropped face images associated with visit events.

---

## Tech stack

- Python 3.11+
- OpenCV
- NumPy
- Ultralytics YOLO
- InsightFace
- ONNX Runtime
- SQLite
- Flask
- psutil

---

## Architecture

Architecture and planning files are included under `docs/`.

| Document | Path |
| --- | --- |
| AI planning | `docs/ai_planning.md` |
| Architecture | `docs/architecture.md` |
| Mermaid source | `docs/architecture_diagram.mmd` |
| Rendered preview | `docs/architecture_diagram.md` |
| Assumptions | `docs/assumptions.md` |
| Compute estimate | `docs/compute_estimate.md` |
| Sample output | `docs/sample_output.md` |
| Submission notes | `docs/submission_notes.md` |

---

## Project structure

```text
Katomaran/
├── main.py
├── config.json
├── requirements.txt
├── README.md
├── docs/
│   ├── ai_planning.md
│   ├── architecture.md
│   ├── architecture_diagram.md
│   ├── architecture_diagram.mmd
│   ├── assumptions.md
│   ├── compute_estimate.md
│   ├── sample_output.md
│   ├── sample_outputs/
│   │   ├── images/
│   │   └── logs/
│   └── submission_notes.md
├── demo/
│   ├── server.py
│   ├── static/
│   └── templates/
├── scripts/
│   ├── init_db.py
│   ├── run_pipeline.py
│   └── run_video_demo.py
├── src/
├── data/
├── logs/
└── models/
```

*(Other scripts and modules omitted for brevity.)*

---

## Requirement → file mapping

| Area | Main module |
| --- | --- |
| Detection | `src/detector_yolo.py` |
| Tracking | `src/tracker.py` |
| Embeddings | `src/embedder.py`, `src/face_aligner.py` |
| Recognition | `src/recognizer.py` |
| Auto-registration | `src/auto_register.py` |
| Visit lifecycle | `src/visit_manager.py` |
| Event logging | `src/event_logger.py` |
| DB persistence | `src/repositories.py`, `src/db.py` |
| Pipeline orchestration | `src/pipeline.py`, `main.py` |
| Demo dashboard | `demo/server.py` |

---

## Installation

From the project root:

```powershell
cd E:\Katomaran
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Notes**

- The detector model path is configured in `config.json`.
- InsightFace may download its model assets on first use.
- The default sample input is `data/test.mp4`.

---

## Configuration

**Main config file:** `config.json`

Default important fields include:

- `input_mode`
- `video_path`
- `rtsp_url`
- Detection thresholds
- Recognition thresholds
- Tracker settings
- Auto-registration settings
- Database path
- Logs directory
- Model path

**Example default behavior**

- `input_mode` = `"video"`
- `video_path` = `"data/test.mp4"`

A fresh run uses the included sample video by default.

---

## How to run

### Option 1: CLI pipeline run

```powershell

.\.venv\Scripts\Activate.ps1
python .\scripts\init_db.py
python .\main.py
```

This will:

- Initialize the SQLite DB
- Process the configured input
- Generate logs
- Generate DB entries
- Save entry/exit crop images
- Print final terminal stats

### Option 2: Dashboard only

```powershell
cd E:\Katomaran
.\.venv\Scripts\Activate.ps1
python .\scripts\init_db.py
python .\demo\server.py
```

Then open: **http://127.0.0.1:5050**

This is enough for dashboard use, including optional browser-triggered runs.

### Option 3: Pipeline + dashboard

```powershell

.\.venv\Scripts\Activate.ps1
python .\scripts\init_db.py
python .\main.py
python .\demo\server.py
```

Use this flow if you want one CLI proof run first, then dashboard viewing afterward.

---

## Dashboard capabilities

The Flask dashboard provides:

- Config summary
- Run status
- Proof summary
- Stats summary
- Cumulative DB-backed event counts

It also supports:

- Local video upload through browser
- RTSP input through browser

Browser-triggered runs use a **temporary** merged config; root **`config.json` is not overwritten**. Only **one** pipeline run at a time from the dashboard; overlapping runs return **409**. `POST /api/run/*` calls are **synchronous** (blocking until the run finishes).

### Supported endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/config` | Summary of validated `config.json` |
| GET | `/api/stats` | Last pipeline stats from `app.log` plus DB snapshot |
| GET | `/api/proof` | Proof facts and cumulative DB event counts |
| GET | `/api/events-summary` | SQLite `events` counts and persons count |
| GET | `/api/run/status` | Current run status |
| POST | `/api/run/video` | Upload and process a video file |
| POST | `/api/run/rtsp` | Process an RTSP stream |

**Important:** Browser-triggered runs are synchronous, so very long videos or streams may take a while and can time out in browser or proxy environments.

---

## Sample verified output

A clean verified run produced the following example final stats:

| Stat | Example value |
| --- | --- |
| Registered Total | 12 |
| Identity Reuse Bindings Total | 17 |
| Unique Visitors | 12 |
| Entries Total | 34 |
| Exits Total | 34 |

These values come from a clean proof run and are also documented under `docs/sample_output.md`.

**Notes**

Exact values can vary depending on:

- Video input
- Configuration
- DB state
- Recognition thresholds

What should remain consistent is the relationship between `persons`, `REGISTERED`, `ENTRY`, `EXIT`, and unique visitor logic.

---

## Sample output proof

Proof artifacts are included in:

- `docs/sample_output.md`
- `docs/sample_outputs/images/`
- `docs/sample_outputs/logs/`

These include:

- Terminal screenshots
- DB proof screenshot
- Entry crop screenshot
- Exit crop screenshot
- Dashboard screenshots
- Sample app log
- Sample events log

---

## Compute estimate

The project includes a hackathon-oriented compute estimate in `docs/compute_estimate.md`.

This estimate explains:

- Where CPU/GPU load mainly occurs
- Why detection and embedding are the main heavy stages
- Why SQLite and logging are lightweight relative to inference
- How config knobs such as frame skip and recognition interval affect practical load

This is an estimate document, not a formal benchmark report.

---

## Resource usage logging

Minimal runtime resource usage proof was added for hackathon documentation.

The pipeline records:

- CPU usage percent
- RAM usage percent
- RAM used in MB

These are logged at pipeline start and pipeline end. This data appears in `logs/app.log`.

---

## Terminal metric semantics

| Metric | Meaning |
| --- | --- |
| Unique Visitors | Distinct `person_id` count from `persons` |
| Registered Total | New identities enrolled in this run |
| Identity Reuse Bindings Total | Duplicate-prevented bindings to existing identities |
| Recognized Total | Confirmed recognition matches |
| Entries Total | Visit opens emitted by `VisitManager` |
| Exits Total | Visit closes emitted by `VisitManager` |

---

## Runtime-generated files

These are generated when the project runs and are not required to already exist in a fresh clone:

- `data/face_tracker.db`
- `logs/app.log`
- `logs/events.log`
- `data/faces/entry/...`
- `data/faces/exit/...`

So anyone can clone the repository and generate these outputs by running:

```powershell
python .\scripts\init_db.py
python .\main.py
```

---

## Limitations

This hackathon build is intentionally focused on correct end-to-end functionality and proof generation.

Current limitations include:

- No full automated test suite
- No production deployment setup
- No advanced production frontend
- No production-grade multi-camera re-identification
- Metrics can vary run to run depending on input and thresholds
- Browser-triggered runs are synchronous
- Dense crowds and heavy motion can still fragment tracks

---

## Future work

Potential future improvements include:

- Production deployment setup
- Async/background processing for long streams
- Multi-camera identity fusion
- More polished frontend experience
- Automated testing and CI
- Exportable reporting for judges and operators

---

## Quick flow

For the fastest verification flow:

```powershell
cd E:\Katomaran
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\scripts\init_db.py
python .\main.py
python .\demo\server.py
```

Then verify:

- Terminal stats
- `data/face_tracker.db`
- `logs/app.log`
- `logs/events.log`
- Dashboard at **http://127.0.0.1:5050**
- Proof doc at `docs/sample_output.md`

---

## Hackathon note

This repository was prepared as a hackathon submission focused on:

- AI-assisted development workflow
- Modular Python implementation
- Real-time video processing
- Structured logging
- Database persistence
- Clear documentation
- Reproducible proof outputs

---

## Footer

This project is part of a hackathon run by [https://katomaran.com](https://katomaran.com).
