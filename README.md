# Intelligent Face Tracker with Auto-Registration and Unique Visitor Counting

An end-to-end **tracker-first** face analytics pipeline for hackathon evaluation. The system detects and tracks multiple faces, recognizes known identities, auto-registers stable unknown faces, prevents duplicate identity creation through identity reuse, records `ENTRY` / `EXIT` visit events, and persists outputs to **SQLite** and structured logs.

It supports both:

- **Local video input**
- **RTSP stream input**

The project also includes a lightweight **Flask demo dashboard** for viewing results and optionally triggering runs from the browser.

---

## Demo video

**YouTube:** [Watch the project demo](https://youtu.be/VDnZWiXQpRQ)

---

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

## Repository navigation

Use these files for quick review:

| Document | Link / path |
| --- | --- |
| Overall overview, setup, run flow, submission mapping | [README.md](README.md) |
| AI planning | [docs/ai_planning.md](docs/ai_planning.md) |
| Architecture explanation | [docs/architecture.md](docs/architecture.md) |
| Rendered architecture diagram | [docs/architecture_diagram.md](docs/architecture_diagram.md) |
| Mermaid source (architecture) | [docs/architecture_diagram.mmd](docs/architecture_diagram.mmd) |
| Assumptions | [docs/assumptions.md](docs/assumptions.md) |
| Compute / load estimate | [docs/compute_estimate.md](docs/compute_estimate.md) |
| Sample output (processed video) | [docs/sample_output.md](docs/sample_output.md) |
| Submission notes | [docs/submission_notes.md](docs/submission_notes.md) |
| Application configuration | [config.json](config.json) |

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

---

## Requirement → file mapping

| Requirement | File / location |
| --- | --- |
| Setup instructions | `README.md` |
| Assumptions made | `docs/assumptions.md` |
| Sample `config.json` structure | `config.json` and [Sample config.json structure](#sample-configjson-structure) below |
| AI planning document | `docs/ai_planning.md` |
| Architecture explanation | `docs/architecture.md` |
| Architecture diagram | `docs/architecture_diagram.md` |
| Mermaid architecture source | `docs/architecture_diagram.mmd` |
| Compute estimate | `docs/compute_estimate.md` |
| Sample output from video file | `docs/sample_output.md` |
| Demo video link | [Demo video](#demo-video) and `docs/submission_notes.md` |
| Demo dashboard | `demo/server.py` |

---

## Setup instructions

From the project root:

```powershell
cd E:\Katomaran
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Initialize the database**

```powershell
python .\scripts\init_db.py
```

**Run the main pipeline**

```powershell
python .\main.py
```

**Run the Flask dashboard**

```powershell
python .\demo\server.py
```

Then open: **http://127.0.0.1:5050**

### Notes

- The detector model path is configured in `config.json`.
- InsightFace may download its model assets on first use.
- The default sample input is `data/test.mp4`.
- A fresh run will generate the SQLite database, logs, and entry/exit crop images.

---

## How to run

### Option 1: CLI pipeline run

```powershell
cd E:\Katomaran
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

### Option 3: Pipeline + dashboard

```powershell
cd E:\Katomaran
.\.venv\Scripts\Activate.ps1
python .\scripts\init_db.py
python .\main.py
python .\demo\server.py
```

---

## Assumptions made

Detailed assumptions are documented in **[docs/assumptions.md](docs/assumptions.md)**.

**Summary**

- The input source is either a local video file or an RTSP stream.
- The sample submission is primarily demonstrated using a provided local video file.
- Face visibility, blur, pose, and occlusion can affect recognition quality.
- Stable short-term tracking is available for associating detections across nearby frames.
- Unique visitors are represented by distinct `person_id` values in the database.
- Entry and exit events are inferred from the visit lifecycle logic implemented in the pipeline.
- Duplicate prevention depends on configured similarity thresholds and embedding quality.
- Browser-triggered runs in the demo dashboard are synchronous and best suited for demo-scale input.

---

## Sample config.json structure

Detailed configuration is available in **[config.json](config.json)**.

A sample structure is shown below:

```json
{
  "input_mode": "video",
  "video_path": "data/test.mp4",
  "rtsp_url": "rtsp://username:password@camera-host:554/stream1",
  "reconnect_enabled": true,
  "reconnect_delay_seconds": 5,
  "recognition_enabled": true,
  "detection_frame_skip": 2,
  "recognition_interval": 3,
  "detection_confidence": 0.45,
  "recognition_threshold": 0.45,
  "recognition_confirmation_window": 3,
  "recognition_confirmation_min_hits": 1,
  "auto_register_min_track_duration_frames": 10,
  "auto_register_min_track_hits": 8,
  "auto_register_min_detection_confidence": 0.45,
  "auto_register_min_unknown_hits": 3,
  "auto_register_min_quality_score": 10.0,
  "auto_register_cooldown_frames": 120,
  "auto_register_identity_reuse_threshold": 0.58,
  "auto_register_fragmentation_merge_slack": 0.1,
  "tracker_max_lost_frames": 30,
  "entry_exit_absence_timeout_seconds": 5,
  "db_path": "data/face_tracker.db",
  "logs_dir": "logs",
  "entry_crop_dir": "data/faces/entry",
  "exit_crop_dir": "data/faces/exit",
  "detector_model_path": "models/yolov8n-face.pt",
  "embedder_model": {
    "provider": "insightface",
    "model_name": "buffalo_l",
    "execution_provider": "CPUExecutionProvider"
  }
}
```

---

## AI planning and architecture

Detailed planning and architecture files:

| Document | Link |
| --- | --- |
| AI planning | [docs/ai_planning.md](docs/ai_planning.md) |
| Architecture | [docs/architecture.md](docs/architecture.md) |
| Architecture diagram (rendered) | [docs/architecture_diagram.md](docs/architecture_diagram.md) |
| Mermaid architecture source | [docs/architecture_diagram.mmd](docs/architecture_diagram.mmd) |

These documents describe:

- Planning approach
- Feature breakdown
- Module interactions
- Pipeline flow
- Decision points for recognition and auto-registration
- Persistence and observability design

---

## Compute estimate

Compute / load estimation is documented in **[docs/compute_estimate.md](docs/compute_estimate.md)**.

This document explains:

- Where CPU/GPU load mainly occurs
- Why detection and embedding are the main heavy stages
- Why SQLite and logging are lightweight relative to inference
- How config knobs such as frame skip and recognition interval affect practical load

---

## Sample output from the video file

Sample output artifacts are documented in **[docs/sample_output.md](docs/sample_output.md)**.

Supporting files are stored in:

- `docs/sample_outputs/images/`
- `docs/sample_outputs/logs/`

These include:

- Terminal run statistics screenshot
- Database output screenshot
- Entry crop screenshot
- Exit crop screenshot
- Dashboard screenshots
- Sample application log
- Sample events log


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

Browser-triggered runs merge overrides into a **temporary config file** per request; root **`config.json` is not modified**. Only **one** run at a time; a second concurrent request returns **HTTP 409**. **`POST /api/run/*`** handlers are **synchronous** (the HTTP request blocks until the pipeline finishes).

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

## Runtime-generated files

These files are generated when the project runs:

- `data/face_tracker.db`
- `logs/app.log`
- `logs/events.log`
- `data/faces/entry/...`
- `data/faces/exit/...`

These runtime outputs may not already exist in a fresh clone if they are excluded by `.gitignore`, but the project will generate them after running:

```powershell
python .\scripts\init_db.py
python .\main.py
```

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

This project is a part of a hackathon run by [https://katomaran.com](https://katomaran.com).
