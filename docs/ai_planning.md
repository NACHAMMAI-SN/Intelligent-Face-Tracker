# AI Planning Document

## 1. Problem understanding

The goal of this project is to build an intelligent video-processing application that can detect faces, track them across frames, recognize known identities, auto-register new stable identities, log meaningful events, and maintain a reliable unique visitor count.

The application should work with either a local video file or an RTSP stream and should persist results in a structured way so they can be verified after a run.

This project is designed as a hackathon-ready AI system that combines computer vision, identity management, event logging, and a lightweight demo interface.

---

## 2. Project objective

The main objective is to create an **Intelligent Face Tracker** that can:

- process video or RTSP input
- detect faces in frames
- track faces over time
- generate embeddings for recognition
- recognize known identities
- auto-register truly new identities
- prevent duplicate person creation when fragmented tracks belong to an existing identity
- log `ENTRY`, `EXIT`, `RECOGNIZED`, and `REGISTERED` events
- store results in SQLite
- provide a lightweight frontend for demo and proof

---

## 3. Planned application flow

The planned end-to-end flow of the application is:

1. Read input from either a video file or RTSP stream.
2. Run face detection on frames using a YOLO-based detector.
3. Track detected faces across frames using a tracker-first pipeline.
4. Extract and prepare face crops for embedding generation.
5. Generate face embeddings using InsightFace.
6. Compare embeddings against known identities for recognition.
7. If the identity is not confidently known, evaluate whether it should:
   - reuse an existing identity, or
   - be auto-registered as a brand-new person
8. Send identity decisions into the visit manager.
9. Create and persist visit lifecycle events such as `ENTRY` and `EXIT`.
10. Write structured logs and store results in SQLite.
11. Expose proof and final statistics through the Flask demo dashboard.

---

## 4. Planned features

The application was planned with the following features:

### Core AI features
- Face detection from video frames
- Multi-frame tracking of faces
- Face embedding generation
- Face recognition using similarity thresholds
- Auto-registration of new identities

### Identity management features
- Duplicate prevention through identity reuse
- Stable unique visitor counting
- Person record persistence in SQLite

### Event and logging features
- `ENTRY` event logging
- `EXIT` event logging
- `RECOGNIZED` event logging
- `REGISTERED` event logging
- App-level run logs and final stats logging

### Demo and usability features
- Local video input support
- RTSP input support
- Flask-based frontend dashboard
- Browser-triggered video processing
- Dashboard proof and summary views

---

## 5. AI components used

This project uses the following AI-related components:

### YOLO face detector
Used to detect faces in incoming frames before tracking and recognition.

### InsightFace embedder / recognizer
Used to generate face embeddings and compare identities through similarity matching.

### Tracker-first video pipeline
Used to maintain continuity of identities across frames and reduce instability caused by per-frame detection alone.

### Auto-registration decision logic
Used to decide whether an unknown face should become:
- a reused existing identity, or
- a brand-new registered person

This logic is important to reduce duplicate identity inflation.

---

## 6. Design decisions taken during planning

The project was planned around the following design choices:

### Tracker-first design
Tracking is placed before identity finalization so that the system can accumulate evidence across frames instead of making isolated per-frame identity decisions.

### SQLite persistence
SQLite is used because it is lightweight, easy to verify during demos, and suitable for hackathon-scale structured storage.

### Structured logging
Logs are treated as proof artifacts, not just debug text. This helps demonstrate the application behavior clearly during evaluation.

### Lightweight frontend
A simple Flask dashboard was chosen instead of a heavy production frontend so that the demo remains easy to run and explain.

### Config-driven runtime
Important runtime behavior such as input mode, thresholds, and frame skipping is controlled through `config.json` for easier testing and explanation.

---

## 7. Expected outputs

The planned outputs of the application are:

- terminal run statistics
- SQLite records for persons, visits, and events
- structured application logs
- saved entry and exit face crops
- dashboard metrics and proof summaries

Important observable event outputs include:

- `ENTRY`
- `EXIT`
- `RECOGNIZED`
- `REGISTERED`

Important summary outputs include:

- detections total
- recognized total
- registered total
- identity reuse bindings total
- entries total
- exits total
- unique visitors

---

## 8. Compute planning approach

During planning, the heaviest expected compute stages were identified as:

- face detection
- embedding generation / recognition

Tracking, SQLite operations, and logging were expected to be much lighter in comparison.

The application was therefore planned to use config-controlled frequency knobs such as:

- detection frame skip
- recognition interval

This allows the demo to remain practical on limited hardware, including CPU-based execution for hackathon proof runs.

A separate compute estimate document is included in:

- `docs/compute_estimate.md`

---

## 9. Risks considered during planning

The following risks were considered:

- duplicate identity creation due to fragmented tracks
- unstable recognition on low-quality or partial face crops
- higher compute cost for continuous live streams
- mismatch between terminal counters and cumulative persisted DB state across multiple runs
- demo complexity becoming too large for a hackathon submission

These risks were handled by:
- identity reuse logic
- threshold-based decisions
- tracker-first design
- structured persistence
- keeping the frontend lightweight

---

## 10. Final planning summary

This application was planned as a modular AI video analytics system focused on:

- face detection
- tracking
- recognition
- auto-registration
- event logging
- unique visitor counting
- proof-oriented persistence
- demo-friendly visualization

The final implementation follows this plan with a modular Python codebase, SQLite-backed persistence, structured logs, and a lightweight Flask dashboard for demonstration.