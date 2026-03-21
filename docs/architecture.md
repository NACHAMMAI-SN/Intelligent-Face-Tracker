# Architecture Notes

## Overview

This project implements a **tracker-first intelligent face analytics pipeline** for hackathon use.  
It processes either a **local video file** or an **RTSP stream**, detects faces, tracks them across frames, extracts embeddings, recognizes known identities, auto-registers stable unknown identities, prevents duplicate registrations through identity reuse, and records visit events in a structured SQLite-backed log flow.

The design goal is to keep the system:
- modular
- easy to demo
- easy to explain
- practical on a single machine

---

## High-level architecture

The runtime flow is:

**Input Source → Detection → Tracking → Face Crop → Embedding → Recognition / Auto-Registration → Visit Management → Logging / Database / Demo Dashboard**

### Main idea

The pipeline is **tracker-first**, not detector-only.  
That means a face is not treated as a fresh event on every frame. Instead:

- the detector finds faces
- the tracker assigns a stable `track_id`
- recognition and registration decisions are made per track over time
- visit events are emitted per person, not per raw detection

This reduces noise and makes identity handling more stable.

---

## Main modules and responsibilities

### 1. VideoSource
Handles input acquisition from:
- local video files
- RTSP streams

This gives the rest of the pipeline a consistent frame-reading interface regardless of source type.

### 2. YOLOFaceDetector
Runs face detection on incoming frames.

Responsibilities:
- locate faces in the frame
- return bounding boxes and confidence scores
- act as the first stage of visual understanding

### 3. FaceTracker
Maintains face tracks across frames.

Responsibilities:
- assign stable `track_id` values
- maintain basic lifecycle information such as:
  - creation frame
  - last seen frame
  - hit count
  - lost count
- reduce fragmentation compared to treating every detection independently

This module is central to the tracker-first design.

### 4. Face crop stage
After tracking, the system crops the face region from the current frame using the tracked bounding box.

Responsibilities:
- create a usable face crop for embedding
- preserve per-track face context
- support entry/exit crop saving

### 5. FaceEmbedder
Generates a face embedding vector from the cropped face.

Responsibilities:
- convert the face image into a numeric representation
- use InsightFace-style embedding inference
- provide the feature vector used for recognition and reuse checks

### 6. FaceRecognizer
Matches embeddings against the in-memory known gallery.

Responsibilities:
- compare current embedding with known identities
- determine whether the face is:
  - confirmed known
  - weakly matched / below threshold
  - unknown
- maintain per-track recognition history where needed

Known identities are seeded from SQLite at startup.

### 7. AutoRegistrar
Handles the unknown-person path.

Responsibilities:
- observe stable unknown tracks over time
- decide whether the track is strong enough to become a new person
- prevent duplicate registrations by checking whether the embedding should reuse an existing identity instead of creating a new one

This module is where **new identity creation** and **duplicate prevention** happen.

### 8. VisitManager
Controls visit lifecycle semantics.

Responsibilities:
- open visits on entry
- close visits on exit
- enforce one active/open visit per person
- use absence timeout logic for exit decisions

This turns raw recognition events into usable ENTRY / EXIT records.

### 9. Repository / SQLite
Provides persistence.

Responsibilities:
- store persons
- store embeddings
- store visits
- store events
- provide counts such as unique visitors
- act as the source of truth for persisted proof

### 10. EventLogger
Writes operational and event logs.

Responsibilities:
- write structured app logs
- write event logs
- support proof during demo
- mirror or avoid duplicate DB writes depending on event path

### 11. Flask Demo Dashboard
Provides a lightweight frontend for demonstration.

Responsibilities:
- show stats and proof data
- read from the same SQLite DB and logs
- optionally trigger browser-driven runs without rewriting the main app architecture

This is a demo layer, not the core analytics engine.

---

## Processing flow in detail

### A. Input and frame reading
The system opens either:
- a configured local video file, or
- a configured RTSP stream

Frames are read through the same abstraction.

### B. Detection
The detector runs on processed frames and outputs face bounding boxes.

### C. Tracking
Detections are passed to the tracker, which returns:
- active tracks
- dead tracks

Each active track carries information such as:
- `track_id`
- bounding box
- confidence
- hit count
- creation / last seen information

### D. Face crop extraction
The system safely crops the face from the tracked bounding box.

If a crop is invalid, the track is skipped for embedding/recognition on that cycle.

### E. Recognition scheduling
Recognition does not have to run on every frame.  
The pipeline uses configuration such as:
- `detection_frame_skip`
- `recognition_interval`

This helps reduce compute load.

### F. Embedding and recognition
When recognition is scheduled:
- the crop is embedded
- the embedding is compared against the known gallery
- the recognizer determines whether the person is already known

If confirmed:
- the track is bound to an existing `person_id`
- the visit manager is updated
- recognition events are logged

### G. Auto-registration / identity reuse
If the recognition result is not confirmed:
- the observation goes to the auto-registrar

The auto-registrar then decides between two paths:

#### Path 1: Identity reuse
If gallery similarity is strong enough under the effective reuse rule:
- the track is bound to an existing `person_id`
- no new person row is created
- no new `REGISTERED` event is emitted for that reuse
- duplicate inflation is prevented

#### Path 2: New identity registration
If the track is stable enough and does not qualify for reuse:
- a new `person_id` is created
- the embedding is stored
- a `REGISTERED` event is written
- the recognizer gallery is updated

This is how the system handles true new visitors.

### H. Visit lifecycle
Once a person is confirmed or registered:
- a visit may be opened if none is currently open
- the system tracks whether the person remains visible
- if missing long enough, the visit may be closed with EXIT
- shutdown finalization can also close pending visits

### I. Logging and proof
The system records:
- app-level operational logs
- event logs
- SQLite rows for persons, visits, embeddings, and events

This supports hackathon-proof demonstration:
- terminal counters
- DB counts
- log-based reuse evidence

---

## Why tracker-first matters

A detector-only approach may repeatedly treat the same real person as unrelated observations.  
This project instead uses track continuity to stabilize identity decisions.

Benefits:
- fewer noisy decisions
- better registration timing
- more reliable entry/exit handling
- reduced duplicate visitor inflation
- easier reasoning about per-person lifecycle

---

## Identity model

The project separates these concepts carefully:

- **track_id** = temporary visual tracking identity
- **person_id** = persistent database identity
- **visit_id** = one visit session for a person entering/exiting the scene

This is important because:
- one person may appear under multiple track fragments
- reuse logic can bind fragmented tracks back to one person
- unique visitor counting should follow `person_id`, not raw tracks

---

## Persistence model

The database is the persisted source of truth.

Main stored entities:
- **persons**
- **face_embeddings**
- **visits**
- **events**

Meaning:
- `persons` represent persistent identities
- `face_embeddings` store recognition vectors
- `visits` represent entry/exit sessions
- `events` record lifecycle and analytics events

---

## Demo and proof architecture

For hackathon evaluation, the project includes a lightweight demo/proof layer:

- the main Python pipeline processes the source
- SQLite stores persistent results
- logs capture proof details
- the Flask dashboard reads the same outputs
- optional frontend-triggered runs can still use the same pipeline backend

This keeps the architecture simple:
- one main backend
- one SQLite DB
- one logging flow
- one lightweight demo UI

---

## Resource and deployment assumptions

This build is intended for:
- single-machine demo execution
- CPU-based or modest local inference
- short proof-oriented runs
- hackathon submission rather than production deployment

It is not yet designed for:
- large-scale multi-camera orchestration
- distributed inference services
- production-grade monitoring
- full enterprise deployment controls

---

## Current strengths

The architecture already supports:
- modular Python components
- real video / RTSP input
- stable face tracking
- embedding-based recognition
- auto-registration
- duplicate prevention via identity reuse
- unique visitor counting
- visit lifecycle persistence
- structured proof through DB and logs
- lightweight dashboard visualization

---

## Current limitations

The architecture is intentionally hackathon-focused.

Not fully productized yet:
- no large automated test suite
- no production deployment stack
- no advanced multi-camera re-identification service
- no high-scale distributed processing
- dashboard is lightweight, not a complete production UI

These are future improvements, not blockers for the current submission.

---

## Summary

This architecture is designed around a simple but strong idea:

**use tracking as the backbone, use embeddings for identity, use reuse logic to prevent duplicates, and use structured persistence to prove the results.**

That makes the project practical for a hackathon demo while still showing clear AI workflow, modular design, and measurable outcomes.