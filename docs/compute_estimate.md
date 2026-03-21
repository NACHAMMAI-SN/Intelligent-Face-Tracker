# Compute Estimate

## Overview

This project runs an end-to-end **tracker-first** pipeline: **YOLO** face detection, per-track crops, **InsightFace** embedding and similarity-based recognition, optional auto-registration, visit lifecycle, and persistence via **SQLite** plus structured logging. Input is **local video** or **RTSP** through the same `VideoSource` abstraction. The reference configuration uses **ONNX Runtime with `CPUExecutionProvider`** for embedding inference in typical demo runs.

## Main compute-heavy stages

- **Face detection (YOLO)** — Runs on each frame according to `detection_frame_skip` in `config.json`. This dominates cost when many faces appear or resolution is high.
- **Embedding / recognition (InsightFace)** — Runs on a schedule (`recognition_interval`); each invocation does neural inference on cropped faces. This is the other major cost alongside detection.
- **Tracking, SQLite, and event logging** — Lightweight relative to the two stages above: mostly CPU-friendly data structures, small queries, and append-style I/O.

## Demo environment assumption

Hackathon demos are assumed to run on a **single machine** with **CPU-only** inference for embeddings (as configured), sufficient RAM for models and frame buffers, and disk space for the SQLite database, optional crops, and logs. No specific FPS, GPU utilization, or wall-clock time is claimed without measurement.

## Practical resource expectations

- **CPU-only demo** is sufficient for the **proof-oriented** runs described in the README (short clip, tuned skip/interval). Throughput depends on hardware, video resolution, face count, and config knobs—not fixed here.
- **RTSP or continuous live streams** may stress the same bottlenecks (detection + embedding) more than file playback; operators can reduce load by increasing **frame skip** or **recognition interval**, or by moving embedding inference to a **GPU** provider if the stack is reconfigured accordingly.

## Scalability note

The current design is a **single-process** pipeline: one video source, one detector, one tracker graph, one gallery-backed recognizer, and one DB. Scaling to many cameras or very high frame rates would require architectural changes (e.g., batching, separate inference services, or hardware acceleration)—out of scope for this submission.

## Honest limitation note

This document does **not** include measured benchmarks (FPS, latency, watts, or GPU memory). Any production sizing should be based on **profiling on target hardware** and the same config parameters that control how often detection and embedding run.
