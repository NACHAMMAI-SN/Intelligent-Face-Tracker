# Practical assumptions

## Tracker-first processing

- **Track IDs** from `FaceTracker` are the primary handle for face crops and identity state in the main loop (`src/pipeline.py`).
- **Recognition** and **auto-registration** run on a configurable interval per track (`recognition_interval`, `detection_frame_skip` in `config.json`).

## Recognition and gallery

- **FaceRecognizer** (`src/recognizer.py`) keeps an in-memory gallery; it is **seeded from SQLite** at startup and updated with **`add_identity`** when new or reused persons get embeddings (`src/pipeline.py`).
- A **confirmed** match (`is_confirmed` + `person_id`) uses the known-person path; **no** auto-register on that frame.

## Auto-registration and duplicate prevention

- **AutoRegistrar** (`src/auto_register.py`) runs only on the **unknown** path (no confirmed binding from recognition).
- **New person:** `create_person`, `store_embedding`, **`REGISTERED`** in DB via `Repository.write_event`, and **`log_registered`** to `events.log` (DB mirror avoided where it would duplicate).
- **Reuse:** Compares the embedding to **`Repository.get_all_known_embeddings()`** (same cosine convention as the recognizer). On success: `update_person_last_seen`, `store_embedding`, **no** `create_person`, **no** `REGISTERED`; logged in **`app.log`** as `duplicate_prevented_existing_identity` / `Existing identity reused for track`.
- **Effective reuse threshold:** `min(auto_register_identity_reuse_threshold, recognition_threshold - auto_register_fragmentation_merge_slack)` so tracks that reach auto-register with best cosine **below** `recognition_threshold` can still match a stored identity. Config: `auto_register_identity_reuse_threshold`, `auto_register_fragmentation_merge_slack`, `recognition_threshold`.

## Unique visitors

- **Unique Visitors** = `Repository.get_unique_visitor_count()` → `COUNT(DISTINCT person_id)` on **`persons`**. **Identity reuse bindings do not insert a new `persons` row.**

## Demo proof (verified behavior)

| Invariant | Description |
| --- | --- |
| **Reuse path** | Binds a new `track_id` to an existing `person_id` — no new person row, no `REGISTERED` for that binding. |
| **`REGISTERED`** | Only **brand-new** identities; `events.log` should align **Registered Total**, not **Identity Reuse Bindings Total**. |
| **Unique Visitors** | Stays aligned with **`persons`** (verified example: **Registered Total 12**, **Reuse 17**, **Unique Visitors 12**). |

## Visits and timeouts

- **`entry_exit_absence_timeout_seconds`:** If a person with an **open** visit is absent from the current frame’s seen set long enough, `VisitManager` may emit **EXIT** (e.g. `visit_timeout_exit` in `app.log`). A larger timeout tends to reduce ENTRY/EXIT flips on the same video vs. a smaller one.
- **5 s** default: looser than **3 s**, stricter than **8 s** in relative terms.

## Outputs and persistence

| Location | Contents |
| --- | --- |
| `data/face_tracker.db` | SQLite: `persons`, `face_embeddings`, `visits`, `events` (`src/db.py`). |
| `logs/events.log` | Mandatory JSONL event stream (`EventLogger`). |
| `logs/app.log` | Operational logs, including reuse proof lines. |
| `data/faces/entry/`, `exit/` | Optional crop snapshots (may be gitignored). |

## Requirements

See **`requirements.txt`**. Python **3.11+** is recommended (`README.md`).
