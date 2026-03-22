# Submission notes



> This is an **Intelligent Face Tracker** with **auto-registration** and **unique visitor counting**.
>
> The design is **tracker-first**: YOLO detects faces, the **tracker** assigns stable **track IDs**, and we crop and embed **per track**. **InsightFace** embeddings feed a **cosine-similarity recognizer** against a gallery loaded from **SQLite**.
>
> If a face is **unknown** but **stable** enough, **auto-registration** can create a new **person** — or, with **duplicate prevention**, **bind the track to an existing person** when gallery similarity passes an **effective threshold**, **without** a new `REGISTERED` row.
>
> **Visits** use **ENTRY** and **EXIT** with strict one-open-visit-per-person rules; events go to **`events.log`** and the database.
>
> At the end, **terminal stats** show **Registered Total** (new IDs only), **Identity Reuse Bindings Total** (reuse path), **Unique Visitors**, and recognition counts.

## How to demo

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



| Artifact | Why |
| --- | --- |
| **Terminal** | Final stats: **Registered Total** vs **Identity Reuse Bindings Total** vs **Unique Visitors**. |
| **`logs\events.log`** | `ENTRY`, `EXIT`, `REGISTERED`, `RECOGNIZED` JSONL. |
| **`logs\app.log`** | Proof of reuse: `Existing identity reused for track`, `duplicate_prevented_existing_identity`, `no_new_person_created`. |
| **`data\face_tracker.db`** | `persons`, `visits`, `events`. |

## Expected output

Example from a **clean run** on the demo clip (reuse on; unique-visitor inflation reduced):

| Stat | Value |
| --- | --- |
| **Registered Total** | **12** |
| **Identity Reuse Bindings Total** | **17** |
| **Unique Visitors** | **12** |
| **Entries Total** | **23** |
| **Exits Total** | **23** |

**`logs/app.log`:** Many lines with `Existing identity reused for track`, `duplicate_prevented_existing_identity`, `no_new_person_created: true`, `effective_reuse_threshold: 0.35`.

**`logs/events.log`:** `REGISTERED` only for true new identities (aligns with **Registered Total**, not reuse bindings).

Numbers can differ with DB seed and config; should verify the **invariants** below.

## Demo proof

- **Reuse:** Fragmented unknown tracks that match the gallery do **not** create new **`persons`** rows.
- **`REGISTERED`:** Only **brand-new** identities; reuse is proven in **`app.log`**, not as `REGISTERED` events.
- **Unique Visitors:** Tied to **`persons`**; reuse bindings do **not** inflate it beyond new enrollments.

## Proof during demo (grep)

```powershell
Set-Location E:\Katomaran
Select-String -Path .\logs\app.log -Pattern "Existing identity reused for track|duplicate_prevented_existing_identity|no_new_person_created"
Select-String -Path .\logs\app.log -Pattern "identity_reuse_bindings_total|Pipeline stopped"
```

Optional — `REGISTERED` lines (new identities only):

```powershell
Select-String -Path .\logs\events.log -Pattern '"event_type": "REGISTERED"'
```
