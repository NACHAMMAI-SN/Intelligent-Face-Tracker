# Architecture Diagram

High-level flow of the Intelligent Face Tracker pipeline and demo dashboard.

```mermaid
flowchart TD
    A[Video File / RTSP Stream] --> B[VideoSource]
    B --> C[YOLO Face Detector]
    C --> D[Face Tracker]
    D --> E[Face Crop]
    E --> F[Face Embedder]
    F --> G[Face Recognizer]

    G -->|Known match| H[Known Identity Binding]
    G -->|Unknown / not confirmed| I[Auto Registeration]

    I -->|Reuse existing identity| J[Duplicate Prevention / Identity Reuse]
    I -->|Brand new identity| K[Register New Person]

    H --> L[Visit Manager]
    J --> L
    K --> L

    L --> M[Repository / SQLite]
    L --> N[Event Logger]

    N --> O[app.log / events.log]
    M --> P[persons / visits / events / embeddings]
    O --> Q[Flask Demo Dashboard]
    P --> Q
