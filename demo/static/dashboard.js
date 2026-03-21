const LABELS = [
  ["frames_seen", "Frames Seen"],
  ["frames_processed", "Frames Processed"],
  ["detections_total", "Detections Total"],
  ["recognized_total", "Recognized Total"],
  ["registered_total", "Registered Total"],
  ["identity_reuse_bindings_total", "Identity Reuse Bindings Total"],
  ["entries_total", "Entries Total"],
  ["exits_total", "Exits Total"],
  ["unique_visitors", "Unique Visitors"],
];

function fmt(v) {
  if (v === null || v === undefined) return "—";
  return String(v);
}

function setRunStatus(text, klass) {
  const el = document.getElementById("run-status");
  el.textContent = text;
  el.className = "status-badge " + (klass || "status-idle");
}

function renderStats(data) {
  const grid = document.getElementById("stats-grid");
  const missing = document.getElementById("stats-missing");
  const paths = document.getElementById("paths");
  const s = data.pipeline_stats_last_run;
  paths.textContent = `DB: ${data.paths?.db_path ?? "?"} · Log: ${data.paths?.app_log ?? "?"}`;
  grid.innerHTML = "";
  if (!s) {
    missing.hidden = false;
    return;
  }
  missing.hidden = true;
  for (const [key, label] of LABELS) {
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.innerHTML = `<div class="label">${label}</div><div class="val">${fmt(s[key])}</div>`;
    grid.appendChild(tile);
  }
}

function renderStatsIntoGrid(gridEl, statsObj) {
  gridEl.innerHTML = "";
  if (!statsObj) return;
  for (const [key, label] of LABELS) {
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.innerHTML = `<div class="label">${label}</div><div class="val">${fmt(statsObj[key])}</div>`;
    gridEl.appendChild(tile);
  }
}

function renderProof(proof) {
  const ul = document.getElementById("proof-list");
  ul.innerHTML = "";
  for (const line of proof.notes || []) {
    const li = document.createElement("li");
    li.textContent = line;
    ul.appendChild(li);
  }
  const dl = document.getElementById("proof-facts");
  const rows = [
    ["Persons table (COUNT)", proof.persons_table_count],
    ["Unique visitors (last run stats)", proof.unique_visitors_last_pipeline_stats],
    ["Aligned (persons == last-run UV)", proof.unique_visitors_matches_persons_table],
    ["Registered Total (last run)", proof.registered_total_last_run],
    ["Reuse bindings (last run stats)", proof.identity_reuse_bindings_last_run],
    ["Reuse log lines (since last Pipeline started)", proof.identity_reuse_log_lines_last_run],
    ["REGISTERED rows in DB (cumulative)", proof.events_registered_total_in_db],
    ["ENTRY rows in DB (cumulative)", proof.events_entry_total_in_db],
    ["EXIT rows in DB (cumulative)", proof.events_exit_total_in_db],
  ];
  dl.innerHTML = "";
  for (const [k, v] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = fmt(v);
    dl.appendChild(dt);
    dl.appendChild(dd);
  }
}

function renderConfigSummary(cfg) {
  const el = document.getElementById("config-summary");
  if (!cfg.ok && cfg.error) {
    el.textContent = `Config (from config.json): error — ${cfg.error}`;
    return;
  }
  el.textContent =
    `Current config summary: input_mode=${fmt(cfg.input_mode)} · video_path=${fmt(cfg.video_path)} · ` +
    `rtsp=${fmt(cfg.rtsp_url_redacted)} · db_path=${fmt(cfg.db_path)} · recognition_enabled=${fmt(cfg.recognition_enabled)}`;
}

function renderRunProof(dlEl, proof) {
  if (!proof) return;
  const rows = [
    ["Persons (persons table)", proof.persons],
    ["REGISTERED events (DB)", proof.registered_events],
    ["ENTRY events (DB)", proof.entries],
    ["EXIT events (DB)", proof.exits],
  ];
  dlEl.innerHTML = "";
  for (const [k, v] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = fmt(v);
    dlEl.appendChild(dt);
    dlEl.appendChild(dd);
  }
}

function showRunSource(source, dbPath) {
  const el = document.getElementById("run-source");
  if (!source) {
    el.textContent = "";
    return;
  }
  if (source.input_mode === "video") {
    el.textContent =
      `Source: video · ${source.uploaded_path_relative || source.uploaded_path_absolute || "?"}` +
      (dbPath ? ` · DB: ${dbPath}` : "");
  } else {
    el.textContent = `Source: RTSP · ${source.rtsp_url || "?"}` + (dbPath ? ` · DB: ${dbPath}` : "");
  }
}

async function loadConfigSummary() {
  const res = await fetch("/api/config");
  const cfg = await res.json();
  renderConfigSummary(cfg);
}

async function loadRunStatus() {
  const res = await fetch("/api/run/status");
  const st = await res.json();
  const map = {
    idle: "status-idle",
    running: "status-running",
    done: "status-done",
    failed: "status-failed",
  };
  setRunStatus(st.status || "idle", map[st.status] || "status-idle");
  const err = document.getElementById("run-error");
  if (st.status === "failed" && st.last_error) {
    err.hidden = false;
    err.textContent = st.last_error;
  } else {
    err.hidden = true;
    err.textContent = "";
  }
}

async function loadAll() {
  const [statsRes, proofRes, evRes] = await Promise.all([
    fetch("/api/stats"),
    fetch("/api/proof"),
    fetch("/api/events-summary"),
  ]);
  const stats = await statsRes.json();
  const proof = await proofRes.json();
  const ev = await evRes.json();
  renderStats(stats);
  renderProof(proof);
  document.getElementById("events-pre").textContent = JSON.stringify(ev, null, 2);
}

function toggleInputRows() {
  const mode = document.getElementById("input-mode").value;
  document.getElementById("row-video").classList.toggle("hidden", mode !== "video");
  document.getElementById("row-rtsp").classList.toggle("hidden", mode !== "rtsp");
}

document.getElementById("input-mode").addEventListener("change", toggleInputRows);
toggleInputRows();

document.getElementById("refresh").addEventListener("click", () => {
  loadConfigSummary();
  loadAll();
  loadRunStatus();
});

document.getElementById("start-run").addEventListener("click", async () => {
  const errEl = document.getElementById("run-error");
  const resultEl = document.getElementById("run-result");
  errEl.hidden = true;
  errEl.textContent = "";
  resultEl.classList.add("hidden");

  const mode = document.getElementById("input-mode").value;
  const btn = document.getElementById("start-run");
  btn.disabled = true;
  let hitServer = false;
  setRunStatus("running", "status-running");

  try {
    let res;
    if (mode === "video") {
      const input = document.getElementById("video-file");
      if (!input.files || !input.files[0]) {
        errEl.hidden = false;
        errEl.textContent = "Choose a video file first.";
        setRunStatus("idle", "status-idle");
        return;
      }
      const fd = new FormData();
      fd.append("file", input.files[0]);
      hitServer = true;
      res = await fetch("/api/run/video", { method: "POST", body: fd });
    } else {
      const url = document.getElementById("rtsp-url").value.trim();
      if (!url) {
        errEl.hidden = false;
        errEl.textContent = "Enter an RTSP URL (must start with rtsp://).";
        setRunStatus("idle", "status-idle");
        return;
      }
      if (!url.toLowerCase().startsWith("rtsp://")) {
        errEl.hidden = false;
        errEl.textContent = "RTSP URL must start with rtsp://";
        setRunStatus("idle", "status-idle");
        return;
      }
      hitServer = true;
      res = await fetch("/api/run/rtsp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rtsp_url: url }),
      });
    }

    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.ok) {
      errEl.hidden = false;
      errEl.textContent = data.error || `Request failed (${res.status})`;
      setRunStatus("failed", "status-failed");
      return;
    }

    errEl.hidden = true;
    errEl.textContent = "";
    setRunStatus("done", "status-done");
    resultEl.classList.remove("hidden");
    renderStatsIntoGrid(document.getElementById("run-stats-grid"), data.stats);
    renderRunProof(document.getElementById("run-proof-dl"), data.proof);
    showRunSource(data.source, data.db_path);
    await loadAll();
  } catch (e) {
    errEl.hidden = false;
    errEl.textContent = e instanceof Error ? e.message : String(e);
    setRunStatus("failed", "status-failed");
  } finally {
    btn.disabled = false;
    if (hitServer) {
      await loadRunStatus();
    }
  }
});

loadConfigSummary();
loadAll();
loadRunStatus();
