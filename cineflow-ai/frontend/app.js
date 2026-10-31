/**
 * CineFlow AI Studio — Frontend Controller (`frontend/app.js`)
 *
 * Manages:
 * - System health polling (`/api/health`) & recent render history (`/api/jobs`)
 * - Drag-and-drop video upload with real-time XHR byte progress (`/api/upload`)
 * - Media metadata inspector & Before/After dual video comparison player
 * - Background render job launch (`/api/process`) & live status polling (`/api/status/{job_id}`)
 * - Interactive multi-track NLE timeline visualizer (V1 Video, A1 Voiceover, A2 Ducking SVG)
 */

const PIPELINE_STEPS = [
  "Queued",
  "Analyzing",
  "Generating Voiceover",
  "Rendering FFmpeg",
  "Done",
];

const state = {
  mediaId: null,
  filename: null,
  sourceMetadata: null,
  sourceVideoUrl: null,
  activeJobId: null,
  pollTimer: null,
  elapsedTimer: null,
  jobStartedAtMs: null,
  isUploading: false,
  isProcessing: false,
  currentTimeline: null,
  duckingLevel: 0.18,
};

// --- DOM References ---
const dom = {
  badgeFfmpeg: document.getElementById("badgeFfmpeg"),
  badgeGemini: document.getElementById("badgeGemini"),
  badgeGroq: document.getElementById("badgeGroq"),
  maxUploadLabel: document.getElementById("maxUploadLabel"),

  btnToggleHistory: document.getElementById("btnToggleHistory"),
  historyDrawer: document.getElementById("historyDrawer"),
  btnRefreshHistory: document.getElementById("btnRefreshHistory"),
  historyList: document.getElementById("historyList"),
  historyCountBadge: document.getElementById("historyCountBadge"),

  videoFileInput: document.getElementById("videoFileInput"),
  uploadDropzone: document.getElementById("uploadDropzone"),
  uploadProgressWrap: document.getElementById("uploadProgressWrap"),
  uploadProgressBar: document.getElementById("uploadProgressBar"),
  uploadProgressText: document.getElementById("uploadProgressText"),
  btnReplaceVideo: document.getElementById("btnReplaceVideo"),

  mediaInspector: document.getElementById("mediaInspector"),
  inspectorThumbVideo: document.getElementById("inspectorThumbVideo"),
  metaFilename: document.getElementById("metaFilename"),
  metaAudioBadge: document.getElementById("metaAudioBadge"),
  metaDuration: document.getElementById("metaDuration"),
  metaResolution: document.getElementById("metaResolution"),
  metaFps: document.getElementById("metaFps"),
  metaCodecs: document.getElementById("metaCodecs"),
  metaAspect: document.getElementById("metaAspect"),
  metaSize: document.getElementById("metaSize"),

  promptInput: document.getElementById("promptInput"),
  promptCharCount: document.getElementById("promptCharCount"),
  selectVoice: document.getElementById("selectVoice"),
  selectAspect: document.getElementById("selectAspect"),
  selectTtsEngine: document.getElementById("selectTtsEngine"),
  rangeMaxDuration: document.getElementById("rangeMaxDuration"),
  labelMaxDuration: document.getElementById("labelMaxDuration"),
  rangeDucking: document.getElementById("rangeDucking"),
  labelDucking: document.getElementById("labelDucking"),
  inlineAlert: document.getElementById("inlineAlert"),
  btnStartRender: document.getElementById("btnStartRender"),
  btnStartRenderLabel: document.getElementById("btnStartRenderLabel"),

  jobStatusBadge: document.getElementById("jobStatusBadge"),
  jobStageDetail: document.getElementById("jobStageDetail"),
  jobElapsed: document.getElementById("jobElapsed"),
  jobPercent: document.getElementById("jobPercent"),
  masterProgressFill: document.getElementById("masterProgressFill"),
  pipelineStepper: document.getElementById("pipelineStepper"),
  consoleDetails: document.getElementById("consoleDetails"),
  logLineCount: document.getElementById("logLineCount"),
  consoleLogOutput: document.getElementById("consoleLogOutput"),

  tabCompare: document.getElementById("tabCompare"),
  tabTheater: document.getElementById("tabTheater"),
  videoStageGrid: document.getElementById("videoStageGrid"),
  chkMuteOriginalSync: document.getElementById("chkMuteOriginalSync"),
  btnSyncPlay: document.getElementById("btnSyncPlay"),
  btnSyncPlayText: document.getElementById("btnSyncPlayText"),
  btnDownloadMp4: document.getElementById("btnDownloadMp4"),

  videoBefore: document.getElementById("videoBefore"),
  placeholderBefore: document.getElementById("placeholderBefore"),
  sourcePaneMeta: document.getElementById("sourcePaneMeta"),

  videoAfter: document.getElementById("videoAfter"),
  placeholderAfter: document.getElementById("placeholderAfter"),
  placeholderAfterText: document.getElementById("placeholderAfterText"),
  outputPaneMeta: document.getElementById("outputPaneMeta"),

  renderSummaryRibbon: document.getElementById("renderSummaryRibbon"),
  statDurationCompare: document.getElementById("statDurationCompare"),
  statSegmentsCount: document.getElementById("statSegmentsCount"),
  statOutputSpecs: document.getElementById("statOutputSpecs"),
  statVoiceEngine: document.getElementById("statVoiceEngine"),

  timelineProjectTitle: document.getElementById("timelineProjectTitle"),
  timelineToneBadge: document.getElementById("timelineToneBadge"),
  timelineVoiceBadge: document.getElementById("timelineVoiceBadge"),
  timelineSummaryText: document.getElementById("timelineSummaryText"),
  timelineEmptyState: document.getElementById("timelineEmptyState"),
  timelineContainer: document.getElementById("timelineContainer"),
  sourceFootprintSpan: document.getElementById("sourceFootprintSpan"),
  sourceFootprintBar: document.getElementById("sourceFootprintBar"),
  timelineRuler: document.getElementById("timelineRuler"),
  trackVideoClips: document.getElementById("trackVideoClips"),
  trackVoiceClips: document.getElementById("trackVoiceClips"),
  trackDuckingSvgWrap: document.getElementById("trackDuckingSvgWrap"),
  modelAttributionLabel: document.getElementById("modelAttributionLabel"),
  segmentCardsList: document.getElementById("segmentCardsList"),
};

// --- Formatting Helpers ---
function formatSeconds(sec) {
  const s = Math.max(0, Number(sec) || 0);
  const mins = Math.floor(s / 60);
  const rem = s - mins * 60;
  if (mins > 0) {
    return `${String(mins).padStart(2, "0")}:${rem.toFixed(1).padStart(4, "0")}`;
  }
  return `${rem.toFixed(1)}s`;
}

function formatClock(ms) {
  const totalSec = Math.max(0, Math.floor(ms / 1000));
  const m = String(Math.floor(totalSec / 60)).padStart(2, "0");
  const s = String(totalSec % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function formatBytes(bytes) {
  const b = Number(bytes) || 0;
  if (b < 1024 * 1024) {
    return `${(b / 1024).toFixed(1)} KB`;
  }
  return `${(b / (1024 * 1024)).toFixed(1)} MB`;
}

function escapeHtml(raw) {
  return String(raw ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function showAlert(message, kind = "error") {
  if (!message) {
    dom.inlineAlert.classList.add("hidden");
    dom.inlineAlert.textContent = "";
    return;
  }
  dom.inlineAlert.textContent = message;
  dom.inlineAlert.className = `alert-banner ${kind === "info" ? "alert-info" : "alert-error"}`;
}

function updateStartButtonState() {
  if (state.isUploading) {
    dom.btnStartRender.disabled = true;
    dom.btnStartRenderLabel.textContent = "Uploading Source Video...";
    return;
  }
  if (state.isProcessing) {
    dom.btnStartRender.disabled = true;
    dom.btnStartRenderLabel.textContent = "AI Editing & Rendering in Progress...";
    return;
  }
  if (!state.mediaId) {
    dom.btnStartRender.disabled = true;
    dom.btnStartRenderLabel.textContent = "Upload a Video to Begin";
    return;
  }
  dom.btnStartRender.disabled = false;
  dom.btnStartRenderLabel.textContent = "Generate AI Edit & Voiceover";
}

// --- System Health & Recent Jobs ---
async function refreshHealthStatus() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) return;
    const data = await res.json();

    if (data.limits?.max_upload_mb) {
      dom.maxUploadLabel.textContent = String(data.limits.max_upload_mb);
    }

    setChip(
      dom.badgeFfmpeg,
      data.ffmpeg?.available ? "chip-ok" : "chip-warn",
      data.ffmpeg?.available ? "FFmpeg Ready" : "FFmpeg Missing"
    );
    setChip(
      dom.badgeGemini,
      data.providers?.gemini_configured ? "chip-ok" : "chip-warn",
      data.providers?.gemini_configured
        ? `Gemini (${data.providers.gemini_model})`
        : "Set GEMINI_API_KEY"
    );
    setChip(
      dom.badgeGroq,
      data.providers?.groq_configured ? "chip-ok" : "chip-info",
      data.providers?.groq_configured ? "Groq LPU + TTS" : "Edge Neural TTS"
    );
  } catch {
    setChip(dom.badgeFfmpeg, "chip-warn", "API Offline");
  }
}

function setChip(el, cls, label) {
  el.className = `status-chip ${cls}`;
  el.innerHTML = `<span class="dot"></span> ${escapeHtml(label)}`;
}

async function loadRecentJobs() {
  try {
    const res = await fetch("/api/jobs?limit=10");
    if (!res.ok) return;
    const payload = await res.json();
    const jobs = payload.jobs || [];

    if (jobs.length > 0) {
      dom.historyCountBadge.textContent = String(jobs.length);
      dom.historyCountBadge.classList.remove("hidden");
    } else {
      dom.historyCountBadge.classList.add("hidden");
    }

    if (jobs.length === 0) {
      dom.historyList.innerHTML = `<p class="empty-hint">No previous renders yet.</p>`;
      return;
    }

    dom.historyList.innerHTML = jobs
      .map((j) => {
        const title = j.timeline?.title || j.prompt.slice(0, 40);
        const statusColor =
          j.status === "Done"
            ? "#6ee7b7"
            : j.status === "Failed"
              ? "#fda4af"
              : "#a5b4fc";
        return `
          <button type="button" class="history-item" data-job-id="${escapeHtml(j.job_id)}">
            <div class="history-item-top">
              <span>${escapeHtml(title)}</span>
              <span style="color:${statusColor};font-family:var(--font-mono);font-size:11px;">
                ${escapeHtml(j.status)}
              </span>
            </div>
            <div class="history-item-sub">${escapeHtml(j.prompt)}</div>
          </button>
        `;
      })
      .join("");

    dom.historyList.querySelectorAll(".history-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        const jobId = btn.getAttribute("data-job-id");
        dom.historyDrawer.classList.add("hidden");
        if (jobId) {
          restoreJobSession(jobId);
        }
      });
    });
  } catch {
    // Non-fatal
  }
}

async function restoreJobSession(jobId) {
  try {
    const res = await fetch(`/api/status/${encodeURIComponent(jobId)}`);
    if (!res.ok) return;
    const job = await res.json();

    state.mediaId = job.media_id;
    state.sourceMetadata = job.source_metadata;
    state.sourceVideoUrl = job.source_video_url;
    state.activeJobId = job.job_id;
    dom.promptInput.value = job.prompt || dom.promptInput.value;
    updatePromptCounter();

    if (job.source_metadata) {
      populateMediaInspector(`Asset ${job.media_id.slice(0, 8)}`, job.source_metadata, job.source_video_url);
    }

    applyJobSnapshot(job);

    if (!["Done", "Failed"].includes(job.status)) {
      state.isProcessing = true;
      updateStartButtonState();
      beginStatusPolling(job.job_id);
    }
  } catch (err) {
    showAlert(`Could not load job ${jobId}: ${err.message}`);
  }
}

// --- Media Upload Workflow ---
function handleSelectedFile(file) {
  if (!file) return;
  showAlert("");

  const allowedExts = [".mp4", ".mov", ".webm", ".mkv", ".m4v"];
  const lowerName = file.name.toLowerCase();
  const hasAllowedExt = allowedExts.some((ext) => lowerName.endsWith(ext));
  if (!hasAllowedExt) {
    showAlert(
      `Unsupported file "${file.name}". Please select an MP4, MOV, WEBM, or MKV video.`
    );
    return;
  }

  state.isUploading = true;
  updateStartButtonState();

  dom.uploadDropzone.classList.remove("hidden");
  dom.mediaInspector.classList.add("hidden");
  dom.uploadProgressWrap.classList.remove("hidden");
  dom.uploadProgressBar.style.width = "0%";
  dom.uploadProgressText.textContent = `Uploading ${file.name}... 0%`;

  const formData = new FormData();
  formData.append("file", file);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload", true);

  xhr.upload.addEventListener("progress", (evt) => {
    if (evt.lengthComputable) {
      const pct = Math.min(99, Math.round((evt.loaded / evt.total) * 100));
      dom.uploadProgressBar.style.width = `${pct}%`;
      dom.uploadProgressText.textContent =
        pct >= 98
          ? "Running ffprobe stream & audio inspection..."
          : `Uploading... ${pct}% (${formatBytes(evt.loaded)} / ${formatBytes(evt.total)})`;
    }
  });

  xhr.addEventListener("load", () => {
    state.isUploading = false;
    dom.uploadProgressWrap.classList.add("hidden");

    let payload = null;
    try {
      payload = JSON.parse(xhr.responseText);
    } catch {
      payload = null;
    }

    if (xhr.status >= 200 && xhr.status < 300 && payload?.media_id) {
      state.mediaId = payload.media_id;
      state.filename = payload.filename;
      state.sourceMetadata = payload.metadata;
      state.sourceVideoUrl = payload.video_url;

      populateMediaInspector(payload.filename, payload.metadata, payload.video_url);
      updateStartButtonState();
    } else {
      const detail = payload?.detail || `Upload failed with HTTP ${xhr.status}.`;
      showAlert(detail);
      updateStartButtonState();
    }
  });

  xhr.addEventListener("error", () => {
    state.isUploading = false;
    dom.uploadProgressWrap.classList.add("hidden");
    showAlert("Network error occurred while uploading video file.");
    updateStartButtonState();
  });

  xhr.send(formData);
}

function populateMediaInspector(filename, meta, videoUrl) {
  dom.uploadDropzone.classList.add("hidden");
  dom.mediaInspector.classList.remove("hidden");
  dom.btnReplaceVideo.classList.remove("hidden");

  dom.metaFilename.textContent = filename;
  dom.metaFilename.title = filename;

  if (meta.has_audio) {
    dom.metaAudioBadge.className = "audio-pill audio-yes";
    dom.metaAudioBadge.textContent = `Audio: ${(meta.audio_codec || "Stereo").toUpperCase()}`;
  } else {
    dom.metaAudioBadge.className = "audio-pill audio-no";
    dom.metaAudioBadge.textContent = "Silent Video (Auto-Synth Track)";
  }

  dom.metaDuration.textContent = formatSeconds(meta.duration);
  dom.metaResolution.textContent = `${meta.width}×${meta.height}`;
  dom.metaFps.textContent = `${meta.fps} fps`;
  dom.metaCodecs.textContent = `${meta.video_codec}${meta.audio_codec ? " / " + meta.audio_codec : ""}`;
  dom.metaAspect.textContent = meta.aspect_ratio_display || "16:9";
  dom.metaSize.textContent = formatBytes(meta.size_bytes);

  dom.inspectorThumbVideo.src = `${videoUrl}#t=0.4`;

  // Load Before comparison player immediately
  dom.videoBefore.src = videoUrl;
  dom.videoBefore.classList.remove("hidden");
  dom.placeholderBefore.classList.add("hidden");
  dom.sourcePaneMeta.textContent = `${meta.width}×${meta.height} • ${formatSeconds(meta.duration)}`;
}

// --- Launching & Polling AI Render Job ---
async function launchRenderJob() {
  if (!state.mediaId || state.isProcessing) return;

  const prompt = dom.promptInput.value.trim();
  if (prompt.length < 4) {
    showAlert("Please enter editing instructions (at least 4 characters).");
    dom.promptInput.focus();
    return;
  }

  showAlert("");
  stopPolling();

  const maxDurVal = Number(dom.rangeMaxDuration.value);
  const duckVal = Number(dom.rangeDucking.value) / 100;
  state.duckingLevel = duckVal;

  const requestBody = {
    media_id: state.mediaId,
    prompt,
    voice: dom.selectVoice.value,
    aspect_ratio: dom.selectAspect.value,
    tts_engine: dom.selectTtsEngine.value,
    ducking_level: Number(duckVal.toFixed(2)),
    max_duration: maxDurVal > 0 ? maxDurVal : null,
  };

  state.isProcessing = true;
  state.jobStartedAtMs = Date.now();
  updateStartButtonState();

  // Reset After player for fresh render
  dom.videoAfter.pause();
  dom.videoAfter.removeAttribute("src");
  dom.videoAfter.classList.add("hidden");
  dom.placeholderAfter.classList.remove("hidden");
  dom.placeholderAfterText.textContent = "Rendering AI edit & voiceover...";
  dom.outputPaneMeta.textContent = "Processing...";
  dom.renderSummaryRibbon.classList.add("hidden");
  dom.btnSyncPlay.disabled = true;
  dom.btnDownloadMp4.classList.add("disabled-link");
  dom.btnDownloadMp4.setAttribute("aria-disabled", "true");

  startElapsedClock();

  try {
    const res = await fetch("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || `Server returned HTTP ${res.status}`);
    }

    state.activeJobId = data.job_id;
    localStorage.setItem("cineflow_last_job_id", data.job_id);
    applyJobSnapshot(data);
    beginStatusPolling(data.job_id);
    loadRecentJobs();
  } catch (err) {
    state.isProcessing = false;
    stopElapsedClock();
    updateStartButtonState();
    showAlert(err.message);
  }
}

function beginStatusPolling(jobId) {
  stopPolling();
  state.pollTimer = window.setInterval(async () => {
    try {
      const res = await fetch(`/api/status/${encodeURIComponent(jobId)}`);
      if (!res.ok) return;
      const snapshot = await res.json();
      applyJobSnapshot(snapshot);

      if (snapshot.status === "Done" || snapshot.status === "Failed") {
        stopPolling();
        stopElapsedClock();
        state.isProcessing = false;
        updateStartButtonState();
        loadRecentJobs();
      }
    } catch {
      // Transient network hiccup; retry on next tick
    }
  }, 1300);
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function startElapsedClock() {
  stopElapsedClock();
  dom.jobElapsed.textContent = "00:00";
  state.elapsedTimer = window.setInterval(() => {
    if (state.jobStartedAtMs) {
      dom.jobElapsed.textContent = formatClock(Date.now() - state.jobStartedAtMs);
    }
  }, 500);
}

function stopElapsedClock() {
  if (state.elapsedTimer) {
    clearInterval(state.elapsedTimer);
    state.elapsedTimer = null;
  }
}

// --- Updating Pipeline UI, Player & Multi-Track Timeline ---
function applyJobSnapshot(job) {
  const pct = Math.max(0, Math.min(100, Number(job.progress) || 0));
  dom.jobPercent.textContent = `${pct}%`;
  dom.masterProgressFill.style.width = `${pct}%`;
  dom.masterProgressFill.parentElement?.setAttribute("aria-valuenow", String(pct));
  dom.jobStageDetail.textContent = job.stage_detail || "";

  // Update status badge
  dom.jobStatusBadge.textContent = job.status;
  if (job.status === "Done") {
    dom.jobStatusBadge.className = "job-badge badge-done";
  } else if (job.status === "Failed") {
    dom.jobStatusBadge.className = "job-badge badge-failed";
  } else {
    dom.jobStatusBadge.className = "job-badge badge-active";
  }

  updateStepperNodes(job.status);

  // Logs console
  const logs = job.logs || [];
  dom.logLineCount.textContent = `${logs.length} events`;
  if (logs.length > 0) {
    dom.consoleLogOutput.textContent = logs.join("\n");
    dom.consoleLogOutput.scrollTop = dom.consoleLogOutput.scrollHeight;
  }

  if (job.status === "Failed" && job.error) {
    showAlert(job.error);
    dom.consoleDetails.open = true;
  }

  // Render timeline as soon as Gemini finishes Analyzing (even before FFmpeg completes!)
  if (job.timeline && Array.isArray(job.timeline.segments)) {
    renderTimelineVisualizer(
      job.timeline,
      job.source_metadata || state.sourceMetadata,
      job.options?.ducking_level ?? state.duckingLevel
    );
  }

  // Activate Final Cut player & download button when Done
  if (job.status === "Done" && job.stream_url && job.download_url) {
    const currentSrc = dom.videoAfter.getAttribute("data-loaded-job");
    if (currentSrc !== job.job_id) {
      dom.videoAfter.src = job.stream_url;
      dom.videoAfter.setAttribute("data-loaded-job", job.job_id);
      dom.videoAfter.classList.remove("hidden");
      dom.placeholderAfter.classList.add("hidden");
    }

    const outMeta = job.output_metadata;
    if (outMeta) {
      dom.outputPaneMeta.textContent = `${outMeta.width}×${outMeta.height} • ${formatSeconds(outMeta.duration)}`;
    }

    dom.btnDownloadMp4.href = job.download_url;
    dom.btnDownloadMp4.classList.remove("disabled-link");
    dom.btnDownloadMp4.setAttribute("aria-disabled", "false");
    dom.btnSyncPlay.disabled = false;

    populateSummaryRibbon(job);
  }
}

function updateStepperNodes(status) {
  const nodes = dom.pipelineStepper.querySelectorAll(".step-node");
  const activeIdx = PIPELINE_STEPS.indexOf(status);

  nodes.forEach((node, idx) => {
    node.classList.remove("is-current", "is-completed", "is-failed");
    if (status === "Failed") {
      if (idx === Math.max(0, activeIdx)) {
        node.classList.add("is-failed");
      }
      return;
    }
    if (status === "Done") {
      node.classList.add("is-completed");
      return;
    }
    if (activeIdx >= 0) {
      if (idx < activeIdx) {
        node.classList.add("is-completed");
      } else if (idx === activeIdx) {
        node.classList.add("is-current");
      }
    }
  });
}

function populateSummaryRibbon(job) {
  const srcDur = job.source_metadata?.duration || 0;
  const outMeta = job.output_metadata;
  const outDur = outMeta?.duration || 0;
  const segCount = job.timeline?.segments?.length || 0;

  let deltaNote = "";
  if (srcDur > 0 && outDur > 0 && outDur < srcDur) {
    const savedPct = Math.round(((srcDur - outDur) / srcDur) * 100);
    deltaNote = ` (-${savedPct}%)`;
  }

  dom.statDurationCompare.textContent = `${formatSeconds(srcDur)} → ${formatSeconds(outDur)}${deltaNote}`;
  dom.statSegmentsCount.textContent = `${segCount} highlight ${segCount === 1 ? "cut" : "cuts"}`;
  dom.statOutputSpecs.textContent = outMeta
    ? `${outMeta.width}×${outMeta.height} • ${formatBytes(outMeta.size_bytes)}`
    : "MP4 H.264/AAC";

  const engines = job.timeline?.tts_engines_used || [];
  if (engines.length > 0) {
    const friendly = engines[0].startsWith("groq:") ? "Groq Orpheus TTS" : "Edge Neural TTS";
    dom.statVoiceEngine.textContent = friendly;
  } else {
    dom.statVoiceEngine.textContent = "Original Audio";
  }

  dom.renderSummaryRibbon.classList.remove("hidden");
}

// --- Multi-Track Nonlinear Timeline Visualizer ---
function renderTimelineVisualizer(timeline, sourceMeta, duckingLevel) {
  state.currentTimeline = timeline;
  const segments = timeline.segments || [];
  if (segments.length === 0) return;

  dom.timelineEmptyState.classList.add("hidden");
  dom.timelineContainer.classList.remove("hidden");

  dom.timelineProjectTitle.textContent = timeline.title || "AI Nonlinear Edit Timeline";
  dom.timelineSummaryText.textContent = timeline.summary || "";

  if (timeline.narrative_tone) {
    dom.timelineToneBadge.textContent = timeline.narrative_tone;
    dom.timelineToneBadge.classList.remove("hidden");
  }
  if (timeline.voice_label) {
    dom.timelineVoiceBadge.textContent = timeline.voice_label.split("—")[0].trim();
    dom.timelineVoiceBadge.classList.remove("hidden");
  }

  dom.modelAttributionLabel.textContent = [
    timeline.model_used ? `Gemini: ${timeline.model_used}` : null,
    timeline.analysis_mode ? `Mode: ${timeline.analysis_mode}` : null,
  ]
    .filter(Boolean)
    .join(" • ");

  const srcDuration = Math.max(
    1,
    sourceMeta?.duration || Math.max(...segments.map((s) => s.end || 1))
  );
  const editedTotal = Math.max(
    0.5,
    segments[segments.length - 1]?.timeline_end ||
      segments.reduce((acc, s) => acc + (s.end - s.start) / (s.playback_speed || 1), 0)
  );

  // 1. Source Extraction Footprint Bar
  dom.sourceFootprintSpan.textContent = `0.0s — ${formatSeconds(srcDuration)} raw source`;
  dom.sourceFootprintBar.innerHTML = segments
    .map((seg, idx) => {
      const leftPct = Math.max(0, Math.min(99.5, (seg.start / srcDuration) * 100));
      const widthPct = Math.max(0.8, Math.min(100 - leftPct, ((seg.end - seg.start) / srcDuration) * 100));
      return `<div
        class="footprint-slice"
        style="left:${leftPct.toFixed(2)}%;width:${widthPct.toFixed(2)}%;"
        data-seg-idx="${idx}"
        title="Cut #${idx + 1}: ${escapeHtml(seg.scene_label)} (${seg.start.toFixed(1)}s - ${seg.end.toFixed(1)}s)"
      ></div>`;
    })
    .join("");

  // 2. Edited Time Ruler
  const tickCount = 6;
  const ticksHtml = [];
  for (let i = 0; i <= tickCount; i += 1) {
    const pct = (i / tickCount) * 100;
    const tVal = (editedTotal * i) / tickCount;
    const align = i === 0 ? "transform:none;" : i === tickCount ? "transform:translateX(-100%);" : "";
    ticksHtml.push(
      `<span class="ruler-tick" style="left:${pct.toFixed(1)}%;${align}">${tVal.toFixed(1)}s</span>`
    );
  }
  dom.timelineRuler.innerHTML = ticksHtml.join("");

  // 3. Track V1 (Video Cuts & Transitions) & Track A1 (Voiceover Clips)
  dom.trackVideoClips.innerHTML = segments
    .map((seg, idx) => {
      const segDur = Math.max(0.25, (seg.timeline_end || 0) - (seg.timeline_start || 0));
      const flexBasis = Math.max(4, (segDur / editedTotal) * 100);
      const actionClass = `clip-action-${seg.action || "cut"}`;
      const speedBadge =
        seg.playback_speed && Math.abs(seg.playback_speed - 1.0) > 0.02
          ? `<span class="mini-tag">${seg.playback_speed}×</span>`
          : "";
      return `
        <div
          class="nle-clip ${escapeHtml(actionClass)}"
          style="flex: ${flexBasis.toFixed(3)} 1 0%;"
          data-seg-idx="${idx}"
          title="Click to seek • ${escapeHtml(seg.scene_label)} (${seg.timeline_start.toFixed(1)}s - ${seg.timeline_end.toFixed(1)}s)"
        >
          <div class="clip-title">#${idx + 1} ${escapeHtml(seg.scene_label)}</div>
          <div class="clip-badges">
            <span class="mini-tag">${escapeHtml(seg.transition || "cut")}</span>
            ${speedBadge}
            <span>${segDur.toFixed(1)}s</span>
          </div>
        </div>
      `;
    })
    .join("");

  dom.trackVoiceClips.innerHTML = segments
    .map((seg, idx) => {
      const segDur = Math.max(0.25, (seg.timeline_end || 0) - (seg.timeline_start || 0));
      const flexBasis = Math.max(4, (segDur / editedTotal) * 100);
      const hasVo = Boolean((seg.voiceover_text || "").trim());
      return `
        <div
          class="vo-clip-slot ${hasVo ? "vo-has-speech" : "vo-silent"}"
          style="flex: ${flexBasis.toFixed(3)} 1 0%;"
          data-seg-idx="${idx}"
          title="${hasVo ? escapeHtml(seg.voiceover_text) : "Original background audio (no narration)"}"
        >
          <span class="vo-clip-text">
            ${hasVo ? `🎙️ ${escapeHtml(seg.voiceover_text)}` : "— Ambient Audio —"}
          </span>
        </div>
      `;
    })
    .join("");

  // 4. Track A2: SVG Dynamic Audio Ducking Envelope Curve
  dom.trackDuckingSvgWrap.innerHTML = buildDuckingEnvelopeSvg(segments, editedTotal, duckingLevel);

  // 5. Detailed Scene & Voiceover Script Cards
  dom.segmentCardsList.innerHTML = segments
    .map((seg, idx) => {
      const hasVo = Boolean((seg.voiceover_text || "").trim());
      const engineShort = seg.tts_engine_used
        ? seg.tts_engine_used.startsWith("groq:")
          ? "Groq Orpheus"
          : "Edge Neural"
        : null;
      return `
        <article class="seg-card" data-seg-idx="${idx}">
          <div class="seg-card-top">
            <span class="seg-index-title">#${idx + 1} &bull; ${escapeHtml(seg.scene_label)}</span>
            <span class="seg-timestamps">${seg.start.toFixed(1)}s → ${seg.end.toFixed(1)}s</span>
          </div>
          <div class="seg-meta-row">
            <span class="seg-pill">Timeline ${seg.timeline_start.toFixed(1)}s–${seg.timeline_end.toFixed(1)}s</span>
            <span class="seg-pill">${escapeHtml((seg.transition || "cut").toUpperCase())}</span>
            <span class="seg-pill">${seg.playback_speed || 1}× speed</span>
            <span class="seg-pill">Energy ${seg.energy_score || 7}/10</span>
            ${engineShort ? `<span class="seg-pill">${escapeHtml(engineShort)}</span>` : ""}
          </div>
          ${
            hasVo
              ? `<p class="seg-script-quote">&ldquo;${escapeHtml(seg.voiceover_text)}&rdquo;</p>`
              : `<p class="seg-script-empty">Un-narrated beat — background audio swells to full level.</p>`
          }
        </article>
      `;
    })
    .join("");

  // Bind click-to-seek handlers across footprint slices, track clips, and script cards
  dom.timelineContainer.querySelectorAll("[data-seg-idx]").forEach((el) => {
    el.addEventListener("click", () => {
      const idx = Number(el.getAttribute("data-seg-idx"));
      const seg = segments[idx];
      if (!seg) return;
      seekPlayersToSegment(seg);
    });
  });
}

function buildDuckingEnvelopeSvg(segments, totalDur, duckLevel) {
  const width = 1000;
  const height = 40;
  const highY = 8; // 92% nominal background audio volume
  const lowY = Math.round(height - 6 - Math.max(0.05, Math.min(0.65, duckLevel)) * 26);

  const points = [`0,${highY}`];

  segments.forEach((seg) => {
    const hasVo = Boolean((seg.voiceover_text || "").trim());
    if (!hasVo) return;
    const startT = seg.timeline_start || 0;
    const voDur =
      seg.voiceover_duration > 0
        ? Math.min(seg.timeline_end - startT, seg.voiceover_duration)
        : Math.max(0.3, (seg.timeline_end - startT) * 0.85);
    const endT = Math.min(totalDur, startT + voDur);

    const x1 = Math.max(0, Math.min(width, (startT / totalDur) * width));
    const xAttack = Math.min(width, x1 + 10);
    const xReleaseStart = Math.max(xAttack, Math.min(width, (endT / totalDur) * width));
    const x2 = Math.min(width, xReleaseStart + 14);

    points.push(`${x1.toFixed(1)},${highY}`);
    points.push(`${xAttack.toFixed(1)},${lowY}`);
    points.push(`${xReleaseStart.toFixed(1)},${lowY}`);
    points.push(`${x2.toFixed(1)},${highY}`);
  });

  points.push(`${width},${highY}`);
  const areaPoints = [`0,${height}`, ...points, `${width},${height}`].join(" ");

  return `
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" width="100%" height="100%" aria-label="Audio ducking gain curve">
      <defs>
        <linearGradient id="duckGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#10b981" stop-opacity="0.42" />
          <stop offset="100%" stop-color="#10b981" stop-opacity="0.04" />
        </linearGradient>
      </defs>
      <polygon points="${areaPoints}" fill="url(#duckGrad)" />
      <polyline points="${points.join(" ")}" fill="none" stroke="#34d399" stroke-width="2" />
    </svg>
  `;
}

function seekPlayersToSegment(seg) {
  if (!dom.videoBefore.classList.contains("hidden") && Number.isFinite(seg.start)) {
    dom.videoBefore.currentTime = seg.start;
  }
  if (!dom.videoAfter.classList.contains("hidden") && Number.isFinite(seg.timeline_start)) {
    dom.videoAfter.currentTime = seg.timeline_start;
    dom.videoAfter.play().catch(() => {});
  }
}

// --- Synchronized Before/After Playback & View Modes ---
function toggleSynchronizedPlayback() {
  const afterReady = !dom.videoAfter.classList.contains("hidden") && dom.videoAfter.src;
  const beforeReady = !dom.videoBefore.classList.contains("hidden") && dom.videoBefore.src;
  if (!afterReady) return;

  const isPlaying = !dom.videoAfter.paused && !dom.videoAfter.ended;
  if (isPlaying) {
    dom.videoAfter.pause();
    if (beforeReady) dom.videoBefore.pause();
    dom.btnSyncPlayText.textContent = "Sync Play Both";
  } else {
    if (beforeReady) {
      dom.videoBefore.muted = dom.chkMuteOriginalSync.checked;
      if (dom.videoAfter.ended) {
        dom.videoAfter.currentTime = 0;
        dom.videoBefore.currentTime = state.currentTimeline?.segments?.[0]?.start || 0;
      }
      dom.videoBefore.play().catch(() => {});
    }
    dom.videoAfter.play().catch(() => {});
    dom.btnSyncPlayText.textContent = "Pause Both";
  }
}

function setPlayerViewMode(mode) {
  if (mode === "theater") {
    dom.videoStageGrid.className = "video-grid mode-theater";
    dom.tabTheater.classList.add("active");
    dom.tabTheater.setAttribute("aria-selected", "true");
    dom.tabCompare.classList.remove("active");
    dom.tabCompare.setAttribute("aria-selected", "false");
  } else {
    dom.videoStageGrid.className = "video-grid mode-compare";
    dom.tabCompare.classList.add("active");
    dom.tabCompare.setAttribute("aria-selected", "true");
    dom.tabTheater.classList.remove("active");
    dom.tabTheater.setAttribute("aria-selected", "false");
  }
}

function updatePromptCounter() {
  dom.promptCharCount.textContent = `${dom.promptInput.value.length} / 1500`;
}

// --- Event Listeners Setup ---
function initEventListeners() {
  // Dropzone click & keyboard trigger
  dom.uploadDropzone.addEventListener("click", () => dom.videoFileInput.click());
  dom.uploadDropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      dom.videoFileInput.click();
    }
  });
  dom.btnReplaceVideo.addEventListener("click", () => dom.videoFileInput.click());

  dom.videoFileInput.addEventListener("change", () => {
    const file = dom.videoFileInput.files?.[0];
    if (file) handleSelectedFile(file);
    dom.videoFileInput.value = "";
  });

  // Drag & Drop handlers
  ["dragenter", "dragover"].forEach((evtName) => {
    dom.uploadDropzone.addEventListener(evtName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dom.uploadDropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((evtName) => {
    dom.uploadDropzone.addEventListener(evtName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dom.uploadDropzone.classList.remove("drag-over");
    });
  });

  dom.uploadDropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) handleSelectedFile(file);
  });

  // Preset prompt chips
  document.querySelectorAll(".preset-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const prompt = chip.getAttribute("data-prompt");
      const aspect = chip.getAttribute("data-aspect");
      const voice = chip.getAttribute("data-voice");
      const maxDur = chip.getAttribute("data-maxdur");

      if (prompt) dom.promptInput.value = prompt;
      if (aspect) dom.selectAspect.value = aspect;
      if (voice) dom.selectVoice.value = voice;
      if (maxDur) {
        dom.rangeMaxDuration.value = maxDur;
        dom.labelMaxDuration.textContent = `${maxDur}s`;
      }
      updatePromptCounter();
    });
  });

  dom.promptInput.addEventListener("input", updatePromptCounter);

  dom.rangeMaxDuration.addEventListener("input", () => {
    const val = Number(dom.rangeMaxDuration.value);
    dom.labelMaxDuration.textContent = val === 0 ? "Auto" : `${val}s`;
  });

  dom.rangeDucking.addEventListener("input", () => {
    const pct = Number(dom.rangeDucking.value);
    const db = Math.round(20 * Math.log10(Math.max(0.01, pct / 100)));
    dom.labelDucking.textContent = `${pct}% (${db} dB)`;
  });

  dom.btnStartRender.addEventListener("click", launchRenderJob);

  // Comparison view tabs & sync play
  dom.tabCompare.addEventListener("click", () => setPlayerViewMode("compare"));
  dom.tabTheater.addEventListener("click", () => setPlayerViewMode("theater"));
  dom.btnSyncPlay.addEventListener("click", toggleSynchronizedPlayback);

  dom.videoAfter.addEventListener("ended", () => {
    dom.btnSyncPlayText.textContent = "Sync Play Both";
    dom.videoBefore.pause();
  });

  dom.chkMuteOriginalSync.addEventListener("change", () => {
    if (!dom.videoAfter.paused) {
      dom.videoBefore.muted = dom.chkMuteOriginalSync.checked;
    }
  });

  // History popover
  dom.btnToggleHistory.addEventListener("click", (e) => {
    e.stopPropagation();
    dom.historyDrawer.classList.toggle("hidden");
  });
  dom.btnRefreshHistory.addEventListener("click", (e) => {
    e.stopPropagation();
    loadRecentJobs();
  });
  document.addEventListener("click", (e) => {
    if (
      !dom.historyDrawer.contains(e.target) &&
      !dom.btnToggleHistory.contains(e.target)
    ) {
      dom.historyDrawer.classList.add("hidden");
    }
  });
}

// --- Bootstrap ---
(async function bootstrapStudio() {
  initEventListeners();
  updatePromptCounter();
  updateStartButtonState();
  await Promise.all([refreshHealthStatus(), loadRecentJobs()]);

  // Automatically resume tracking if a previous render job is still running
  const savedJobId = localStorage.getItem("cineflow_last_job_id");
  if (savedJobId) {
    try {
      const res = await fetch(`/api/status/${encodeURIComponent(savedJobId)}`);
      if (res.ok) {
        const job = await res.json();
        if (!["Done", "Failed"].includes(job.status)) {
          restoreJobSession(savedJobId);
        }
      }
    } catch {
      // Ignore stale localStorage job id
    }
  }
})();
