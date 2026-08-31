const API_BASE = "/api/v1";
const GAUGE_RADIUS = 68;
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * GAUGE_RADIUS;
const CATEGORIES = ["Low", "Medium", "High", "Viral"];
const CATEGORY_COLORS = {
  Low: "#94a3b8",
  Medium: "#38bdf8",
  High: "#f59e0b",
  Viral: "#10b981",
};

const state = {
  videoFile: null,
};

const els = {
  form: document.getElementById("virality-form"),
  dropZone: document.getElementById("drop-zone"),
  fileInput: document.getElementById("video-file-input"),
  fileInfo: document.getElementById("file-info"),
  fileName: document.getElementById("file-name"),
  removeFileBtn: document.getElementById("remove-file-btn"),
  captionInput: document.getElementById("caption-input"),
  captionMeta: document.getElementById("caption-meta"),
  durationSlider: document.getElementById("duration-slider"),
  durationDisplay: document.getElementById("duration-display"),
  uploadHour: document.getElementById("upload-hour-select"),
  weekendToggle: document.getElementById("is-weekend-toggle"),
  weekendLabel: document.getElementById("weekend-label"),
  categorySelect: document.getElementById("category-select"),
  analyzeBtn: document.getElementById("analyze-btn"),
  sampleViralBtn: document.getElementById("sample-viral-btn"),
  sampleLowBtn: document.getElementById("sample-low-btn"),
  statusBadge: document.getElementById("api-status-badge"),
  statusText: document.getElementById("api-status-text"),
  loadingOverlay: document.getElementById("loading-overlay"),
  loadingStage: document.getElementById("loading-stage-text"),
  emptyState: document.getElementById("empty-state"),
  resultsContainer: document.getElementById("results-container"),
  gaugeFill: document.getElementById("gauge-fill"),
  scoreNumber: document.getElementById("score-number"),
  categoryBadge: document.getElementById("category-badge"),
  probBars: document.getElementById("prob-bars-container"),
  shapWaterfall: document.getElementById("shap-waterfall"),
  positiveDrivers: document.getElementById("positive-drivers-list"),
  negativeDrivers: document.getElementById("negative-drivers-list"),
  recommendations: document.getElementById("recommendations-list"),
};

function setStatus(kind, text) {
  els.statusBadge.className = `status-badge status-${kind}`;
  els.statusText.textContent = text;
}

async function pollHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) {
      setStatus("offline", "API Offline");
      return;
    }
    const data = await res.json();
    if (data.status === "healthy" && data.model_loaded) {
      setStatus("healthy", "API Healthy");
    } else {
      setStatus("degraded", "API Degraded");
    }
  } catch (_err) {
    setStatus("offline", "API Offline");
  }
}

function updateCaptionMeta() {
  const text = els.captionInput.value || "";
  const hashtags = (text.match(/#\w+/g) || []).length;
  els.captionMeta.textContent = `${text.length} chars | ${hashtags} hashtags`;
}

function setVideoFile(file) {
  state.videoFile = file || null;
  if (file) {
    els.fileName.textContent = file.name;
    els.fileInfo.classList.remove("hidden");
  } else {
    els.fileName.textContent = "";
    els.fileInfo.classList.add("hidden");
    els.fileInput.value = "";
  }
}

function collectPayload() {
  return {
    caption: els.captionInput.value.trim(),
    duration_sec: Number(els.durationSlider.value),
    upload_hour: Number(els.uploadHour.value),
    is_weekend: els.weekendToggle.checked ? 1 : 0,
    category_id: Number(els.categorySelect.value),
  };
}

function showLoading(stage) {
  els.loadingOverlay.classList.remove("hidden");
  els.loadingStage.textContent = stage;
}

function hideLoading() {
  els.loadingOverlay.classList.add("hidden");
}

function setGauge(score) {
  const clamped = Math.max(0, Math.min(100, Number(score) || 0));
  const offset = GAUGE_CIRCUMFERENCE * (1 - clamped / 100);
  els.gaugeFill.style.strokeDasharray = String(GAUGE_CIRCUMFERENCE);
  els.gaugeFill.style.strokeDashoffset = String(offset);
  els.gaugeFill.style.stroke =
    clamped >= 75 ? CATEGORY_COLORS.Viral : clamped >= 50 ? CATEGORY_COLORS.High : clamped >= 25 ? CATEGORY_COLORS.Medium : CATEGORY_COLORS.Low;
  els.scoreNumber.textContent = clamped.toFixed(1);
}

function setCategoryBadge(category) {
  const name = category || "Low";
  els.categoryBadge.textContent = name;
  els.categoryBadge.className = `category-badge category-${name.toLowerCase()}`;
}

function renderProbabilities(probabilities) {
  const probs = probabilities || {};
  els.probBars.innerHTML = CATEGORIES.map((cat) => {
    const value = Number(probs[cat] || 0);
    const pct = Math.round(value * 100);
    return `
      <div class="prob-row">
        <span>${cat}</span>
        <div class="prob-track">
          <div class="prob-fill" style="width:${pct}%;background:${CATEGORY_COLORS[cat]}"></div>
        </div>
        <span>${pct}%</span>
      </div>
    `;
  }).join("");
}

function collectShapDrivers(result) {
  const shap = result.shap_explanation || {};
  let items = [
    ...(shap.top_positive_features || []),
    ...(shap.top_negative_features || []),
  ];
  if (!items.length && result.explainability_breakdown) {
    items = Object.entries(result.explainability_breakdown).map(([feature, shap_value]) => ({
      feature,
      shap_value,
    }));
  }
  return items.sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));
}

function renderWaterfall(items) {
  const maxAbs = Math.max(0.01, ...items.map((item) => Math.abs(Number(item.shap_value) || 0)));
  els.shapWaterfall.innerHTML = items
    .map((item) => {
      const value = Number(item.shap_value) || 0;
      const widthPct = (Math.abs(value) / maxAbs) * 50;
      const isPos = value >= 0;
      const left = isPos ? 50 : 50 - widthPct;
      return `
        <div class="waterfall-row">
          <span class="waterfall-label" title="${item.feature}">${item.feature}</span>
          <div class="waterfall-track">
            <div class="waterfall-axis"></div>
            <div class="waterfall-bar ${isPos ? "positive" : "negative"}" style="left:${left}%;width:${widthPct}%"></div>
          </div>
          <span class="waterfall-value ${isPos ? "positive" : "negative"}">${value >= 0 ? "+" : ""}${value.toFixed(2)}</span>
        </div>
      `;
    })
    .join("");
}

function renderDriverList(target, items) {
  if (!items.length) {
    target.innerHTML = `<div class="driver-item"><span>None</span></div>`;
    return;
  }
  target.innerHTML = items
    .map(
      (item) => `
      <div class="driver-item">
        <span>${item.feature}</span>
        <strong>${Number(item.shap_value) >= 0 ? "+" : ""}${Number(item.shap_value).toFixed(2)}</strong>
      </div>
    `
    )
    .join("");
}

function renderRecommendations(recs) {
  const list = recs || [];
  if (!list.length) {
    els.recommendations.innerHTML = `<div class="rec-card"><p>No recommendations available.</p></div>`;
    return;
  }
  els.recommendations.innerHTML = list
    .map((rec) => {
      const priority = String(rec.priority || "LOW").toUpperCase();
      return `
        <article class="rec-card">
          <div class="rec-header">
            <span class="rec-feature">${rec.feature || "overall"}</span>
            <span class="priority-badge priority-${priority.toLowerCase()}">${priority}</span>
          </div>
          <p>${rec.suggestion || ""}</p>
        </article>
      `;
    })
    .join("");
}

function renderResults(result) {
  els.emptyState.classList.add("hidden");
  els.resultsContainer.classList.remove("hidden");
  setGauge(result.virality_score);
  setCategoryBadge(result.predicted_category);
  renderProbabilities(result.probabilities);
  const shapItems = collectShapDrivers(result);
  renderWaterfall(shapItems);
  const shap = result.shap_explanation || {};
  renderDriverList(els.positiveDrivers, shap.top_positive_features || shapItems.filter((i) => i.shap_value > 0));
  renderDriverList(els.negativeDrivers, shap.top_negative_features || shapItems.filter((i) => i.shap_value < 0));
  renderRecommendations(result.recommendations);
}

async function submitAnalysis(event) {
  event.preventDefault();
  const payload = collectPayload();
  if (!payload.caption) {
    els.captionInput.focus();
    return;
  }

  els.analyzeBtn.disabled = true;
  showLoading("Extracting multimodal features...");
  const stages = [
    "Running virality inference...",
    "Computing SHAP feature drivers...",
    "Generating creator recommendations...",
  ];
  let stageIdx = 0;
  const stageTimer = setInterval(() => {
    els.loadingStage.textContent = stages[stageIdx % stages.length];
    stageIdx += 1;
  }, 900);

  try {
    let response;
    if (state.videoFile) {
      const formData = new FormData();
      formData.append("caption", payload.caption);
      formData.append("duration_sec", String(payload.duration_sec));
      formData.append("upload_hour", String(payload.upload_hour));
      formData.append("is_weekend", String(payload.is_weekend));
      formData.append("category_id", String(payload.category_id));
      formData.append("file", state.videoFile);
      response = await fetch(`${API_BASE}/predict/video`, {
        method: "POST",
        body: formData,
      });
    } else {
      response = await fetch(`${API_BASE}/predict/json`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    }

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: "Prediction failed" }));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }
    const result = await response.json();
    renderResults(result);
  } catch (error) {
    els.emptyState.classList.remove("hidden");
    els.resultsContainer.classList.add("hidden");
    els.emptyState.querySelector("h3").textContent = "Analysis failed";
    els.emptyState.querySelector("p").textContent = String(error.message || error);
  } finally {
    clearInterval(stageTimer);
    hideLoading();
    els.analyzeBtn.disabled = false;
  }
}

function loadSample(kind) {
  if (kind === "viral") {
    els.captionInput.value = "Unbelievable trick shot! You will not believe this 🔥 #trending #fyp #viral #dance";
    els.durationSlider.value = "16";
    els.uploadHour.value = "20";
    els.weekendToggle.checked = true;
    els.categorySelect.value = "1";
  } else {
    els.captionInput.value = "Quiet morning walk around the block.";
    els.durationSlider.value = "48";
    els.uploadHour.value = "3";
    els.weekendToggle.checked = false;
    els.categorySelect.value = "0";
  }
  els.durationDisplay.textContent = `${els.durationSlider.value}s`;
  els.weekendLabel.textContent = els.weekendToggle.checked
    ? "Yes (Saturday/Sunday)"
    : "No (Weekday)";
  updateCaptionMeta();
}

function bindEvents() {
  els.dropZone.addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", () => {
    const file = els.fileInput.files && els.fileInput.files[0];
    setVideoFile(file || null);
  });
  ["dragenter", "dragover"].forEach((evt) => {
    els.dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      els.dropZone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    els.dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      els.dropZone.classList.remove("dragover");
    });
  });
  els.dropZone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    setVideoFile(file || null);
  });
  els.removeFileBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    setVideoFile(null);
  });

  els.captionInput.addEventListener("input", updateCaptionMeta);
  els.durationSlider.addEventListener("input", () => {
    els.durationDisplay.textContent = `${els.durationSlider.value}s`;
  });
  els.weekendToggle.addEventListener("change", () => {
    els.weekendLabel.textContent = els.weekendToggle.checked
      ? "Yes (Saturday/Sunday)"
      : "No (Weekday)";
  });
  els.form.addEventListener("submit", submitAnalysis);
  els.sampleViralBtn.addEventListener("click", () => loadSample("viral"));
  els.sampleLowBtn.addEventListener("click", () => loadSample("low"));
}

els.gaugeFill.style.strokeDasharray = String(GAUGE_CIRCUMFERENCE);
els.gaugeFill.style.strokeDashoffset = String(GAUGE_CIRCUMFERENCE);
bindEvents();
updateCaptionMeta();
pollHealth();
setInterval(pollHealth, 10000);
