/* ─────────────────────────────────────────────
   ViralityAI  —  app.js
   GPT-Style SPA: Input → Results routing
   localStorage history persistence
   ───────────────────────────────────────────── */

const API_ORIGIN = (window.VIRALITY_API_URL || window.location.origin).replace(/\/$/, "");
const API_BASE = `${API_ORIGIN}/api/v1`;
const GAUGE_RADIUS = 68;
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * GAUGE_RADIUS;
const HISTORY_KEY = "virality-history";
const HISTORY_MAX = 50;

const CATEGORIES = ["Low", "Medium", "High", "Viral"];
const CAT_COLORS = {
  Low:    "#F43F5E",
  Medium: "#94A3B8",
  High:   "#F59E0B",
  Viral:  "#10B981",
};
const SHAP_POS = "#10B981";
const SHAP_NEG = "#F43F5E";

const ABLATION_LABELS = ["Metadata", "Text", "Audio", "Visual", "Full Fusion"];
const ABLATION_KEYS   = ["Metadata", "Text", "Audio", "Visual", "Multimodal Full Fusion"];

/* ── Chart instances ── */
const charts = { prob: null, shap: null, ablation: null };
const chartState = { probs: null, shapItems: null, ablation: null };

/* ── DOM refs ── */
const $ = id => document.getElementById(id);
const els = {
  // Pages
  pageInput:   $("page-input"),
  pageResults: $("page-results"),
  // Form
  form:           $("virality-form"),
  dropZone:       $("drop-zone"),
  fileInput:      $("video-file-input"),
  dzIdle:         $("dz-idle"),
  dzFile:         $("dz-file"),
  fileName:       $("file-name"),
  removeFileBtn:  $("remove-file-btn"),
  captionInput:   $("caption-input"),
  captionMeta:    $("caption-meta"),
  durationSlider: $("duration-slider"),
  durationDisplay:$("duration-display"),
  uploadHour:     $("upload-hour-select"),
  weekendToggle:  $("is-weekend-toggle"),
  weekendLabel:   $("weekend-label"),
  categorySelect: $("category-select"),
  analyzeBtn:     $("analyze-btn"),
  sampleViralBtn: $("sample-viral-btn"),
  sampleLowBtn:   $("sample-low-btn"),
  // Results
  loadingOverlay: $("loading-overlay"),
  loadingStage:   $("loading-stage-text"),
  resultsContent: $("results-content"),
  resultsMetaBar: $("results-meta-bar"),
  gaugeFill:      $("gauge-fill"),
  scoreNumber:    $("score-number"),
  categoryBadge:  $("category-badge"),
  positiveDrivers:$("positive-drivers-list"),
  negativeDrivers:$("negative-drivers-list"),
  recommendations:$("recommendations-list"),
  backBtn:        $("back-btn"),
  // Sidebar
  sidebar:          $("sidebar"),
  sidebarCloseBtn:  $("sidebar-close-btn"),
  sidebarOpenBtn:   $("sidebar-open-btn"),
  newAnalysisBtn:   $("new-analysis-btn"),
  historyList:      $("history-list"),
  historyEmpty:     $("history-empty"),
  sidebarModelStatus: $("sidebar-model-status"),
  // Status + theme
  statusBadge:  $("api-status-badge"),
  statusText:   $("api-status-text"),
  themeToggle:  $("theme-toggle"),
  toastContainer: $("toast-container"),
};

const state = {
  videoFile: null,
  currentHistoryId: null,
};

/* ════════════════════════════════════════
   THEME
   ════════════════════════════════════════ */
function isLight() {
  return document.documentElement.getAttribute("data-theme") === "light";
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("virality-theme", theme);
  applyChartDefaults();
  if (chartState.ablation) renderAblationChart(chartState.ablation);
  if (chartState.probs)    renderProbChart(chartState.probs);
  if (chartState.shapItems)renderShapChart(chartState.shapItems);
}

(function initTheme() {
  const saved = localStorage.getItem("virality-theme");
  applyTheme(saved === "light" ? "light" : "dark");
})();

/* ════════════════════════════════════════
   PARTICLE CANVAS
   ════════════════════════════════════════ */
(function initParticles() {
  const canvas = $("ambient-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener("resize", resize);

  const count = Math.min(42, Math.floor(window.innerWidth / 32));
  const pts = Array.from({ length: count }, () => ({
    x: Math.random() * window.innerWidth,
    y: Math.random() * window.innerHeight,
    r: Math.random() * 1.6 + 0.4,
    vx: (Math.random() - 0.5) * 0.35,
    vy: (Math.random() - 0.5) * 0.35,
    a: Math.random() * 0.4 + 0.12,
  }));

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const rgb = isLight() ? "107,59,250" : "140,110,255";
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      p.x = (p.x + p.vx + canvas.width) % canvas.width;
      p.y = (p.y + p.vy + canvas.height) % canvas.height;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${rgb},${p.a})`;
      ctx.fill();
      for (let j = i + 1; j < pts.length; j++) {
        const q = pts[j];
        const d = Math.hypot(p.x - q.x, p.y - q.y);
        if (d < 105) {
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(q.x, q.y);
          ctx.strokeStyle = `rgba(${rgb},${(1 - d / 105) * 0.1})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }
  draw();
})();

/* ════════════════════════════════════════
   SPA PAGE ROUTING
   ════════════════════════════════════════ */
function showPage(page) {
  if (page === "input") {
    els.pageInput.classList.remove("hidden");
    els.pageResults.classList.add("hidden");
    // Reset results page scroll
    els.pageResults.scrollTop = 0;
  } else {
    els.pageInput.classList.add("hidden");
    els.pageResults.classList.remove("hidden");
    els.pageResults.scrollTop = 0;
  }
}

/* ════════════════════════════════════════
   SIDEBAR (mobile toggle)
   ════════════════════════════════════════ */
let overlay = null;
function ensureOverlay() {
  if (overlay) return overlay;
  overlay = document.createElement("div");
  overlay.className = "sidebar-overlay";
  overlay.addEventListener("click", closeSidebar);
  document.body.appendChild(overlay);
  return overlay;
}

function openSidebar() {
  els.sidebar.classList.add("open");
  ensureOverlay().classList.add("visible");
}

function closeSidebar() {
  els.sidebar.classList.remove("open");
  if (overlay) overlay.classList.remove("visible");
}

/* ════════════════════════════════════════
   API STATUS
   ════════════════════════════════════════ */
function setStatus(kind, text) {
  if (els.statusBadge) {
    els.statusBadge.className = `status-badge status-${kind}`;
    els.statusText.textContent = text;
  }
  if (els.sidebarModelStatus) {
    const dot = els.sidebarModelStatus.previousElementSibling;
    els.sidebarModelStatus.textContent = text;
    if (dot && dot.classList.contains("stat-dot")) {
      dot.className = "stat-dot " + (kind === "healthy" ? "healthy" : kind === "offline" ? "offline" : "");
    }
  }
}

async function pollHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) { setStatus("offline", "Service offline"); return; }
    const data = await res.json();
    if (data.status === "healthy" && data.model_loaded) {
      setStatus("healthy", "Pipeline ready");
    } else {
      setStatus("degraded", "Model degraded");
    }
  } catch {
    setStatus("offline", "Service offline");
  }
}

/* ════════════════════════════════════════
   TOAST
   ════════════════════════════════════════ */
function showToast(title, msg, type = "error") {
  if (!els.toastContainer) return;
  const t = document.createElement("div");
  t.className = `toast toast-${type}`;
  t.innerHTML = `<div class="toast-title">${title}</div><div class="toast-msg">${msg}</div>`;
  els.toastContainer.appendChild(t);
  setTimeout(() => {
    t.style.transition = "all 0.28s ease";
    t.style.opacity = "0";
    t.style.transform = "translateX(40px)";
    setTimeout(() => t.remove(), 300);
  }, 4000);
}

/* ════════════════════════════════════════
   FORM UTILS
   ════════════════════════════════════════ */
function updateCaptionMeta() {
  const txt = els.captionInput.value || "";
  const tags = (txt.match(/#\w+/g) || []).length;
  els.captionMeta.textContent = `${txt.length} chars · ${tags} #tags`;
}

function setVideoFile(file) {
  state.videoFile = file || null;
  if (file) {
    els.fileName.textContent = file.name;
    els.dzIdle.classList.add("hidden");
    els.dzFile.classList.remove("hidden");
  } else {
    els.dzIdle.classList.remove("hidden");
    els.dzFile.classList.add("hidden");
    els.fileInput.value = "";
  }
}

function collectPayload() {
  return {
    caption:     els.captionInput.value.trim(),
    duration_sec:Number(els.durationSlider.value),
    upload_hour: Number(els.uploadHour.value),
    is_weekend:  els.weekendToggle.checked ? 1 : 0,
    category_id: Number(els.categorySelect.value),
  };
}

function loadSample(kind) {
  if (kind === "viral") {
    els.captionInput.value = "Unbelievable trick shot! 🔥 #trending #fyp #viral #dance #wow";
    els.durationSlider.value = "15";
    els.uploadHour.value = "20";
    els.weekendToggle.checked = true;
    els.categorySelect.value = "1";
  } else {
    els.captionInput.value = "Quiet morning walk around the neighborhood.";
    els.durationSlider.value = "48";
    els.uploadHour.value = "3";
    els.weekendToggle.checked = false;
    els.categorySelect.value = "0";
  }
  els.durationDisplay.textContent = `${els.durationSlider.value}s`;
  els.weekendLabel.textContent = els.weekendToggle.checked ? "Saturday / Sunday" : "Weekday (Mon–Fri)";
  updateCaptionMeta();
}

/* ════════════════════════════════════════
   LOADING STATE
   ════════════════════════════════════════ */
function showLoading(stage) {
  els.loadingOverlay.classList.remove("hidden");
  els.loadingStage.textContent = stage;
}
function hideLoading() {
  els.loadingOverlay.classList.add("hidden");
}

/* ════════════════════════════════════════
   HISTORY (localStorage)
   ════════════════════════════════════════ */
function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveHistory(list) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(list.slice(0, HISTORY_MAX)));
}

function addToHistory(result, payload) {
  const id = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
  const entry = {
    id,
    timestamp: Date.now(),
    caption:   (payload.caption || "Untitled").slice(0, 60),
    score:     Number(result.virality_score || 0),
    category:  result.predicted_category || "Low",
    payload,
    result,
  };
  const list = getHistory();
  list.unshift(entry);
  saveHistory(list);
  state.currentHistoryId = id;
  renderHistoryList();
  return id;
}

function deleteHistoryEntry(id) {
  const list = getHistory().filter(e => e.id !== id);
  saveHistory(list);
  renderHistoryList();
  if (state.currentHistoryId === id) {
    showPage("input");
  }
}

function renderHistoryList() {
  const list = getHistory();
  if (!els.historyList) return;

  // Clear children except the empty state div
  Array.from(els.historyList.children).forEach(c => {
    if (c.id !== "history-empty") c.remove();
  });

  if (!list.length) {
    els.historyEmpty.classList.remove("hidden");
    return;
  }
  els.historyEmpty.classList.add("hidden");

  list.forEach(entry => {
    const div = document.createElement("div");
    div.className = "history-item" + (entry.id === state.currentHistoryId ? " active" : "");
    div.dataset.id = entry.id;

    const cat = (entry.category || "low").toLowerCase();
    const iconClass = `h-icon-${cat}`;
    const initials = { viral: "V", high: "H", medium: "M", low: "L" };
    const score = Number(entry.score).toFixed(1);

    div.innerHTML = `
      <div class="history-icon ${iconClass}">${(initials[cat] || "?")}</div>
      <div class="history-meta">
        <div class="history-caption">${escapeHtml(entry.caption || "Untitled")}</div>
        <div class="history-score">${entry.category} · ${score}/100</div>
      </div>
      <button class="history-delete" title="Delete" aria-label="Delete this entry">×</button>
    `;

    div.querySelector(".history-delete").addEventListener("click", e => {
      e.stopPropagation();
      deleteHistoryEntry(entry.id);
    });

    div.addEventListener("click", () => restoreHistoryEntry(entry.id));

    els.historyList.appendChild(div);
  });
}

function restoreHistoryEntry(id) {
  const list = getHistory();
  const entry = list.find(e => e.id === id);
  if (!entry) return;

  state.currentHistoryId = id;
  renderHistoryList();

  // Restore form fields
  if (entry.payload) {
    els.captionInput.value      = entry.payload.caption || "";
    els.durationSlider.value    = entry.payload.duration_sec || 18;
    els.uploadHour.value        = entry.payload.upload_hour || 20;
    els.weekendToggle.checked   = entry.payload.is_weekend === 1;
    els.categorySelect.value    = entry.payload.category_id || 0;
    els.durationDisplay.textContent = `${entry.payload.duration_sec || 18}s`;
    els.weekendLabel.textContent = els.weekendToggle.checked ? "Saturday / Sunday" : "Weekday (Mon–Fri)";
    updateCaptionMeta();
  }

  showPage("results");
  renderResults(entry.result, false /* no animation for history restore */);
  closeSidebar();
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/* ════════════════════════════════════════
   CHART HELPERS
   ════════════════════════════════════════ */
function chartTheme() {
  const light = isLight();
  return {
    text:   light ? "#0F172A" : "#F5F7FF",
    muted:  light ? "#64748B" : "#94A3B8",
    grid:   light ? "rgba(100,116,139,0.15)" : "rgba(255,255,255,0.06)",
  };
}

function applyChartDefaults() {
  if (typeof Chart === "undefined") return;
  const t = chartTheme();
  Chart.defaults.color = t.text;
  Chart.defaults.borderColor = t.grid;
  Chart.defaults.font.family = "'DM Sans', system-ui, sans-serif";
  Chart.defaults.font.size = 12;
}

function destroyChart(key) {
  if (charts[key]) { charts[key].destroy(); charts[key] = null; }
}

/* ════════════════════════════════════════
   GAUGE ANIMATION
   ════════════════════════════════════════ */
function animateScore(target) {
  const start = performance.now();
  const dur = 1100;
  function step(now) {
    const p = Math.min((now - start) / dur, 1);
    const ease = 1 - Math.pow(1 - p, 3);
    const val = target * ease;
    els.scoreNumber.textContent = val.toFixed(1);
    if (p < 1) requestAnimationFrame(step);
    else els.scoreNumber.textContent = target.toFixed(1);
  }
  requestAnimationFrame(step);
}

function setGauge(score, animate = true) {
  const s = Math.max(0, Math.min(100, Number(score) || 0));
  const offset = GAUGE_CIRCUMFERENCE * (1 - s / 100);
  els.gaugeFill.style.strokeDasharray = GAUGE_CIRCUMFERENCE;
  els.gaugeFill.style.strokeDashoffset = offset;

  // Color by tier
  if (s >= 75)      els.gaugeFill.style.stroke = CAT_COLORS.Viral;
  else if (s >= 50) els.gaugeFill.style.stroke = CAT_COLORS.High;
  else if (s >= 25) els.gaugeFill.style.stroke = CAT_COLORS.Medium;
  else              els.gaugeFill.style.stroke = CAT_COLORS.Low;

  if (animate) animateScore(s);
  else els.scoreNumber.textContent = s.toFixed(1);
}

function setCategoryBadge(category) {
  const c = (category || "Low").toLowerCase();
  els.categoryBadge.textContent = category || "Low";
  els.categoryBadge.className = `cat-badge cat-${c}`;
}

/* ════════════════════════════════════════
   CHART RENDERS
   ════════════════════════════════════════ */
function renderProbChart(probs) {
  chartState.probs = probs;
  const canvas = $("prob-chart");
  if (!canvas || typeof Chart === "undefined") return;
  const vals = CATEGORIES.map(c => Number(probs[c] || 0));
  const t = chartTheme();
  destroyChart("prob");

  charts.prob = new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: CATEGORIES,
      datasets: [{
        data: vals,
        backgroundColor: Object.values(CAT_COLORS),
        borderColor: isLight() ? "#ffffff" : "#060A14",
        borderWidth: 2,
        hoverOffset: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "68%",
      plugins: {
        legend: {
          position: "right",
          labels: {
            color: t.text,
            boxWidth: 11,
            padding: 12,
            font: { family: "'DM Sans', sans-serif", weight: "600", size: 11 },
          },
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${(ctx.parsed * 100).toFixed(1)}%`,
          },
        },
      },
    },
  });
}

function collectShapItems(result) {
  const shap = result.shap_explanation || {};
  let items = [
    ...(shap.top_positive_features || []),
    ...(shap.top_negative_features || []),
  ];
  if (!items.length && result.explainability_breakdown) {
    items = Object.entries(result.explainability_breakdown).map(([feature, shap_value]) => ({ feature, shap_value }));
  }
  return items.sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));
}

function renderShapChart(items) {
  chartState.shapItems = items;
  const canvas = $("shap-chart");
  if (!canvas || typeof Chart === "undefined") return;
  const top = items.slice(0, 8);
  const t = chartTheme();
  destroyChart("shap");

  charts.shap = new Chart(canvas, {
    type: "bar",
    data: {
      labels: top.map(i => i.feature),
      datasets: [{
        label: "SHAP impact",
        data: top.map(i => Number(i.shap_value) || 0),
        backgroundColor: top.map(i => i.shap_value >= 0 ? SHAP_POS : SHAP_NEG),
        borderRadius: 5,
        borderWidth: 0,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` Impact: ${ctx.parsed.x >= 0 ? "+" : ""}${ctx.parsed.x.toFixed(3)}` } },
      },
      scales: {
        x: {
          ticks: { color: t.muted, font: { family: "'JetBrains Mono', monospace", size: 10 } },
          grid:  { color: t.grid },
          title: { display: true, text: "SHAP marginal contribution", color: t.muted, font: { size: 11 } },
        },
        y: {
          ticks: { color: t.text, font: { family: "'JetBrains Mono', monospace", size: 10 } },
          grid: { display: false },
        },
      },
    },
  });
}

function renderAblationChart(payload) {
  chartState.ablation = payload;
  const canvas = $("ablation-chart");
  if (!canvas || typeof Chart === "undefined") return;
  const mods = (payload && payload.modalities) || {};
  const f1  = ABLATION_KEYS.map(k => Number((mods[k] || {}).f1_macro || 0));
  const auc = ABLATION_KEYS.map(k => Number((mods[k] || {}).roc_auc_ovr_macro || 0));
  const t = chartTheme();
  destroyChart("ablation");

  charts.ablation = new Chart(canvas, {
    type: "bar",
    data: {
      labels: ABLATION_LABELS,
      datasets: [
        { label: "Macro F1", data: f1,  backgroundColor: "rgba(107,59,250,0.82)", borderRadius: 5 },
        { label: "ROC-AUC",  data: auc, backgroundColor: "rgba(6,182,212,0.82)",  borderRadius: 5 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: t.text, font: { family: "'DM Sans', sans-serif", weight: "600" } } },
      },
      scales: {
        x: { ticks: { color: t.text },  grid: { color: t.grid } },
        y: { beginAtZero: true, max: 1, ticks: { color: t.muted, font: { family: "'JetBrains Mono', monospace", size: 10 } }, grid: { color: t.grid } },
      },
    },
  });
}

async function fetchAblationChart() {
  try {
    const res = await fetch(`${API_BASE}/metrics/ablation`);
    if (!res.ok) return;
    const payload = await res.json();
    renderAblationChart(payload);
  } catch { /* silent — benchmarks optional */ }
}

/* ════════════════════════════════════════
   DRIVER LISTS
   ════════════════════════════════════════ */
function renderDriverList(target, items, isPos) {
  if (!items || !items.length) {
    target.innerHTML = `<div class="driver-item"><span class="driver-name">None detected</span></div>`;
    return;
  }
  target.innerHTML = items.slice(0, 5).map(item => {
    const v = Number(item.shap_value);
    const cls = v >= 0 ? "positive" : "negative";
    return `
      <div class="driver-item">
        <span class="driver-name">${escapeHtml(item.feature)}</span>
        <span class="driver-val ${cls}">${v >= 0 ? "+" : ""}${v.toFixed(3)}</span>
      </div>
    `;
  }).join("");
}

/* ════════════════════════════════════════
   RECOMMENDATIONS
   ════════════════════════════════════════ */
function renderRecommendations(recs) {
  const list = recs || [];
  if (!list.length) {
    els.recommendations.innerHTML = `<div class="rec-card"><p>No targeted recommendations for this content profile.</p></div>`;
    return;
  }
  els.recommendations.innerHTML = list.map(rec => {
    const pri = String(rec.priority || "LOW").toUpperCase();
    return `
      <article class="rec-card">
        <div class="rec-header">
          <span class="rec-feature"># ${escapeHtml(rec.feature || "overall")}</span>
          <span class="priority-badge priority-${pri.toLowerCase()}">${pri}</span>
        </div>
        <p>${escapeHtml(rec.suggestion || "")}</p>
      </article>
    `;
  }).join("");
}

/* ════════════════════════════════════════
   RENDER RESULTS
   ════════════════════════════════════════ */
function renderResults(result, animate = true) {
  setGauge(result.virality_score, animate);
  setCategoryBadge(result.predicted_category);

  const confidenceEl = $("confidence-badge");
  if (confidenceEl) {
    const confidence = Math.round(Number(result.confidence || 0) * 100);
    const band = String(result.uncertainty_band || "unknown");
    confidenceEl.textContent = `Confidence proxy: ${confidence}% · ${band} uncertainty`;
  }

  renderProbChart(result.probabilities || {});

  const shapItems = collectShapItems(result);
  renderShapChart(shapItems);

  const shap = result.shap_explanation || {};
  const positives = shap.top_positive_features || shapItems.filter(i => i.shap_value > 0);
  const negatives = shap.top_negative_features || shapItems.filter(i => i.shap_value < 0);
  renderDriverList(els.positiveDrivers, positives, true);
  renderDriverList(els.negativeDrivers, negatives, false);

  renderRecommendations(result.recommendations);
}

/* ════════════════════════════════════════
   SUBMIT ANALYSIS
   ════════════════════════════════════════ */
async function submitAnalysis(e) {
  e.preventDefault();
  const payload = collectPayload();

  if (!payload.caption) {
    showToast("Caption required", "Enter a caption with hashtags to enable NLP analysis.", "error");
    els.captionInput.focus();
    return;
  }

  els.analyzeBtn.disabled = true;

  // Switch to results page immediately, show loader
  showPage("results");
  showLoading("Synthesizing multimodal representations…");

  const stages = [
    "Downsampling frames & running Librosa acoustic analysis…",
    "Computing SBERT semantic embeddings…",
    "Running XGBoost multimodal ensemble…",
    "Computing TreeSHAP game-theoretic attributions…",
    "Generating prescriptive creator advice…",
  ];
  let si = 0;
  const stageTimer = setInterval(() => {
    els.loadingStage.textContent = stages[si % stages.length];
    si++;
  }, 900);

  try {
    let res;
    if (state.videoFile) {
      const fd = new FormData();
      Object.entries(payload).forEach(([k, v]) => fd.append(k, String(v)));
      fd.append("file", state.videoFile);
      res = await fetch(`${API_BASE}/predict/video`, { method: "POST", body: fd });
    } else {
      res = await fetch(`${API_BASE}/predict/json`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Prediction pipeline failed." }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const result = await res.json();

    // Update meta bar
    if (els.resultsMetaBar) {
      const cat = result.predicted_category || "?";
      const score = Number(result.virality_score || 0).toFixed(1);
      els.resultsMetaBar.textContent = `${escapeHtml(payload.caption.slice(0, 55))}${payload.caption.length > 55 ? "…" : ""} · ${cat} · ${score}/100`;
    }

    renderResults(result, true);
    const id = addToHistory(result, payload);
    state.currentHistoryId = id;

    showToast(
      `${result.predicted_category || "Result"} — ${Number(result.virality_score).toFixed(1)}/100`,
      "Prediction saved to history.",
      "success"
    );

  } catch (err) {
    showPage("input");
    showToast("Inference failed", String(err.message || err), "error");
  } finally {
    clearInterval(stageTimer);
    hideLoading();
    els.analyzeBtn.disabled = false;
  }
}

/* ════════════════════════════════════════
   EVENT BINDING
   ════════════════════════════════════════ */
function bindEvents() {
  // Form submit
  els.form.addEventListener("submit", submitAnalysis);

  // Back button
  els.backBtn?.addEventListener("click", () => {
    showPage("input");
    state.currentHistoryId = null;
    renderHistoryList();
  });

  // New analysis button (sidebar)
  els.newAnalysisBtn?.addEventListener("click", () => {
    showPage("input");
    state.currentHistoryId = null;
    renderHistoryList();
    closeSidebar();
  });

  // Sidebar open/close
  els.sidebarOpenBtn?.addEventListener("click", openSidebar);
  els.sidebarCloseBtn?.addEventListener("click", closeSidebar);

  // Theme toggle
  els.themeToggle?.addEventListener("click", () => {
    applyTheme(isLight() ? "dark" : "light");
  });

  // Sample buttons
  els.sampleViralBtn?.addEventListener("click", () => loadSample("viral"));
  els.sampleLowBtn?.addEventListener("click", () => loadSample("low"));

  // Video dropzone
  els.dropZone.addEventListener("click", () => els.fileInput.click());
  els.dropZone.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.fileInput.click(); }
  });
  els.fileInput.addEventListener("change", () => setVideoFile(els.fileInput.files?.[0] || null));

  ["dragenter", "dragover"].forEach(ev =>
    els.dropZone.addEventListener(ev, e => { e.preventDefault(); els.dropZone.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach(ev =>
    els.dropZone.addEventListener(ev, e => { e.preventDefault(); els.dropZone.classList.remove("dragover"); })
  );
  els.dropZone.addEventListener("drop", e => setVideoFile(e.dataTransfer.files?.[0] || null));
  els.removeFileBtn?.addEventListener("click", e => { e.stopPropagation(); setVideoFile(null); });

  // Form controls
  els.captionInput.addEventListener("input", updateCaptionMeta);
  els.durationSlider.addEventListener("input", () => {
    els.durationDisplay.textContent = `${els.durationSlider.value}s`;
  });
  els.weekendToggle.addEventListener("change", () => {
    els.weekendLabel.textContent = els.weekendToggle.checked ? "Saturday / Sunday" : "Weekday (Mon–Fri)";
  });

  // Ablation chart resize on details toggle
  $("benchmarks-details")?.addEventListener("toggle", () => {
    if (charts.ablation) setTimeout(() => charts.ablation.resize(), 50);
  });
}

/* ════════════════════════════════════════
   INIT
   ════════════════════════════════════════ */
els.gaugeFill.style.strokeDasharray = GAUGE_CIRCUMFERENCE;
els.gaugeFill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;

applyChartDefaults();
bindEvents();
updateCaptionMeta();

// Load history into sidebar
renderHistoryList();

// Poll API health
pollHealth();
setInterval(pollHealth, 12000);

// Fetch ablation benchmarks
fetchAblationChart();

// Start on input page
showPage("input");
