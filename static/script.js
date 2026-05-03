/* ================================================================
   Healthcare AI — Unified Script
   Pages: Heart Disease | Brain Tumor
   Detect page via: <body data-page="heart|brain">
   ================================================================ */

"use strict";

/* ─────────────────────────────────────────────
   1. PAGE DETECTION
───────────────────────────────────────────── */
const PAGE_TYPE = document.body.dataset.page || "heart";

/* ─────────────────────────────────────────────
   2. SLIDER SYNC  (heart only)
───────────────────────────────────────────── */
const sliderPairs = [
  ["age-slider",        "age"],
  ["cigsPerDay-slider", "cigsPerDay"],
  ["totChol-slider",    "totChol"],
  ["sysBP-slider",      "sysBP"],
  ["diaBP-slider",      "diaBP"],
  ["heartRate-slider",  "heartRate"],
  ["BMI-slider",        "BMI"],
  ["glucose-slider",    "glucose"],
];

if (PAGE_TYPE === "heart") {
  sliderPairs.forEach(([sliderId, inputId]) => {
    const slider = document.getElementById(sliderId);
    const input  = document.getElementById(inputId);
    if (!slider || !input) return;
    slider.addEventListener("input", () => { input.value = slider.value; });
    input.addEventListener("input",  () => {
      slider.value = Math.min(Math.max(input.value, slider.min), slider.max);
    });
  });
}

/* ─────────────────────────────────────────────
   3. ADVANCED SECTION TOGGLE  (heart only)
───────────────────────────────────────────── */
function toggleAdvanced() {
  const section = document.getElementById("advanced-section");
  const icon    = document.getElementById("adv-icon");
  if (!section || !icon) return;
  const isOpen = section.classList.toggle("open");
  icon.style.transform = isOpen ? "rotate(90deg)" : "rotate(0deg)";
}

/* ─────────────────────────────────────────────
   4. UI STATE HELPERS
───────────────────────────────────────────── */
function showState(stateId) {
  ["idle-state", "loading-state", "result-content", "error-state"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle("hidden", id !== stateId);
  });
}

function resetError() { showState("idle-state"); }

function showError(msg) {
  const el = document.getElementById("error-msg");
  if (el) el.textContent = msg;
  showState("error-state");
}

/* ─────────────────────────────────────────────
   5. ANIMATION HELPERS
───────────────────────────────────────────── */
function animateNumber(el, from, to, duration, suffix = "") {
  if (!el) return;
  const start = performance.now();
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased    = 1 - Math.pow(1 - progress, 3);
    el.textContent = (from + (to - from) * eased).toFixed(1) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/* ─────────────────────────────────────────────
   6. SHARED RENDER HELPERS
───────────────────────────────────────────── */

/** Verdict block */
function renderVerdict(data, subText) {
  const isDisease = data.prediction === 1;
  const label     = data.label || data.result || (isDisease ? "Disease Detected" : "Normal");
  const sub       = subText || (isDisease
    ? "Please consult a specialist promptly."
    : "Maintain your healthy lifestyle.");

  const verdictEl = document.getElementById("result-verdict");
  if (verdictEl) verdictEl.className = "result-verdict " + (isDisease ? "disease" : "healthy");

  const iconEl = document.getElementById("verdict-icon");
  if (iconEl) iconEl.textContent = isDisease ? "⚠" : "✓";

  const labelEl = document.getElementById("verdict-label");
  if (labelEl) labelEl.textContent = label;

  const subEl = document.getElementById("verdict-sub");
  if (subEl) subEl.textContent = sub;

  // Brain: show tumor type badge if present
  const badgeEl = document.getElementById("tumor-badge");
  if (badgeEl) {
    badgeEl.textContent = data.result ? `Type: ${data.result}` : "";
    badgeEl.style.display = data.result ? "inline-block" : "none";
  }
}

/** Confidence bar */
function renderConfidence(confidence, isDisease) {
  const confBar = document.getElementById("conf-bar");
  if (confBar) {
    confBar.className   = "conf-bar " + (isDisease ? "disease-bar" : "healthy-bar");
    confBar.style.width = "0%";
    setTimeout(() => { confBar.style.width = confidence + "%"; }, 100);
  }
  animateNumber(document.getElementById("conf-value"), 0, confidence, 1000, "%");
}

/** Feature importance bars (heart only) */
function renderFeatureImportance(topFeatures) {
  const featList = document.getElementById("feat-list");
  if (!featList || !Array.isArray(topFeatures) || topFeatures.length === 0) return;

  featList.innerHTML = "";
  const maxImp = Math.max(...topFeatures.map(f => f.importance));

  topFeatures.forEach((f, i) => {
    const div = document.createElement("div");
    div.className = "feat-item animate-in";
    div.style.animationDelay = (i * 80) + "ms";
    div.innerHTML = `
      <div class="feat-row">
        <span class="feat-name">${f.feature}</span>
        <span class="feat-pct">${f.importance}%</span>
      </div>
      <div class="feat-bar-bg">
        <div class="feat-bar-fill" style="width:0%" data-width="${(f.importance / maxImp * 100).toFixed(1)}%"></div>
      </div>`;
    featList.appendChild(div);
  });

  setTimeout(() => {
    document.querySelectorAll(".feat-bar-fill").forEach(el => {
      el.style.width = el.dataset.width;
    });
  }, 150);
}

/** Recommendations list */
function renderRecommendations(recommendations) {
  const recList = document.getElementById("rec-list");
  if (!recList || !Array.isArray(recommendations) || recommendations.length === 0) return;
  recList.innerHTML = "";
  recommendations.forEach((r, i) => {
    const li = document.createElement("li");
    li.textContent = r;
    li.className   = "animate-in";
    li.style.animationDelay = (i * 60) + "ms";
    recList.appendChild(li);
  });
}

/** Diet plan grid */
function renderDietPlan(dietPlan) {
  const dietGrid = document.getElementById("diet-grid");
  if (!dietGrid || !dietPlan || typeof dietPlan !== "object") return;

  const mealIcons = {
    "Morning":     "🌅",
    "Mid-Morning": "☀️",
    "Lunch":       "🍽️",
    "Evening":     "🌆",
    "Dinner":      "🌙",
    "Avoid":       "🚫",
  };

  dietGrid.innerHTML = "";
  Object.entries(dietPlan).forEach(([meal, items], i) => {
    if (!Array.isArray(items)) return;
    const div = document.createElement("div");
    div.className = "diet-meal animate-in" + (meal === "Avoid" ? " avoid" : "");
    div.style.animationDelay = (i * 70) + "ms";
    const icon = mealIcons[meal] || "🥗";
    div.innerHTML = `
      <div class="diet-meal-title">${icon} ${meal}</div>
      <ul>${items.map(it => `<li>${it}</li>`).join("")}</ul>`;
    dietGrid.appendChild(div);
  });
}

/* ─────────────────────────────────────────────
   7. TUMOR PROBABILITY CHART  (brain only)
───────────────────────────────────────────── */
let tumorChart = null;

function renderTumorChart(classProbs) {
  const canvas = document.getElementById("tumorChart");
  if (!canvas || !classProbs || typeof classProbs !== "object") return;

  const labels = Object.keys(classProbs);
  const values = Object.values(classProbs);

  // Color each bar differently; highlight the winning class
  const maxVal = Math.max(...values);
  const colors = values.map(v =>
    v === maxVal
      ? "rgba(167, 139, 250, 0.85)"   // purple — highest
      : "rgba(99, 179, 237, 0.55)"    // blue — others
  );
  const borders = values.map(v =>
    v === maxVal ? "#a78bfa" : "#63b3ed"
  );

  if (tumorChart) tumorChart.destroy();

  tumorChart = new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Probability (%)",
        data: values,
        backgroundColor: colors,
        borderColor: borders,
        borderWidth: 2,
        borderRadius: 8,
      }],
    },
    options: {
      animation: { duration: 900 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.parsed.y.toFixed(1)}%`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#cbd5e1", font: { family: "DM Sans, sans-serif" } },
          grid:  { color: "rgba(255,255,255,0.06)" },
        },
        y: {
          beginAtZero: true,
          max: 100,
          ticks: { color: "#94a3b8", callback: v => v + "%" },
          grid:  { color: "rgba(255,255,255,0.06)" },
        },
      },
    },
  });
}

/* ─────────────────────────────────────────────
   8. MASTER RENDER
───────────────────────────────────────────── */
function renderResult(data, subText) {
  const isDisease = data.prediction === 1;
  renderVerdict(data, subText);
  renderConfidence(data.confidence, isDisease);
  renderFeatureImportance(data.top_features);          // heart only; safely skipped
  renderTumorChart(data.class_probabilities);           // brain only; safely skipped
  renderRecommendations(data.recommendations);
  renderDietPlan(data.diet_plan);
  showState("result-content");
}

/* ─────────────────────────────────────────────
   9. IMAGE UPLOAD PREVIEW  (brain only)
───────────────────────────────────────────── */
function initImageUpload() {
  const fileInput   = document.getElementById("image-input");   // ★ fixed id
  const preview     = document.getElementById("image-preview");
  const placeholder = document.getElementById("upload-placeholder");
  const dropZone    = document.getElementById("drop-zone");

  if (!fileInput) return;

  function displayPreview(file) {
    if (!file || !file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      if (preview) {
        preview.src = e.target.result;
        preview.classList.remove("hidden");
      }
      if (placeholder) placeholder.classList.add("hidden");
    };
    reader.readAsDataURL(file);
  }

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) displayPreview(fileInput.files[0]);
  });

  if (dropZone) {
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("dragover");
    });
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      const file = e.dataTransfer.files[0];
      if (file) {
        // Assign dropped file to input
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
        displayPreview(file);
      }
    });
  }
}

/* ─────────────────────────────────────────────
   10. HEART DISEASE FORM SUBMIT
───────────────────────────────────────────── */
function initHeartForm() {
  const form = document.getElementById("predict-form");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = document.getElementById("submit-btn");
    if (btn) btn.disabled = true;
    showState("loading-state");

    const getVal = (id) => { const el = document.getElementById(id); return el ? el.value : ""; };

    // Required selects — always visible, must be chosen
    const requiredSelects = { sex: "Sex", currentSmoker: "Current Smoker" };
    for (const id of Object.keys(requiredSelects)) {
      if (getVal(id) === "") {
        showError(`Please select a value for "${requiredSelects[id]}"`);
        if (btn) btn.disabled = false;
        return;
      }
    }

    // Advanced selects — hidden by default, default to "No" (0) if not selected
    const advancedSelects = ["BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"];
    for (const id of advancedSelects) {
      const el = document.getElementById(id);
      if (el && el.value === "") el.value = "0";
    }

    const payload = {
      age:             parseFloat(getVal("age")),
      sex:             parseFloat(getVal("sex")),
      currentSmoker:   parseFloat(getVal("currentSmoker")),
      cigsPerDay:      parseFloat(getVal("cigsPerDay")),
      BPMeds:          parseFloat(getVal("BPMeds")),
      prevalentStroke: parseFloat(getVal("prevalentStroke")),
      prevalentHyp:    parseFloat(getVal("prevalentHyp")),
      diabetes:        parseFloat(getVal("diabetes")),
      totChol:         parseFloat(getVal("totChol")),
      sysBP:           parseFloat(getVal("sysBP")),
      diaBP:           parseFloat(getVal("diaBP")),
      BMI:             parseFloat(getVal("BMI")),
      heartRate:       parseFloat(getVal("heartRate")),
      glucose:         parseFloat(getVal("glucose")),
    };

    for (const [key, val] of Object.entries(payload)) {
      if (isNaN(val)) {
        showError(`Invalid value for field: ${key}`);
        if (btn) btn.disabled = false;
        return;
      }
    }

    try {
      const res  = await fetch("/predict", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload),
      });
      const data = await res.json();

      if (!res.ok || data.error) {
        showError(data.error || "Server error. Please try again.");
        if (btn) btn.disabled = false;
        return;
      }

      const sub = data.prediction === 1
        ? "Please consult a cardiologist promptly."
        : "Maintain your healthy lifestyle.";
      renderResult(data, sub);

    } catch (err) {
      showError("Network error. Is the Flask server running?");
    }

    if (btn) btn.disabled = false;
  });
}

/* ─────────────────────────────────────────────
   11. IMAGE MODEL FORM SUBMIT  (brain)
   
   FIXED:
   • Reads file from id="image-input" (name="file")
   • Sends as FormData to /predict_brain
   • Fully renders all result sections including chart
───────────────────────────────────────────── */
function initImageForm(endpoint, specialist) {
  const form = document.getElementById("predict-form");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const btn       = document.getElementById("submit-btn");
    const fileInput = document.getElementById("image-input");   // ★ fixed id

    if (!fileInput || !fileInput.files[0]) {
      showError("Please upload an MRI image before submitting.");
      return;
    }

    if (btn) btn.disabled = true;
    showState("loading-state");

    // FormData field name must be "file" — matches app.py request.files["file"]
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
      const res  = await fetch(endpoint, { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok || data.error) {
        showError(data.error || "Server error. Please try again.");
        if (btn) btn.disabled = false;
        return;
      }

      const isNoTumor = data.result === "No Tumor";
      const sub = isNoTumor
        ? "No abnormality detected. Maintain regular check-ups."
        : `Please consult a ${specialist} promptly.`;

      renderResult(data, sub);

    } catch (err) {
      showError("Network error. Is the Flask server running?");
    }

    if (btn) btn.disabled = false;
  });
}

/* ─────────────────────────────────────────────
   12. BOOTSTRAP
───────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  switch (PAGE_TYPE) {
    case "heart":
      initHeartForm();
      break;

    case "brain":
      initImageUpload();
      initImageForm("/predict_brain", "neurologist");
      break;

    default:
      console.warn(`[HealthcareAI] Unknown page type: "${PAGE_TYPE}". Expected heart | brain.`);
  }
});
