/* ============================================================
   main.js — Shared utilities used across all pages
   ============================================================ */

// ── Toast notification ────────────────────────────────────────
function showToast(msg, type = "info") {
  const t = document.getElementById("toast");
  if (!t) return;
  t.textContent = msg;
  t.className = `show ${type}`;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => { t.className = ""; }, 3200);
}

// ── Copy to clipboard ─────────────────────────────────────────
function copyText(text) {
  navigator.clipboard.writeText(text).then(() => {
    showToast("Copied to clipboard!", "success");
  }).catch(() => {
    showToast("Copy failed.", "error");
  });
}

// ── Word count live display ───────────────────────────────────
function initWordCount(textareaId, displayId) {
  const ta = document.getElementById(textareaId);
  const el = document.getElementById(displayId);
  if (!ta || !el) return;

  function update() {
    const words = ta.value.trim() ? ta.value.trim().split(/\s+/).length : 0;
    el.textContent = words > 0 ? `${words} words` : "";
    el.style.color = words >= 30 ? "var(--green)" : words > 0 ? "var(--orange)" : "";
  }

  ta.addEventListener("input", update);
  update();
}

// ── Slider live value display ─────────────────────────────────
function initSlider(sliderId, displayId) {
  const s = document.getElementById(sliderId);
  const d = document.getElementById(displayId);
  if (!s || !d) return;
  d.textContent = s.value;
  s.addEventListener("input", () => { d.textContent = s.value; });
}

// ── Tab switcher ──────────────────────────────────────────────
function initTabs(containerSelector) {
  const containers = document.querySelectorAll(containerSelector);
  containers.forEach(container => {
    const btns   = container.querySelectorAll(".tab-btn");
    const panels = container.querySelectorAll(".tab-panel");
    btns.forEach(btn => {
      btn.addEventListener("click", () => {
        btns.forEach(b => b.classList.remove("active"));
        panels.forEach(p => p.classList.remove("active"));
        btn.classList.add("active");
        const target = container.querySelector(`#${btn.dataset.tab}`);
        if (target) target.classList.add("active");
      });
    });
  });
}

// ── Active nav link ───────────────────────────────────────────
function setActiveNav() {
  const path = window.location.pathname;
  document.querySelectorAll(".nav-links a").forEach(a => {
    a.classList.remove("active");
    if (a.getAttribute("href") === path ||
       (path === "/" && a.getAttribute("href") === "/")) {
      a.classList.add("active");
    }
  });
}

// ── Poll model status ─────────────────────────────────────────
async function pollModelStatus(badgeId) {
  const badge = document.getElementById(badgeId);
  if (!badge) return;

  async function check() {
    try {
      const r = await fetch("/api/status");
      const d = await r.json();
      if (d.bart_loaded) {
        badge.classList.add("loaded");
        badge.querySelector(".dot").style.background = "var(--green)";
        badge.querySelector(".label").textContent    = "BART Ready";
        clearInterval(poller);
      }
    } catch (_) {}
  }

  const poller = setInterval(check, 3000);
  check();
}

// ── ROUGE chart (canvas) ──────────────────────────────────────
function drawRougeChart(canvasId, extScores, absScores) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const labels  = ["ROUGE-1", "ROUGE-2", "ROUGE-L"];
  const extVals = [extScores.rouge1.f1, extScores.rouge2.f1, extScores.rougeL.f1];
  const absVals = [absScores.rouge1.f1, absScores.rouge2.f1, absScores.rougeL.f1];

  // Clear
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const W = canvas.width, H = canvas.height;
  const pad = { top: 40, right: 20, bottom: 50, left: 50 };
  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top  - pad.bottom;
  const barW = chartW / (labels.length * 2 + labels.length + 1) * 0.9;
  const groupW = chartW / labels.length;

  // Background
  ctx.fillStyle = "#161b22";
  ctx.roundRect(0, 0, W, H, 12);
  ctx.fill();

  // Grid lines
  ctx.strokeStyle = "#30363d";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (chartH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(W - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = "#8b949e";
    ctx.font = "11px system-ui";
    ctx.textAlign = "right";
    ctx.fillText((1 - i / 4).toFixed(2), pad.left - 6, y + 4);
  }

  // Bars
  labels.forEach((label, i) => {
    const gx    = pad.left + i * groupW + groupW * 0.15;
    const extH  = extVals[i] * chartH;
    const absH  = absVals[i] * chartH;

    // Extractive bar (green)
    ctx.fillStyle = "#238636";
    ctx.beginPath();
    ctx.roundRect(gx, pad.top + chartH - extH, barW, extH, [4, 4, 0, 0]);
    ctx.fill();

    // Abstractive bar (purple)
    ctx.fillStyle = "#6e40c9";
    ctx.beginPath();
    ctx.roundRect(gx + barW + 6, pad.top + chartH - absH, barW, absH, [4, 4, 0, 0]);
    ctx.fill();

    // Values above bars
    ctx.font = "bold 11px system-ui";
    ctx.textAlign = "center";
    ctx.fillStyle = "#3fb950";
    ctx.fillText(extVals[i].toFixed(3), gx + barW / 2, pad.top + chartH - extH - 6);
    ctx.fillStyle = "#bc8cff";
    ctx.fillText(absVals[i].toFixed(3), gx + barW + 6 + barW / 2, pad.top + chartH - absH - 6);

    // X labels
    ctx.fillStyle = "#e6edf3";
    ctx.font = "12px system-ui";
    ctx.textAlign = "center";
    ctx.fillText(label, gx + barW + 3, H - 14);
  });

  // Legend
  const lx = pad.left + 4;
  const ly = 16;
  ctx.fillStyle = "#238636";
  ctx.fillRect(lx, ly - 9, 14, 10);
  ctx.fillStyle = "#e6edf3";
  ctx.font = "11px system-ui";
  ctx.textAlign = "left";
  ctx.fillText("Extractive", lx + 18, ly);

  ctx.fillStyle = "#6e40c9";
  ctx.fillRect(lx + 100, ly - 9, 14, 10);
  ctx.fillStyle = "#e6edf3";
  ctx.fillText("BART Abstractive", lx + 118, ly);
}

// ── Download helpers ──────────────────────────────────────────
function downloadText(filename, content) {
  const blob = new Blob([content], { type: "text/plain" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadCSV(filename, rows) {
  // rows = array of objects
  const headers = Object.keys(rows[0]);
  const lines = [
    headers.join(","),
    ...rows.map(r => headers.map(h => `"${String(r[h]).replace(/"/g, '""')}"`).join(","))
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ── On DOM ready ──────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  setActiveNav();
  initTabs(".tab-container");
  pollModelStatus("model-badge");
});
