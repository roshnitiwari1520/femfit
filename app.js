/* =========================================================
   app.js — FemFit Frontend Logic
   Handles form → API → results rendering
   ========================================================= */

// ── Config ──────────────────────────────────────────────
const API_BASE = '';  // empty = relative to current origin (served via FastAPI)

// ── Utility: animate a number from 0 to target ──────────
function animateNumber(el, target, decimals = 0, duration = 900) {
  const start = performance.now();
  const from = 0;
  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased    = 1 - Math.pow(1 - progress, 3);
    const current  = from + (target - from) * eased;
    el.textContent = current.toFixed(decimals);
    if (progress < 1) requestAnimationFrame(tick);
    else el.textContent = target.toFixed(decimals);
  }
  requestAnimationFrame(tick);
}

// ── Toast notification ───────────────────────────────────
function showToast(message) {
  let toast = document.getElementById('toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'toast';
    toast.className = 'toast';
    toast.innerHTML = '<span>⚠️</span><span id="toast-msg"></span>';
    document.body.appendChild(toast);
  }
  document.getElementById('toast-msg').textContent = message;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 4500);
}

// ── HR Zone colour mapping ────────────────────────────────
const ZONE_DATA = {
  'Rest':      { color: '#5c8af7', pct: 18 },
  'Fat Burn':  { color: '#3de0c8', pct: 36 },
  'Cardio':    { color: '#ffb830', pct: 60 },
  'Peak':      { color: '#ff4fa0', pct: 80 },
  'Maximum':   { color: '#c84bff', pct: 100 },
};

function zoneBadgeHtml(zone, isFemfit = false) {
  const z = ZONE_DATA[zone] || { color: '#8b82b0', pct: 50 };
  return `<span class="zone-badge" style="color:${z.color}; border-color:${z.color}40; background:${z.color}18">
    ${isFemfit ? '💜 ' : '⌚ '} ${zone}
  </span>`;
}

// ── cycle phase display ───────────────────────────────────
const PHASE_META = {
  follicular: { icon: '🌸', color: '#e040fb', label: 'Follicular Phase' },
  ovulatory:  { icon: '✨', color: '#ffb830', label: 'Ovulatory Phase'  },
  luteal:     { icon: '🌙', color: '#9b59f7', label: 'Luteal Phase'     },
  menstrual:  { icon: '🔴', color: '#ff4fa0', label: 'Menstrual Phase'  },
};

const FITNESS_META = {
  high:     { icon: '🏆', label: 'High Fitness' },
  moderate: { icon: '⚡', label: 'Moderate Fitness' },
  low:      { icon: '🌱', label: 'Low Fitness' },
};

// ── Save/load result from sessionStorage ─────────────────
function saveResult(data) {
  sessionStorage.setItem('femfit_result', JSON.stringify(data));
}

function loadResult() {
  const raw = sessionStorage.getItem('femfit_result');
  return raw ? JSON.parse(raw) : null;
}

// ═══════════════════════════════════════════════════════════
//  PAGE 1 — Form Logic
// ═══════════════════════════════════════════════════════════
function initFormPage() {
  const form    = document.getElementById('femfit-form');
  if (!form) return;

  // Range slider: cycle length
  const cycleRange = document.getElementById('cycle_length');
  const cycleVal   = document.getElementById('cycle_val');
  if (cycleRange && cycleVal) {
    function updateRange() {
      const min = +cycleRange.min, max = +cycleRange.max, val = +cycleRange.value;
      const pct = ((val - min) / (max - min)) * 100;
      cycleRange.style.setProperty('--pct', pct + '%');
      cycleVal.textContent = val;
    }
    cycleRange.addEventListener('input', updateRange);
    updateRange();
  }

  // Default date to today
  const dateInput = document.getElementById('last_period_date');
  if (dateInput && !dateInput.value) {
    const today = new Date();
    dateInput.value = today.toISOString().split('T')[0];
  }

  // Form submit
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('submit-btn');

    const payload = {
      age:               parseInt(document.getElementById('age').value),
      weight_kg:         parseFloat(document.getElementById('weight_kg').value),
      height_cm:         parseFloat(document.getElementById('height_cm').value),
      resting_hr:        parseFloat(document.getElementById('resting_hr').value),
      hemoglobin_g_dl:   parseFloat(document.getElementById('hemoglobin_g_dl').value),
      last_period_date:  document.getElementById('last_period_date').value,
      cycle_length:      parseInt(document.getElementById('cycle_length').value),
    };

    // Basic validation
    for (const [k, v] of Object.entries(payload)) {
      if (v === null || v === undefined || (typeof v === 'number' && isNaN(v)) || (typeof v === 'string' && !v)) {
        showToast(`Please fill in: ${k.replace(/_/g, ' ')}`);
        return;
      }
    }

    // Loading state
    btn.classList.add('loading');
    btn.disabled = true;

    try {
      const res  = await fetch(`${API_BASE}/calculate`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `API error ${res.status}`);
      }

      const data = await res.json();
      saveResult(data);

      // Animate out and navigate
      document.querySelector('.page-wrapper').style.opacity = '0';
      document.querySelector('.page-wrapper').style.transition = 'opacity 0.3s ease';
      setTimeout(() => { window.location.href = 'results.html'; }, 300);

    } catch (err) {
      showToast(err.message || 'Could not connect to FemFit API. Is the server running?');
      btn.classList.remove('loading');
      btn.disabled = false;
    }
  });
}

// ═══════════════════════════════════════════════════════════
//  PAGE 2 — Results Logic
// ═══════════════════════════════════════════════════════════
function initResultsPage() {
  const container = document.getElementById('results-root');
  if (!container) return;

  const data = loadResult();
  if (!data) {
    container.innerHTML = `
      <div style="text-align:center; padding: 4rem; color: var(--clr-text-muted);">
        <p style="font-size:3rem; margin-bottom:1rem;">🔍</p>
        <p>No results found. Please fill in the form first.</p>
        <a href="index.html" class="btn-back" style="margin-top:1.5rem; display:inline-flex;">← Back to Form</a>
      </div>`;
    return;
  }

  const { wearable_says: W, femfit_says: F, insights, shap_explanation } = data;
  const phase   = PHASE_META[F.cycle_phase]   || PHASE_META.follicular;
  const fitness = FITNESS_META[F.fitness_label] || FITNESS_META.moderate;

  container.innerHTML = buildResultsHTML(W, F, phase, fitness, insights, shap_explanation);

  // Animate numbers after DOM insertion
  requestAnimationFrame(() => {
    animateMetric('w-bmr',      W.bmr,            0);
    animateMetric('f-bmr',      F.bmr,            0);
    animateMetric('w-cal',      W.calories_burned, 1);
    animateMetric('f-cal',      F.calories_burned, 1);
    animateMetric('w-maxhr',    W.max_hr,          0);
    animateMetric('f-maxhr',    F.effective_max_hr,0);
    animateMetric('f-vo2',      F.vo2,             1);
    animateMetric('f-vo2-lo',   F.vo2_lower,       1);
    animateMetric('f-vo2-hi',   F.vo2_upper,       1);

    // VO2 confidence band
    const vo2Min = 10, vo2Max = 60;
    const leftPct  = ((F.vo2_lower - vo2Min) / (vo2Max - vo2Min)) * 100;
    const widthPct = ((F.vo2_upper - F.vo2_lower) / (vo2Max - vo2Min)) * 100;
    const bar = document.getElementById('vo2-band');
    if (bar) {
      bar.style.setProperty('--left',  Math.max(0, leftPct)  + '%');
      bar.style.setProperty('--width', Math.min(100, widthPct) + '%');
    }

    // HR zone bars
    renderHRZoneBars(F.hr_zones);
  });

  // SHAP toggle
  const toggle = document.getElementById('shap-toggle');
  const body   = document.getElementById('shap-body');
  if (toggle && body) {
    toggle.addEventListener('click', () => {
      toggle.classList.toggle('open');
      body.classList.toggle('open');
    });
  }
}

function animateMetric(id, value, decimals) {
  const el = document.getElementById(id);
  if (el) animateNumber(el, value, decimals, 1000);
}

function buildResultsHTML(W, F, phase, fitness, insights, shap) {

  const bmrDelta  = F.bmr_delta;
  const calDelta  = F.calories_delta;
  const vo2Delta  = F.vo2_delta;

  function deltaHtml(val, unitLabel) {
    if (Math.abs(val) < 0.5) return '';
    const cls  = val < 0 ? 'negative' : 'positive';
    const sign = val < 0 ? '↓' : '↑';
    return `<span class="metric-delta ${cls}">${sign} ${Math.abs(val).toFixed(1)} ${unitLabel}</span>`;
  }

  const insightCards = insights.map((ins, i) => `
    <div class="insight-card delay-${Math.min(i + 1, 6)}" style="animation: fade-in-up 0.5s var(--ease-smooth) ${0.05 * i}s both">
      <div class="insight-category">
        <div class="insight-category-dot"></div>
        <span class="insight-category-name">${ins.category}</span>
      </div>
      <p class="insight-message">${ins.message}</p>
    </div>
  `).join('');

  const shapItems = shap.map(s => `<div class="shap-item">${s}</div>`).join('');

  return `
    <a href="index.html" class="btn-back">← Back to Form</a>

    <!-- Phase Banner -->
    <div class="phase-banner">
      <div class="phase-info">
        <div class="phase-icon">${phase.icon}</div>
        <div>
          <div class="phase-label">Current Cycle Phase</div>
          <div class="phase-name" style="color:${phase.color}">${F.cycle_phase}</div>
        </div>
      </div>
      <div class="phase-meta">
        <div class="meta-chip">
          ${fitness.icon}&nbsp; Fitness: <span class="chip-val">&nbsp;${F.fitness_label}</span>
        </div>
        <div class="meta-chip">
          ❤️&nbsp; VO2 Max: <span class="chip-val">&nbsp;${F.vo2.toFixed(1)} ml/kg/min</span>
        </div>
        <div class="meta-chip">
          📉&nbsp; Calorie bias: <span class="chip-val">&nbsp;${F.calories_bias_pct.toFixed(1)}%</span>
        </div>
      </div>
    </div>

    <!-- Column Headers -->
    <p class="comparison-section-title">Side-by-Side Comparison</p>
    <div class="comparison-header-row">
      <div class="column-header standard">
        <span class="col-icon">⌚</span>
        Standard Wearable
      </div>
      <div class="column-header femfit">
        <span class="col-icon">💜</span>
        FemFit (Female-Calibrated)
      </div>
    </div>

    <!-- Metric Grid -->
    <div class="comparison-grid">

      <!-- BMR -->
      <div class="metric-card delay-1">
        <div class="metric-label">Basal Metabolic Rate</div>
        <div class="metric-value"><span id="w-bmr">—</span></div>
        <div class="metric-unit">kcal / day</div>
        <div style="margin-top:6px; font-size:0.78rem; color:var(--clr-text-faint)">Harris-Benedict (1919)</div>
      </div>
      <div class="metric-card femfit-card delay-2">
        <div class="metric-label">Basal Metabolic Rate</div>
        <div class="metric-value femfit-value"><span id="f-bmr">—</span></div>
        <div class="metric-unit">kcal / day</div>
        ${deltaHtml(bmrDelta, 'kcal')}
      </div>

      <!-- Calories -->
      <div class="metric-card delay-3">
        <div class="metric-label">Calories Burned (30 min)</div>
        <div class="metric-value"><span id="w-cal">—</span></div>
        <div class="metric-unit">kcal</div>
      </div>
      <div class="metric-card femfit-card delay-4">
        <div class="metric-label">Calories Burned (30 min)</div>
        <div class="metric-value femfit-value"><span id="f-cal">—</span></div>
        <div class="metric-unit">kcal</div>
        ${deltaHtml(calDelta, 'kcal')}
      </div>

      <!-- Max HR -->
      <div class="metric-card delay-5">
        <div class="metric-label">Maximum Heart Rate</div>
        <div class="metric-value"><span id="w-maxhr">—</span></div>
        <div class="metric-unit">bpm &nbsp;·&nbsp; 220 − age formula</div>
      </div>
      <div class="metric-card femfit-card delay-6">
        <div class="metric-label">Effective Max HR (Phase-Adjusted)</div>
        <div class="metric-value femfit-value"><span id="f-maxhr">—</span></div>
        <div class="metric-unit">bpm &nbsp;·&nbsp; Gulati female formula</div>
      </div>

      <!-- HR Zone -->
      <div class="metric-card delay-1">
        <div class="metric-label">HR Zone @ 75% Effort</div>
        ${zoneBadgeHtml(W.hr_zone_at_75pct, false)}
      </div>
      <div class="metric-card femfit-card delay-2">
        <div class="metric-label">HR Zone @ 75% Effort (Corrected)</div>
        ${zoneBadgeHtml(F.hr_zone_at_75pct, true)}
      </div>

      <!-- VO2 Max (femfit only) -->
      <div class="metric-card delay-3" style="opacity:0.4; pointer-events:none;">
        <div class="metric-label">VO2 Max</div>
        <div class="metric-value" style="font-size:1.2rem; color:var(--clr-text-faint)">Not calculated</div>
        <div class="metric-unit">Standard wearables don't correct for female physiology</div>
      </div>
      <div class="metric-card femfit-card delay-4">
        <div class="metric-label">VO2 Max (Female-Calibrated)</div>
        <div class="metric-value femfit-value"><span id="f-vo2">—</span></div>
        <div class="metric-unit">ml/kg/min</div>
        <div class="vo2-range-wrap">
          <div style="font-size:0.72rem; color:var(--clr-text-faint); margin-top:6px">
            95% CI: <span id="f-vo2-lo">—</span> – <span id="f-vo2-hi">—</span> ml/kg/min
          </div>
          <div class="vo2-range-bar">
            <div class="vo2-range-fill" id="vo2-band"></div>
          </div>
          <div class="vo2-ci-label">Confidence interval wider in luteal/anemia conditions</div>
        </div>
        ${deltaHtml(vo2Delta, 'ml/kg/min')}
      </div>

    </div>

    <!-- HR Zones Breakdown -->
    <div class="hr-zones-wrap">
      <div class="hr-zones-title">❤️ FemFit Heart Rate Zones (Phase-Adjusted)</div>
      <div id="hr-zones-list"></div>
    </div>

    <div class="divider"></div>

    <!-- Insights -->
    ${insights.length > 0 ? `
    <div class="insights-section">
      <p class="comparison-section-title">Personalised Insights</p>
      <div class="insights-grid">${insightCards}</div>
    </div>` : ''}

    <!-- SHAP -->
    ${shap.length > 0 ? `
    <div class="shap-section">
      <button class="shap-toggle" id="shap-toggle">
        <span>🔎 Why did FemFit predict this calorie burn?</span>
        <span class="chevron">▾</span>
      </button>
      <div class="shap-body" id="shap-body">${shapItems}</div>
    </div>` : ''}
  `;
}

// ── Build HR zone rows from API hr_zones dict ─────────────
function renderHRZoneBars(zones) {
  const list = document.getElementById('hr-zones-list');
  if (!list || !zones) return;

  const order  = ['Fat Burn', 'Cardio', 'Peak', 'Maximum'];
  // API returns keys like "fat_burn": { low: 95, high: 114 }
  const keyMap = {
    'Fat Burn': 'fat_burn',
    'Cardio':   'cardio',
    'Peak':     'peak',
    'Maximum':  'maximum',
  };

  const allHighs = Object.values(zones).map(z => z.high || 0).filter(Boolean);
  const maxHr    = Math.max(...allHighs, 200);

  let html = '';
  for (const zoneName of order) {
    const z     = zones[keyMap[zoneName]];
    if (!z) continue;
    const meta  = ZONE_DATA[zoneName] || {};
    const fillPct = ((z.high / maxHr) * 100).toFixed(1);
    html += `
      <div class="hr-zone-row">
        <div class="hr-zone-dot" style="background:${meta.color || '#888'}"></div>
        <div class="hr-zone-name">${zoneName}</div>
        <div class="hr-zone-bar-wrap">
          <div class="hr-zone-bar-fill" style="width:0%; background:${meta.color || '#888'}" data-target="${fillPct}"></div>
        </div>
        <div class="hr-zone-range">${z.low}–${z.high} <span style="color:var(--clr-text-faint)">bpm</span></div>
      </div>`;
  }
  list.innerHTML = html;

  // Animate bars
  requestAnimationFrame(() => {
    list.querySelectorAll('.hr-zone-bar-fill').forEach(bar => {
      const target = bar.dataset.target;
      bar.style.transition = 'width 1.1s cubic-bezier(0.4,0,0.2,1)';
      requestAnimationFrame(() => { bar.style.width = target + '%'; });
    });
  });
}

// ── Boot ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initFormPage();
  initResultsPage();
});
