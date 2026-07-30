/* =====================================================
   NeuroViz — AI Research Command Center
   Interactive JavaScript — Canvas Animations, Live Data
   ===================================================== */

'use strict';

// =====================================================
// 0. LOAD REAL AutoML DATA FROM dashboard_data.json
// =====================================================
const FI_COLORS = ['#7c3aed', '#06b6d4', '#ec4899', '#10b981', '#f59e0b'];
const RANK_CLASS = ['gold', 'silver', 'bronze'];
const RANK_LABEL = ['#1', '#2', '#3', '#4'];

async function loadRealData() {
  try {
    const res = await fetch('./dashboard_data.json?v=' + Date.now());
    if (!res.ok) throw new Error('Not found');
    const data = await res.json();
    renderRealData(data);
  } catch (e) {
    console.warn('[NeuroViz] dashboard_data.json not loaded:', e.message);
    document.getElementById('dataSource').textContent = 'Run export_dashboard_data.py to load real metrics';
  }
}

function renderRealData(d) {
  // Summary banner
  document.getElementById('rProblemType').textContent = d.problem_type || '—';
  document.getElementById('rTarget').textContent      = d.target        || '—';
  document.getElementById('rRows').textContent        = d.total_rows    || '—';
  document.getElementById('rFeatures').textContent    = d.total_features || '—';
  document.getElementById('rBestModel').textContent   = d.best_model    || '—';
  const acc = d.best_metrics && d.best_metrics.accuracy != null
    ? (d.best_metrics.accuracy * 100).toFixed(1) + '%'
    : '—';
  document.getElementById('rAccuracy').textContent = acc;

  const ts = d.generated_at ? new Date(d.generated_at).toLocaleString() : '';
  document.getElementById('dataSource').textContent =
    `Loaded from reports/automl_report.md + model.joblib  •  ${ts}`;

  // Leaderboard
  const lb = document.getElementById('leaderboardList');
  if (d.leaderboard && d.leaderboard.length) {
    lb.innerHTML = '';
    d.leaderboard.forEach((item, i) => {
      const rankCls  = RANK_CLASS[i] || '';
      const rankLbl  = RANK_LABEL[i] || `#${i+1}`;
      const metrics  = item.metrics || {};
      const chips = Object.entries(metrics)
        .slice(0, 3)
        .map(([k, v]) => `<span class="lb-metric-chip">${k}: ${(typeof v === 'number' ? v.toFixed(3) : v)}</span>`)
        .join('');
      lb.innerHTML += `
        <div class="lb-row">
          <span class="lb-rank ${rankCls}">${rankLbl}</span>
          <span class="lb-name">${item.model}</span>
          <div class="lb-metrics">${chips}</div>
        </div>`;
    });
  }

  // Feature Importance bars
  const fi = document.getElementById('fiBars');
  if (d.feature_drivers && d.feature_drivers.length) {
    fi.innerHTML = '';
    const maxImp = Math.max(...d.feature_drivers.map(x => x.importance));
    d.feature_drivers.forEach((driver, i) => {
      const color = FI_COLORS[i % FI_COLORS.length];
      const pct   = ((driver.importance / maxImp) * 100).toFixed(0);
      fi.innerHTML += `
        <div class="fi-row">
          <div class="fi-header">
            <span class="fi-feature">${driver.feature}</span>
            <span class="fi-pct">${driver.importance}%</span>
          </div>
          <div class="fi-bar-bg">
            <div class="fi-bar-fill" style="width:0%; background:${color}; box-shadow:0 0 6px ${color}44"
                 data-width="${pct}"></div>
          </div>
        </div>`;
    });
    // Animate bars in
    setTimeout(() => {
      document.querySelectorAll('.fi-bar-fill').forEach(el => {
        el.style.width = el.dataset.width + '%';
      });
    }, 300);
  }

  // Confusion Matrix
  if (d.confusion_matrix && d.class_labels) {
    drawConfusionMatrix('cmCanvas', d.confusion_matrix, d.class_labels);
    const labelsEl = document.getElementById('cmLabels');
    labelsEl.textContent = `Classes: ${d.class_labels.join(' / ')}`;
  }

  // Cleaning Log
  const cl = document.getElementById('cleaningList');
  if (d.cleaning_log && d.cleaning_log.length) {
    cl.innerHTML = '';
    d.cleaning_log.forEach(entry => {
      cl.innerHTML += `<div class="cleaning-entry">${entry}</div>`;
    });
  } else {
    cl.innerHTML = '<div class="lb-loading">No cleaning steps recorded.</div>';
  }
}

function drawConfusionMatrix(canvasId, matrix, labels) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const n   = matrix.length;
  const W   = canvas.offsetWidth || 300;
  const H   = 200;
  canvas.width = W; canvas.height = H;

  const cellW = (W - 60) / n;
  const cellH = (H - 40) / n;
  const offsetX = 50, offsetY = 20;

  // Find max for color scaling
  const flat = matrix.flat();
  const maxVal = Math.max(...flat) || 1;

  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const val   = matrix[r][c];
      const ratio = val / maxVal;
      const isCorrect = r === c;
      const alpha = 0.15 + ratio * 0.7;
      const color = isCorrect ? `rgba(16,185,129,${alpha})` : `rgba(239,68,68,${alpha})`;

      ctx.fillStyle = color;
      ctx.fillRect(offsetX + c * cellW, offsetY + r * cellH, cellW - 2, cellH - 2);

      ctx.fillStyle = '#fff';
      ctx.font = `bold ${Math.min(cellH * 0.4, 16)}px JetBrains Mono`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(val, offsetX + c * cellW + cellW / 2, offsetY + r * cellH + cellH / 2);
    }
  }

  // Labels
  ctx.font = '11px JetBrains Mono';
  ctx.fillStyle = 'rgba(148,163,184,0.7)';
  ctx.textAlign = 'center';
  labels.forEach((lbl, i) => {
    ctx.fillText(lbl, offsetX + i * cellW + cellW / 2, offsetY - 5);
  });

  ctx.textAlign = 'right';
  labels.forEach((lbl, i) => {
    ctx.fillText(lbl, offsetX - 4, offsetY + i * cellH + cellH / 2);
  });
}

window.addEventListener('load', loadRealData);

// =====================================================
// 1. CURSOR GLOW EFFECT
// =====================================================
const cursorGlow = document.getElementById('cursorGlow');
document.addEventListener('mousemove', (e) => {
  cursorGlow.style.left = e.clientX + 'px';
  cursorGlow.style.top = e.clientY + 'px';
});

// =====================================================
// 2. NEURAL NETWORK BACKGROUND CANVAS
// =====================================================
(function initNeuralBackground() {
  const canvas = document.getElementById('neuralCanvas');
  const ctx = canvas.getContext('2d');
  let W, H, nodes, animId;

  function resize() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function createNodes(count) {
    return Array.from({ length: count }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      r: Math.random() * 2 + 1,
      color: Math.random() > 0.5 ? '#7c3aed' : '#06b6d4',
    }));
  }

  function drawFrame() {
    ctx.clearRect(0, 0, W, H);

    // Update positions
    for (const n of nodes) {
      n.x += n.vx;
      n.y += n.vy;
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;
    }

    // Draw edges
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 150) {
          const alpha = (1 - dist / 150) * 0.12;
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.strokeStyle = `rgba(124, 58, 237, ${alpha})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }

    // Draw nodes
    for (const n of nodes) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = n.color + '55';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r * 0.5, 0, Math.PI * 2);
      ctx.fillStyle = n.color;
      ctx.fill();
    }

    animId = requestAnimationFrame(drawFrame);
  }

  resize();
  nodes = createNodes(60);
  drawFrame();
  window.addEventListener('resize', () => { resize(); nodes = createNodes(60); });
})();

// =====================================================
// 3. ANIMATED COUNTER for Hero Stats
// =====================================================
function animateCount(el, target, duration = 1800, decimal = 0, suffix = '') {
  const start = performance.now();
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = target * eased;
    el.textContent = decimal > 0
      ? value.toFixed(decimal) + suffix
      : Math.floor(value) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// Trigger counters with IntersectionObserver
const statNums = document.querySelectorAll('.stat-num');
const statsObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      const el = e.target;
      const count = parseFloat(el.dataset.count);
      const decimal = parseInt(el.dataset.decimal || '0');
      const suffix = el.dataset.suffix || '';
      animateCount(el, count, 1800, decimal, suffix);
      statsObserver.unobserve(el);
    }
  });
}, { threshold: 0.5 });

statNums.forEach(n => statsObserver.observe(n));

// =====================================================
// 4. TRAINING LOSS CHART (Canvas Sparkline)
// =====================================================
function createSparklineChart(canvasId, color, initialData, isDescending) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');
  let data = [...initialData];

  function draw() {
    const W = canvas.offsetWidth || 400;
    const H = canvas.offsetHeight || 120;
    canvas.width = W;
    canvas.height = H;
    ctx.clearRect(0, 0, W, H);

    const pad = 12;
    const min = Math.min(...data) * 0.95;
    const max = Math.max(...data) * 1.05;
    const range = max - min || 1;

    const toX = (i) => pad + (i / (data.length - 1)) * (W - pad * 2);
    const toY = (v) => H - pad - ((v - min) / range) * (H - pad * 2);

    // Grid lines
    for (let i = 0; i < 4; i++) {
      const y = pad + (i / 3) * (H - pad * 2);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.strokeStyle = 'rgba(255,255,255,0.04)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Gradient fill
    const grad = ctx.createLinearGradient(0, pad, 0, H);
    grad.addColorStop(0, color + '40');
    grad.addColorStop(1, color + '00');

    ctx.beginPath();
    ctx.moveTo(toX(0), toY(data[0]));
    for (let i = 1; i < data.length; i++) {
      const cpx = (toX(i - 1) + toX(i)) / 2;
      ctx.bezierCurveTo(cpx, toY(data[i - 1]), cpx, toY(data[i]), toX(i), toY(data[i]));
    }
    ctx.lineTo(toX(data.length - 1), H);
    ctx.lineTo(toX(0), H);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(data[0]));
    for (let i = 1; i < data.length; i++) {
      const cpx = (toX(i - 1) + toX(i)) / 2;
      ctx.bezierCurveTo(cpx, toY(data[i - 1]), cpx, toY(data[i]), toX(i), toY(data[i]));
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Current point glow
    const lx = toX(data.length - 1);
    const ly = toY(data[data.length - 1]);
    const glowGrad = ctx.createRadialGradient(lx, ly, 0, lx, ly, 10);
    glowGrad.addColorStop(0, color + 'cc');
    glowGrad.addColorStop(1, color + '00');
    ctx.beginPath();
    ctx.arc(lx, ly, 10, 0, Math.PI * 2);
    ctx.fillStyle = glowGrad;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(lx, ly, 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(lx, ly, 2, 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();
  }

  draw();

  return {
    update(newVal) {
      data.push(newVal);
      if (data.length > 40) data.shift();
      draw();
    }
  };
}

// Generate initial smooth data
function smoothData(points, isDesc) {
  const data = [];
  let v = isDesc ? 0.6 : 0.5;
  for (let i = 0; i < points; i++) {
    v += isDesc ? -(Math.random() * 0.015) : (Math.random() * 0.012);
    v = Math.max(0.03, Math.min(1, v));
    data.push(v);
  }
  return data;
}

let lossChart, accChart;

window.addEventListener('load', () => {
  const lossData = smoothData(30, true);
  const accData = smoothData(30, false);

  lossChart = createSparklineChart('lossChart', '#7c3aed', lossData, true);
  accChart = createSparklineChart('accChart', '#06b6d4', accData, false);

  // Draw gauges
  drawGauge('gpuGauge', 0.87, '#06b6d4', '#7c3aed');
  drawGauge('vramGauge', 0.855, '#ec4899', '#7c3aed');
});

// =====================================================
// 5. GAUGE CHARTS
// =====================================================
function drawGauge(canvasId, value, color1, color2) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = 140; const H = 80;
  canvas.width = W; canvas.height = H;

  const cx = W / 2;
  const cy = H - 10;
  const r = 60;
  const startAngle = Math.PI;
  const endAngle = Math.PI * 2;

  // Background arc
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, endAngle);
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 10;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Value arc with gradient
  const grad = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
  grad.addColorStop(0, color1);
  grad.addColorStop(1, color2);

  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, startAngle + value * Math.PI);
  ctx.strokeStyle = grad;
  ctx.lineWidth = 10;
  ctx.lineCap = 'round';
  ctx.shadowColor = color1;
  ctx.shadowBlur = 12;
  ctx.stroke();
  ctx.shadowBlur = 0;
}

// =====================================================
// 6. LIVE SIMULATION - Tick Updates
// =====================================================
let simActive = false;
let simInterval = null;
let lossVal = 0.043;
let accVal = 96.2;
let gpuVal = 87;
let epochCount = 89;

function tick() {
  // Update loss
  lossVal = Math.max(0.01, lossVal + (Math.random() - 0.52) * 0.003);
  document.getElementById('lossValue').textContent = lossVal.toFixed(4);
  if (lossChart) lossChart.update(lossVal);

  // Update accuracy
  accVal = Math.min(99.9, Math.max(80, accVal + (Math.random() - 0.45) * 0.15));
  document.getElementById('accValue').textContent = accVal.toFixed(1) + '%';
  if (accChart) accChart.update(accVal / 100);

  // Update GPU
  gpuVal = Math.max(60, Math.min(99, gpuVal + (Math.random() - 0.5) * 5));
  document.getElementById('gpuValue').textContent = Math.round(gpuVal) + '%';
  drawGauge('gpuGauge', gpuVal / 100, '#06b6d4', '#7c3aed');

  // Update VRAM
  const vram = 34.2 + (Math.random() - 0.5) * 0.8;
  document.getElementById('vramValue').textContent = vram.toFixed(1) + ' GB';
  drawGauge('vramGauge', vram / 40, '#ec4899', '#7c3aed');

  // Occasionally add a log entry
  if (Math.random() < 0.3) addLogEntry();
}

function addLogEntry() {
  const log = document.getElementById('activityLog');
  const now = new Date();
  const time = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}`;

  const types = [
    { cls: 'info', badge: 'info-badge', label: 'INFO', msgs: [
      `AutoML Agent — Step ${Math.floor(Math.random()*5000+1000)} | Loss: ${lossVal.toFixed(4)}`,
      `Checkpoint saved to /checkpoints/epoch_${epochCount}.pt`,
      `Learning rate adjusted to ${(Math.random()*1e-3).toFixed(2)}e-3`,
    ]},
    { cls: 'success', badge: 'success-badge', label: 'SUCCESS', msgs: [
      `Validation accuracy improved to ${accVal.toFixed(1)}%`,
      `Model ensemble achieved new best F1: ${(0.88 + Math.random()*0.1).toFixed(3)}`,
    ]},
    { cls: 'warn', badge: 'warn-badge', label: 'WARN', msgs: [
      `High gradient norm detected: ${(Math.random()*5+3).toFixed(2)}`,
      `VRAM usage above 85% threshold`,
    ]},
  ];

  const type = types[Math.floor(Math.random() * types.length)];
  const msg = type.msgs[Math.floor(Math.random() * type.msgs.length)];

  const entry = document.createElement('div');
  entry.className = `log-entry ${type.cls}`;
  entry.innerHTML = `
    <span class="log-time">${time}</span>
    <span class="log-badge ${type.badge}">${type.label}</span>
    <span class="log-msg">${msg}</span>
  `;

  log.insertBefore(entry, log.firstChild);

  // Keep max 20 entries
  while (log.children.length > 20) {
    log.removeChild(log.lastChild);
  }
}

document.getElementById('startSimulation').addEventListener('click', function () {
  simActive = !simActive;
  if (simActive) {
    this.textContent = '⏸ Pause Simulation';
    this.style.background = 'linear-gradient(135deg, #059669, #10b981)';
    simInterval = setInterval(tick, 800);
  } else {
    this.innerHTML = `<svg viewBox="0 0 24 24" fill="none" width="16" height="16"><polygon points="5,3 19,12 5,21" fill="white"/></svg> Run Simulation`;
    this.style.background = '';
    clearInterval(simInterval);
  }
});

document.getElementById('clearLog').addEventListener('click', () => {
  document.getElementById('activityLog').innerHTML = '';
});

// =====================================================
// 7. NEURAL NETWORK ARCHITECTURE VISUALIZER
// =====================================================
const architectures = {
  automl: {
    layers: [
      { name: 'Input', nodes: 4, type: 'input' },
      { name: 'Encoder', nodes: 8, type: 'hidden' },
      { name: 'Hidden', nodes: 12, type: 'hidden' },
      { name: 'Attention', nodes: 6, type: 'attention' },
      { name: 'Hidden', nodes: 8, type: 'hidden' },
      { name: 'Policy', nodes: 4, type: 'output' },
    ]
  },
  vqa: {
    layers: [
      { name: 'Image', nodes: 5, type: 'input' },
      { name: 'ViT', nodes: 9, type: 'hidden' },
      { name: 'Cross-Attn', nodes: 7, type: 'attention' },
      { name: 'Text', nodes: 5, type: 'input' },
      { name: 'BERT', nodes: 9, type: 'hidden' },
      { name: 'Fusion', nodes: 6, type: 'hidden' },
      { name: 'Answer', nodes: 3, type: 'output' },
    ]
  },
  medical: {
    layers: [
      { name: 'DICOM', nodes: 4, type: 'input' },
      { name: 'CNN', nodes: 8, type: 'hidden' },
      { name: 'Self-Attn', nodes: 7, type: 'attention' },
      { name: 'Cross-Attn', nodes: 6, type: 'attention' },
      { name: 'Decoder', nodes: 8, type: 'hidden' },
      { name: 'LLM', nodes: 5, type: 'hidden' },
      { name: 'Report', nodes: 3, type: 'output' },
    ]
  }
};

let nnAnimId = null;
let nnPulse = 0;

function getNodeColor(type) {
  return {
    input: '#06b6d4',
    hidden: '#7c3aed',
    attention: '#f59e0b',
    output: '#10b981',
  }[type] || '#7c3aed';
}

function drawNNViz(archKey) {
  const canvas = document.getElementById('nnViz');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const W = canvas.offsetWidth || 800;
  const H = canvas.offsetHeight || 300;
  canvas.width = W;
  canvas.height = H;

  if (nnAnimId) cancelAnimationFrame(nnAnimId);

  const arch = architectures[archKey] || architectures.automl;
  const layers = arch.layers;
  const layerW = W / (layers.length + 1);
  const maxNodes = Math.max(...layers.map(l => l.nodes));
  const nodeR = Math.min(12, H / (maxNodes * 2.5));

  // Precompute positions
  const positions = layers.map((layer, li) => {
    const x = layerW * (li + 1);
    return Array.from({ length: layer.nodes }, (_, ni) => {
      const spacing = H / (layer.nodes + 1);
      const y = spacing * (ni + 1);
      return { x, y, type: layer.type, name: layer.name };
    });
  });

  function drawFrame() {
    ctx.clearRect(0, 0, W, H);
    nnPulse += 0.03;

    // Draw edges with signal animation
    for (let li = 0; li < positions.length - 1; li++) {
      for (const from of positions[li]) {
        for (const to of positions[li + 1]) {
          // Skip if too many connections (VQA has cross-layer)
          if (Math.random() > 0.5 && layers[li].nodes > 6) continue;

          const pulseFactor = (Math.sin(nnPulse + from.x * 0.01 + from.y * 0.01) + 1) / 2;
          const alpha = 0.03 + pulseFactor * 0.08;

          ctx.beginPath();
          ctx.moveTo(from.x, from.y);
          ctx.lineTo(to.x, to.y);
          ctx.strokeStyle = `rgba(124,58,237,${alpha})`;
          ctx.lineWidth = 0.8;
          ctx.stroke();

          // Animated signal packet
          if (Math.random() < 0.002) {
            const t = (nnPulse * 0.3) % 1;
            const px = from.x + (to.x - from.x) * t;
            const py = from.y + (to.y - from.y) * t;
            ctx.beginPath();
            ctx.arc(px, py, 2, 0, Math.PI * 2);
            ctx.fillStyle = getNodeColor(from.type);
            ctx.fill();
          }
        }
      }
    }

    // Draw nodes
    for (const layer of positions) {
      for (const node of layer) {
        const pulse = (Math.sin(nnPulse + node.x * 0.02 + node.y * 0.02) + 1) / 2;
        const color = getNodeColor(node.type);

        // Outer glow
        const grd = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, nodeR * 2.5);
        grd.addColorStop(0, color + '30');
        grd.addColorStop(1, color + '00');
        ctx.beginPath();
        ctx.arc(node.x, node.y, nodeR * 2.5, 0, Math.PI * 2);
        ctx.fillStyle = grd;
        ctx.fill();

        // Node circle
        ctx.beginPath();
        ctx.arc(node.x, node.y, nodeR + pulse * 2, 0, Math.PI * 2);
        ctx.fillStyle = color + '30';
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.fill();
        ctx.stroke();

        // Inner bright dot
        ctx.beginPath();
        ctx.arc(node.x, node.y, nodeR * 0.4, 0, Math.PI * 2);
        ctx.fillStyle = '#fff';
        ctx.fill();
      }
    }

    // Layer labels
    ctx.font = '11px JetBrains Mono, monospace';
    for (let li = 0; li < layers.length; li++) {
      const x = layerW * (li + 1);
      ctx.fillStyle = 'rgba(148,163,184,0.6)';
      ctx.textAlign = 'center';
      ctx.fillText(layers[li].name, x, H - 6);
    }

    nnAnimId = requestAnimationFrame(drawFrame);
  }

  drawFrame();
}

// Init visualizer
window.addEventListener('load', () => {
  setTimeout(() => drawNNViz('automl'), 200);
});

document.getElementById('archSelect').addEventListener('change', function () {
  drawNNViz(this.value);
});

window.addEventListener('resize', () => {
  const sel = document.getElementById('archSelect');
  drawNNViz(sel.value);
});

// =====================================================
// 8. PROGRESS BAR ANIMATION on Scroll
// =====================================================
const progressFills = document.querySelectorAll('.progress-fill');
const progressObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      const el = e.target;
      const targetWidth = el.dataset.width + '%';
      setTimeout(() => { el.style.width = targetWidth; }, 200);
      progressObserver.unobserve(el);
    }
  });
}, { threshold: 0.3 });

progressFills.forEach(el => progressObserver.observe(el));

// =====================================================
// 9. NAV LINK HIGHLIGHT on Scroll
// =====================================================
const navLinks = document.querySelectorAll('.nav-link');
const sections = ['section-overview', 'section-metrics', 'section-projects', 'section-visualizer'];

const navObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      navLinks.forEach(a => a.classList.remove('active'));
      const id = e.target.id;
      const link = document.querySelector(`.nav-link[href="#${id}"]`);
      if (link) link.classList.add('active');
    }
  });
}, { threshold: 0.4 });

sections.forEach(id => {
  const el = document.getElementById(id);
  if (el) navObserver.observe(el);
});

// =====================================================
// 10. NAV RANGE TOGGLE
// =====================================================
document.querySelectorAll('.ctrl-btn').forEach(btn => {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.ctrl-btn').forEach(b => b.classList.remove('active'));
    this.classList.add('active');
  });
});

// =====================================================
// 11. PROJECT CARD TILT EFFECT
// =====================================================
document.querySelectorAll('.project-card').forEach(card => {
  card.addEventListener('mousemove', (e) => {
    const rect = card.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width - 0.5) * 2;
    const y = ((e.clientY - rect.top) / rect.height - 0.5) * 2;
    card.style.transform = `perspective(1000px) rotateX(${-y * 4}deg) rotateY(${x * 4}deg) translateY(-6px)`;
  });
  card.addEventListener('mouseleave', () => {
    card.style.transform = '';
  });
});

// =====================================================
// 12. START AUTO SIMULATION AFTER 2s
// =====================================================
setTimeout(() => {
  document.getElementById('startSimulation').click();
}, 2000);

console.log('%c⚡ NeuroViz AI Dashboard Loaded', 'color: #a855f7; font-size: 16px; font-weight: bold;');
console.log('%cBuilt with Pure Canvas API — Zero Dependencies', 'color: #06b6d4; font-size: 12px;');
