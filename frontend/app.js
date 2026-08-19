// ---------------------------------------------------------------------------
// Cupping Lab frontend
// Talks to the from-scratch decision tree API (see backend/app/).
// No framework, no build step, just fetch() and a hand-rolled SVG tree layout.
// ---------------------------------------------------------------------------

const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? "http://127.0.0.1:8000"
  : ""; // if you deploy the backend elsewhere, hardcode its URL here

const CLASS_COLORS = {
  Washed: "#4fb2c9",
  Natural: "#c1432f",
  Honey: "#e0a339",
  "Semi-Washed": "#8a9a4b",
};

const FEATURE_LABELS = {
  aroma: "Aroma", flavor: "Flavor", aftertaste: "Aftertaste", acidity: "Acidity",
  body: "Body", balance: "Balance", uniformity: "Uniformity", clean_cup: "Clean Cup",
  sweetness: "Sweetness", moisture: "Moisture %", defects: "Category 2 defects",
  altitude_m: "Altitude (m)", country: "Country",
};

const PRESETS = [
  {
    label: "Classic Ethiopian washed",
    values: { aroma: 8.3, flavor: 8.3, aftertaste: 8.0, acidity: 8.3, body: 7.7, balance: 8.0,
      uniformity: 10, clean_cup: 10, sweetness: 10, moisture: 0.11, defects: 0, altitude_m: 2050, country: "Ethiopia" },
  },
  {
    label: "Brazilian natural, big body",
    values: { aroma: 7.7, flavor: 7.6, aftertaste: 7.4, acidity: 7.3, body: 7.9, balance: 7.5,
      uniformity: 10, clean_cup: 10, sweetness: 9.3, moisture: 0.10, defects: 2, altitude_m: 1100, country: "Brazil" },
  },
  {
    label: "Central American honey",
    values: { aroma: 7.9, flavor: 7.9, aftertaste: 7.6, acidity: 7.8, body: 7.8, balance: 7.7,
      uniformity: 10, clean_cup: 10, sweetness: 10, moisture: 0.12, defects: 0, altitude_m: 1650, country: "Costa Rica" },
  },
  {
    label: "Borderline / hard case",
    values: { aroma: 7.4, flavor: 7.4, aftertaste: 7.2, acidity: 7.3, body: 7.4, balance: 7.3,
      uniformity: 10, clean_cup: 9.3, sweetness: 9.3, moisture: 0.11, defects: 4, altitude_m: 1400, country: "Guatemala" },
  },
];

let datasetSummary = null;
let currentTree = null;   // full response from /api/train
let nodePositions = {};   // node_id -> {x, y, w, h}

// ---------------------------------------------------------------------------
// Bootstrapping
// ---------------------------------------------------------------------------

async function init() {
  wireControls();
  try {
    datasetSummary = await fetchJSON("/api/dataset/summary");
    renderDatasetStat();
    renderLegend();
    buildCupForm();
    buildPresets();
  } catch (e) {
    document.getElementById("datasetStat").innerHTML =
      `<span style="color:#c1432f">Can't reach the API at ${API_BASE || "(same origin)"}, is uvicorn running?</span>`;
    console.error(e);
  }
}

async function fetchJSON(path, options) {
  const res = await fetch(API_BASE + path, options);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json();
}

function renderDatasetStat() {
  const s = datasetSummary;
  const counts = Object.entries(s.class_counts).map(([k, v]) => `${k} <b>${v}</b>`).join(" &nbsp;·&nbsp; ");
  document.getElementById("datasetStat").innerHTML =
    `${s.n_samples} graded lots &nbsp;·&nbsp; ${counts}`;
}

function renderLegend() {
  const el = document.getElementById("legend");
  el.innerHTML = Object.entries(CLASS_COLORS).map(([label, color]) => `
    <span class="legend-chip"><span class="legend-dot" style="background:${color}"></span>${label}</span>
  `).join("");
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

function wireControls() {
  const depth = document.getElementById("maxDepth");
  const split = document.getElementById("minSplit");
  const leaf = document.getElementById("minLeaf");
  const gain = document.getElementById("minGain");

  depth.addEventListener("input", () => document.getElementById("maxDepthVal").textContent = depth.value);
  split.addEventListener("input", () => document.getElementById("minSplitVal").textContent = split.value);
  leaf.addEventListener("input", () => document.getElementById("minLeafVal").textContent = leaf.value);
  gain.addEventListener("input", () => document.getElementById("minGainVal").textContent = Number(gain.value).toFixed(3));

  document.querySelectorAll("#criterionToggle .seg-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#criterionToggle .seg-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
    });
  });

  document.getElementById("trainBtn").addEventListener("click", trainTree);
  document.getElementById("cupForm").addEventListener("submit", (e) => {
    e.preventDefault();
    predictSample();
  });
}

function currentParams() {
  return {
    criterion: document.querySelector("#criterionToggle .seg-btn.active").dataset.value,
    max_depth: Number(document.getElementById("maxDepth").value),
    min_samples_split: Number(document.getElementById("minSplit").value),
    min_samples_leaf: Number(document.getElementById("minLeaf").value),
    min_impurity_decrease: Number(document.getElementById("minGain").value),
  };
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

async function trainTree() {
  const btn = document.getElementById("trainBtn");
  const status = document.getElementById("trainStatus");
  btn.disabled = true;
  status.textContent = "growing the tree from the training split…";
  try {
    const result = await fetchJSON("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentParams()),
    });
    currentTree = result;
    status.textContent = `trained on ${result.train_metrics.n} lots, tested on ${result.test_metrics.n}`;
    renderMetrics(result);
    renderImportance(result);
    renderTree(result.tree);
    document.getElementById("prediction").hidden = true;
  } catch (e) {
    status.textContent = "training failed, check the console";
    console.error(e);
  } finally {
    btn.disabled = false;
  }
}

function renderMetrics(result) {
  document.getElementById("metrics").hidden = false;
  document.getElementById("testAcc").textContent = (result.test_metrics.accuracy * 100).toFixed(1) + "%";
  document.getElementById("baselineAcc").textContent = (result.baseline_accuracy * 100).toFixed(1) + "%";
  document.getElementById("depthLeaves").textContent = `${result.depth} / ${result.leaf_count}`;
  const delta = (result.test_metrics.accuracy - result.baseline_accuracy) * 100;
  const note = document.getElementById("metricNote");
  note.textContent = delta > 0.5
    ? `beats "always guess Washed" by ${delta.toFixed(1)} points`
    : "barely beats the majority-class guess, try a bit more depth";
}

function renderImportance(result) {
  const wrap = document.getElementById("importance");
  wrap.hidden = false;
  const entries = Object.entries(result.feature_importances).filter(([, v]) => v > 0).slice(0, 6);
  document.getElementById("importanceBars").innerHTML = entries.map(([f, v]) => `
    <div class="imp-row">
      <span>${FEATURE_LABELS[f] || f}</span>
      <span class="imp-track"><span class="imp-fill" style="width:${(v * 100).toFixed(0)}%"></span></span>
      <span>${(v * 100).toFixed(0)}%</span>
    </div>
  `).join("");
}

// ---------------------------------------------------------------------------
// Tree layout + SVG rendering
// ---------------------------------------------------------------------------

const UNIT_W = 148;     // horizontal space per leaf slot
const ROW_H = 118;      // vertical space per depth level
const NODE_W = 128;
const NODE_H_INTERNAL = 72;
const NODE_H_LEAF = 84;

function computeLayout(node) {
  // returns leaf-slot width of this subtree, and fills node.__x (in slot units, center)
  if (node.is_leaf) {
    node.__slots = 1;
    return 1;
  }
  const yesW = computeLayout(node.yes);
  const noW = computeLayout(node.no);
  node.__slots = yesW + noW;
  return node.__slots;
}

function assignX(node, leftSlot) {
  if (node.is_leaf) {
    node.__x = leftSlot + 0.5;
    return;
  }
  assignX(node.yes, leftSlot);
  assignX(node.no, leftSlot + node.yes.__slots);
  node.__x = (node.yes.__x + node.no.__x) / 2;
}

function renderTree(root) {
  document.getElementById("treeEmpty").hidden = true;
  computeLayout(root);
  assignX(root, 0);

  const totalSlots = root.__slots;
  const width = Math.max(totalSlots * UNIT_W, 500);
  const maxDepth = maxDepthOf(root);
  const height = (maxDepth + 1) * ROW_H + 60;

  const svg = document.getElementById("treeSvg");
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";
  nodePositions = {};

  const edgesLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  const nodesLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  svg.appendChild(edgesLayer);
  svg.appendChild(nodesLayer);

  walkAndDraw(root, edgesLayer, nodesLayer, width);

  // auto scroll to show the root
  document.getElementById("treeScroll").scrollLeft = (width - document.getElementById("treeScroll").clientWidth) / 2;
}

function maxDepthOf(node) {
  if (node.is_leaf) return node.depth;
  return Math.max(maxDepthOf(node.yes), maxDepthOf(node.no));
}

function pxFor(node, totalWidth) {
  return { cx: node.__x * UNIT_W, cy: node.depth * ROW_H + 40 };
}

function walkAndDraw(node, edgesLayer, nodesLayer, totalWidth) {
  const { cx, cy } = pxFor(node, totalWidth);
  const h = node.is_leaf ? NODE_H_LEAF : NODE_H_INTERNAL;
  const x = cx - NODE_W / 2, y = cy - h / 2;
  nodePositions[node.id] = { cx, cy, x, y, w: NODE_W, h };

  if (!node.is_leaf) {
    [["yes", node.yes], ["no", node.no]].forEach(([answer, child]) => {
      const { cx: ccx, cy: ccy } = pxFor(child, totalWidth);
      const childH = child.is_leaf ? NODE_H_LEAF : NODE_H_INTERNAL;
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const startY = cy + h / 2, endY = ccy - childH / 2;
      const midY = (startY + endY) / 2;
      path.setAttribute("d", `M ${cx} ${startY} C ${cx} ${midY}, ${ccx} ${midY}, ${ccx} ${endY}`);
      path.setAttribute("class", "tedge");
      path.dataset.parent = node.id;
      path.dataset.child = child.id;
      path.dataset.answer = answer;
      edgesLayer.appendChild(path);

      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", (cx + ccx) / 2 + (answer === "yes" ? -14 : 14));
      label.setAttribute("y", midY + 4);
      label.setAttribute("class", "tedge-label");
      label.setAttribute("text-anchor", "middle");
      label.textContent = answer;
      edgesLayer.appendChild(label);

      walkAndDraw(child, edgesLayer, nodesLayer, totalWidth);
    });
  }

  drawNodeCard(node, x, y, nodesLayer);
}

function drawNodeCard(node, x, y, layer) {
  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
  g.dataset.nodeId = node.id;
  g.setAttribute("transform", `translate(${x}, ${y})`);

  const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  rect.setAttribute("width", NODE_W);
  rect.setAttribute("height", node.is_leaf ? NODE_H_LEAF : NODE_H_INTERNAL);
  rect.setAttribute("rx", 9);
  rect.setAttribute("class", "tnode-card" + (node.is_leaf ? " leaf" : ""));
  if (node.is_leaf) rect.style.stroke = CLASS_COLORS[node.prediction] || "#4d3520";
  g.appendChild(rect);

  if (node.is_leaf) {
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", 10); label.setAttribute("y", 20);
    label.setAttribute("class", "tnode-question");
    label.setAttribute("font-size", "12.5");
    label.style.fill = CLASS_COLORS[node.prediction] || "#f4ead9";
    label.textContent = node.prediction;
    g.appendChild(label);

    const sub = document.createElementNS("http://www.w3.org/2000/svg", "text");
    sub.setAttribute("x", 10); sub.setAttribute("y", 35);
    sub.setAttribute("class", "tnode-sub");
    sub.setAttribute("font-size", "10");
    sub.textContent = `n=${node.n_samples} · entropy ${node.entropy}`;
    g.appendChild(sub);

    // mini stacked bar of class distribution
    const total = Object.values(node.class_counts).reduce((a, b) => a + b, 0) || 1;
    let cursor = 0;
    const barY = 46, barH = 8, barW = NODE_W - 20;
    Object.entries(node.class_counts).forEach(([cls, cnt]) => {
      if (cnt === 0) return;
      const w = (cnt / total) * barW;
      const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      r.setAttribute("x", 10 + cursor); r.setAttribute("y", barY);
      r.setAttribute("width", Math.max(w, 0.5)); r.setAttribute("height", barH);
      r.setAttribute("rx", 2);
      r.style.fill = CLASS_COLORS[cls] || "#666";
      g.appendChild(r);
      cursor += w;
    });

    const acc = document.createElementNS("http://www.w3.org/2000/svg", "text");
    acc.setAttribute("x", 10); acc.setAttribute("y", 70);
    acc.setAttribute("class", "tnode-sub");
    acc.setAttribute("font-size", "9.5");
    acc.textContent = `leaf purity ${(node.accuracy * 100).toFixed(0)}%`;
    g.appendChild(acc);
  } else {
    const q = document.createElementNS("http://www.w3.org/2000/svg", "text");
    q.setAttribute("x", 10); q.setAttribute("y", 22);
    q.setAttribute("class", "tnode-question");
    q.setAttribute("font-size", "12");
    q.textContent = truncate(prettyQuestion(node), 22);
    g.appendChild(q);

    const sub = document.createElementNS("http://www.w3.org/2000/svg", "text");
    sub.setAttribute("x", 10); sub.setAttribute("y", 39);
    sub.setAttribute("class", "tnode-sub");
    sub.setAttribute("font-size", "10");
    sub.textContent = `n=${node.n_samples} · gain ${node.gain}`;
    g.appendChild(sub);

    const sub2 = document.createElementNS("http://www.w3.org/2000/svg", "text");
    sub2.setAttribute("x", 10); sub2.setAttribute("y", 54);
    sub2.setAttribute("class", "tnode-sub");
    sub2.setAttribute("font-size", "10");
    sub2.textContent = `entropy ${node.entropy}  gini ${node.gini}`;
    g.appendChild(sub2);
  }

  layer.appendChild(g);
}

function prettyQuestion(node) {
  if (node.feature_type === "numeric") {
    return `${FEATURE_LABELS[node.feature] || node.feature} ≤ ${node.threshold}?`;
  }
  return `${FEATURE_LABELS[node.feature] || node.feature} = ${node.category}?`;
}

function truncate(s, n) { return s.length > n ? s.slice(0, n - 1) + "…" : s; }

// ---------------------------------------------------------------------------
// Cupping form
// ---------------------------------------------------------------------------

function buildCupForm() {
  const form = document.getElementById("cupForm");
  const numeric = datasetSummary.features.filter(f => datasetSummary.feature_types[f] === "numeric");
  const rangesByFeature = datasetSummary.numeric_ranges;

  const scoreFields = ["aroma", "flavor", "aftertaste", "acidity", "body", "balance", "uniformity", "clean_cup", "sweetness"];
  const otherNumeric = numeric.filter(f => !scoreFields.includes(f));

  let html = "";
  scoreFields.forEach(f => {
    if (!rangesByFeature[f]) return;
    html += cupSlider(f, rangesByFeature[f], 0.1, 7.5);
  });
  otherNumeric.forEach(f => {
    const r = rangesByFeature[f];
    const step = f === "moisture" ? 0.01 : (f === "defects" ? 1 : 10);
    const def = f === "defects" ? r.min : (f === "moisture" ? 0.11 : Math.round((r.min + r.max) / 2));
    html += cupSlider(f, r, step, def);
  });

  html += `
    <div class="cup-field">
      <label for="country">Country</label>
      <select id="country" name="country" style="grid-column: 2 / span 2;">
        ${datasetSummary.countries.map(c => `<option value="${c}">${c}</option>`).join("")}
      </select>
    </div>
  `;
  form.innerHTML = html;
  document.getElementById("country").value = "Ethiopia";

  form.querySelectorAll('input[type="range"]').forEach(input => {
    const out = form.querySelector(`output[for="${input.id}"]`);
    input.addEventListener("input", () => out.textContent = input.value);
  });
}

function cupSlider(f, range, step, def) {
  const min = f === "defects" ? 0 : range.min;
  const max = f === "defects" ? Math.max(range.max, 10) : range.max;
  return `
    <div class="cup-field">
      <label for="${f}">${FEATURE_LABELS[f] || f}</label>
      <input type="range" id="${f}" name="${f}" min="${min}" max="${max}" step="${step}" value="${def}" />
      <output for="${f}">${def}</output>
    </div>
  `;
}

function buildPresets() {
  const el = document.getElementById("presets");
  el.innerHTML = PRESETS.map((p, i) => `<button type="button" class="preset-chip" data-i="${i}">${p.label}</button>`).join("");
  el.querySelectorAll(".preset-chip").forEach(btn => {
    btn.addEventListener("click", () => applyPreset(PRESETS[Number(btn.dataset.i)].values));
  });
}

function applyPreset(values) {
  Object.entries(values).forEach(([k, v]) => {
    const input = document.getElementById(k);
    if (!input) return;
    input.value = v;
    const out = document.querySelector(`output[for="${k}"]`);
    if (out) out.textContent = v;
  });
}

function readCupForm() {
  const form = document.getElementById("cupForm");
  const data = {};
  form.querySelectorAll("input, select").forEach(el => {
    data[el.name] = el.type === "range" ? Number(el.value) : el.value;
  });
  return data;
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

async function predictSample() {
  if (!currentTree) {
    alert("Train the tree first (left panel) before sending a sample down it.");
    return;
  }
  clearHighlights();
  const sample = readCupForm();
  let result;
  try {
    result = await fetchJSON("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(sample),
    });
  } catch (e) {
    console.error(e);
    alert("Prediction failed, check the console.");
    return;
  }

  document.getElementById("prediction").hidden = false;
  const color = CLASS_COLORS[result.prediction] || "#e0a339";
  document.getElementById("verdict").innerHTML = `
    <span class="verdict-label">Predicted process</span>
    <span style="color:${color}">${result.prediction}</span>
  `;

  const probaEl = document.getElementById("probaBars");
  const sorted = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  probaEl.innerHTML = sorted.map(([cls, p]) => `
    <div class="proba-row">
      <span>${cls}</span>
      <span class="proba-track"><span class="proba-fill" style="width:${(p * 100).toFixed(0)}%; background:${CLASS_COLORS[cls]}"></span></span>
      <span>${(p * 100).toFixed(0)}%</span>
    </div>
  `).join("");

  const pathEl = document.getElementById("pathList");
  pathEl.innerHTML = result.path.map(step => {
    if (step.leaf) {
      return `<li>Landed in a leaf → predicts <span class="leaf-final">${step.prediction}</span></li>`;
    }
    const cls = step.answer === "yes" ? "answer-yes" : "answer-no";
    return `<li>${step.question} → <span class="${cls}">${step.answer}</span></li>`;
  }).join("");

  highlightPath(result.path);
  scrollToPathBottom(result.path);
}

function clearHighlights() {
  document.querySelectorAll(".tedge.lit").forEach(e => e.classList.remove("lit"));
  document.querySelectorAll(".tnode-card.lit").forEach(e => e.classList.remove("lit"));
}

function highlightPath(path) {
  for (let i = 0; i < path.length; i++) {
    const nodeG = document.querySelector(`g[data-node-id="${path[i].node_id}"]`);
    if (nodeG) {
      const rect = nodeG.querySelector("rect");
      setTimeout(() => rect && rect.classList.add("lit"), i * 160);
    }
    if (i < path.length - 1) {
      const edge = document.querySelector(`path[data-parent="${path[i].node_id}"][data-child="${path[i + 1].node_id}"]`);
      if (edge) setTimeout(() => edge.classList.add("lit"), i * 160 + 80);
    }
  }
}

function scrollToPathBottom(path) {
  const lastId = path[path.length - 1].node_id;
  const pos = nodePositions[lastId];
  if (!pos) return;
  const scroller = document.getElementById("treeScroll");
  setTimeout(() => {
    scroller.scrollTo({
      left: Math.max(pos.cx - scroller.clientWidth / 2, 0),
      top: Math.max(pos.cy - scroller.clientHeight / 2, 0),
      behavior: "smooth",
    });
  }, path.length * 160);
}

init();
