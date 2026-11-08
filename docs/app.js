/* OTiS web UI. No framework: canvas + fetch against server.py, or plant.js offline. */
(() => {
  "use strict";

  // ------------------------------------------------------------------ utils
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const fmt = (v, d = 3) => (v == null || !isFinite(v) ? "—" : Number(v).toFixed(d));
  const fmtSig = (v) => {
    if (v == null || !isFinite(v)) return "—";
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a < 1e-3) return v.toExponential(1);
    if (a < 1) return v.toFixed(4);
    if (a < 100) return v.toFixed(3);
    return v.toFixed(0);
  };
  const fmtParams = (n) => (n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(0) + "K" : String(n));
  const api = async (path, body) => {
    const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    const r = await fetch(path, opt);
    let data = {};
    try { data = await r.json(); } catch { /* empty or non-JSON body */ }
    if (!r.ok) {
      const err = new Error(data.error || `HTTP ${r.status}`);
      err.http = r.status; err.kind = data.kind; err.data = data;
      throw err;
    }
    return data;
  };
  let toastTimer;
  const toast = (msg) => {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("show"), 2200);
  };

  const COLORS = () => ({
    neural: css("--blue"), convex: css("--green"), e_guidance: css("--orange"), pursuit: css("--teal"),
    start: css("--purple"), goal: css("--red"), vel: css("--indigo"), thrust: css("--pink"),
    grid: css("--grid"), gridStrong: css("--grid-strong"), gridLine: "rgba(255,255,255,0.075)", gridAxis: "rgba(255,255,255,0.16)", bg: css("--canvas-bg"),
    label: css("--label"), label2: css("--label-2"), label3: css("--label-3"), elev: css("--bg-elev"),
  });
  let C = COLORS();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { C = COLORS(); drawAll(); });

  const NAMES = { neural: "OTiS", convex: "Convex opt.", e_guidance: "E-guidance", pursuit: "Pursuit" };
  const SHORT = { neural: "OTiS", convex: "Convex", e_guidance: "E-guidance", pursuit: "Pursuit" };
  const METHOD_INFO = {
    neural: { name: "OTiS", detail: "Neural network · one forward pass", swatch: "neural" },
    convex: { name: "Convex optimization", detail: "Direct transcription · same plant", swatch: "convex" },
    e_guidance: { name: "E-guidance", detail: "Closed-loop ZEM law", swatch: "eguid" },
    pursuit: { name: "Pursuit", detail: "Geometric bicycle guidance", swatch: "pursuit" },
  };

  // ------------------------------------------------------------------ state
  const S = {
    cfg: { position_limit: 2, velocity_limit: 0.5, n_steps: 50, dt: 0.08, sizes: {} },
    x0: [-1.6, -1.2, 0.3, 0.0],
    goal: [1.5, 1.3],
    plans: {},
    optimalLoss: null,
    show: { neural: true, convex: true, e_guidance: true, pursuit: true, thrust: true },
    t: 50, playing: false,
    server: null,
    snapshots: [], evoIndex: -1, evoPlaying: false, evoLive: true,
    size: "large",
    tab: "simulator",
    vehicle: "rocket",
    staticMode: false,
    catalog: null,        // vehicles.json: {schema, vehicles}
    params: {},           // per-vehicle user overrides {vehicle: {key: value}}
    scen: {},             // per-vehicle scenario {vehicle: {x0, goal}}
    spec: null,           // last resolved spec from the planner (params, derived, workspace...)
    neural: null,         // neural_status from the planner
  };

  // ------------------------------------------------------------------ parameters
  const STORE_KEY = "otis.params.v1";
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
    if (saved && typeof saved === "object") { S.params = saved.params || {}; S.scen = saved.scen || {}; S.vehicle = saved.vehicle || "rocket"; }
  } catch { /* storage unavailable: start from presets */ }
  const persist = () => { try { localStorage.setItem(STORE_KEY, JSON.stringify({ params: S.params, scen: S.scen, vehicle: S.vehicle })); } catch { /* ignore */ } };
  const preset = (v = S.vehicle) => S.catalog?.vehicles?.[v];
  // Fields marked "hidden" (gravity) keep their preset value: not shown, not editable, not imported.
  const schema = (v = S.vehicle) => (S.catalog?.schema?.[preset(v)?.kind] || []).filter((f) => !f.hidden);
  function currentParams(v = S.vehicle) {
    const base = preset(v)?.params || {};
    return { ...base, ...(S.params[v] || {}) };
  }
  const WS = () => S.spec?.workspace ?? +currentParams().workspace ?? 2;
  const HORIZON = () => S.spec?.horizon ?? +currentParams().horizon ?? 4;
  const DTS = () => HORIZON() / (S.cfg.n_steps || 50);
  const velScale = () => 0.3 * HORIZON(); // world metres of handle per m/s
  function defaultScenario(v = S.vehicle) {
    const p = currentParams(v), L = +p.workspace || 2;
    return { x0: [-0.8 * L, -0.6 * L, 0.3 * (+p.start_speed || 0) * 1.2, 0], goal: [0.75 * L, 0.65 * L] };
  }
  function loadScenario() {
    const sc = S.scen[S.vehicle] || defaultScenario();
    S.x0 = sc.x0.slice(); S.goal = sc.goal.slice();
  }
  function saveScenario() { S.scen[S.vehicle] = { x0: S.x0.slice(), goal: S.goal.slice() }; persist(); }

  // ------------------------------------------------------------------ segmented controls
  function segment(el, onChange) {
    const thumb = el.querySelector(".seg-thumb") || Object.assign(document.createElement("span"), { className: "seg-thumb" });
    if (!thumb.parentNode) el.appendChild(thumb);
    const place = () => {
      const a = el.querySelector("button.active");
      if (!a) return;
      thumb.style.width = a.offsetWidth + "px";
      thumb.style.transform = `translateX(${a.offsetLeft - 2}px)`;
    };
    el.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b || b.disabled) return;
      el.querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b));
      place();
      onChange && onChange(b);
    });
    new ResizeObserver(place).observe(el);
    requestAnimationFrame(place);
    return place;
  }

  // The vehicle bar lives in the active view's title row (left column); hidden on About.
  function placeVehicleBar() {
    const bar = $("#nav-sub"), hero = $(`#view-${S.tab} > .hero`);
    bar.classList.toggle("hidden", S.tab === "about");
    if (hero && bar.parentNode !== hero) hero.prepend(bar);
  }
  placeVehicleBar();
  segment($("#tabs"), (b) => {
    S.tab = b.dataset.tab;
    $$(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + S.tab));
    placeVehicleBar();
    if (S.tab === "training") loadEvolution();
    requestAnimationFrame(() => { resizeAll(); drawAll(); });
  });

  function currentVehicle() {
    return preset() || { methods: ["neural", "convex", "e_guidance"], blurb: "", kind: "point_mass", draw: "rocket" };
  }
  function renderMethods() {
    const holder = $("#method-list");
    if (!holder) return;
    const methods = currentVehicle().methods || [];
    holder.innerHTML = methods.map((m) => {
      const info = METHOD_INFO[m] || { name: m, detail: "", swatch: m };
      const on = S.show[m] !== false;
      return `<label class="row"><span class="swatch ${info.swatch}"></span><span class="grow">${info.name}<small>${info.detail}</small></span><input type="checkbox" class="switch" data-method="${m}" ${on ? "checked" : ""}></label>`;
    }).join("");
    holder.querySelectorAll(".switch[data-method]").forEach((sw) => sw.addEventListener("change", () => {
      S.show[sw.dataset.method] = sw.checked; renderCompare(); drawThrustChart();
    }));
    const blurb = $("#vehicle-blurb");
    if (blurb) blurb.textContent = currentVehicle().blurb || "";
  }
  if ($("#vehicle-tabs")) {
    segment($("#vehicle-tabs"), (b) => {
      saveScenario();
      S.vehicle = b.dataset.vehicle;
      S.spec = null; S.plans = {};
      loadScenario();
      persist();
      renderMethods();
      buildParamsPanel();
      renderTrainPlant();
      S.t = 0; S.playing = true;
      requestPlan(true);
    });
  }

  // ------------------------------------------------------------------ canvas helpers
  // Everything on the page is drawn 1.2x larger. Canvases keep a backing store at
  // the real device ratio; the returned unit scales strokes, markers and text.
  const UI_SCALE = 1.2;
  function fitCanvas(cv) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = cv.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width * dpr));
    // Setting cv.height rewrites the height attribute, so remember the design
    // height once; reading it back each frame made the chart grow without bound.
    if (cv.dataset.baseH === undefined) cv.dataset.baseH = cv.hasAttribute("height") && !cv.closest(".stage") ? cv.getAttribute("height") : "";
    const cssH = cv.dataset.baseH ? +cv.dataset.baseH * UI_SCALE : r.height;
    const h = Math.max(1, Math.round(cssH * dpr));
    if (cv.dataset.baseH) cv.style.height = cssH + "px";
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    return dpr * UI_SCALE;
  }

  function makeView(cv, Lw) {
    const L = Lw ?? WS();
    const span = L * 1.45;
    const w = cv.width, h = cv.height;
    const scale = Math.min(w, h * 1.0) / (2 * span) * 0.92;
    const cx = w / 2, cy = h / 2 - h * 0.02;
    return {
      scale, L,
      toPx: (x, y) => [cx + x * scale, cy - y * scale],
      toWorld: (px, py) => [(px - cx) / scale, -(py - cy) / scale],
    };
  }

  function drawGrid(ctx, V, dpr) {
    const cv = ctx.canvas;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, cv.width, cv.height);
    const [ax, ay] = V.toPx(-V.L, V.L), [bx, by] = V.toPx(V.L, -V.L);
    // Solid gridlines at a "nice" spacing (1, 2 or 5 x 10^k), about 20 cells across the workspace.
    const raw = (2 * V.L) / 20, mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw * 0.999);
    const [wx0, wy1] = V.toWorld(0, 0), [wx1, wy0] = V.toWorld(cv.width, cv.height);
    ctx.save();
    ctx.lineWidth = 1 * dpr;
    for (let x = Math.ceil(wx0 / step) * step; x <= wx1; x += step) {
      const px = Math.round(V.toPx(x, 0)[0]) + 0.5;
      ctx.strokeStyle = Math.abs(x) < step * 1e-6 ? C.gridAxis : C.gridLine;
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, cv.height); ctx.stroke();
    }
    for (let y = Math.ceil(wy0 / step) * step; y <= wy1; y += step) {
      const py = Math.round(V.toPx(0, y)[1]) + 0.5;
      ctx.strokeStyle = Math.abs(y) < step * 1e-6 ? C.gridAxis : C.gridLine;
      ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(cv.width, py); ctx.stroke();
    }
    ctx.restore();
    ctx.save();
    ctx.strokeStyle = C.gridStrong;
    ctx.lineWidth = 1 * dpr;
    roundRect(ctx, ax, ay, bx - ax, by - ay, 0);
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = C.label3;
    ctx.font = `500 ${11 * dpr}px ${css("--font")}`;
    ctx.fillText(`workspace ±${+V.L.toPrecision(4)} m`, ax + 10 * dpr, ay + 18 * dpr);
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function drawPath(ctx, V, path, color, dpr, opts = {}) {
    if (!path || path.length < 2) return;
    ctx.save();
    ctx.globalAlpha = opts.alpha ?? 1;
    ctx.strokeStyle = color;
    ctx.lineWidth = (opts.width ?? 3) * dpr;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (opts.dash) ctx.setLineDash(opts.dash.map((d) => d * dpr));
    ctx.beginPath();
    const upto = opts.upto ?? path.length - 1;
    path.forEach((p, i) => {
      if (i > Math.ceil(upto)) return;
      let [x, y] = [p[0], p[1]];
      if (i === Math.ceil(upto) && upto % 1 !== 0) {
        const q = path[i - 1], f = upto % 1;
        x = lerp(q[0], p[0], f); y = lerp(q[1], p[1], f);
      }
      const [px, py] = V.toPx(x, y);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    ctx.stroke();
    ctx.restore();
  }

  function drawThrustVectors(ctx, V, path, controls, dpr, upto) {
    if (!path || !controls) return;
    ctx.save();
    ctx.strokeStyle = C.thrust;
    ctx.globalAlpha = 0.75;
    ctx.lineWidth = 1.5 * dpr;
    ctx.lineCap = "round";
    const len = 0.11 * V.L;
    for (let k = 0; k < controls.length; k += 2) {
      if (k > upto) break;
      const [ux, uy] = controls[k];
      const p = path[k];
      const [x0, y0] = V.toPx(p[0], p[1]);
      const [x1, y1] = V.toPx(p[0] + ux * len, p[1] + uy * len);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    }
    ctx.restore();
  }

  function samplePath(path, t) {
    const i = clamp(Math.floor(t), 0, path.length - 1), j = Math.min(i + 1, path.length - 1), f = t - i;
    return [lerp(path[i][0], path[j][0], f), lerp(path[i][1], path[j][1], f), lerp(path[i][2], path[j][2], f), lerp(path[i][3], path[j][3], f)];
  }

  // Screen-space heading from the path itself: look ahead (or back at the end)
  // until the craft has moved a visible distance, so a near-stop never jitters.
  function headingAt(V, path, t) {
    const p = samplePath(path, t), [x0, y0] = V.toPx(p[0], p[1]);
    const end = path.length - 1;
    for (let dt = 0.25; dt <= end; dt += 0.25) {
      const q = samplePath(path, Math.min(end, t + dt)), [x1, y1] = V.toPx(q[0], q[1]);
      if (Math.hypot(x1 - x0, y1 - y0) > 2) return Math.atan2(y1 - y0, x1 - x0);
      if (t + dt >= end) break;
    }
    for (let dt = 0.25; dt <= end; dt += 0.25) {
      const q = samplePath(path, Math.max(0, t - dt)), [x1, y1] = V.toPx(q[0], q[1]);
      if (Math.hypot(x0 - x1, y0 - y1) > 2) return Math.atan2(y0 - y1, x0 - x1);
      if (t - dt <= 0) break;
    }
    return Math.atan2(-p[3], p[2]);
  }

  function drawCarBody(ctx, dpr, color) {
    ctx.fillStyle = C.elev; ctx.strokeStyle = color; ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    ctx.roundRect(-11 * dpr, -6 * dpr, 22 * dpr, 12 * dpr, 3 * dpr);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = color; ctx.globalAlpha = 0.28;
    ctx.fillRect(-3 * dpr, -4.5 * dpr, 9 * dpr, 9 * dpr);
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    [[-7, -7], [6, -7], [-7, 7], [6, 7]].forEach(([x, y]) => {
      ctx.beginPath(); ctx.arc(x * dpr, y * dpr, 2.1 * dpr, 0, Math.PI * 2); ctx.fill();
    });
  }
  function drawUavBody(ctx, dpr, color) {
    ctx.strokeStyle = color; ctx.lineWidth = 2 * dpr; ctx.fillStyle = C.elev;
    ctx.beginPath(); ctx.moveTo(-9 * dpr, 0); ctx.lineTo(9 * dpr, 0); ctx.moveTo(0, -9 * dpr); ctx.lineTo(0, 9 * dpr); ctx.stroke();
    [[-9, -9], [9, -9], [-9, 9], [9, 9]].forEach(([x, y]) => {
      ctx.beginPath(); ctx.arc(x * dpr, y * dpr, 3.4 * dpr, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    });
    ctx.beginPath(); ctx.arc(0, 0, 3.2 * dpr, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  }
  function drawUsvBody(ctx, dpr, color) {
    ctx.fillStyle = C.elev; ctx.strokeStyle = color; ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    ctx.moveTo(12 * dpr, 0);
    ctx.quadraticCurveTo(4 * dpr, -7 * dpr, -10 * dpr, -5 * dpr);
    ctx.lineTo(-10 * dpr, 5 * dpr);
    ctx.quadraticCurveTo(4 * dpr, 7 * dpr, 12 * dpr, 0);
    ctx.fill(); ctx.stroke();
  }
  function drawCraft(ctx, V, path, controls, t, color, dpr, kindOverride) {
    if (!path) return;
    const s = samplePath(path, t);
    const k = clamp(Math.floor(t), 0, (controls?.length || 1) - 1);
    const u = controls ? controls[k] : [0, 0];
    const [px, py] = V.toPx(s[0], s[1]);
    const thrustMag = Math.hypot(u[0], u[1]);
    const kind = kindOverride || currentVehicle().draw || S.vehicle || "rocket";
    // Every vehicle points along its direction of travel (path tangent), so the
    // rocket no longer flips when the plan reverses thrust to brake.
    const ang = headingAt(V, path, t);
    ctx.save();
    ctx.translate(px, py);
    // exhaust plume opposite to the actual thrust (a braking burn shows at the nose)
    if (kind === "rocket" && thrustMag > 0.02 && t < path.length - 1) {
      ctx.save();
      ctx.rotate(Math.atan2(-u[1], u[0]));
      const fl = (10 + 22 * thrustMag) * dpr * (0.85 + 0.15 * Math.sin(performance.now() / 45));
      const g = ctx.createLinearGradient(-6 * dpr, 0, -6 * dpr - fl, 0);
      g.addColorStop(0, css("--yellow"));
      g.addColorStop(0.45, css("--orange"));
      g.addColorStop(1, "rgba(255,120,0,0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.moveTo(-6 * dpr, -4 * dpr);
      ctx.quadraticCurveTo(-6 * dpr - fl * 0.6, 0, -6 * dpr - fl, 0);
      ctx.quadraticCurveTo(-6 * dpr - fl * 0.6, 0, -6 * dpr, 4 * dpr);
      ctx.fill();
      ctx.restore();
    }
    ctx.rotate(ang);
    ctx.shadowColor = "rgba(0,0,0,.25)";
    ctx.shadowBlur = 8 * dpr;
    ctx.fillStyle = C.elev;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * dpr;
    if (kind === "car") drawCarBody(ctx, dpr, color);
    else if (kind === "uav") drawUavBody(ctx, dpr, color);
    else if (kind === "usv") drawUsvBody(ctx, dpr, color);
    else {
      ctx.beginPath();
      ctx.moveTo(11 * dpr, 0);
      ctx.quadraticCurveTo(2 * dpr, -8 * dpr, -7 * dpr, -6 * dpr);
      ctx.lineTo(-7 * dpr, 6 * dpr);
      ctx.quadraticCurveTo(2 * dpr, 8 * dpr, 11 * dpr, 0);
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawStart(ctx, V, x0, dpr, hot, opts = {}) {
    const [px, py] = V.toPx(x0[0], x0[1]);
    // velocity handle
    const vs = opts.velScale ?? velScale();
    const [hx, hy] = V.toPx(x0[0] + x0[2] * vs, x0[1] + x0[3] * vs);
    ctx.save();
    ctx.strokeStyle = C.vel;
    ctx.fillStyle = C.vel;
    ctx.lineWidth = 2 * dpr;
    ctx.setLineDash([3 * dpr, 4 * dpr]);
    ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(hx, hy); ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(hx, hy, (hot === "vel" ? 8 : 6) * dpr, 0, Math.PI * 2);
    ctx.globalAlpha = 0.18; ctx.fill(); ctx.globalAlpha = 1;
    ctx.beginPath(); ctx.arc(hx, hy, 4 * dpr, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
    // start puck
    ctx.save();
    ctx.fillStyle = C.start;
    ctx.globalAlpha = 0.16;
    ctx.beginPath(); ctx.arc(px, py, (hot === "start" ? 17 : 14) * dpr, 0, Math.PI * 2); ctx.fill();
    ctx.globalAlpha = 1;
    ctx.shadowColor = "rgba(0,0,0,.25)"; ctx.shadowBlur = 6 * dpr;
    ctx.beginPath(); ctx.arc(px, py, 7 * dpr, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#fff";
    ctx.beginPath(); ctx.arc(px, py, 2.6 * dpr, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }

  function drawGoal(ctx, V, goal, dpr, hot) {
    const [px, py] = V.toPx(goal[0], goal[1]);
    const pulse = 1 + 0.06 * Math.sin(performance.now() / 380);
    ctx.save();
    ctx.strokeStyle = C.goal;
    ctx.fillStyle = C.goal;
    ctx.globalAlpha = 0.12;
    ctx.beginPath(); ctx.arc(px, py, (hot === "goal" ? 22 : 19) * dpr * pulse, 0, Math.PI * 2); ctx.fill();
    ctx.globalAlpha = 1;
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath(); ctx.arc(px, py, 12 * dpr, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(px, py, 5 * dpr, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }

  // ------------------------------------------------------------------ simulator view
  const simCv = $("#sim-canvas");
  const simCtx = simCv.getContext("2d");
  let simDpr = 1, simV = null, hot = null, drag = null;

  function drawSim() {
    simDpr = fitCanvas(simCv);
    simV = makeView(simCv);
    const ctx = simCtx, V = simV, d = simDpr;
    drawGrid(ctx, V, d);
    const order = ["e_guidance", "convex", "pursuit", "neural"];
    const style = { neural: { width: 3.2 }, convex: { width: 2.4, dash: [1, 0] }, e_guidance: { width: 2.2, dash: [7, 6] }, pursuit: { width: 2.6, dash: [1, 0] } };
    order.forEach((m) => {
      const p = S.plans[m];
      if (!p || !S.show[m]) return;
      drawPath(ctx, V, p.path, C[m], d, { ...style[m], alpha: 0.28, width: style[m].width * 0.8, glow: false });
      drawPath(ctx, V, p.path, C[m], d, { ...style[m], upto: S.t });
    });
    if (S.show.thrust && S.plans.neural && S.show.neural) drawThrustVectors(ctx, V, S.plans.neural.path, S.plans.neural.controls, d, S.t);
    drawGoal(ctx, V, S.goal, d, hot);
    drawStart(ctx, V, S.x0, d, hot);
    order.forEach((m) => {
      const p = S.plans[m];
      if (p && S.show[m]) drawCraft(ctx, V, p.path, p.controls, S.t, C[m], d);
    });
    // legend
    $("#sim-legend").innerHTML = (S.planError ? "<span>Last valid plan</span>" : "") + (Object.keys(NAMES).filter((m) => S.plans[m] && S.show[m])
      .map((m) => `<span><i style="background:${C[m]}"></i>${NAMES[m]}</span>`).join("") || "<span>No plans</span>");
  }

  function hitTest(px, py) {
    if (!simV) return null;
    const r = 18 * simDpr;
    const dist = (x, y) => { const [a, b] = simV.toPx(x, y); return Math.hypot(a - px, b - py); };
    const ds = dist(S.x0[0], S.x0[1]);
    // The start puck's own centre always wins, so a short velocity arrow can
    // never hide it; otherwise take the nearest handle within reach.
    if (ds < 9 * simDpr) return "start";
    const c = [["vel", dist(S.x0[0] + S.x0[2] * velScale(), S.x0[1] + S.x0[3] * velScale())], ["goal", dist(S.goal[0], S.goal[1])], ["start", ds]]
      .filter(([, d]) => d < r).sort((a, b) => a[1] - b[1]);
    return c.length ? c[0][0] : null;
  }

  const evPos = (e) => { const r = simCv.getBoundingClientRect(); const k = simCv.width / Math.max(1, r.width); return [(e.clientX - r.left) * k, (e.clientY - r.top) * k]; };
  simCv.addEventListener("pointerdown", (e) => {
    const [px, py] = evPos(e);
    drag = hitTest(px, py);
    if (drag) { simCv.setPointerCapture(e.pointerId); simCv.classList.add("grabbing"); $("#sim-hint").style.opacity = 0; }
  });
  simCv.addEventListener("pointermove", (e) => {
    const [px, py] = evPos(e);
    if (!drag) {
      const h = hitTest(px, py);
      if (h !== hot) { hot = h; simCv.classList.toggle("grab", !!h); }
      return;
    }
    const [wx, wy] = simV.toWorld(px, py);
    const L = WS();
    if (drag === "start") { S.x0[0] = clamp(wx, -L, L); S.x0[1] = clamp(wy, -L, L); }
    if (drag === "goal") { S.goal = [clamp(wx, -L, L), clamp(wy, -L, L)]; }
    if (drag === "vel") {
      const vmax = Math.max(2 * (+currentParams().start_speed || 0), L / HORIZON());
      S.x0[2] = clamp((wx - S.x0[0]) / velScale(), -vmax, vmax);
      S.x0[3] = clamp((wy - S.x0[1]) / velScale(), -vmax, vmax);
    }
    renderScenario();
    S.t = S.cfg.n_steps;
    requestPlan();
  });
  const endDrag = () => { if (drag) { drag = null; simCv.classList.remove("grabbing"); requestPlan(true); } };
  simCv.addEventListener("pointerup", endDrag);
  simCv.addEventListener("pointercancel", endDrag);
  simCv.addEventListener("dblclick", (e) => {
    const [wx, wy] = simV.toWorld(...evPos(e));
    const L = WS();
    S.goal = [clamp(wx, -L, L), clamp(wy, -L, L)];
    S.t = 0; S.playing = true;
    requestPlan(true);
  });

  // planning requests: at most one in flight, latest wins
  let inflight = false, pending = false;
  function applyLocalPlan(res, vehicle, rev) {
    if (vehicle !== S.vehicle) return; // a stale reply for a tab the user already left
    if (res.error) { showPlanError(res.error, res.spec); return; }
    S.planError = null;
    S.notices = res.notices || [];
    S.plans = res.plans || {};
    S.optimalLoss = res.optimal_loss;
    S.spec = res.spec || null;
    S.neural = res.neural_status || null;
    // Adopt the server's (workspace-clipped) start/goal only if nothing was
    // edited after this request left; otherwise the newer edit wins and the
    // pending request re-plans it.
    if (!drag && res.x0 && (rev === undefined || rev === S.rev)) { S.x0 = res.x0; S.goal = res.goal; saveScenario(); }
    updateStatus(S.server);
    renderCompare();
    renderScenario();
    renderDerived();
    renderNeural();
    renderTrainPlant();
    drawThrustChart();
  }
  const planLocal = () => window.OTiSPlant.planLocal(S.vehicle, S.x0, S.goal, currentParams());
  // Keep the last good plans on screen (marked stale) and say what is wrong.
  function showPlanError(message, spec) {
    S.planError = message;
    // Keep result metadata with the retained map; diagnostics use current fields.
    S.neural = { available: false, trainable: false, reason: message };
    updateStatus(S.server);
    renderCompare(); renderDerived(); renderNeural();
    drawAll();
  }
  async function requestPlan(force) {
    S.rev = (S.rev || 0) + 1;
    if (inflight) { pending = true; return; }
    inflight = true;
    try {
      if (!S.catalog) return;
      const vehicle = S.vehicle, rev = S.rev;
      if (S.staticMode && window.OTiSPlant) {
        applyLocalPlan(planLocal(), vehicle, rev);
      } else {
        const res = await api("/api/plan", { x0: S.x0.slice(), goal: S.goal.slice(), vehicle, params: currentParams(vehicle) });
        applyLocalPlan(res, vehicle, rev);
      }
    } catch (err) {
      if (err.http) {
        // The server answered: invalid physics (4xx) or a planner failure (5xx).
        // Stay on the server; never swap silently to the browser copy.
        showPlanError(err.message);
      } else if (!S.staticMode && window.OTiSPlant && S.catalog) {
        // The server is unreachable: fall back to the in-browser planners, visibly.
        console.warn("Planner server unreachable; using the in-browser planners.", err);
        S.staticMode = true;
        try { applyLocalPlan(planLocal(), S.vehicle); } catch (e) { showPlanError(e.message); }
        toast("Server unreachable · planning in the browser (no network)");
      } else {
        showPlanError(err.message);
      }
    } finally {
      inflight = false;
      if (pending) { pending = false; requestPlan(); }
    }
  }

  const fmtIn = (v) => (v == null || !isFinite(v) ? "" : String(+Number(v).toPrecision(5)));
  function renderScenario(force = false) {
    $$("[data-sc]").forEach((el) => {
      if (!force && el === document.activeElement) return;
      const [k, i] = el.dataset.sc.split(".");
      el.value = fmtIn(S[k][+i]);
      el.classList.remove("invalid");
    });
  }

  function deltaCell(v, ref, lowerBetter = true) {
    if (v == null || ref == null || !isFinite(ref)) return `<span class="delta neutral"></span>`;
    if (Math.abs(ref) < 1e-6 || Math.abs(v - ref) > 100 * Math.abs(ref)) return `<span class="delta neutral">${v >= ref ? "+" : ""}${fmtSig(v - ref)} abs</span>`;
    const pct = (100 * (v - ref)) / Math.abs(ref);
    const cls = Math.abs(pct) < 0.5 ? "neutral" : (pct < 0) === lowerBetter ? "good" : "bad";
    return `<span class="delta ${cls}">${pct >= 0 ? "+" : ""}${pct.toFixed(Math.abs(pct) < 10 ? 1 : 0)}%</span>`;
  }

  function renderCompare() {
    const ref = S.plans.convex;
    const rows = Object.keys(NAMES).filter((m) => S.plans[m]).map((m) => {
      const p = S.plans[m];
      const isRef = m === "convex";
      return `<div class="row"><div class="cmp-grid">
        <div class="cmp-name"><span class="swatch ${(METHOD_INFO[m] || {}).swatch || m}"></span>${SHORT[m] || m}</div>
        <div class="num">${fmtSig(p.miss)}</div><div class="num">${fmtSig(p.fuel)}</div><div class="num">${p.ms < 0.01 ? "<0.01" : p.ms < 1 ? p.ms.toFixed(2) : p.ms.toFixed(1)}</div>
        <div class="delta neutral">${isRef ? "reference" : ""}</div>
        ${isRef ? "<div></div><div></div><div></div>" : `${deltaCell(p.miss, ref?.miss)}${deltaCell(p.fuel, ref?.fuel)}${deltaCell(p.ms, ref?.ms)}`}
      </div></div>`;
    });
    $("#compare").innerHTML = (S.planError ? `<div class="note err">Last valid plan · results do not match the current fields.</div>` : "") + `<div class="row"><div class="cmp-grid cmp-head"><div>Method</div><div class="num">Miss m</div><div class="num">Control effort Σu²</div><div class="num">ms</div></div></div>` + rows.join("");
    const ng = networkGap();
    $("#opt-note").textContent = !ng ? "" : ng.pct != null
      ? `network gap to optimum ${ng.pct >= 0 ? "+" : ""}${Math.abs(ng.pct) >= 100 ? Math.round(ng.pct).toLocaleString() : ng.pct.toFixed(2)}%`
      : `network excess cost +${fmtSig(Math.max(0, ng.excess))} (optimum ≈ 0)`;
  }

  // How far the network's plan is from the exact optimum on this mission.
  // When the optimum costs almost nothing (goal at the start), a percentage
  // is meaningless, so the gap is reported as an absolute excess instead.
  function networkGap() {
    const pn = S.plans?.neural, opt = S.optimalLoss;
    if (!pn || opt == null || !isFinite(opt) || S.planError) return null;
    const L = WS(), floor = 1e-4 * L * L;
    const excess = pn.loss - opt;
    const tiny = opt < floor;
    const pct = tiny ? null : (100 * excess) / opt;
    return { excess, pct, miss: pn.miss, optMiss: S.plans.convex?.miss, significant: tiny ? (excess > floor || (opt < 1e-6 && excess > 1e-5)) : pct > 5 };
  }

  // thrust profile chart
  const thCv = $("#thrust-canvas"), thCtx = thCv.getContext("2d");
  function drawThrustChart() {
    const d = fitCanvas(thCv);
    const ctx = thCtx, w = thCv.width, h = thCv.height;
    ctx.clearRect(0, 0, w, h);
    const pad = 6 * d, top = 8 * d, bottom = h - 16 * d;
    const n = S.cfg.n_steps;
    const X = (k) => pad + (k / n) * (w - 2 * pad);
    const Y = (u) => lerp(bottom, top, (u + 1) / 2);
    ctx.strokeStyle = C.gridStrong; ctx.lineWidth = 1 * d;
    [-1, 0, 1].forEach((u) => { ctx.beginPath(); ctx.moveTo(pad, Y(u)); ctx.lineTo(w - pad, Y(u)); ctx.stroke(); });
    ctx.fillStyle = C.label3; ctx.font = `${10 * d}px ${css("--font")}`;
    ctx.fillText("+1", pad, Y(1) + 11 * d); ctx.fillText("−1", pad, Y(-1) - 3 * d);
    const axisLab = currentVehicle().kind === "bicycle" ? "throttle solid · steering dashed" : "throttle x solid · y dashed";
    ctx.fillText(axisLab, w - pad - ctx.measureText(axisLab).width, h - 3 * d);
    ["convex", "e_guidance", "pursuit", "neural"].forEach((m) => {
      const p = S.plans[m];
      if (!p || !S.show[m]) return;
      [0, 1].forEach((axis) => {
        ctx.save();
        ctx.strokeStyle = C[m];
        ctx.lineWidth = (m === "neural" ? 2 : 1.4) * d;
        if (axis === 1) ctx.setLineDash([4 * d, 3 * d]);
        ctx.beginPath();
        p.controls.forEach((u, k) => {
          ctx.lineTo(X(k), Y(u[axis])); ctx.lineTo(X(k + 1), Y(u[axis]));
        });
        ctx.stroke();
        ctx.restore();
      });
    });
    // time cursor
    ctx.strokeStyle = C.label2; ctx.lineWidth = 1 * d;
    ctx.beginPath(); ctx.moveTo(X(S.t), top); ctx.lineTo(X(S.t), bottom); ctx.stroke();
  }

  // transport
  const slider = $("#time-slider");
  function syncSlider() {
    slider.max = S.cfg.n_steps;
    slider.value = S.t;
    slider.style.setProperty("--p", (100 * S.t) / S.cfg.n_steps + "%");
    const tNow = S.t * DTS();
    $("#time-label").textContent = `t = ${tNow < 10 ? tNow.toFixed(2) : tNow.toFixed(1)} s`;
    $("#play-btn").innerHTML = S.playing ? '<svg viewBox="0 0 24 24"><path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z"/></svg>' : '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
  }
  slider.addEventListener("input", () => { S.t = +slider.value; S.playing = false; drawThrustChart(); });
  $("#play-btn").addEventListener("click", togglePlay);
  function togglePlay() {
    if (!S.playing && S.t >= S.cfg.n_steps - 0.01) S.t = 0;
    S.playing = !S.playing;
  }

  $$(".switch[data-method]").forEach((sw) => sw.addEventListener("change", () => { S.show[sw.dataset.method] = sw.checked; renderCompare(); drawThrustChart(); }));
  $("#show-thrust").addEventListener("change", (e) => { S.show.thrust = e.target.checked; });

  function randomScenario() {
    const L = WS(), V = +currentParams().start_speed || 0;
    const r = (a) => (Math.random() * 2 - 1) * a;
    S.x0 = [r(L), r(L), r(V), r(V)];
    S.goal = [r(L), r(L)];
    S.t = 0; S.playing = true;
    requestPlan(true);
  }
  $("#btn-random").addEventListener("click", randomScenario);
  $("#btn-zero").addEventListener("click", () => { S.x0[2] = 0; S.x0[3] = 0; S.t = 0; S.playing = true; renderScenario(); requestPlan(true); });
  window.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea")) return;
    if (S.tab !== "simulator") return;
    if (e.key === "r" || e.key === "R") randomScenario();
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
  });

  // ------------------------------------------------------------------ training view
  const trCv = $("#train-canvas"), trCtx = trCv.getContext("2d");
  const lossCv = $("#loss-canvas"), lossCtx = lossCv.getContext("2d");
  let trT = 0;

  function currentTrainingFrame() {
    const st = S.server;
    if (!S.evoLive && S.snapshots.length && S.evoIndex >= 0) return { ...S.snapshots[S.evoIndex], ghosts: S.snapshots.slice(Math.max(0, S.evoIndex - 10), S.evoIndex) };
    if (st && st.trajectory) return { path: st.trajectory, controls: st.controls, x0: st.x0, goal: st.goal, step: st.step, loss: st.current_loss, distance: st.distance, ghosts: S.snapshots.slice(-10) };
    if (S.snapshots.length) return { ...S.snapshots[S.snapshots.length - 1], ghosts: S.snapshots.slice(-11, -1) };
    return null;
  }

  function trainingPlant() {
    const tp = S.server?.plant;
    if (tp && tp.params) return tp;
    return { vehicle: S.vehicle, name: currentVehicle().name, params: currentParams() };
  }
  function drawTrain() {
    const d = fitCanvas(trCv);
    const tp = trainingPlant();
    const V = makeView(trCv, +tp.params.workspace || WS()), ctx = trCtx;
    const tvs = 0.3 * (+tp.params.horizon || HORIZON());
    drawGrid(ctx, V, d);
    const f = currentTrainingFrame();
    if (!f) {
      ctx.fillStyle = C.label2;
      ctx.font = `500 ${15 * d}px ${css("--font")}`;
      const msg = S.server?.phase === "compiling" ? "Compiling the differentiable simulator on the GPU…" : "Press Train to start. The plan will go from chaos to a clean burn.";
      ctx.fillText(msg, (trCv.width - ctx.measureText(msg).width) / 2, trCv.height / 2);
      return;
    }
    (f.ghosts || []).forEach((g, i, arr) => drawPath(ctx, V, g.path, C.neural, d, { width: 1.6, alpha: 0.06 + 0.22 * (i / Math.max(1, arr.length)) }));
    drawPath(ctx, V, f.path, C.neural, d, { width: 3.2 });
    drawThrustVectors(ctx, V, f.path, f.controls, d, 1e9);
    drawGoal(ctx, V, f.goal, d);
    drawStart(ctx, V, f.x0, d, null, { velScale: tvs });
    trT = (trT + 0.35) % (S.cfg.n_steps + 12);
    drawCraft(ctx, V, f.path, f.controls, Math.min(trT, S.cfg.n_steps), C.neural, d, S.catalog?.vehicles?.[tp.vehicle]?.draw);
    $("#train-badge").innerHTML = `Step ${(f.step ?? 0).toLocaleString()} &nbsp;·&nbsp; miss ${fmtSig(f.distance)} &nbsp;·&nbsp; loss ${fmtSig(f.loss)}`;
  }

  function drawLoss() {
    const d = fitCanvas(lossCv);
    const ctx = lossCtx, w = lossCv.width, h = lossCv.height;
    ctx.clearRect(0, 0, w, h);
    const hist = S.server?.history || {};
    const steps = hist.step || [];
    const series = [[hist.val_loss || [], C.neural], [hist.val_terminal || [], C.convex]];
    const vals = series.flatMap(([s]) => s).filter((v) => v > 0 && isFinite(v));
    const padL = 34 * d, padR = 8 * d, top = 8 * d, bottom = h - 18 * d;
    ctx.font = `${12 * d}px ${css("--font")}`;
    if (steps.length < 2 || !vals.length) {
      ctx.fillStyle = C.label3; ctx.fillText("No training history yet", padL, h / 2); return;
    }
    let lo = Math.floor(Math.log10(Math.min(...vals))), hi = Math.ceil(Math.log10(Math.max(...vals)));
    if (hi === lo) hi = lo + 1;
    const maxStep = Math.max(S.server?.total_steps || 0, steps[steps.length - 1]);
    const X = (s) => padL + (s / maxStep) * (w - padL - padR);
    const Y = (v) => lerp(bottom, top, (Math.log10(v) - lo) / (hi - lo));
    for (let e = lo; e <= hi; e++) {
      ctx.strokeStyle = C.grid; ctx.lineWidth = 1 * d;
      ctx.beginPath(); ctx.moveTo(padL, Y(10 ** e)); ctx.lineTo(w - padR, Y(10 ** e)); ctx.stroke();
      ctx.fillStyle = C.label3; ctx.fillText(`1e${e}`, 2 * d, Y(10 ** e) + 3 * d);
    }
    ctx.fillText("0", padL, h - 4 * d);
    const endLab = maxStep.toLocaleString();
    ctx.fillText(endLab, w - padR - ctx.measureText(endLab).width, h - 4 * d);
    series.forEach(([s, col]) => {
      ctx.save();
      ctx.strokeStyle = col; ctx.lineWidth = 2 * d; ctx.lineJoin = "round";
      ctx.beginPath();
      let started = false;
      s.forEach((v, i) => { if (!(v > 0)) return; const p = [X(steps[i]), Y(v)]; started ? ctx.lineTo(...p) : ctx.moveTo(...p); started = true; });
      ctx.stroke();
      ctx.restore();
    });
  }

  // size presets
  function approxParams(w, depth, n = S.cfg.n_steps) {
    return 10 * w + w + depth * (2 * (w * w + w) + 2 * w) + 2 * w + w * 2 * n + 2 * n;
  }
  function buildSizes() {
    const seg = $("#size-seg");
    const sizes = S.cfg.sizes || {};
    seg.innerHTML = Object.entries(sizes).map(([k, [w, dpt]]) => `<button data-size="${k}" class="${k === S.size ? "active" : ""}" title="${k}">${k === "xl" ? "XL" : k[0].toUpperCase() + k.slice(1)}<small style="margin:0;font-size:12.6px">${fmtParams(approxParams(w, dpt))}</small></button>`).join("");
    const note = () => { const [w, dpt] = sizes[S.size] || [0, 0]; $("#size-note").textContent = `Residual MLP · width ${w} · ${dpt} blocks · ${approxParams(w, dpt).toLocaleString()} parameters`; };
    note();
    segment(seg, (b) => { S.size = b.dataset.size; note(); });
  }

  const stepsEl = $("#steps");
  const sliderFill = (el) => el.style.setProperty("--p", (100 * (el.value - el.min)) / (el.max - el.min) + "%");
  stepsEl.addEventListener("input", () => { $("#steps-val").textContent = (+stepsEl.value).toLocaleString(); sliderFill(stepsEl); });

  async function startTraining() {
    if (S.staticMode) return toast("Training needs the Python server: run python main.py");
    if (currentVehicle().kind !== "point_mass") return toast("Network training needs a thrust-driven plant (Rocket, UAV or USV)");
    let res;
    try {
      res = await api("/api/train", { size: S.size, steps: +stepsEl.value, vehicle: S.vehicle, params: currentParams(), x0: S.x0, goal: S.goal });
    } catch (err) {
      return toast(err.data?.message || err.message || "Could not start training");
    }
    if (!res.ok) return toast(res.message || "Could not start training");
    S.snapshots = []; S.evoLive = true; S.evoIndex = -1;
    toast(`Training ${S.size} model on the ${currentVehicle().name} · ${(+stepsEl.value).toLocaleString()} steps`);
  }
  $("#btn-train").addEventListener("click", startTraining);
  $("#btn-pause").addEventListener("click", async () => {
    const paused = S.server?.phase === "paused";
    await api(paused ? "/api/resume" : "/api/pause", {});
  });
  $("#btn-stop").addEventListener("click", () => api("/api/stop", {}));

  // evolution scrubber
  const evo = $("#evo-slider");
  async function loadEvolution() {
    const r = await api("/api/evolution");
    S.snapshots = r.snapshots || [];
    evo.max = Math.max(0, S.snapshots.length - 1);
    if (S.evoLive) { evo.value = evo.max; }
    sliderFill(evo);
    $("#evo-label").textContent = S.evoLive ? "live" : `step ${S.snapshots[S.evoIndex]?.step ?? 0}`;
  }
  evo.addEventListener("input", () => {
    S.evoLive = +evo.value >= +evo.max && S.server?.phase === "training";
    S.evoIndex = +evo.value; S.evoPlaying = false;
    sliderFill(evo);
    $("#evo-label").textContent = S.evoLive ? "live" : `step ${S.snapshots[S.evoIndex]?.step ?? 0}`;
  });
  $("#evo-play").addEventListener("click", async () => {
    await loadEvolution();
    if (!S.snapshots.length) return toast("Train first to record the evolution");
    S.evoLive = false; S.evoIndex = 0; S.evoPlaying = true;
  });

  // ------------------------------------------------------------------ polling
  let lastPhase = null, lastVersion = -1, pollDelay = 400;
  async function poll() {
    try {
      const st = await api("/api/state");
      S.server = st;
      const firstLoad = !S.cfg.sizes || !Object.keys(S.cfg.sizes).length;
      S.cfg = { ...S.cfg, ...st.config };
      if (!S.catalog && st.config?.catalog) setCatalog(st.config.catalog);
      if (S.staticMode) { S.staticMode = false; requestPlan(true); }
      if (firstLoad) {
        S.size = st.config.size || "large";
        stepsEl.value = st.config.steps || 6000; stepsEl.dispatchEvent(new Event("input"));
        buildSizes();
        requestPlan(true);
      }
      updateStatus(st);
      const training = ["training", "compiling", "paused"].includes(st.phase);
      if (st.version !== lastVersion) {
        lastVersion = st.version;
        if (S.tab === "training" && training && st.snapshots !== S.snapshots.length) loadEvolution();
      }
      if (lastPhase !== st.phase) renderTrainPlant();
      if (lastPhase && lastPhase !== st.phase && st.phase === "done") {
        toast(`Training finished · best step ${st.best_step}`);
        loadEvolution();
        requestPlan(true);
      }
      // keep the simulator's neural plan fresh while weights change
      if (training && S.tab === "simulator" && !drag && st.step % 50 < 12) requestPlan();
      lastPhase = st.phase;
      pollDelay = training ? 250 : 1500;
    } catch {
      if (!S.staticMode) {
        S.staticMode = true;
        if (!S.catalog) {
          try { setCatalog(await (await fetch("vehicles.json")).json()); } catch { /* no catalog: nothing to plan */ }
        }
        renderMethods();
        renderNeural();
        requestPlan(true);
      }
      updateStatus(null);
      pollDelay = 4000;
    }
    setTimeout(poll, pollDelay);
  }

  function updateStatus(st) {
    const el = $("#status"), tx = $("#status-text");
    el.className = "status";
    if (!st) { tx.textContent = S.staticMode ? "Static demo" : "Server offline"; el.classList.add(S.staticMode ? "ready" : "warn"); return; }
    const dev = (st.config?.device || "").toLowerCase().includes("cuda") ? "GPU" : "CPU";
    // Describe the network for the plant selected now, not the last training run.
    const nm = S.neural?.available ? S.neural.model : null;
    const net = S.neural?.live ? "live network" : nm ? `${fmtParams(nm.n_params)} ${currentVehicle().name} network` : "no network for these parameters";
    const map = {
      idle: ["warn", `Ready · ${net}`],
      pretrained: ["ready", `Ready · ${net}`],
      compiling: ["live", `Compiling · ${dev}`],
      training: ["live", `Training on ${dev}`],
      paused: ["warn", "Training paused"],
      done: ["ready", `Ready · ${net}`],
      error: ["warn", "Training error · see terminal"],
    };
    let [cls, text] = map[st.phase] || ["", st.phase];
    if (S.planError && !["compiling", "training", "paused"].includes(st.phase)) { cls = "warn"; text = "Planner error · see Parameters"; }
    el.classList.add(cls); tx.textContent = text;
    el.title = `Selected plant: ${net}. Plans use frozen weights on the CPU (about 1 ms); training runs on the ${dev} (${st.config?.device || "?"}).`;

    const training = ["training", "compiling", "paused"].includes(st.phase);
    $("#btn-train").disabled = training || currentVehicle().kind !== "point_mass";
    $("#btn-pause").disabled = !training || st.phase === "compiling";
    $("#btn-stop").disabled = !training;
    $("#btn-pause").textContent = st.phase === "paused" ? "Resume" : "Pause";
    const h = st.history || {};
    const lastVal = h.val_loss?.[h.val_loss.length - 1];
    const lastDist = h.val_distance?.[h.val_distance.length - 1];
    $("#k-loss").textContent = fmtSig(lastVal);
    $("#k-dist").textContent = fmtSig(lastDist);
    $("#k-rate").textContent = st.rate ? st.rate.toFixed(0) : "—";
    $("#k-params").textContent = st.n_params || st.model?.n_params ? fmtParams(st.n_params || st.model.n_params) : "—";
    renderLog(st);
    $("#progress-bar").style.width = st.total_steps ? `${(100 * st.step) / st.total_steps}%` : st.phase === "pretrained" ? "100%" : "0";
  }

  // ------------------------------------------------------------------ training log table
  // The trainer evaluates every 50 steps; the table samples that history at a
  // coarser interval (always keeping step 0 and the latest row).
  let logEvery = 250, logLastStep = -1;
  function logRows(h) {
    const steps = h.step || [];
    const rows = [];
    steps.forEach((s, i) => { if (s % logEvery === 0 || i === steps.length - 1) rows.push(i); });
    if (rows.length > 1 && steps[rows[rows.length - 1]] === steps[rows[rows.length - 2]]) rows.pop();
    return rows.map((i, k) => {
      const prev = k > 0 ? h.val_loss[rows[k - 1]] : null;
      const v = h.val_loss[i];
      return {
        step: steps[i], time: h.time ? h.time[i] : null, train: h.train_loss?.[i], val: v,
        dval: prev ? (100 * (v - prev)) / prev : null,
        miss: h.val_distance?.[i], fuel: h.val_fuel?.[i], grad: h.grad_norm?.[i], lr: h.lr?.[i],
      };
    });
  }
  const sig = (v, n = 4) => (v == null || !isFinite(v) ? "—" : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : (+v.toPrecision(n)).toString());
  function renderLog(st) {
    const h = st?.history || {};
    const rows = logRows(h);
    const tb = $("#log-table");
    if (!tb) return;
    const head = "<thead><tr><th>Step</th><th>Time (s)</th><th>Train loss</th><th>Val loss</th><th>Δ val</th><th>Mean miss (m)</th><th>Control effort Σu²</th><th>Grad norm</th><th>Learning rate</th></tr></thead>";
    if (!rows.length) { $("#log-sub").textContent = st?.phase === "compiling" ? "compiling…" : "updates live while training"; tb.innerHTML = head + `<tbody><tr><td class="empty" colspan="9">${S.staticMode ? "Training runs need the local Python server (python main.py)." : "No training history yet. Press Train to start a run."}</td></tr></tbody>`; return; }
    const last = rows[rows.length - 1].step;
    const isNew = last !== logLastStep;
    logLastStep = last;
    const body = rows.slice().reverse().map((r, k) => {
      const cls = [k === 0 ? "latest" : "", k === 0 && isNew ? "fresh" : ""].join(" ").trim();
      const d = r.dval == null ? "—" : `${r.dval >= 0 ? "+" : ""}${r.dval.toFixed(1)}%`;
      const dc = r.dval == null ? "" : r.dval < 0 ? "good" : "bad";
      return `<tr class="${cls}"><td>${r.step.toLocaleString()}</td><td>${r.time == null ? "—" : r.time.toFixed(1)}</td><td>${sig(r.train)}</td><td>${sig(r.val)}</td><td class="${dc}">${d}</td><td>${sig(r.miss)}</td><td>${sig(r.fuel)}</td><td>${sig(r.grad, 3)}</td><td>${sig(r.lr, 3)}</td></tr>`;
    }).join("");
    const keep = tb.closest(".log-wrap").scrollTop;
    tb.innerHTML = head + "<tbody>" + body + "</tbody>";
    tb.closest(".log-wrap").scrollTop = keep;
    const live = ["training", "compiling", "paused"].includes(st.phase);
    $("#log-sub").textContent = live ? `live · ${rows.length} rows` : `${rows.length} rows · last run`;
  }
  segment($("#log-every"), (b) => { logEvery = +b.dataset.every; logLastStep = -1; renderLog(S.server); });
  $("#log-csv").addEventListener("click", () => {
    const rows = logRows(S.server?.history || {});
    if (!rows.length) return toast("No training history to export yet");
    const cols = ["step", "time", "train", "val", "dval", "miss", "fuel", "grad", "lr"];
    const hdr = "step,time_s,train_loss,val_loss,delta_val_pct,mean_miss_m,control_effort_sum_u2,grad_norm,learning_rate";
    const csv = [hdr, ...rows.map((r) => cols.map((c) => (r[c] == null ? "" : r[c])).join(","))].join("\n");
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([csv], { type: "text/csv" })), download: `otis-training-log-every-${logEvery}.csv` });
    document.body.appendChild(a); a.click(); a.remove();
  });

  // ------------------------------------------------------------------ main loop
  function resizeAll() { [simCv, trCv, thCv, lossCv].forEach(fitCanvas); }
  function drawAll() {
    if (S.tab === "simulator") { drawSim(); }
    if (S.tab === "training") { drawTrain(); drawLoss(); }
  }
  let lastT = performance.now(), evoAcc = 0;
  function frame(now) {
    const dt = Math.min(0.05, (now - lastT) / 1000);
    lastT = now;
    if (S.playing) {
      S.t += dt * 14; // ~3.6 s to fly a 4 s mission
      if (S.t >= S.cfg.n_steps) { S.t = S.cfg.n_steps; S.playing = false; }
      drawThrustChart();
    }
    if (S.evoPlaying && S.snapshots.length) {
      evoAcc += dt;
      const cur = S.snapshots[S.evoIndex]?.step ?? 0;
      if (evoAcc > (cur <= 40 ? 0.16 : 0.05)) {
        evoAcc = 0;
        S.evoIndex++;
        if (S.evoIndex >= S.snapshots.length) { S.evoIndex = S.snapshots.length - 1; S.evoPlaying = false; }
        evo.value = S.evoIndex; sliderFill(evo);
        $("#evo-label").textContent = `step ${S.snapshots[S.evoIndex].step}`;
      }
    }
    syncSlider();
    drawAll();
    requestAnimationFrame(frame);
  }
  window.addEventListener("resize", () => { resizeAll(); drawThrustChart(); });
  // ------------------------------------------------------------------ parameters panel
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  function setCatalog(cat) {
    S.catalog = cat;
    if (window.OTiSPlant) window.OTiSPlant.setCatalog(cat);
    if (!cat.vehicles[S.vehicle]) S.vehicle = "rocket";
    // Drop any saved edits to hidden fields so every vehicle uses its preset gravity.
    Object.entries(cat.vehicles).forEach(([id, v]) => {
      const hidden = (cat.schema[v.kind] || []).filter((f) => f.hidden).map((f) => f.key);
      if (S.params[id]) hidden.forEach((k) => delete S.params[id][k]);
    });
    persist();
    $$("#vehicle-tabs button").forEach((b) => b.classList.toggle("active", b.dataset.vehicle === S.vehicle));
    window.dispatchEvent(new Event("resize"));
    loadScenario();
    renderMethods();
    buildParamsPanel();
    renderTrainPlant();
    renderScenario();
  }

  let planTimer = null;
  const planSoon = () => { clearTimeout(planTimer); planTimer = setTimeout(() => requestPlan(true), 120); };
  const parseNum = (t) => { const v = Number(String(t).trim().replace(",", ".")); return String(t).trim() === "" ? NaN : v; };

  function buildParamsPanel() {
    const holder = $("#param-groups");
    if (!holder || !S.catalog) return;
    const base = preset(), locked = base.locked || [], cur = currentParams(), edits = S.params[S.vehicle] || {};
    const groups = {};
    schema().forEach((f) => (groups[f.group] = groups[f.group] || []).push(f));
    holder.innerHTML = `<div class="list">` + Object.entries(groups).map(([g, fields]) => `
      <div class="param-sub">${esc(g)}</div>
      <div class="param-list">${fields.map((f) => {
        const isLocked = locked.includes(f.key);
        return `<div class="field" data-key="${f.key}">
          <span class="field-lab" title="${esc(f.help)}${isLocked ? " (fixed for this vehicle)" : ""}">${esc(f.label)}</span>
          <input class="num-in${f.key in edits ? " edited" : ""}" type="text" inputmode="decimal" aria-label="${esc(f.label)}" data-key="${f.key}" value="${fmtIn(cur[f.key])}" ${isLocked ? "disabled" : ""}>
          <span class="unit">${esc(f.unit)}</span>
        </div>`;
      }).join("")}</div>`).join("") + `</div>`;
    holder.querySelectorAll("input[data-key]").forEach((el) => {
      const f = schema().find((x) => x.key === el.dataset.key);
      const row = el.closest(".field");
      const showNote = (msg) => {
        row.querySelector(".field-note")?.remove();
        if (msg) row.insertAdjacentHTML("beforeend", `<div class="field-note">${esc(msg)}</div>`);
      };
      el.addEventListener("input", () => {
        const v = parseNum(el.value);
        const ok = isFinite(v) && v >= f.min && v <= f.max;
        el.classList.toggle("invalid", !ok);
        showNote(ok ? "" : `Enter a number from ${f.min} to ${f.max.toLocaleString()}${f.unit ? " " + f.unit : ""}`);
        if (!ok) return;
        const presetVal = +preset().params[f.key];
        S.params[S.vehicle] = S.params[S.vehicle] || {};
        if (Math.abs(v - presetVal) < 1e-12) delete S.params[S.vehicle][f.key]; else S.params[S.vehicle][f.key] = v;
        el.classList.toggle("edited", f.key in S.params[S.vehicle]);
        if (f.key === "workspace") {
          // keep the scenario inside the new square
          const L = v;
          S.x0[0] = clamp(S.x0[0], -L, L); S.x0[1] = clamp(S.x0[1], -L, L);
          S.goal = [clamp(S.goal[0], -L, L), clamp(S.goal[1], -L, L)];
          renderScenario();
        }
        persist();
        renderTrainPlant(); renderDerived();
        planSoon();
      });
      el.addEventListener("change", () => { if (!el.classList.contains("invalid")) el.value = fmtIn(currentParams()[f.key]); });
      el.addEventListener("keydown", (e) => {
        if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
        e.preventDefault();
        const step = (e.shiftKey ? 10 : 1) * (f.step || 1);
        const v = clamp((parseNum(el.value) || 0) + (e.key === "ArrowUp" ? step : -step), f.min, f.max);
        el.value = fmtIn(+v.toPrecision(10));
        el.dispatchEvent(new Event("input"));
      });
    });
    renderDerived();
  }

  // start / velocity / goal numeric entry
  $$("[data-sc]").forEach((el) => {
    el.addEventListener("input", () => {
      const v = parseNum(el.value);
      const [k, i] = el.dataset.sc.split(".");
      const L = WS();
      const isPos = !(k === "x0" && +i >= 2);
      const ok = isFinite(v) && (!isPos || Math.abs(v) <= L + 1e-9);
      el.classList.toggle("invalid", !ok);
      el.title = ok ? "" : isPos ? `Must lie inside the workspace (±${+L.toPrecision(4)} m)` : "Enter a number";
      if (!ok) return;
      S[k][+i] = v;
      S.t = S.cfg.n_steps;
      saveScenario();
      planSoon();
    });
    el.addEventListener("change", () => renderScenario());
  });

  function renderDerived() {
    const box = $("#derived");
    if (!box) return;
    const d = window.OTiSPlant && S.catalog ? window.OTiSPlant.resolve(S.vehicle, currentParams()).derived : S.spec?.derived;
    if (!d) { box.innerHTML = ""; return; }
    const n = (v, digits = 3) => (v == null || !isFinite(v) ? "—" : +Number(v).toPrecision(digits) + "");
    const rows = [["Time step", `${n(d.dt)} s`]];
    if (currentVehicle().kind === "point_mass") {
      rows.push(["Max acceleration", `${n(d.max_accel)} m/s²`]);
      if (d.thrust_to_weight != null) rows.push(["Thrust-to-weight", n(d.thrust_to_weight)]);
      if (d.top_speed != null) rows.push(["Top speed (drag-limited)", `${n(d.top_speed)} m/s`]);
      rows.push([`Reach in the mission time`, `${n(d.reach)} m`]);
      rows.push(["Stiffness κ", `<span title="Condition number of the planning problem. The published network's plant is ≈170; above ≈10⁴ training needs many more steps.">${d.stiffness >= 1e4 ? Number(d.stiffness).toExponential(1) : Math.round(d.stiffness).toLocaleString()}</span>`]);
    } else {
      rows.push(["Min turn radius", `${n(d.min_turn_radius)} m`]);
      rows.push(["Reach from rest", `${n(d.reach)} m`]);
    }
    const errs = S.planError ? `<div class="note err"><b>Planner error.</b> ${esc(S.planError)}${Object.keys(S.plans || {}).length ? " The map shows the last valid plan." : ""}</div>` : "";
    const notes = (S.notices || []).map((w) => `<div class="note info">${esc(w)}</div>`).join("");
    const warns = errs + notes + (d.warnings || []).filter((w) => w !== S.planError).map((w) => `<div class="note">${esc(w)}</div>`).join("");
    box.innerHTML = rows.map(([l, v]) => `<div class="drow"><span class="dlab">${l}</span><span class="mono">${v}</span></div>`).join("") +
      (warns ? `<div class="notes">${warns}</div>` : "");
  }

  function renderNeural() {
    const box = $("#neural-card");
    if (!box) return;
    const st = S.neural;
    const training = ["training", "compiling", "paused"].includes(S.server?.phase);
    const trainingThis = training && S.server?.plant?.signature && S.server.plant.signature === S.spec?.signature;
    let cls = "off", title = "No network for these parameters", body = "", action = "";
    if (S.planError) { title = "Current parameters not validated"; body = "<p>Planning failed for the current fields. Retained plans are from the last valid request, not evidence of a matching trained network.</p>"; }
    else if (!st) { body = "<p>Waiting for the planner…</p>"; }
    else if (trainingThis) {
      cls = "live"; title = "Training on these parameters";
      body = `<p>The plan updates live from the current weights. Watch it in the Training tab.</p>`;
      action = `<button class="btn" data-go="training">Open Training</button>`;
    } else if (st.available) {
      const m = st.model || {};
      cls = "on"; title = "Network trained for these parameters";
      body = `<p>${m.size ? esc(m.size) + " · " : ""}${m.n_params ? fmtParams(m.n_params) + " parameters · " : ""}${m.step ? "best step " + m.step.toLocaleString() : ""}${m.workspace ? ` · trained on ±${m.workspace} m` : ""}</p>` +
        (st.warning ? `<div class="note">${esc(st.warning)}</div>` : "");
      const ng = networkGap();
      if (ng?.significant) {
        const how = ng.pct != null ? `costs ${ng.pct >= 100 ? Math.round(ng.pct).toLocaleString() : ng.pct.toFixed(1)}% more than the optimum` : `misses by ${fmtSig(ng.miss)} m where the optimum misses by ${fmtSig(ng.optMiss ?? 0)} m`;
        body += `<div class="note">On this mission the network's plan ${how}. The network is under-trained here; the convex plan is the reliable one. Training longer or a larger model narrows the gap.</div>`;
      }
    } else {
      body = `<p>${esc(st.reason || "")}</p>`;
      if (st.trainable && !S.staticMode) {
        body += `<p>Classical planners already run on your parameters. Train to add the network: it learns this plant from scratch in about ${S.size === "small" ? "20" : "40"} s per 3,000 steps on the GPU.</p>`;
        action = `<button class="btn primary" data-train="1" ${training ? "disabled" : ""}>Train on these parameters</button>`;
      }
    }
    box.innerHTML = `<div class="neural-body"><div class="state ${cls}"><i></i>${title}</div>${body}${action}</div>`;
    box.querySelector("[data-train]")?.addEventListener("click", () => { switchTab("training"); startTraining(); });
    box.querySelector("[data-go]")?.addEventListener("click", () => switchTab("training"));
  }

  function switchTab(name) {
    $(`#tabs button[data-tab="${name}"]`)?.click();
  }

  function renderTrainPlant() {
    const box = $("#train-plant");
    if (!box || !S.catalog) return;
    const training = ["training", "compiling", "paused"].includes(S.server?.phase);
    const tp = training && S.server?.plant ? S.server.plant : { vehicle: S.vehicle, name: currentVehicle().name, params: currentParams() };
    const kind = S.catalog.vehicles[tp.vehicle]?.kind;
    const fields = (S.catalog.schema[kind] || []).filter((f) => !f.hidden);
    const chips = fields.map((f) => `<span class="chip"><b>${esc(f.label.replace(/ per axis| half-width| range/g, ""))}</b>${fmtIn(tp.params[f.key])}${f.unit ? " " + esc(f.unit) : ""}</span>`).join("");
    let note = "";
    if (kind !== "point_mass") note = `<div class="note">The ${esc(tp.name)} is planned classically; network training needs a thrust-driven plant (Rocket, UAV or USV).</div>`;
    else if (training) note = `<div class="note info">Training the ${esc(tp.name)}. Change parameters in the Simulator after this run to train another plant.</div>`;
    else {
      const k = S.spec?.derived?.stiffness;
      if (k > 1e4) note = `<div class="note">Stiffness κ ≈ ${k.toExponential(1)}: expect a slower climb to the optimum. A larger fuel weight λ trains faster.</div>`;
    }
    const last = S.server?.plant;
    const lastLine = !training && last?.name && ["done", "error"].includes(S.server?.phase)
      ? `<div class="note info">The map, progress and log show the <b>last run</b>: ${esc(last.name)}${S.server.n_params || S.server.model?.n_params ? " · " + fmtParams(S.server.n_params || S.server.model.n_params) + " parameters" : ""}${S.server.best_step != null ? " · best step " + Number(S.server.best_step).toLocaleString() : ""}${last.signature !== S.spec?.signature ? ", not the plant above" : ""}.</div>` : "";
    box.innerHTML = `<div class="row"><span class="grow">${esc(tp.name)}</span><span class="muted">${training ? "this run" : "next run"}</span></div><div class="plant-summary">${chips}</div>${note || lastLine ? `<div class="notes">${note}${lastLine}</div>` : ""}`;
    $("#btn-train").disabled = training || kind !== "point_mass";
  }

  // reset / export / import
  $("#btn-reset").addEventListener("click", () => {
    delete S.params[S.vehicle];
    delete S.scen[S.vehicle];
    loadScenario(); persist();
    clearTimeout(planTimer);
    buildParamsPanel(); renderScenario(true); renderTrainPlant();
    S.t = 0; S.playing = true;
    requestPlan(true);
    toast(`${currentVehicle().name} preset restored`);
  });
  $("#btn-export").addEventListener("click", () => {
    const doc = { otis: 1, vehicle: S.vehicle, params: currentParams(), scenario: { x0: S.x0, goal: S.goal } };
    const url = URL.createObjectURL(new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: `otis-${S.vehicle}-mission.json` });
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  $("#btn-import").addEventListener("click", () => $("#import-file").click());
  $("#import-file").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    try {
      const doc = JSON.parse(await file.text());
      if (!doc || !S.catalog.vehicles[doc.vehicle]) throw new Error("unknown vehicle");
      const veh = S.catalog.vehicles[doc.vehicle];
      const fields = S.catalog.schema[veh.kind].filter((f) => !f.hidden && !(veh.locked || []).includes(f.key));
      const params = {}, adjusted = [], bad = [];
      fields.forEach((f) => {
        if (!doc.params || !(f.key in doc.params)) return;
        const v = Number(doc.params[f.key]);
        if (doc.params[f.key] === null || doc.params[f.key] === "" || !isFinite(v)) { bad.push(f.label); return; }
        const c = clamp(v, f.min, f.max);
        if (c !== v) adjusted.push(`${f.label} ${+v.toPrecision(6)} → ${+c.toPrecision(6)}`);
        params[f.key] = c;
      });
      if (bad.length) throw new Error(`not a number: ${bad.join(", ")}`);
      const L = +(params.workspace ?? veh.params.workspace);
      const sc = doc.scenario || {};
      let scen = null;
      if (sc.x0 !== undefined || sc.goal !== undefined) {
        if (!(Array.isArray(sc.x0) && sc.x0.length === 4 && Array.isArray(sc.goal) && sc.goal.length === 2 && [...sc.x0, ...sc.goal].every((v) => v !== null && v !== "" && isFinite(+v))))
          throw new Error("scenario needs x0 [x, y, vx, vy] and goal [x, y] as numbers");
        const x0 = sc.x0.map(Number), goal = sc.goal.map(Number);
        [0, 1].forEach((i) => { const c = clamp(x0[i], -L, L); if (c !== x0[i]) adjusted.push(`start ${"xy"[i]} → ${c}`); x0[i] = c; });
        [0, 1].forEach((i) => { const c = clamp(goal[i], -L, L); if (c !== goal[i]) adjusted.push(`goal ${"xy"[i]} → ${c}`); goal[i] = c; });
        scen = { x0, goal };
      }
      S.params[doc.vehicle] = params;
      if (scen) S.scen[doc.vehicle] = scen;
      if (doc.vehicle === S.vehicle) {
        // Same vehicle: apply directly (a tab click would save the old scenario over it).
        if (scen) { S.x0 = scen.x0.slice(); S.goal = scen.goal.slice(); }
        S.spec = null; persist();
        buildParamsPanel(); renderTrainPlant(); renderScenario();
        S.t = S.cfg.n_steps; requestPlan(true);
      } else {
        $(`#vehicle-tabs button[data-vehicle="${doc.vehicle}"]`).click();
      }
      toast(adjusted.length ? `Loaded ${file.name} · adjusted to limits: ${adjusted.join("; ")}` : `Loaded ${file.name}`);
    } catch (err) {
      toast(`Could not read that file: ${err.message}`);
    }
  });
  $$(".linkbtn").forEach((b) => b.addEventListener("mousedown", (e) => e.preventDefault()));

  renderScenario();
  poll();
  requestAnimationFrame(frame);
})();
