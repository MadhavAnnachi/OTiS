/* OTiS client-side plants: a line-for-line mirror of otis/vehicles.py, used
   when the Python server is offline (GitHub Pages). Presets and the
   parameter schema come from vehicles.json, the same file Python reads.
   The network is not available here; the classical planners are. */
(() => {
  "use strict";
  const N = 50;
  // Legacy stiffness-domain guard, same admission rule as otis/config.py.
  // Exact point-mass propagation below no longer needs RK4 sub-steps.
  const MAX_DECAY = 0.5, MAX_SUB = 64;
  const substepsFor = (drag, dt) => {
    const n = Math.max(1, Math.ceil(Math.max(drag, 0) * dt / MAX_DECAY - 1e-12));
    return n <= MAX_SUB ? n : 0;
  };
  let CATALOG = null;

  const clip = (v, a, b) => Math.max(a, Math.min(b, v));
  const wrap = (a) => { let t = (a + Math.PI) % (2 * Math.PI); if (t < 0) t += 2 * Math.PI; return t - Math.PI; };
  const add = (a, b, k) => a.map((v, i) => v + k * b[i]);
  const rk4 = (s, f, dt) => {
    const k1 = f(s), k2 = f(add(s, k1, dt / 2)), k3 = f(add(s, k2, dt / 2)), k4 = f(add(s, k3, dt));
    return s.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
  };

  // ------------------------------------------------------------ parameters -> spec
  function resolve(vehicleId, overrides) {
    const base = CATALOG.vehicles[vehicleId] || CATALOG.vehicles.rocket;
    const kind = base.kind;
    const params = { ...base.params };
    const clamped = [];
    for (const f of CATALOG.schema[kind]) {
      if (!overrides || !(f.key in overrides) || (base.locked || []).includes(f.key)) continue;
      const v = Number(overrides[f.key]);
      if (!isFinite(v) || overrides[f.key] === "" || overrides[f.key] === null) continue;
      const b = clip(v, f.min, f.max);
      if (b !== v) clamped.push(f.key);
      params[f.key] = b;
    }
    const horizon = +params.horizon;
    const spec = {
      id: base.id, name: base.name, kind, methods: base.methods.slice(), params, clamped,
      horizon, dt: horizon / N, workspace: +params.workspace, start_speed: +params.start_speed, lambda_fuel: +params.lambda_fuel,
    };
    if (kind === "point_mass") {
      spec.gain = params.max_thrust / params.mass;
      spec.gravity = +params.gravity;
      spec.drag = params.drag / params.mass;
    } else {
      spec.wheelbase = +params.wheelbase;
      spec.max_accel = +params.max_accel;
      spec.max_steer = (params.max_steer * Math.PI) / 180;
      spec.speed_limit = +params.speed_limit;
      spec.drag = +params.drag;
    }
    spec.substeps = kind === "point_mass" ? substepsFor(spec.drag, spec.dt) : 1;
    spec.error = spec.substeps ? null : `Drag/mass × time step = ${+(spec.drag * spec.dt).toPrecision(3)} is too stiff to simulate reliably (the limit is ${MAX_DECAY * MAX_SUB}). Lower the drag or the mission time, or raise the mass.`;
    spec.signature = kind === "point_mass" ? "pm:" + [spec.gain, spec.gravity, spec.drag, spec.horizon, spec.lambda_fuel].map((v) => +v.toPrecision(6)).join(",") : null;
    spec.derived = derived(spec);
    if (spec.error) spec.derived.warnings.push(spec.error);
    return spec;
  }

  function derived(spec) {
    const T = spec.horizon, L = spec.workspace;
    const out = { dt: spec.dt, workspace_diagonal: 2 * Math.SQRT2 * L, warnings: [] };
    if (spec.kind === "point_mass") {
      const a = spec.gain, g = spec.gravity, k = spec.drag;
      out.max_accel = a;
      out.thrust_to_weight = g > 0 ? a / g : null;
      out.top_speed = k > 0 ? a / k : null;
      out.reach = k > 1e-9 ? (a / k) * (T - (1 - Math.exp(-k * T)) / k) : 0.5 * a * T * T;
      out.can_hover = g === 0 || a > g;
      const w = influence(spec);
      out.stiffness = 1 + w.reduce((s, v) => s + v * v, 0) / spec.lambda_fuel;
      if (g > 0 && a <= g) out.warnings.push("Thrust cannot hold the vehicle up: max acceleration is below gravity.");
      if (out.reach < 2 * L) out.warnings.push(`Reach in ${+T.toPrecision(6)} s is about ${+out.reach.toPrecision(3)} m, less than the ${+(2 * L).toPrecision(3)} m workspace width: far goals cannot be reached.`);
      if (out.stiffness > 1e4) out.warnings.push(`Stiff problem (κ ≈ ${out.stiffness.toPrecision(2)}): the network will need many more training steps to match the optimum. A larger fuel weight λ trains much faster.`);
    } else {
      const v = spec.speed_limit, acc = spec.max_accel, tAcc = v / acc;
      out.min_turn_radius = spec.wheelbase / Math.tan(spec.max_steer);
      out.reach = T > tAcc ? v * (T - 0.5 * tAcc) : 0.5 * acc * T * T;
      out.top_speed = v;
      if (out.reach < 2 * L) out.warnings.push(`From rest the car covers about ${+out.reach.toPrecision(3)} m in ${+T.toPrecision(6)} s, less than the ${+(2 * L).toPrecision(3)} m workspace width.`);
    }
    return out;
  }

  // ------------------------------------------------------------ point mass
  // Exact zero-order-hold transition; Taylor form avoids tiny-drag cancellation.
  function pmCoefficients(drag, dt) {
    const z = drag * dt, e = Math.exp(-z);
    const tau = Math.abs(z) < 1e-3 ? dt * (1 - z/2 + z*z/6 - z**3/24 + z**4/120) : -Math.expm1(-z) / drag;
    const q = Math.abs(z) < 1e-3 ? dt*dt * (0.5 - z/6 + z*z/24 - z**3/120 + z**4/720) : (dt - tau) / drag;
    return [e, tau, q];
  }
  function stepPm(s, u, spec) {
    if (!(spec.substeps || substepsFor(spec.drag, spec.dt))) throw new Error(spec.error || "plant too stiff to integrate");
    const [e, tau, q] = pmCoefficients(spec.drag, spec.dt);
    const ax = spec.gain * clip(u[0], -1, 1), ay = spec.gain * clip(u[1], -1, 1) - spec.gravity;
    return [s[0] + tau*s[2] + q*ax, s[1] + tau*s[3] + q*ay, e*s[2] + tau*ax, e*s[3] + tau*ay];
  }
  function simulatePm(controls, x0, spec) {
    let s = x0.slice(0, 4).map(Number);
    const path = [s.slice()];
    for (const u of controls) { s = stepPm(s, u, spec); path.push(s.slice()); }
    return path;
  }
  const bCache = {};
  function influence(spec) {
    const key = [spec.gain, spec.drag, spec.dt].map((v) => v.toPrecision(12)).join("|");
    if (bCache[key]) return bCache[key];
    // Time-invariant plant: impulse at step k = one forced step + (N-1-k) free steps.
    const [e, tau, q] = pmCoefficients(spec.drag, spec.dt);
    let p = spec.gain * q, v = spec.gain * tau;
    const xs = [p];
    for (let j = 1; j < N; j++) { p += tau * v; v *= e; xs.push(p); }
    return (bCache[key] = xs.reverse());
  }
  function residual(x0, goal, spec) {
    const free = simulatePm(Array.from({ length: N }, () => [0, 0]), x0, spec)[N];
    return [goal[0] - free[0], goal[1] - free[1]];
  }
  function qpSolve(x0, goal, spec) {
    const lam = spec.lambda_fuel, r = residual(x0, goal, spec), w = influence(spec);
    const lo = [Math.min(0, r[0]), Math.min(0, r[1])], hi = [Math.max(0, r[0]), Math.max(0, r[1])];
    const ctrl = (s, a) => w.map((bk) => clip((bk * s[a]) / lam, -1, 1));
    for (let it = 0; it < 100; it++) {
      const s = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
      for (let a = 0; a < 2; a++) {
        const h = s[a] + ctrl(s, a).reduce((acc, uk, i) => acc + uk * w[i], 0) - r[a];
        if (h > 0) hi[a] = s[a]; else lo[a] = s[a];
      }
    }
    const s = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
    const ux = ctrl(s, 0), uy = ctrl(s, 1);
    return ux.map((v, i) => [v, uy[i]]);
  }
  function eGuidance(x0, goal, spec) {
    const { dt, horizon, gain, gravity, drag } = spec;
    let s = x0.slice(0, 4).map(Number);
    const controls = [];
    for (let k = 0; k < N; k++) {
      const tgo = horizon - k * dt;
      const ax = (3 * (goal[0] - (s[0] + s[2] * tgo))) / (tgo * tgo) + drag * s[2];
      const ay = (3 * (goal[1] - (s[1] + s[3] * tgo))) / (tgo * tgo) + gravity + drag * s[3];
      const u = [clip(ax / gain, -1, 1), clip(ay / gain, -1, 1)];
      controls.push(u);
      s = stepPm(s, u, spec);
    }
    return controls;
  }

  // ------------------------------------------------------------ bicycle
  function bikeFromX0(x0, goal) {
    const [x, y, vx, vy] = x0.map(Number);
    const speed = Math.hypot(vx, vy);
    return [x, y, speed > 1e-6 ? Math.atan2(vy, vx) : Math.atan2(goal[1] - y, goal[0] - x), speed];
  }
  const toXyv = (s) => [s[0], s[1], s[3] * Math.cos(s[2]), s[3] * Math.sin(s[2])];
  function stepBike(s, u, spec) {
    const acc = clip(u[0], -1, 1) * spec.max_accel, steer = clip(u[1], -1, 1) * spec.max_steer;
    const n = rk4(s, (x) => [x[3] * Math.cos(x[2]), x[3] * Math.sin(x[2]), (x[3] / spec.wheelbase) * Math.tan(steer), acc - spec.drag * x[3]], spec.dt);
    n[3] = Math.max(n[3], 0);
    return n;
  }
  function simulateBike(controls, x0, spec, goal) {
    let s = bikeFromX0(x0, goal);
    const path = [toXyv(s)];
    for (const u of controls) { s = stepBike(s, u, spec); path.push(toXyv(s)); }
    return path;
  }
  function pursuit(x0, goal, spec) {
    const { dt, horizon } = spec;
    const radius = spec.wheelbase / Math.tan(spec.max_steer);
    let s = bikeFromX0(x0, goal);
    const controls = [];
    for (let k = 0; k < N; k++) {
      const tgo = Math.max(horizon - k * dt, dt);
      const dx = goal[0] - s[0], dy = goal[1] - s[1], d = Math.hypot(dx, dy);
      const alpha = wrap(Math.atan2(dy, dx) - s[2]);
      let delta;
      if (Math.abs(alpha) <= Math.PI / 2) delta = Math.atan2(2 * spec.wheelbase * Math.sin(alpha), Math.max(d, 1e-6));
      else if (d < 2 * radius * Math.abs(Math.sin(alpha))) delta = 0;
      else delta = Math.sign(alpha) * spec.max_steer;
      const target = Math.min(spec.speed_limit, d / tgo);
      const tau = Math.max(4 * dt, 0.3);
      const u = [clip((target - s[3]) / (tau * spec.max_accel), -1, 1), clip(delta / spec.max_steer, -1, 1)];
      controls.push(u);
      s = stepBike(s, u, spec);
    }
    return controls;
  }

  // ------------------------------------------------------------ scoring + entry point
  function score(controls, x0, goal, spec) {
    controls = controls.map((u) => [clip(u[0], -1, 1), clip(u[1], -1, 1)]);
    const path = spec.kind === "bicycle" ? simulateBike(controls, x0, spec, goal) : simulatePm(controls, x0, spec);
    const last = path[path.length - 1];
    const miss = Math.hypot(last[0] - goal[0], last[1] - goal[1]);
    const fuel = controls.reduce((a, u) => a + u[0] * u[0] + u[1] * u[1], 0);
    const delta_v = spec.kind === "bicycle"
      ? controls.reduce((a, u) => a + Math.abs(u[0]), 0) * spec.max_accel * spec.dt
      : controls.reduce((a, u) => a + Math.hypot(u[0], u[1]), 0) * spec.gain * spec.dt;
    return { controls, path, miss, fuel, delta_v, loss: miss * miss + spec.lambda_fuel * fuel, ms: 0 };
  }
  const timed = (fn) => { const t0 = performance.now(); const r = fn(); return [r, performance.now() - t0]; };

  function planLocal(vehicleId, x0, goal, overrides) {
    const spec = resolve(vehicleId, overrides);
    const L = spec.workspace;
    x0 = x0.slice(0, 4).map(Number);
    x0[0] = clip(x0[0], -L, L); x0[1] = clip(x0[1], -L, L);
    goal = [clip(+goal[0], -L, L), clip(+goal[1], -L, L)];
    const plans = {};
    let optimal = null;
    if (spec.error) {
      return { x0, goal, plans, optimal_loss: null, error: spec.error,
        spec: { params: spec.params, horizon: spec.horizon, dt: spec.dt, workspace: spec.workspace, start_speed: spec.start_speed, lambda_fuel: spec.lambda_fuel, signature: spec.signature, derived: spec.derived, clamped: spec.clamped },
        neural_status: { available: false, trainable: false, reason: spec.error } };
    }
    if (spec.kind === "point_mass") {
      optimal = score(qpSolve(x0, goal, spec), x0, goal, spec).loss;
      const [u, ms] = timed(() => qpSolve(x0, goal, spec));
      plans.convex = { ...score(u, x0, goal, spec), ms, solver: "KKT transcription" };
      const [ue, mse] = timed(() => eGuidance(x0, goal, spec));
      plans.e_guidance = { ...score(ue, x0, goal, spec), ms: mse };
    } else {
      const [u, ms] = timed(() => pursuit(x0, goal, spec));
      plans.pursuit = { ...score(u, x0, goal, spec), ms, solver: "pure pursuit" };
    }
    return {
      x0, goal, plans, optimal_loss: optimal,
      spec: { params: spec.params, horizon: spec.horizon, dt: spec.dt, workspace: spec.workspace, start_speed: spec.start_speed, lambda_fuel: spec.lambda_fuel, signature: spec.signature, derived: spec.derived, clamped: spec.clamped },
      neural_status: { available: false, trainable: false, reason: "The network needs the Python server (python main.py). This static copy runs the classical planners in your browser." },
    };
  }

  window.OTiSPlant = {
    N,
    setCatalog(cat) { CATALOG = cat; },
    get catalog() { return CATALOG; },
    resolve: (id, o) => resolve(id, o),
    planLocal,
  };
})();
