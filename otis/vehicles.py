"""User-parameterized vehicle plants and their classical planners.

Presets and the editable parameter schema live in ``web/vehicles.json`` so
the Python server and the browser read the same numbers. ``resolve`` merges
a user's overrides into a preset and returns a *spec*: the parameters in SI
units plus the derived plant coefficients the simulators use.

Point-mass plants (Rocket, UAV, USV)
    v_dot = (F_max / m) * u - (c / m) * v - [0, g],   |u_i| <= 1
    A linear plant, so the status-quo convex program and its exact KKT
    optimum apply to any parameter choice, and the network can be trained
    on it through the differentiable JAX simulator.

Kinematic bicycle (Car)
    x_dot = v cos(psi), y_dot = v sin(psi), psi_dot = v / L * tan(delta),
    v_dot = a_max * u_a - k v
    Nonlinear; planned with pure-pursuit steering (Coulter 1992).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

try:
    from . import config
except ImportError:
    import config


CATALOG_PATH = Path(__file__).resolve().parent / "web" / "vehicles.json"
CATALOG = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
SCHEMA: dict = CATALOG["schema"]
VEHICLES: dict = CATALOG["vehicles"]
N = config.N_STEPS


def substeps_for(drag: float, dt: float) -> int:
    """Same rule as ``simulator.substeps_for`` (kept JAX-free for the planners)."""

    z = max(float(drag), 0.0) * float(dt)
    n = max(1, math.ceil(z / config.MAX_DECAY_PER_SUBSTEP - 1e-12))
    return n if n <= config.MAX_SUBSTEPS else 0


# --------------------------------------------------------------------------
# Parameters -> spec
# --------------------------------------------------------------------------

def resolve(vehicle_id: str | None, overrides: dict | None = None) -> dict:
    """Preset + validated user overrides -> spec with derived coefficients."""

    base = VEHICLES.get(vehicle_id or "rocket") or VEHICLES["rocket"]
    kind = base["kind"]
    params = dict(base["params"])
    clamped = []
    for field in SCHEMA[kind]:
        key = field["key"]
        if not overrides or key not in overrides or key in base.get("locked", []):
            continue
        try:
            value = float(overrides[key])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        bounded = min(max(value, field["min"]), field["max"])
        if bounded != value:
            clamped.append(key)
        params[key] = bounded

    horizon = float(params["horizon"])
    spec = {
        "id": base["id"],
        "name": base["name"],
        "kind": kind,
        "methods": list(base["methods"]),
        "params": params,
        "clamped": clamped,
        "horizon": horizon,
        "dt": horizon / N,
        "workspace": float(params["workspace"]),
        "start_speed": float(params["start_speed"]),
        "lambda_fuel": float(params["lambda_fuel"]),
    }
    if kind == "point_mass":
        spec["gain"] = float(params["max_thrust"]) / float(params["mass"])
        spec["gravity"] = float(params["gravity"])
        spec["drag"] = float(params["drag"]) / float(params["mass"])
    else:
        spec["wheelbase"] = float(params["wheelbase"])
        spec["max_accel"] = float(params["max_accel"])
        spec["max_steer"] = math.radians(float(params["max_steer"]))
        spec["speed_limit"] = float(params["speed_limit"])
        spec["drag"] = float(params["drag"])
    if kind == "point_mass":
        n_sub = substeps_for(spec["drag"], spec["dt"])
        spec["substeps"] = n_sub
        spec["error"] = None if n_sub else (
            f"Drag/mass × time step = {spec['drag'] * spec['dt']:.3g} is too stiff to simulate reliably "
            f"(the limit is {config.MAX_DECAY_PER_SUBSTEP * config.MAX_SUBSTEPS:g}). "
            "Lower the drag or the mission time, or raise the mass."
        )
    else:
        spec["substeps"], spec["error"] = 1, None
    spec["signature"] = signature(spec)
    spec["derived"] = derived_quantities(spec)
    if spec["error"]:
        spec["derived"]["warnings"].append(spec["error"])
    return spec


def plant_tuple(spec: dict) -> tuple:
    """``(accel_gain, gravity, drag)`` for the JAX simulator."""

    return (spec["gain"], spec["gravity"], spec["drag"])


def signature(spec: dict) -> str | None:
    """Identity of the *physics + objective* a network must be trained on.

    Two parameter sets with the same thrust/mass ratio, gravity, drag/mass
    ratio, horizon and fuel weight pose the identical optimization problem,
    so one trained network serves both.
    """

    if spec["kind"] != "point_mass":
        return None
    vals = (spec["gain"], spec["gravity"], spec["drag"], spec["horizon"], spec["lambda_fuel"])
    return "pm:" + ",".join(f"{v:.6g}" for v in vals)


def signature_slug(sig: str) -> str:
    return hashlib.sha1(sig.encode()).hexdigest()[:10]


def derived_quantities(spec: dict) -> dict:
    """Readouts that help a user sanity-check their parameters."""

    T = spec["horizon"]
    L = spec["workspace"]
    out = {"dt": spec["dt"], "workspace_diagonal": 2 * math.sqrt(2) * L}
    if spec["kind"] == "point_mass":
        a, g, k = spec["gain"], spec["gravity"], spec["drag"]
        out["max_accel"] = a
        out["thrust_to_weight"] = a / g if g > 0 else None
        out["top_speed"] = a / k if k > 0 else None
        # Farthest displacement from rest at full throttle on one axis.
        if k > 1e-9:
            reach = (a / k) * (T - (1 - math.exp(-k * T)) / k)
        else:
            reach = 0.5 * a * T * T
        out["reach"] = reach
        out["can_hover"] = g == 0 or a > g
        # Condition number of the planning QP, 1 + |B|^2 / lambda. It sets how
        # hard the fuel/accuracy trade-off is to learn by gradient descent:
        # the published unit plant is ~170; above ~1e4 training slows sharply.
        weights = influence_weights(spec)
        out["stiffness"] = 1.0 + float(weights @ weights) / spec["lambda_fuel"]
        warnings = []
        if g > 0 and a <= g:
            warnings.append("Thrust cannot hold the vehicle up: max acceleration is below gravity.")
        if reach < 2 * L:
            warnings.append(
                f"Reach in {T:g} s is about {reach:.3g} m, less than the {2 * L:.3g} m workspace width: "
                "far goals cannot be reached."
            )
        if out["stiffness"] > 1e4:
            warnings.append(
                f"Stiff problem (κ ≈ {out['stiffness']:.2g}): the network will need many more training "
                "steps to match the optimum. A larger fuel weight λ trains much faster."
            )
        out["warnings"] = warnings
    else:
        v, acc = spec["speed_limit"], spec["max_accel"]
        out["min_turn_radius"] = spec["wheelbase"] / math.tan(spec["max_steer"])
        t_acc = v / acc  # from rest at full throttle, then cruise at the limit
        out["reach"] = v * (T - 0.5 * t_acc) if T > t_acc else 0.5 * acc * T * T
        out["top_speed"] = v
        out["warnings"] = (
            [f"From rest the car covers about {out['reach']:.3g} m in {T:g} s, less than the {2 * L:.3g} m workspace width."]
            if out["reach"] < 2 * L else []
        )
    return out


def vehicle_public(spec_or_base: dict) -> dict:
    keys = ("id", "name", "kind", "draw", "blurb", "methods", "params", "locked")
    return {k: spec_or_base[k] for k in keys if k in spec_or_base}


def catalog_public() -> dict:
    return {"schema": SCHEMA, "vehicles": {k: vehicle_public(v) for k, v in VEHICLES.items()}}


# --------------------------------------------------------------------------
# Point-mass plant (numpy, float64): scoring and classical planners
# --------------------------------------------------------------------------

def _deriv_pm(state, control, gain, gravity, drag):
    return np.array(
        [
            state[2],
            state[3],
            gain * control[0] - drag * state[2],
            gain * control[1] - gravity - drag * state[3],
        ],
        dtype=np.float64,
    )


def point_mass_coefficients(drag, dt):
    """Exact ZOH coefficients: exp(-k h), integral exp, double integral exp."""
    z = drag * dt
    decay = math.exp(-z)
    if abs(z) < 1e-3:
        tau = dt * (1 - z / 2 + z*z / 6 - z**3 / 24 + z**4 / 120)
        q = dt*dt * (0.5 - z / 6 + z*z / 24 - z**3 / 120 + z**4 / 720)
    else:
        tau = -math.expm1(-z) / drag
        q = (dt - tau) / drag
    return decay, tau, q


def rk4_point_mass(state, control, spec):
    """Exact linear-plant control step (legacy name retained for callers).

    Keep the existing parameter-domain guard; no RK4 approximation is needed.
    """
    gain, gravity, drag = spec["gain"], spec["gravity"], spec["drag"]
    if not (spec.get("substeps") or substeps_for(drag, spec["dt"])):
        raise ValueError(spec.get("error") or "plant too stiff to integrate")
    e, tau, q = point_mass_coefficients(drag, spec["dt"])
    control = np.clip(np.asarray(control, dtype=np.float64), -1.0, 1.0)
    acc = gain * control - np.array([0.0, gravity])
    state = np.asarray(state, dtype=np.float64)
    return np.r_[state[:2] + tau * state[2:] + q * acc, e * state[2:] + tau * acc]


def simulate_point_mass(controls, x0, spec):
    state = np.asarray(x0, dtype=np.float64).copy()
    path = [state.copy()]
    for control in np.asarray(controls, dtype=np.float64):
        state = rk4_point_mass(state, control, spec)
        path.append(state.copy())
    return np.stack(path)


_B_CACHE: dict[tuple, np.ndarray] = {}


def influence_weights(spec: dict) -> np.ndarray:
    """Terminal x-displacement caused by unit throttle at each step.

    The plant is linear, so the terminal position is
    ``free_flight(x0) + sum_k b_k u_k`` per axis. ``b`` is measured with the
    same exact transition used for scoring, so the convex program and the scorer agree.
    """

    key = (round(spec["gain"], 12), round(spec["drag"], 12), round(spec["dt"], 12), spec.get("substeps", 1))
    cached = _B_CACHE.get(key)
    if cached is not None:
        return cached
    # The plant is time-invariant, so a unit impulse at step k moves the
    # endpoint by the x-position after one forced step and N-1-k free steps.
    # One rollout gives all N weights, with the same integrator as scoring.
    e, tau, q = point_mass_coefficients(spec["drag"], spec["dt"])
    position, velocity = spec["gain"] * q, spec["gain"] * tau
    xs = [position]
    for _ in range(N - 1):
        position += tau * velocity
        velocity *= e
        xs.append(position)
    weights = np.array(xs[::-1], dtype=np.float64)
    _B_CACHE[key] = weights
    return weights


def residual_target(x0, goal, spec) -> np.ndarray:
    """What thrust must add to free flight (gravity, drag, initial velocity)."""

    free = simulate_point_mass(np.zeros((N, 2)), x0, spec)
    return np.asarray(goal, dtype=np.float64) - free[-1, :2]


def qp_solve_point_mass(x0, goal, spec) -> np.ndarray:
    """Exact optimum of  min |r - B u|^2 + lambda |u|^2,  |u| <= 1  (KKT bisection)."""

    lam = spec["lambda_fuel"]
    residual = residual_target(x0, goal, spec)
    weights = influence_weights(spec)
    lo = np.minimum(0.0, residual)
    hi = np.maximum(0.0, residual)

    def controls_for(scale):
        return np.clip(weights[None, :] * scale[:, None] / lam, -1.0, 1.0)

    for _ in range(100):
        scale = 0.5 * (lo + hi)
        gap = scale + controls_for(scale) @ weights - residual
        hi = np.where(gap > 0, scale, hi)
        lo = np.where(gap > 0, lo, scale)
    return controls_for(0.5 * (lo + hi)).T.copy()


def e_guidance_point_mass(x0, goal, spec) -> np.ndarray:
    """Apollo-heritage E-guidance: a = 3 ZEM / t_go^2, plus gravity and drag
    compensation, converted to throttle and clipped to the thrust limit."""

    dt, horizon = spec["dt"], spec["horizon"]
    gain, gravity, drag = spec["gain"], spec["gravity"], spec["drag"]
    state = np.asarray(x0, dtype=np.float64).copy()
    goal = np.asarray(goal, dtype=np.float64)
    controls = []
    for step in range(N):
        t_go = horizon - step * dt
        zem = goal - (state[:2] + state[2:] * t_go)
        accel = 3.0 * zem / (t_go * t_go) + np.array([0.0, gravity]) + drag * state[2:]
        command = np.clip(accel / gain, -1.0, 1.0)
        controls.append(command)
        state = rk4_point_mass(state, command, spec)
    return np.stack(controls)


# --------------------------------------------------------------------------
# Kinematic bicycle
# --------------------------------------------------------------------------

def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def bicycle_from_x0(x0, goal):
    x, y, vx, vy = [float(v) for v in np.asarray(x0, dtype=np.float64)[:4]]
    speed = math.hypot(vx, vy)
    heading = math.atan2(vy, vx) if speed > 1e-6 else math.atan2(float(goal[1]) - y, float(goal[0]) - x)
    return np.array([x, y, heading, speed], dtype=np.float64)


def _to_xyv(state):
    x, y, heading, speed = state
    return np.array([x, y, speed * math.cos(heading), speed * math.sin(heading)], dtype=np.float64)


def _deriv_bicycle(state, control, spec):
    _x, _y, heading, speed = state
    accel = float(np.clip(control[0], -1.0, 1.0)) * spec["max_accel"] - spec["drag"] * speed
    steer = float(np.clip(control[1], -1.0, 1.0)) * spec["max_steer"]
    return np.array(
        [
            speed * math.cos(heading),
            speed * math.sin(heading),
            (speed / spec["wheelbase"]) * math.tan(steer),
            accel,
        ],
        dtype=np.float64,
    )


def rk4_bicycle(state, control, spec):
    dt = spec["dt"]
    k1 = _deriv_bicycle(state, control, spec)
    k2 = _deriv_bicycle(state + 0.5 * dt * k1, control, spec)
    k3 = _deriv_bicycle(state + 0.5 * dt * k2, control, spec)
    k4 = _deriv_bicycle(state + dt * k3, control, spec)
    new = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    new[3] = max(new[3], 0.0)  # no reversing
    return new


def simulate_bicycle(controls, x0, spec, goal):
    state = bicycle_from_x0(x0, goal)
    path = [_to_xyv(state)]
    for control in np.asarray(controls, dtype=np.float64):
        state = rk4_bicycle(state, control, spec)
        path.append(_to_xyv(state))
    return np.stack(path)


def pursuit_car(x0, goal, spec) -> np.ndarray:
    """Pure-pursuit steering toward the goal plus a time-to-go speed profile.

    Steering: delta = atan(2 L sin(alpha) / d), the pure-pursuit curvature
    with the goal as the look-ahead point. A goal behind the car gets full
    lock toward it, unless it sits inside the turning circle, in which case
    the car drives straight to open up room first. Speed: cover the
    remaining distance in the remaining time, capped by the speed limit.
    (A kinematic bicycle's turn radius does not depend on speed, so there
    is no reason to slow for turns.)
    """

    dt, horizon = spec["dt"], spec["horizon"]
    state = bicycle_from_x0(x0, goal)
    goal = np.asarray(goal, dtype=np.float64)
    radius = spec["wheelbase"] / math.tan(spec["max_steer"])
    controls = []
    for step in range(N):
        t_go = max(horizon - step * dt, dt)
        dx, dy = goal[0] - state[0], goal[1] - state[1]
        distance = math.hypot(dx, dy)
        alpha = _wrap(math.atan2(dy, dx) - state[2])
        if abs(alpha) <= math.pi / 2:
            delta = math.atan2(2.0 * spec["wheelbase"] * math.sin(alpha), max(distance, 1e-6))
        elif distance < 2 * radius * abs(math.sin(alpha)):
            delta = 0.0
        else:
            delta = math.copysign(spec["max_steer"], alpha)
        target_speed = min(spec["speed_limit"], distance / t_go)
        tau = max(4 * dt, 0.3)  # speed-loop time constant
        accel = float(np.clip((target_speed - state[3]) / (tau * spec["max_accel"]), -1.0, 1.0))
        steer = float(np.clip(delta / spec["max_steer"], -1.0, 1.0))
        control = np.array([accel, steer], dtype=np.float64)
        controls.append(control)
        state = rk4_bicycle(state, control, spec)
    return np.stack(controls)


# --------------------------------------------------------------------------
# Scoring and the planning entry point
# --------------------------------------------------------------------------

def score_plan(controls, x0, goal, spec) -> dict:
    controls = np.clip(np.asarray(controls, dtype=np.float64), -1.0, 1.0)
    if spec["kind"] == "bicycle":
        path = simulate_bicycle(controls, x0, spec, goal)
        delta_v = float(np.sum(np.abs(controls[:, 0])) * spec["max_accel"] * spec["dt"])
    else:
        path = simulate_point_mass(controls, x0, spec)
        delta_v = float(np.sum(np.linalg.norm(controls, axis=-1)) * spec["gain"] * spec["dt"])
    goal = np.asarray(goal, dtype=np.float64)
    miss = float(np.linalg.norm(path[-1, :2] - goal))
    fuel = float(np.sum(controls ** 2))
    return {
        "controls": _sig(controls, digits=12),
        "path": _sig(path),
        "loss": miss * miss + spec["lambda_fuel"] * fuel,
        "miss": miss,
        "fuel": fuel,
        "delta_v": delta_v,
    }


def _sig(arr, digits=9):
    """Round to significant digits (not decimals) so tiny values keep precision."""

    return [[float(f"{v:.{digits}g}") for v in row] for row in np.asarray(arr, dtype=np.float64)]


def _best_of(fn, n=3):
    """Best-of-n wall time in ms (filters HTTP-thread jitter)."""

    best, result = float("inf"), None
    for _ in range(n):
        started = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - started)
    return result, 1e3 * best


def plan_vehicle(spec, x0, goal, methods=None, neural_fn=None, convex=None) -> dict:
    """Plan one mission with every applicable method and score each plan
    in the same simulator with the same objective.

    ``neural_fn(x0, goal) -> run`` is supplied by the server only when a
    network trained on this exact plant exists; it stages the inputs on the
    device and returns a zero-argument ``run()`` so only inference is timed.
    ``convex`` is a ``baselines.ConvexGuidance`` built for this plant
    (CVXPY + Clarabel); if absent, the KKT solution stands in.
    """

    chosen = set(methods or spec["methods"]) & set(spec["methods"])
    x0 = np.asarray(x0, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    payload = {"x0": np.round(x0, 6).tolist(), "goal": np.round(goal, 6).tolist(), "plans": {}}

    def pack(name, controls, ms, extra=None):
        packed = score_plan(controls, x0, goal, spec)
        packed["ms"] = ms
        packed.update(extra or {})
        payload["plans"][name] = packed
        return packed

    if spec["kind"] == "point_mass":
        if "neural" in chosen and neural_fn is not None:
            run = neural_fn(x0, goal)
            run()  # warm
            controls, ms = _best_of(run)
            pack("neural", np.asarray(controls), ms)
        optimum = qp_solve_point_mass(x0, goal, spec)
        payload["optimal_loss"] = score_plan(optimum, x0, goal, spec)["loss"]
        if "convex" in chosen:
            solved = False
            if convex is not None:
                residual = residual_target(x0, goal, spec)
                try:
                    (controls, _sec), ms = _best_of(lambda: convex.solve_residual(residual))
                    pack("convex", controls, ms, {"solver": "CVXPY + Clarabel"})
                    solved = True
                except Exception as exc:  # numerical failure: report it, fall back to the exact optimum
                    payload.setdefault("notices", []).append(
                        f"Clarabel could not solve this mission ({exc}); the convex row shows the exact KKT optimum instead.")
            if solved:
                pass
            else:
                controls, ms = _best_of(lambda: qp_solve_point_mass(x0, goal, spec))
                pack("convex", controls, ms, {"solver": "KKT transcription"})
        if "e_guidance" in chosen:
            controls, ms = _best_of(lambda: e_guidance_point_mass(x0, goal, spec))
            pack("e_guidance", controls, ms)
    elif "pursuit" in chosen:
        controls, ms = _best_of(lambda: pursuit_car(x0, goal, spec))
        pack("pursuit", controls, ms, {"solver": "pure pursuit"})
    return payload
