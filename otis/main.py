"""Entry point for OTiS (Optimized Trajectory Simulator).

``python main.py`` starts the interactive web simulator at
http://localhost:8765 (training runs on a background thread, driven from the
browser). ``--no-viz`` trains in the foreground, prints the five-case demo,
and writes a summary figure. ``--benchmark`` compares the trained network
with industry baselines and writes BENCHMARK.md. ``--legacy-window`` opens
the old Matplotlib window.

``--check`` runs the physics identity, a finite-difference gradient check,
a short train, and a headless animation frame.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import traceback
from pathlib import Path

# JAX grabs 75% of GPU memory up front by default; allocate on demand instead
# so the UI server, a benchmark run and the desktop can share the 16 GB.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from . import config
except ImportError:
    import config


PACKAGE_ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="OTiS: train a vehicle controller by differentiating through RK4."
    )
    parser.add_argument("--steps", type=int, default=config.TRAIN_STEPS, help=f"Optimizer updates (default {config.TRAIN_STEPS}).")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help=f"Problems per update (default {config.BATCH_SIZE}).")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Adam learning rate.")
    parser.add_argument("--lambda-fuel", type=float, default=config.LAMBDA_FUEL, help="Weight on sum of squared thrust.")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--position-limit",
        type=float,
        default=config.POSITION_LIMIT,
        help="Half-width of the start/goal square. Default 2 stays inside the 4s reachable set.",
    )
    parser.add_argument("--velocity-limit", type=float, default=config.VELOCITY_LIMIT,
                        help="Initial velocity is uniform in +/- this per axis (0 = start at rest).")
    parser.add_argument("--size", choices=sorted(config.MODEL_SIZES), default=config.DEFAULT_SIZE,
                        help="Network size preset (default large, ~13M parameters).")
    parser.add_argument("--benchmark", action="store_true",
                        help="Compare the checkpoint with convex optimization and E-guidance; write BENCHMARK.md.")
    parser.add_argument("--export-benchmark", action="store_true",
                        help="Write exports/benchmark.html (standalone, editable) from the last benchmark run, then exit.")
    parser.add_argument("--port", type=int, default=8765, help="Web UI port.")
    parser.add_argument("--host", default="127.0.0.1", help="Web UI bind address.")
    parser.add_argument("--autotrain", action="store_true", help="Start training as soon as the UI server starts.")
    parser.add_argument("--legacy-window", action="store_true", help="Old Matplotlib live window instead of the web UI.")
    parser.add_argument(
        "--write-benchmark",
        action="store_true",
        help="Write the judge-facing start/goal benchmark JSON, then exit.",
    )
    parser.add_argument("--log-every", type=int, default=config.LOG_EVERY)
    parser.add_argument("--no-viz", action="store_true", help="Train without a live window.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Physics, gradient, short-train, and headless animation checks.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path for the best-parameter bytes.")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory for checkpoints and figures.")
    parser.add_argument("--demo-cases", type=int, default=config.DEMO_CASES)
    return parser.parse_args(argv)


def should_visualize(args) -> bool:
    """True only when a live window can actually open."""

    if args.no_viz or args.check:
        return False
    backend = os.environ.get("MPLBACKEND", "").strip().lower()
    if backend in {"agg", "svg", "pdf", "ps", "template"}:
        return False
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return False
    return True


def _configure_backend(live: bool) -> None:
    import matplotlib

    if not live:
        matplotlib.use("Agg", force=True)
        return
    current = str(matplotlib.get_backend()).lower()
    if current in {"agg", "svg", "pdf", "ps", "template"} or sys.platform.startswith("win"):
        for name in ("TkAgg", "QtAgg", "Qt5Agg"):
            try:
                matplotlib.use(name, force=True)
                break
            except Exception:
                continue


def _banner(args, device: str, reach: float) -> None:
    diagonal = 2.0 * args.position_limit * math.sqrt(2.0)
    horizon = config.N_STEPS * config.DT
    print("OTiS", flush=True)
    print(f"  device: {device}", flush=True)
    print(
        f"  horizon: {config.N_STEPS} steps x {config.DT:.2f} s = {horizon:.2f} s",
        flush=True,
    )
    print(
        f"  thrust limit: ±{config.MAX_THRUST:.1f}    reachable range: {reach:.3f}",
        flush=True,
    )
    print(
        f"  workspace: [{-args.position_limit:.1f}, {args.position_limit:.1f}]^2"
        f"    diagonal {diagonal:.3f}",
        flush=True,
    )
    width, depth = config.MODEL_SIZES[args.size]
    print(
        f"  policy: {args.size} residual MLP, width {width}, {depth} blocks -> "
        f"{config.N_STEPS * 2} outputs (tanh, then scale)",
        flush=True,
    )
    print(f"  initial velocity: +/-{args.velocity_limit:g} per axis", flush=True)
    print(
        f"  train: {args.steps} steps, batch {args.batch_size}, "
        f"Adam lr {args.lr:g}, fuel weight {args.lambda_fuel:g}",
        flush=True,
    )
    print(f"  seed: {args.seed}", flush=True)
    if diagonal > reach + 1e-6:
        print(
            "  warning: the workspace diagonal is longer than the craft can fly. "
            "Some goals cannot be reached. Lower --position-limit or the 0.5 target will fail.",
            flush=True,
        )
    print(flush=True)


def _paths(args):
    save_dir = Path(args.save_dir) if args.save_dir else PACKAGE_ROOT
    checkpoint = Path(args.checkpoint) if args.checkpoint else save_dir / "checkpoints" / "best_params.pkl"
    report = save_dir / "outputs" / "demo.png"
    return checkpoint, report


def _load_runtime():
    import jax

    try:
        from .simulator import max_travel_distance
        from .train import evaluate_cases, train
        from .visualize import TrainingMonitor, save_report
    except ImportError:
        from simulator import max_travel_distance
        from train import evaluate_cases, train
        from visualize import TrainingMonitor, save_report

    return jax, max_travel_distance, train, evaluate_cases, TrainingMonitor, save_report


def _print_summary(cases, probe, result) -> None:
    print("\nFinal demo", flush=True)
    distances = []
    fuels = []
    for i, case in enumerate(cases, start=1):
        distances.append(case["distance"])
        fuels.append(case["fuel"])
        mark = "ok" if case["distance"] < 0.5 else "miss"
        print(
            f"  case {i}: distance {case['distance']:.4f}   fuel {case['fuel']:.3f}   {mark}",
            flush=True,
        )
    n_ok = sum(d < 0.5 for d in distances)
    print(
        f"Average terminal error (L2 distance): {sum(distances) / len(distances):.4f}",
        flush=True,
    )
    print(
        f"Average fuel consumption (sum of squared thrust): {sum(fuels) / len(fuels):.3f}",
        flush=True,
    )
    print(f"Within 0.5: {n_ok}/{len(cases)}", flush=True)
    if probe:
        probe_ok = sum(c["distance"] < 0.5 for c in probe)
        probe_mean = sum(c["distance"] for c in probe) / len(probe)
        print(
            f"Held-out {len(probe)}: mean distance {probe_mean:.4f}   within 0.5: {probe_ok}/{len(probe)}",
            flush=True,
        )
    print(f"Total training time: {result.elapsed_sec:.1f} s", flush=True)
    print(
        f"Best checkpoint: step {result.best_step}   "
        f"val loss {result.best_val_loss:.4f}   val distance {result.best_val_distance:.3f}",
        flush=True,
    )
    if result.history["val_distance"]:
        print(
            f"Val distance: {result.history['val_distance'][0]:.3f} -> {result.history['val_distance'][-1]:.3f}",
            flush=True,
        )


def _run_demo(result, args, evaluate_cases):
    cases = evaluate_cases(
        result.rollout,
        result.best_params,
        n=args.demo_cases,
        seed=args.seed,
        position_limit=args.position_limit,
        velocity_limit=args.velocity_limit,
        stream=1,
    )
    probe = evaluate_cases(
        result.rollout,
        result.best_params,
        n=config.HELDOUT_CASES,
        seed=args.seed,
        position_limit=args.position_limit,
        velocity_limit=args.velocity_limit,
        stream=2,
    )
    return cases, probe


def run_headless(args) -> None:
    jax, max_travel_distance, train, evaluate_cases, _monitor, save_report = _load_runtime()
    checkpoint, report = _paths(args)
    _banner(args, str(jax.devices()[0]), max_travel_distance())
    result = train(
        n_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lambda_fuel=args.lambda_fuel,
        seed=args.seed,
        position_limit=args.position_limit,
        velocity_limit=args.velocity_limit,
        size=args.size,
        log_every=args.log_every,
        checkpoint_path=checkpoint,
    )
    cases, probe = _run_demo(result, args, evaluate_cases)
    _print_summary(cases, probe, result)
    save_report(
        report,
        result.history,
        cases,
        args.position_limit,
        note=f"best step {result.best_step}   val loss {result.best_val_loss:.4f}",
    )
    print(f"Wrote {report}", flush=True)
    print(f"Wrote {checkpoint}", flush=True)


def run_live(args) -> None:
    import numpy as np

    jax, max_travel_distance, train, evaluate_cases, TrainingMonitor, save_report = _load_runtime()
    checkpoint, report = _paths(args)
    _banner(args, str(jax.devices()[0]), max_travel_distance())
    scenario_box: dict = {}
    params_box: dict = {}
    rng = np.random.default_rng(args.seed + 17)

    def _clip_xy(xy):
        lim = float(args.position_limit)
        return np.array([np.clip(xy[0], -lim, lim), np.clip(xy[1], -lim, lim)], dtype=np.float32)

    def _publish_plan(x0, goal, phase="playground"):
        rollout_fn = params_box.get("rollout")
        params = params_box.get("params")
        if rollout_fn is None or params is None:
            view.publish(x0=np.asarray(x0), goal=np.asarray(goal), scenario_id=f"play-{id(goal)}")
            return
        controls, path, _final, total, _term, fuel, distance = rollout_fn(
            params, np.asarray(x0), np.asarray(goal)
        )
        view.publish(
            phase=phase,
            scenario_id=f"play-{float(goal[0]):.3f}-{float(goal[1]):.3f}",
            trajectory=np.asarray(path),
            controls=np.asarray(controls),
            x0=np.asarray(x0),
            goal=np.asarray(goal),
            current_loss=float(total),
            distance=float(distance),
            fuel=float(fuel),
            message="interactive",
        )

    def on_set_goal(xy):
        goal = _clip_xy(xy)
        scenario_box["goal"] = goal
        x0 = scenario_box.get("x0")
        if x0 is None:
            x0 = np.zeros((4,), dtype=np.float32)
            scenario_box["x0"] = x0
        _publish_plan(x0, goal)

    def on_set_start(xy):
        start = _clip_xy(xy)
        x0 = np.array([start[0], start[1], 0.0, 0.0], dtype=np.float32)
        scenario_box["x0"] = x0
        goal = scenario_box.get("goal")
        if goal is None:
            goal = np.array([0.5, 0.5], dtype=np.float32)
            scenario_box["goal"] = goal
        _publish_plan(x0, goal)

    def on_random():
        pos = rng.uniform(-args.position_limit, args.position_limit, size=2).astype(np.float32)
        goal = rng.uniform(-args.position_limit, args.position_limit, size=2).astype(np.float32)
        x0 = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float32)
        scenario_box["x0"] = x0
        scenario_box["goal"] = goal
        _publish_plan(x0, goal)

    def on_replay():
        x0 = scenario_box.get("x0")
        goal = scenario_box.get("goal")
        if x0 is None or goal is None:
            on_random()
            return
        _publish_plan(x0, goal)

    view = TrainingMonitor(
        args.steps,
        args.position_limit,
        on_set_goal=on_set_goal,
        on_set_start=on_set_start,
        on_random=on_random,
        on_replay=on_replay,
    )
    stop_event = threading.Event()
    holder: dict = {}

    def _legacy_publish(**kw):
        kw.pop("params", None)
        kw.pop("rollout", None)
        hist = kw.pop("history", None)
        if hist is not None:
            kw["loss_steps"] = np.asarray(hist["step"], dtype=np.float64)
            kw["loss_totals"] = np.asarray(hist["val_loss"], dtype=np.float64)
            kw["loss_terminals"] = np.asarray(hist["val_terminal"], dtype=np.float64)
        kw.setdefault("scenario_id", "train")
        view.publish(**kw)

    def worker():
        try:
            result = train(
                n_steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                lambda_fuel=args.lambda_fuel,
                seed=args.seed,
                position_limit=args.position_limit,
                velocity_limit=args.velocity_limit,
                size=args.size,
                log_every=args.log_every,
                checkpoint_path=checkpoint,
                on_update=_legacy_publish,
                stop_event=stop_event,
                scenario_box=scenario_box,
            )
            holder["result"] = result
            params_box["params"] = result.best_params
            params_box["rollout"] = result.rollout
            if result.steps_completed == 0:
                return
            cases, probe = _run_demo(result, args, evaluate_cases)
            holder["cases"] = cases
            holder["probe"] = probe
            _print_summary(cases, probe, result)
            view.publish(
                phase="demo",
                demos=cases,
                step=result.steps_completed,
                loss_steps=np.asarray(result.history["step"], dtype=np.float64),
                loss_totals=np.asarray(result.history["val_loss"], dtype=np.float64),
                loss_terminals=np.asarray(result.history["val_terminal"], dtype=np.float64),
            )
        except Exception as exc:  # surfaced in the window and the terminal
            traceback.print_exc()
            view.publish(phase="error", error=str(exc), message=str(exc))
            holder["error"] = exc

    thread = threading.Thread(target=worker, name="otis-train", daemon=True)
    thread.start()
    try:
        view.show(stop_event)
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        stop_event.set()
        thread.join(timeout=15)
    if "error" in holder:
        raise SystemExit(1) from holder["error"]
    result = holder.get("result")
    cases = holder.get("cases")
    if result is not None and cases:
        save_report(
            report,
            result.history,
            cases,
            args.position_limit,
            note=f"best step {result.best_step}   val loss {result.best_val_loss:.4f}",
        )
        print(f"Wrote {report}", flush=True)


def _fit_open_loop(trajectory_loss, simulate, x0, goal, steps: int = 600):
    import jax
    import jax.numpy as jnp
    import optax

    controls = jnp.zeros((config.N_STEPS, 2), dtype=jnp.float32)
    opt = optax.adam(0.08)
    state = opt.init(controls)

    @jax.jit
    def body(c, st):
        loss, grads = jax.value_and_grad(lambda u: trajectory_loss(u, x0, goal))(c)
        updates, st = opt.update(grads, st, c)
        c = optax.apply_updates(c, updates)
        c = jnp.clip(c, -config.MAX_THRUST, config.MAX_THRUST)
        return c, st, loss

    loss = jnp.array(0.0)
    for _ in range(steps):
        controls, state, loss = body(controls, state)
    loss.block_until_ready()
    final, _ = simulate(controls, x0)
    distance = float(jnp.linalg.norm(final[:2] - goal))
    return distance, float(loss)


def run_checks(args) -> None:
    """Fail with a non-zero status if physics, gradients, or a short train break."""

    import inspect

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    try:
        from .loss import batch_network_loss, trajectory_loss
        from .network import apply_policy, build_policy, init_params
        from .simulator import max_travel_distance, rk4_step, simulate
        from .train import load_params, train
        from .visualize import TrainingMonitor, save_report
    except ImportError:
        from loss import batch_network_loss, trajectory_loss
        from network import apply_policy, build_policy, init_params
        from simulator import max_travel_distance, rk4_step, simulate
        from train import load_params, train
        from visualize import TrainingMonitor, save_report

    print("Physics check", flush=True)
    dt_default = inspect.signature(simulate).parameters["dt"].default
    fuel_default = inspect.signature(trajectory_loss).parameters["lambda_fuel"].default
    if dt_default != config.DT or fuel_default != 0.01:
        raise SystemExit(f"Unexpected defaults dt={dt_default} lambda_fuel={fuel_default}")

    dt = jnp.float32(config.DT)
    controls = jnp.concatenate(
        [
            jnp.full((config.N_STEPS, 1), 0.3, dtype=jnp.float32),
            jnp.full((config.N_STEPS, 1), -0.5, dtype=jnp.float32),
        ],
        axis=1,
    )
    x0 = jnp.zeros((4,), dtype=jnp.float32)
    final, trajectory = simulate(controls, x0, dt)
    final.block_until_ready()
    horizon = dt * config.N_STEPS
    expected_v = controls[0] * horizon
    expected_p = 0.5 * controls[0] * horizon * horizon
    if trajectory.shape != (config.N_STEPS, 4):
        raise SystemExit(f"trajectory shape {trajectory.shape}")
    if not np.allclose(np.asarray(final[:2]), np.asarray(expected_p), atol=1e-4):
        raise SystemExit(f"position mismatch {final[:2]} vs {expected_p}")
    if not np.allclose(np.asarray(final[2:]), np.asarray(expected_v), atol=1e-4):
        raise SystemExit(f"velocity mismatch {final[2:]} vs {expected_v}")
    # A single RK4 step matches the closed form too.
    one = rk4_step(x0, controls[0], dt)
    one_p = 0.5 * controls[0] * dt * dt
    one_v = controls[0] * dt
    if not np.allclose(np.asarray(one[:2]), np.asarray(one_p), atol=1e-5):
        raise SystemExit(f"rk4 position {one[:2]} vs {one_p}")
    if not np.allclose(np.asarray(one[2:]), np.asarray(one_v), atol=1e-5):
        raise SystemExit(f"rk4 velocity {one[2:]} vs {one_v}")
    reach = max_travel_distance()
    print(f"  RK4 matches the double-integrator closed form. Reach {reach:.3f}.", flush=True)

    print("Gradient check", flush=True)
    key = jax.random.PRNGKey(0)
    key, ctrl_key = jax.random.split(key)
    guess = jax.random.uniform(ctrl_key, (config.N_STEPS, 2), minval=-0.4, maxval=0.4, dtype=jnp.float32)
    sample_x0 = jnp.array([0.2, -0.3, 0.0, 0.0], dtype=jnp.float32)
    sample_goal = jnp.array([-0.4, 0.5], dtype=jnp.float32)

    def scalar(c):
        return trajectory_loss(c, sample_x0, sample_goal)

    grads = jax.grad(scalar)(guess)
    eps = 1e-2
    for idx in ((0, 0), (10, 1), (config.N_STEPS - 1, 0)):
        bump = jnp.zeros_like(guess).at[idx].set(eps)
        fd = (scalar(guess + bump) - scalar(guess - bump)) / (2.0 * eps)
        err = abs(float(fd) - float(grads[idx]))
        print(f"  dL/du{idx} autodiff {float(grads[idx]): .5f}  finite-diff {float(fd): .5f}  abs {err:.2e}", flush=True)
        if err > 2e-3:
            raise SystemExit(f"Gradient mismatch at {idx}: {err}")
    if not np.all(np.isfinite(np.asarray(grads))):
        raise SystemExit("Non-finite control gradient")

    print("Network check", flush=True)
    model = build_policy(args.size)
    key, init_key = jax.random.split(key)
    params = init_params(model, init_key)
    pred = apply_policy(model, params, sample_x0, sample_goal)
    if tuple(pred.shape) != (config.N_STEPS, 2):
        raise SystemExit(f"policy shape {pred.shape}")
    if float(jnp.max(jnp.abs(pred))) > config.MAX_THRUST + 1e-5:
        raise SystemExit("policy exceeded the thrust limit")
    leaves = jax.tree_util.tree_leaves(params)
    n_params = int(sum(leaf.size for leaf in leaves))
    print(f"  output {tuple(pred.shape)} within ±{config.MAX_THRUST}, parameters {n_params}", flush=True)

    print("Open-loop feasibility (corner to corner)", flush=True)
    lim = float(args.position_limit)
    corner_x0 = jnp.array([-lim, -lim, 0.0, 0.0], dtype=jnp.float32)
    corner_goal = jnp.array([lim, lim], dtype=jnp.float32)
    corner_dist, corner_loss = _fit_open_loop(trajectory_loss, simulate, corner_x0, corner_goal)
    diagonal = 2.0 * lim * math.sqrt(2.0)
    print(f"  direct-control distance {corner_dist:.4f}  loss {corner_loss:.4f}", flush=True)
    if diagonal <= reach + 1e-6 and corner_dist >= 0.5:
        raise SystemExit(
            f"Even an optimized open-loop plan misses the corner by {corner_dist:.3f}. "
            "The workspace is too large for the thrust limit."
        )

    print("Batched value_and_grad", flush=True)

    def apply_fn(p, x0, goal):
        return apply_policy(model, p, x0, goal)

    def batch_loss(p, x0s, goals):
        return batch_network_loss(p, apply_fn, x0s, goals, config.LAMBDA_FUEL)

    loss_and_grad = jax.jit(jax.value_and_grad(batch_loss, argnums=0))  # grad w.r.t. network params
    key, batch_key = jax.random.split(key)
    key_pos, key_goal = jax.random.split(batch_key)
    pos = jax.random.uniform(key_pos, (32, 2), minval=-lim, maxval=lim, dtype=jnp.float32)
    x0s = jnp.concatenate([pos, jnp.zeros((32, 2), dtype=jnp.float32)], axis=-1)
    goals = jax.random.uniform(key_goal, (32, 2), minval=-lim, maxval=lim, dtype=jnp.float32)
    loss, param_grads = loss_and_grad(params, x0s, goals)
    grad_norm = float(optax.global_norm(param_grads))
    print(f"  batch loss {float(loss):.4f}  grad norm {grad_norm:.3e}", flush=True)
    if not math.isfinite(float(loss)) or not math.isfinite(grad_norm) or grad_norm <= 0.0:
        raise SystemExit("Batched gradient is zero or non-finite")

    steps = min(args.steps, 300) if args.steps == config.TRAIN_STEPS else args.steps
    print(f"Short train ({steps} steps)", flush=True)
    check_dir = PACKAGE_ROOT / "outputs" / "check"
    checkpoint = check_dir / "best_params.pkl"
    report = check_dir / "demo.png"
    result = train(
        n_steps=steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lambda_fuel=args.lambda_fuel,
        seed=args.seed,
        position_limit=args.position_limit,
        velocity_limit=args.velocity_limit,
        size=args.size,
        log_every=max(1, min(args.log_every, 100)),
        checkpoint_path=checkpoint,
    )
    start_loss = result.history["val_loss"][0]
    end_loss = result.history["val_loss"][-1]
    start_dist = result.history["val_distance"][0]
    end_dist = result.history["val_distance"][-1]
    print(
        f"  val loss {start_loss:.4f} -> {end_loss:.4f}   "
        f"val distance {start_dist:.3f} -> {end_dist:.3f}",
        flush=True,
    )
    if not (end_loss < start_loss and end_dist < start_dist):
        raise SystemExit("Validation loss did not decrease")
    restored = load_params(checkpoint, result.best_params)
    for left, right in zip(
        jax.tree_util.tree_leaves(result.best_params),
        jax.tree_util.tree_leaves(restored),
    ):
        if not np.allclose(np.asarray(left), np.asarray(right)):
            raise SystemExit("Checkpoint round-trip changed a parameter")
    print("  checkpoint round-trip ok", flush=True)

    print("Animation frame check", flush=True)
    monitor = TrainingMonitor(steps, args.position_limit)
    monitor.publish(
        phase="training",
        step=steps,
        total_steps=steps,
        scenario_id="check",
        loss_steps=np.asarray(result.history["step"], dtype=np.float64),
        loss_totals=np.asarray(result.history["val_loss"], dtype=np.float64),
        loss_terminals=np.asarray(result.history["val_terminal"], dtype=np.float64),
        trajectory=np.asarray(simulate(pred, sample_x0)[1]),
        controls=np.asarray(pred),
        x0=np.asarray(sample_x0),
        goal=np.asarray(sample_goal),
        current_loss=float(scalar(pred)),
        distance=float(jnp.linalg.norm(simulate(pred, sample_x0)[0][:2] - sample_goal)),
        fuel=float(jnp.sum(pred**2)),
    )
    # The published trajectory omitted x0; prepend it so the craft starts on the dot.
    rolled = monitor.snapshot()
    path = np.concatenate([rolled["x0"][None, :], rolled["trajectory"]], axis=0)
    monitor.publish(trajectory=path)
    fig = monitor.render_frames(4)
    frame_path = report.parent / "anim_check.png"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(frame_path, dpi=120)
    import matplotlib.pyplot as plt

    plt.close(fig)
    save_report(
        report,
        result.history,
        [
            {
                "trajectory": path,
                "controls": np.asarray(pred),
                "x0": np.asarray(sample_x0),
                "goal": np.asarray(sample_goal),
                "distance": float(jnp.linalg.norm(simulate(pred, sample_x0)[0][:2] - sample_goal)),
                "loss": float(scalar(pred)),
                "fuel": float(jnp.sum(pred**2)),
            }
        ],
        args.position_limit,
        note="self-check frame (untrained policy on one case, plus the short-train curve)",
    )
    print(f"  wrote {frame_path}", flush=True)
    print("Checks passed.", flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.benchmark:
        try:
            from .evaluate import run_benchmark
        except ImportError:
            from evaluate import run_benchmark
        checkpoint, _ = _paths(args)
        if not checkpoint.exists():
            raise SystemExit(f"No checkpoint at {checkpoint}. Train first: python main.py --no-viz")
        run_benchmark(checkpoint)
        return
    if args.export_benchmark:
        try:
            from .export_benchmark import export
        except ImportError:
            from export_benchmark import export
        export()
        return
    if args.write_benchmark:
        try:
            from .benchmarks import write_benchmark
        except ImportError:
            from benchmarks import write_benchmark
        dest = (
            Path(args.save_dir) / "data" / "benchmark_problems.json"
            if args.save_dir
            else PACKAGE_ROOT / "data" / "benchmark_problems.json"
        )
        print(f"Wrote {write_benchmark(dest)}", flush=True)
        return
    if not (args.no_viz or args.check or args.legacy_window):
        try:
            from .server import serve
        except ImportError:
            from server import serve
        checkpoint, _ = _paths(args)
        serve(
            checkpoint,
            {
                "size": args.size,
                "steps": args.steps,
                "lr": args.lr,
                "lambda_fuel": args.lambda_fuel,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "position_limit": args.position_limit,
                "velocity_limit": args.velocity_limit,
            },
            host=args.host,
            port=args.port,
            autotrain=args.autotrain,
        )
        return
    live = should_visualize(args)
    _configure_backend(live)
    if args.check:
        run_checks(args)
        return
    if live:
        run_live(args)
    else:
        if not args.no_viz:
            print("No display detected; training without a live window. Pass --no-viz to silence this.", flush=True)
        run_headless(args)


if __name__ == "__main__":
    main()
