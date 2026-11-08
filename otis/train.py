"""Optax training loop for the thrust network.

Each step draws a fresh batch of problems, runs the network, rolls the
predicted thrust through the differentiable RK4 simulator, and applies AdamW
to the exact gradient of the mean loss. Keys from ``jax.random`` are split on
every use and never reused.

The loop lives in Python so it can be paused, stopped, and observed by the
web UI. Everything inside one step — the batched rollout and the backward
pass — is ``jax.jit`` compiled. A fixed validation batch is scored every
``log_every`` steps; the weights with the lowest validation loss are kept.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

try:
    from . import config
    from .loss import batch_network_loss, batch_network_metrics, per_example_metrics
    from .network import apply_policy, build_policy, count_params, init_params
    from .simulator import DEFAULT_PLANT
except ImportError:
    import config
    from loss import batch_network_loss, batch_network_metrics, per_example_metrics
    from network import apply_policy, build_policy, count_params, init_params
    from simulator import DEFAULT_PLANT


@dataclass
class TrainResult:
    """Everything callers need after the loop finishes."""

    model: Any
    params: Any
    best_params: Any
    best_step: int
    best_val_loss: float
    best_val_distance: float
    history: dict
    elapsed_sec: float
    position_limit: float
    lambda_fuel: float
    rollout: Any
    steps_completed: int
    size: str
    n_params: int


def sample_problems(
    key,
    batch_size: int,
    position_limit: float,
    velocity_limit: float = config.VELOCITY_LIMIT,
):
    """Sample ``batch_size`` starts and goals.

    Position is uniform in ``[-position_limit, position_limit]^2`` and
    velocity uniform in ``[-velocity_limit, velocity_limit]^2``. The goal is
    drawn independently from the position square. This online distribution
    *is* the training data: there is no recorded dataset.
    """

    key_pos, key_goal, key_vel = jax.random.split(key, 3)
    pos = jax.random.uniform(
        key_pos, (batch_size, 2), minval=-position_limit, maxval=position_limit, dtype=jnp.float32
    )
    if velocity_limit > 0:
        vel = jax.random.uniform(
            key_vel, (batch_size, 2), minval=-velocity_limit, maxval=velocity_limit, dtype=jnp.float32
        )
    else:
        vel = jnp.zeros((batch_size, 2), dtype=jnp.float32)
    x0 = jnp.concatenate([pos, vel], axis=-1)
    goal = jax.random.uniform(
        key_goal, (batch_size, 2), minval=-position_limit, maxval=position_limit, dtype=jnp.float32
    )
    return x0, goal


def sample_one(key, position_limit: float, velocity_limit: float = config.VELOCITY_LIMIT):
    """One ``(x0, goal)`` pair. ``x0`` has shape ``(4,)``."""

    x0, goal = sample_problems(key, 1, position_limit, velocity_limit)
    return x0[0], goal[0]


def _copy_params(params):
    return jax.tree_util.tree_map(jnp.copy, params)


def _save_params(path: Path, params, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(params))
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load_params(path, template):
    """Restore parameters written by ``train``."""

    return serialization.from_bytes(template, Path(path).read_bytes())


def load_checkpoint(path):
    """Rebuild ``(model, params, meta)`` from a checkpoint and its sidecar."""

    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    plant = meta.get("plant") or {}
    model = build_policy(
        meta.get("size", "small"),
        position_limit=meta.get("position_limit", config.POSITION_LIMIT),
        velocity_limit=meta.get("velocity_limit", config.VELOCITY_LIMIT),
        horizon=plant.get("horizon", config.N_STEPS * config.DT),
        gravity=plant.get("gravity", 0.0),
    )
    template = init_params(model, jax.random.PRNGKey(0))
    return model, load_params(path, template), meta


def _save_history(path: Path, history: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2))


def plant_from_meta(meta: dict):
    """``(plant tuple, dt)`` a checkpoint was trained on (default: unit plant)."""

    plant = meta.get("plant") or {}
    tup = (
        float(plant.get("gain", DEFAULT_PLANT[0])),
        float(plant.get("gravity", DEFAULT_PLANT[1])),
        float(plant.get("drag", DEFAULT_PLANT[2])),
    )
    horizon = float(plant.get("horizon", config.N_STEPS * config.DT))
    return tup, horizon / config.N_STEPS


def make_rollout(model, lambda_fuel: float, dt: float = config.DT, plant=DEFAULT_PLANT):
    """JIT-compiled single-problem rollout used by the demo, UI and benchmark."""

    plant = tuple(float(v) for v in plant)

    @jax.jit
    def rollout(p, x0, goal):
        controls = apply_policy(model, p, x0, goal)
        total, terminal, fuel, distance, final_state, path = per_example_metrics(
            controls, x0, goal, lambda_fuel, dt, plant
        )
        return controls, path, final_state, total, terminal, fuel, distance

    return rollout


def make_optimizer(learning_rate: float, n_steps: int, warmup: int = config.WARMUP_STEPS):
    warmup = int(min(warmup, max(1, n_steps // 5)))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=learning_rate * 0.05,
        peak_value=learning_rate,
        warmup_steps=warmup,
        decay_steps=max(n_steps, warmup + 1),
        end_value=learning_rate * 0.02,
    )
    return optax.chain(
        optax.clip_by_global_norm(config.GRAD_CLIP),
        optax.adamw(schedule, weight_decay=config.WEIGHT_DECAY),
    ), schedule


def train(
    *,
    n_steps: int = config.TRAIN_STEPS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    lambda_fuel: float = config.LAMBDA_FUEL,
    seed: int = config.SEED,
    position_limit: float = config.POSITION_LIMIT,
    velocity_limit: float = config.VELOCITY_LIMIT,
    size: str = config.DEFAULT_SIZE,
    log_every: int = config.LOG_EVERY,
    viz_every: int = config.VIZ_EVERY,
    val_batch_size: int = config.VAL_BATCH_SIZE,
    checkpoint_path: Path | None = None,
    on_update: Callable[..., None] | None = None,
    stop_event: Any = None,
    pause_event: Any = None,
    scenario_box: dict | None = None,
    verbose: bool = True,
    plant=DEFAULT_PLANT,
    horizon: float | None = None,
    plant_meta: dict | None = None,
) -> TrainResult:
    """Train the policy and return the best weights.

    ``on_update`` receives keyword fields (trajectory, loss curve, step, …).
    ``scenario_box`` lets a UI swap the showcase start/goal mid-training.
    ``pause_event`` (set = paused) and ``stop_event`` (set = stop) come from
    the server.

    ``plant`` is ``(accel_gain, gravity, drag)`` and ``horizon`` the mission
    length in seconds (``dt = horizon / N_STEPS``); both default to the
    published unit plant. ``plant_meta`` is copied into the checkpoint
    sidecar (vehicle id, user parameters) so the UI can match it later.
    """

    if batch_size < 1 or n_steps < 1:
        raise ValueError("batch_size and n_steps must be positive")

    log = print if verbose else (lambda *a, **k: None)
    key = jax.random.PRNGKey(seed)
    key, init_key, viz_key, val_key = jax.random.split(key, 4)

    plant = tuple(float(v) for v in plant)
    horizon = float(config.N_STEPS * config.DT if horizon is None else horizon)
    dt = horizon / config.N_STEPS
    model = build_policy(
        size,
        position_limit=position_limit,
        velocity_limit=velocity_limit,
        horizon=horizon,
        gravity=plant[1],
    )
    params = init_params(model, init_key)
    n_params = count_params(params)
    # Squared miss grows with the workspace size. Rescaling the whole
    # objective by a constant leaves its minimizer unchanged but keeps the
    # gradient magnitude comparable to the published unit plant, so the same
    # clipping and learning rate work for a 2 m or a 50 m workspace.
    loss_scale = (config.POSITION_LIMIT / position_limit) ** 2

    def apply_fn(p, x0, goal):
        return apply_policy(model, p, x0, goal)

    def batch_loss(p, x0s, goals):
        return loss_scale * batch_network_loss(p, apply_fn, x0s, goals, lambda_fuel, dt, plant)

    tx, schedule = make_optimizer(learning_rate, n_steps)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(p, state, x0s, goals):
        # Exact gradient of the batched loss w.r.t. the network weights,
        # backpropagated through every RK4 stage of every rollout.
        loss, grads = jax.value_and_grad(batch_loss)(p, x0s, goals)
        updates, state = tx.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, loss, optax.global_norm(grads)

    @jax.jit
    def eval_batch(p, x0s, goals):
        return batch_network_metrics(p, apply_fn, x0s, goals, lambda_fuel, dt, plant)

    rollout = make_rollout(model, lambda_fuel, dt, plant)

    viz_x0, viz_goal = sample_one(viz_key, position_limit, velocity_limit)
    val_x0, val_goals = sample_problems(val_key, val_batch_size, position_limit, velocity_limit)
    if scenario_box is not None:
        if scenario_box.get("x0") is None:
            scenario_box["x0"] = np.asarray(viz_x0)
            scenario_box["goal"] = np.asarray(viz_goal)

    history = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "val_terminal": [],
        "val_distance": [],
        "val_fuel": [],
        "grad_norm": [],
        "lr": [],
        "time": [],
    }
    clock = {"t0": None, "paused": 0.0}
    best_params = _copy_params(params)
    best_val_loss = float("inf")
    best_val_distance = float("inf")
    best_step = 0

    meta_base = {
        "seed": seed,
        "size": size,
        "n_params": n_params,
        "position_limit": position_limit,
        "velocity_limit": velocity_limit,
        "lambda_fuel": lambda_fuel,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_steps_horizon": config.N_STEPS,
        "dt": dt,
        "device": str(jax.devices()[0]),
        "plant": {"gain": plant[0], "gravity": plant[1], "drag": plant[2], "horizon": horizon},
        **(plant_meta or {}),
    }

    def publish(phase: str, step: int, current_params, **extra):
        nonlocal viz_x0, viz_goal
        if on_update is None:
            return
        if scenario_box is not None and scenario_box.get("x0") is not None:
            viz_x0 = jnp.asarray(scenario_box["x0"], dtype=jnp.float32)
            viz_goal = jnp.asarray(scenario_box["goal"], dtype=jnp.float32)
        controls, path, _, total, _, fuel, distance = rollout(current_params, viz_x0, viz_goal)
        payload = {
            "phase": phase,
            "step": step,
            "total_steps": n_steps,
            "history": history,
            "trajectory": np.asarray(path),
            "controls": np.asarray(controls),
            "x0": np.asarray(viz_x0),
            "goal": np.asarray(viz_goal),
            "current_loss": float(total),
            "distance": float(distance),
            "fuel": float(fuel),
            "params": current_params,
            "rollout": rollout,
            "n_params": n_params,
            "size": size,
        }
        payload.update(extra)
        on_update(**payload)

    def record_eval(step: int, train_loss, grad_norm, current_params):
        nonlocal best_params, best_val_loss, best_val_distance, best_step
        val_loss, val_terminal, val_fuel, val_distance = (
            float(v) for v in eval_batch(current_params, val_x0, val_goals)
        )
        history["step"].append(step)
        history["train_loss"].append(None if train_loss is None else float(train_loss))
        history["val_loss"].append(val_loss)
        history["val_terminal"].append(val_terminal)
        history["val_distance"].append(val_distance)
        history["val_fuel"].append(val_fuel)
        history["grad_norm"].append(None if grad_norm is None else float(grad_norm))
        history["lr"].append(float(schedule(step)))
        history["time"].append(0.0 if clock["t0"] is None else time.perf_counter() - clock["t0"] - clock["paused"])
        if val_loss < best_val_loss:
            best_val_loss, best_val_distance, best_step = val_loss, val_distance, step
            best_params = _copy_params(current_params)
        return val_loss, val_terminal, val_distance, val_fuel

    if on_update is not None:
        on_update(phase="compiling", step=0, total_steps=n_steps, n_params=n_params, size=size)

    log(f"Compiling the differentiated trainer ({n_params:,} parameters)...", flush=True)
    compile_t0 = time.perf_counter()
    record_eval(0, None, None, params)
    key, batch_key = jax.random.split(key)
    wx, wg = sample_problems(batch_key, batch_size, position_limit, velocity_limit)
    # Compile once on a throwaway copy so step 1 still starts from init.
    _p, _s, _l, gnorm = train_step(params, opt_state, wx, wg)
    gnorm.block_until_ready()
    del _p, _s, _l
    log(f"Compiled in {time.perf_counter() - compile_t0:.1f}s", flush=True)
    publish("training", 0, params)

    t_run = time.perf_counter()
    clock["t0"] = t_run
    paused_total = 0.0
    steps_completed = 0
    for step in range(1, n_steps + 1):
        if pause_event is not None and pause_event.is_set():
            t_pause = time.perf_counter()
            publish("paused", steps_completed, params)
            while pause_event.is_set() and not (stop_event is not None and stop_event.is_set()):
                time.sleep(0.05)
            paused_total += time.perf_counter() - t_pause
            clock["paused"] = paused_total
        if stop_event is not None and stop_event.is_set():
            log(f"Stopped at step {steps_completed}.", flush=True)
            break
        key, batch_key = jax.random.split(key)
        x0s, goals = sample_problems(batch_key, batch_size, position_limit, velocity_limit)
        params, opt_state, loss, grad_norm = train_step(params, opt_state, x0s, goals)
        steps_completed = step

        should_log = step % log_every == 0 or step == n_steps
        if should_log:
            train_loss, grad_norm_f = float(loss), float(grad_norm)
            if not np.isfinite(train_loss) or not np.isfinite(grad_norm_f):
                raise RuntimeError(f"Non-finite training signal at step {step}")
            val_loss, val_terminal, val_distance, _ = record_eval(step, train_loss, grad_norm_f, params)
            rate = step / max(time.perf_counter() - t_run - paused_total, 1e-9)
            log(
                f"step {step:5d} | train {train_loss:.4f} | val {val_loss:.4f} | "
                f"terminal {val_terminal:.5f} | dist {val_distance:.4f} | "
                f"grad {grad_norm_f:.2e} | {rate:.1f} step/s",
                flush=True,
            )
        # The plan changes fastest in the first few dozen updates, so the
        # live view gets every one of those, then every ``viz_every``.
        if step <= 40 or step % viz_every == 0 or should_log:
            publish("training", step, params, rate=step / max(time.perf_counter() - t_run - paused_total, 1e-9))

    elapsed_sec = time.perf_counter() - t_run - paused_total

    if checkpoint_path is not None:
        _save_params(
            Path(checkpoint_path),
            best_params,
            {
                **meta_base,
                "step": best_step,
                "val_loss": best_val_loss,
                "val_distance": best_val_distance,
                "steps_completed": steps_completed,
                "train_seconds": elapsed_sec,
            },
        )
        _save_history(Path(checkpoint_path).parent / "history.json", history)

    return TrainResult(
        model=model,
        params=params,
        best_params=best_params,
        best_step=best_step,
        best_val_loss=best_val_loss,
        best_val_distance=best_val_distance,
        history=history,
        elapsed_sec=elapsed_sec,
        position_limit=position_limit,
        lambda_fuel=lambda_fuel,
        rollout=rollout,
        steps_completed=steps_completed,
        size=size,
        n_params=n_params,
    )


def evaluate_cases(
    rollout,
    params,
    *,
    n: int,
    seed: int,
    position_limit: float,
    stream: int,
    velocity_limit: float = config.VELOCITY_LIMIT,
):
    """Roll out ``n`` fresh problems from an independent key stream."""

    key = jax.random.fold_in(jax.random.PRNGKey(seed), stream)
    cases = []
    for _ in range(n):
        key, sub = jax.random.split(key)
        x0, goal = sample_one(sub, position_limit, velocity_limit)
        controls, path, _final, loss, terminal, fuel, distance = rollout(params, x0, goal)
        cases.append(
            {
                "x0": np.asarray(x0),
                "goal": np.asarray(goal),
                "controls": np.asarray(controls),
                "trajectory": np.asarray(path),
                "loss": float(loss),
                "terminal": float(terminal),
                "fuel": float(fuel),
                "distance": float(distance),
            }
        )
    return cases
