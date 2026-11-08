"""Judge-facing start/goal benchmark set.

Training samples problems online: there is no external labeled spacecraft
corpus for this 2D double integrator. This module writes fold 0 of the exact
test set that ``python main.py --benchmark`` scores (same PRNG stream, never
used in training), together with the provably optimal objective for every
case, so anyone can re-score a controller against it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from . import config
except ImportError:
    import config


def write_benchmark(path=None, n: int = config.BENCHMARK_CASES, seed: int = config.SEED):
    """Write ``n`` start/goal pairs (plus the optimal objective) as JSON."""

    try:
        from .baselines import qp_solve_exact, score
        from .evaluate import TEST_STREAM_BASE, _fold
    except ImportError:
        from baselines import qp_solve_exact, score
        from evaluate import TEST_STREAM_BASE, _fold

    dest = Path(path) if path is not None else Path(__file__).resolve().parent / "data" / "benchmark_problems.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    lim, vlim = float(config.POSITION_LIMIT), float(config.VELOCITY_LIMIT)
    x0, goal = _fold(seed, 0, int(n), lim, vlim)
    opt = score(qp_solve_exact(x0, goal), x0, goal)
    horizon = float(config.N_STEPS * config.DT)
    cases = [
        {
            "id": i,
            "x0": [round(float(v), 7) for v in x0[i]],
            "goal": [round(float(v), 7) for v in goal[i]],
            "optimal_objective": round(float(opt["loss"][i]), 8),
            "optimal_miss": round(float(opt["miss"][i]), 8),
        }
        for i in range(int(n))
    ]
    payload = {
        "name": "OTiS synthetic double-integrator benchmark (fold 0)",
        "description": (
            "Start/goal pairs for a unit-mass 2D double integrator, drawn with "
            f"jax.random.fold_in(PRNGKey({seed}), {TEST_STREAM_BASE}) - a stream the trainer never uses. "
            "optimal_objective is the global optimum of J = |p_T - goal|^2 + lambda * sum(u^2), |u_i| <= 1, "
            "from the exact KKT solver (cross-checked against CVXPY/Clarabel and OSQP)."
        ),
        "seed": int(seed),
        "n": int(n),
        "position_limit": lim,
        "velocity_limit": vlim,
        "n_steps": int(config.N_STEPS),
        "dt": float(config.DT),
        "max_thrust": float(config.MAX_THRUST),
        "lambda_fuel": float(config.LAMBDA_FUEL),
        "reachable_range_from_rest": 0.5 * float(config.MAX_THRUST) * horizon * horizon,
        "state": "[x, y, vx, vy]",
        "cases": cases,
    }
    dest.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return dest
