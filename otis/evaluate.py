"""Benchmark the trained network against industry baselines, and cross-validate.

    python main.py --benchmark [--checkpoint PATH]

Writes ``outputs/benchmark_results.json`` and ``../BENCHMARK.md``.

Methods compared, all scored by the same RK4 simulator and the same loss:

- ``neural``      the trained network (one forward pass per problem)
- ``convex``      CVXPY + Clarabel direct transcription (G-FOLD-style), per problem
- ``optimal``     exact KKT optimum of the same QP (ground truth, provable)
- ``e_guidance``  Apollo-heritage closed-loop ZEM law, clipped to the thrust box

Cross-validation:

1. Integrator: our float32 JAX RK4 vs the closed form vs SciPy DOP853 (float64).
2. Gradient: autodiff vs central finite differences through the simulator.
3. Solver: Clarabel vs OSQP vs the exact KKT solution.
4. Generalization: ``BENCHMARK_FOLDS`` disjoint test folds (never seen in
   training), mean ± std and a bootstrap 95 % CI of the optimality gap.
5. Stress: out-of-distribution starts (2x the training velocity range).
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

try:
    from . import config
    from .baselines import ConvexGuidance, e_guidance_batch, qp_solve_exact, score
    from .loss import trajectory_loss
    from .network import apply_policy
    from .simulator import simulate, terminal_map
    from .train import load_checkpoint, sample_problems
except ImportError:
    import config
    from baselines import ConvexGuidance, e_guidance_batch, qp_solve_exact, score
    from loss import trajectory_loss
    from network import apply_policy
    from simulator import simulate, terminal_map
    from train import load_checkpoint, sample_problems

PACKAGE_ROOT = Path(__file__).resolve().parent
TEST_STREAM_BASE = 10_000  # training uses PRNGKey(seed) splits; folds use fold_in(…, 10000+i)


def _fold(seed, i, n, position_limit, velocity_limit):
    key = jax.random.fold_in(jax.random.PRNGKey(seed), TEST_STREAM_BASE + i)
    x0, g = sample_problems(key, n, position_limit, velocity_limit)
    return np.asarray(x0, dtype=np.float32), np.asarray(g, dtype=np.float32)


def _summ(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def _bootstrap_ci(values, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


# --------------------------------------------------------------------------
# Cross-validation of the physics, gradients and solvers
# --------------------------------------------------------------------------

def crossval_integrator(n=200, seed=0):
    from scipy.integrate import solve_ivp

    rng = np.random.default_rng(seed)
    b, c, horizon = terminal_map()
    err_closed, err_scipy, err_closed_vs_scipy = [], [], []
    for _ in range(n):
        u = rng.uniform(-1, 1, size=(config.N_STEPS, 2))
        x0 = np.r_[rng.uniform(-2, 2, 2), rng.uniform(-0.5, 0.5, 2)]
        final, _ = simulate(jnp.asarray(u, jnp.float32), jnp.asarray(x0, jnp.float32))
        final = np.asarray(final, dtype=np.float64)
        closed = np.r_[x0[:2] + x0[2:] * horizon + b @ u, x0[2:] + c @ u]
        # DOP853 integrating the continuous ODE, one zero-order-hold step at a time
        s = x0.copy()
        for k in range(config.N_STEPS):
            sol = solve_ivp(
                lambda t, y, uk=u[k]: [y[2], y[3], uk[0] / config.MASS, uk[1] / config.MASS],
                (0.0, config.DT), s, method="DOP853", rtol=1e-12, atol=1e-12,
            )
            s = sol.y[:, -1]
        err_closed.append(np.abs(final - closed).max())
        err_scipy.append(np.abs(final - s).max())
        err_closed_vs_scipy.append(np.abs(closed - s).max())
    return {
        "cases": n,
        "max_abs_rk4_vs_closed_form": float(max(err_closed)),
        "max_abs_rk4_vs_scipy_dop853": float(max(err_scipy)),
        "max_abs_closed_form_vs_dop853": float(max(err_closed_vs_scipy)),
        "note": "RK4 runs in float32 (production precision); closed form and DOP853 in float64.",
    }


def crossval_gradient(n=5, seed=1):
    rng = np.random.default_rng(seed)
    worst = 0.0
    checks = 0
    for _ in range(n):
        u = jnp.asarray(rng.uniform(-0.5, 0.5, (config.N_STEPS, 2)), jnp.float32)
        x0 = jnp.asarray(np.r_[rng.uniform(-2, 2, 2), rng.uniform(-0.5, 0.5, 2)], jnp.float32)
        g = jnp.asarray(rng.uniform(-2, 2, 2), jnp.float32)
        f = lambda c: trajectory_loss(c, x0, g)
        grad = np.asarray(jax.grad(f)(u))
        eps = 1e-2
        for idx in [(0, 0), (config.N_STEPS // 2, 1), (config.N_STEPS - 1, 0)]:
            bump = jnp.zeros_like(u).at[idx].set(eps)
            fd = (float(f(u + bump)) - float(f(u - bump))) / (2 * eps)
            worst = max(worst, abs(fd - grad[idx]))
            checks += 1
    return {"checks": checks, "max_abs_autodiff_vs_finite_difference": float(worst)}


def crossval_solvers(x0, g, n=100):
    exact = qp_solve_exact(x0[:n], g[:n])
    out = {"cases": n}
    for name in ("CLARABEL", "OSQP"):
        solver = ConvexGuidance(solver=name)
        us, ts = [], []
        for i in range(n):
            u, t = solver.solve(x0[i], g[i])
            us.append(u)
            ts.append(t)
        us = np.asarray(us)
        se, ss = score(exact, x0[:n], g[:n]), score(us, x0[:n], g[:n])
        out[name.lower()] = {
            "max_abs_control_diff_vs_exact": float(np.abs(us - exact).max()),
            "max_abs_loss_diff_vs_exact": float(np.abs(ss["loss"] - se["loss"]).max()),
            "median_solve_ms": float(1e3 * np.median(ts)),
        }
    return out


# --------------------------------------------------------------------------
# Head-to-head benchmark
# --------------------------------------------------------------------------

def kkt_consistency(meta, x0, g, lambda_fuel):
    """Confirm the KKT referee solves the same discretized problem as training.

    Checks (1) the checkpoint was trained on the plant the benchmark scores
    (dt, N, unit gain, no gravity or drag), and (2) the terminal positions the
    KKT solver *predicts* from its closed-form map b_k equal the terminal
    positions the training RK4 simulator *produces* for those same controls.
    """

    plant = meta.get("plant") or {}
    trained_horizon = float(plant.get("horizon", config.N_STEPS * config.DT))
    trained_dt = trained_horizon / config.N_STEPS
    same_plant = (
        abs(trained_dt - config.DT) < 1e-12
        and abs(float(plant.get("gain", 1.0)) - 1.0) < 1e-12
        and abs(float(plant.get("gravity", 0.0))) < 1e-12
        and abs(float(plant.get("drag", 0.0))) < 1e-12
    )
    u = qp_solve_exact(x0, g, lambda_fuel)
    b, _c, horizon = terminal_map(config.N_STEPS, config.DT)
    p_pred = np.asarray(x0, np.float64)[:, :2] + np.asarray(x0, np.float64)[:, 2:] * horizon + np.einsum("k,bka->ba", b, u)
    finals, _ = jax.vmap(lambda c, x: simulate(c, x))(jnp.asarray(u, jnp.float32), jnp.asarray(x0, jnp.float32))
    err = float(np.abs(np.asarray(finals)[:, :2] - p_pred).max())
    return {
        "kkt_dt": config.DT, "kkt_n_steps": config.N_STEPS,
        "training_dt": trained_dt, "training_n_steps": config.N_STEPS,
        "checkpoint_plant_matches": bool(same_plant),
        "max_abs_terminal_pos_kkt_map_vs_rk4": err,
        "cases": int(len(x0)),
        "consistent": bool(same_plant and err < 1e-4),
    }


def _method_stats(s, tolerances):
    d = {k: _summ(s[k]) for k in ("loss", "miss", "fuel", "delta_v")}
    d["success"] = {f"{t:g}": float((s["miss"] < t).mean()) for t in tolerances}
    return d


def run_fold(model, params, x0, g, lambda_fuel, convex_subset):
    apply_batch = jax.jit(lambda p, a, b: apply_policy(model, p, a, b))

    # Neural network: one batched forward pass (timed after compile)
    u_nn = apply_batch(params, x0, g)
    u_nn.block_until_ready()
    t0 = time.perf_counter()
    u_nn = apply_batch(params, x0, g)
    u_nn.block_until_ready()
    nn_batch_ms = 1e3 * (time.perf_counter() - t0)
    s_nn = score(np.asarray(u_nn), x0, g, lambda_fuel)

    t0 = time.perf_counter()
    u_opt = qp_solve_exact(x0, g, lambda_fuel)
    exact_ms = 1e3 * (time.perf_counter() - t0)
    s_opt = score(u_opt, x0, g, lambda_fuel)

    u_eg = e_guidance_batch(jnp.asarray(x0), jnp.asarray(g))
    u_eg.block_until_ready()
    t0 = time.perf_counter()
    u_eg = e_guidance_batch(jnp.asarray(x0), jnp.asarray(g))
    u_eg.block_until_ready()
    eg_ms = 1e3 * (time.perf_counter() - t0)
    s_eg = score(np.asarray(u_eg), x0, g, lambda_fuel)

    solver = ConvexGuidance(lambda_fuel=lambda_fuel, solver="CLARABEL")
    m = len(x0) if convex_subset is None else min(convex_subset, len(x0))
    u_cv, t_cv = [], []
    for i in range(m):
        u, t = solver.solve(x0[i], g[i])
        u_cv.append(u)
        t_cv.append(t)
    s_cv = score(np.asarray(u_cv), x0[:m], g[:m], lambda_fuel) if m else None

    return {
        "scores": {"neural": s_nn, "optimal": s_opt, "e_guidance": s_eg, "convex": s_cv},
        "timing_ms_per_problem": {
            "neural_batched_gpu": nn_batch_ms / len(x0),
            "optimal_kkt_vectorized": exact_ms / len(x0),
            "e_guidance_batched": eg_ms / len(x0),
            "convex_clarabel_median": float(1e3 * np.median(t_cv)) if m else float("nan"),
        },
        "convex_n": m,
    }


def neural_single_latency_ms(model, params, x0, g, reps=300, device=None):
    """Median latency of one plan, weights compiled in as constants.

    Passing the 13 M-parameter tree as a jit argument adds ~6 ms of
    per-call argument handling under WSL; a deployed controller would bake
    the frozen weights in, so that is what we time.
    """

    device = device or jax.devices()[0]
    p_dev = jax.device_put(params, device)
    f = jax.jit(lambda a, b: apply_policy(model, p_dev, a, b))
    xs = jax.device_put(jnp.asarray(x0), device)
    gs = jax.device_put(jnp.asarray(g), device)
    f(xs[0], gs[0]).block_until_ready()
    ts = []
    for i in range(reps):
        a, b = xs[i % len(xs)], gs[i % len(gs)]
        t0 = time.perf_counter()
        f(a, b).block_until_ready()
        ts.append(time.perf_counter() - t0)
    return float(1e3 * np.median(ts))


def run_benchmark(checkpoint, out_json=None, out_md=None, n=config.BENCHMARK_CASES,
                  folds=config.BENCHMARK_FOLDS, convex_subset=None, seed=config.SEED):
    checkpoint = Path(checkpoint)
    model, params, meta = load_checkpoint(checkpoint)
    lam = meta.get("lambda_fuel", config.LAMBDA_FUEL)
    plim = meta.get("position_limit", config.POSITION_LIMIT)
    vlim = meta.get("velocity_limit", config.VELOCITY_LIMIT)
    tols = config.SUCCESS_TOLERANCES
    print(f"Benchmark: {folds} folds x {n} problems, checkpoint {checkpoint}", flush=True)

    print("Cross-validating integrator, gradients…", flush=True)
    cv = {"integrator": crossval_integrator(), "gradient": crossval_gradient()}

    fold_results = []
    pooled = {k: {m: [] for m in ("loss", "miss", "fuel", "delta_v")} for k in ("neural", "optimal", "e_guidance")}
    gaps = []
    timing = []
    convex_pool = {m: [] for m in ("loss", "miss", "fuel", "delta_v")}
    convex_vs_opt = []
    for i in range(folds):
        x0, g = _fold(seed, i, n, plim, vlim)
        if i == 0:
            print("Cross-validating convex solvers…", flush=True)
            cv["solvers"] = crossval_solvers(x0, g)
            cv["kkt_consistency"] = kkt_consistency(meta, x0, g, lam)
            print(f"  KKT vs training discretization: {cv['kkt_consistency']}", flush=True)
        r = run_fold(model, params, x0, g, lam, convex_subset)
        sc = r["scores"]
        gap = (sc["neural"]["loss"] - sc["optimal"]["loss"]) / sc["optimal"]["loss"]
        gaps.append(gap)
        for k in pooled:
            for m in pooled[k]:
                pooled[k][m].append(sc[k][m])
        for m in convex_pool:
            convex_pool[m].append(sc["convex"][m])
        convex_vs_opt.append(np.abs(sc["convex"]["loss"] - sc["optimal"]["loss"][: r["convex_n"]]))
        timing.append(r["timing_ms_per_problem"])
        fold_results.append({
            "fold": i,
            "neural_mean_loss": float(sc["neural"]["loss"].mean()),
            "optimal_mean_loss": float(sc["optimal"]["loss"].mean()),
            "e_guidance_mean_loss": float(sc["e_guidance"]["loss"].mean()),
            "neural_gap_pct_mean": float(100 * gap.mean()),
            "neural_mean_miss": float(sc["neural"]["miss"].mean()),
        })
        print(f"  fold {i}: gap {100 * gap.mean():.3f}%  NN miss {sc['neural']['miss'].mean():.4f}", flush=True)

    S = {k: {m: np.concatenate(v[m]) for m in v} for k, v in pooled.items()}
    S["convex"] = {m: np.concatenate(v) for m, v in convex_pool.items()}
    gaps = np.concatenate(gaps)
    fold_gap_means = [f["neural_gap_pct_mean"] for f in fold_results]

    methods = {k: _method_stats(S[k], tols) for k in ("neural", "convex", "optimal", "e_guidance")}

    def deltas(a, b):
        out = {}
        for m in ("loss", "miss", "fuel", "delta_v"):
            ma, mb = S[a][m].mean(), S[b][m].mean()
            out[m] = {"abs": float(ma - mb), "pct": float(100 * (ma - mb) / mb) if mb else None}
        return out

    timing_avg = {k: float(np.mean([t[k] for t in timing])) for k in timing[0]}
    lat_x, lat_g = _fold(seed, 0, 64, plim, vlim)
    timing_avg["neural_single_gpu_latency"] = neural_single_latency_ms(model, params, lat_x, lat_g)
    timing_avg["neural_single_cpu_latency"] = neural_single_latency_ms(model, params, lat_x, lat_g, device=jax.devices("cpu")[0])
    timing_avg["neural_single_best_latency"] = min(timing_avg["neural_single_gpu_latency"], timing_avg["neural_single_cpu_latency"])

    # Paired per-problem comparison NN vs E-guidance on the same problems
    nn_beats_eg = float((S["neural"]["loss"] < S["e_guidance"]["loss"]).mean())

    # Stress: 2x training velocity range (out of distribution)
    xs, gs = _fold(seed, 99, n, plim, 2 * vlim)
    rs = run_fold(model, params, xs, gs, lam, 0)["scores"]
    stress = {
        "velocity_limit": 2 * vlim,
        "neural_gap_pct_mean": float(100 * ((rs["neural"]["loss"] - rs["optimal"]["loss"]) / rs["optimal"]["loss"]).mean()),
        "neural_mean_miss": float(rs["neural"]["miss"].mean()),
        "optimal_mean_miss": float(rs["optimal"]["miss"].mean()),
        "e_guidance_gap_pct_mean": float(100 * ((rs["e_guidance"]["loss"] - rs["optimal"]["loss"]) / rs["optimal"]["loss"]).mean()),
    }

    cv["convex_vs_exact_on_benchmark"] = {"max_abs_loss_diff": float(np.concatenate(convex_vs_opt).max())}
    cv["generalization"] = {
        "folds": folds,
        "problems_per_fold": n,
        "gap_pct_mean_per_fold": fold_gap_means,
        "gap_pct_mean": float(np.mean(fold_gap_means)),
        "gap_pct_std_across_folds": float(np.std(fold_gap_means, ddof=1)) if folds > 1 else 0.0,
        "gap_pct_bootstrap_95ci": [100 * v for v in _bootstrap_ci(gaps)],
        "gap_pct_median": float(100 * np.median(gaps)),
        "gap_pct_p95": float(100 * np.percentile(gaps, 95)),
        "gap_pct_p99": float(100 * np.percentile(gaps, 99)),
        "gap_definition": "per mission (J_nn - J*) / J*, then averaged (mean per-mission gap)",
        "delta_pct_definition": "(mean J_a - mean J_b) / mean J_b (ratio of means)",
    }

    results = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(checkpoint),
        "model": {k: meta.get(k) for k in ("size", "n_params", "step", "steps_completed", "train_seconds", "batch_size", "learning_rate", "device")},
        "problem": {
            "n_steps": config.N_STEPS, "dt": config.DT, "horizon": config.N_STEPS * config.DT,
            "max_thrust": config.MAX_THRUST, "lambda_fuel": lam,
            "position_limit": plim, "velocity_limit": vlim,
            "dynamics": "planar point mass (double integrator), state [x, y, vx, vy], "
                        "v' = (F/m) u with F/m = 1 m/s^2, no gravity or drag; "
                        "thrust held constant over each step, integrated with RK4",
            "objective": "J = ||p_N - goal||^2 + lambda * sum_k ||u_k||^2",
            "control_bound": "|u_x|, |u_y| <= 1 per step",
            "terminal_condition": "soft: terminal position error is penalized in J, terminal velocity is free; "
                                  "E-guidance instead drives the terminal miss to zero by construction (exact rendezvous)",
            "test_missions": folds * n,
            "sampling": f"{folds} disjoint folds x {n} missions; start and goal positions uniform in "
                        f"[-{plim}, {plim}]^2 m, initial velocity uniform in [-{vlim}, {vlim}]^2 m/s, "
                        f"drawn from PRNG streams fold_in(seed, {TEST_STREAM_BASE}+i) that training never uses",
        },
        "environment": {"platform": platform.platform(), "jax": jax.__version__, "device": str(jax.devices()[0])},
        "methods": methods,
        "deltas": {
            "neural_minus_convex": deltas("neural", "convex"),
            "neural_minus_optimal": deltas("neural", "optimal"),
            "neural_minus_e_guidance": deltas("neural", "e_guidance"),
            "e_guidance_minus_optimal": deltas("e_guidance", "optimal"),
        },
        "neural_beats_e_guidance_fraction": nn_beats_eg,
        "timing_ms_per_problem": timing_avg,
        "speedup_neural_vs_convex": {
            "batched": timing_avg["convex_clarabel_median"] / max(timing_avg["neural_batched_gpu"], 1e-9),
            "single": timing_avg["convex_clarabel_median"] / max(timing_avg["neural_single_best_latency"], 1e-9),
        },
        "stress_out_of_distribution": stress,
        "cross_validation": cv,
        "folds": fold_results,
    }

    out_json = Path(out_json or PACKAGE_ROOT / "outputs" / "benchmark_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    out_md = Path(out_md or PACKAGE_ROOT.parent / "BENCHMARK.md")
    out_md.write_text(render_markdown(results), encoding="utf-8")
    print(f"Wrote {out_json}\nWrote {out_md}", flush=True)
    return results


def _f(v, d=4):
    return "—" if v is None else f"{v:.{d}f}"


def render_markdown(r):
    M, D, T, CV = r["methods"], r["deltas"], r["timing_ms_per_problem"], r["cross_validation"]
    names = {"neural": "OTiS (neural net)", "convex": "Convex optimization (CVXPY + Clarabel)",
             "optimal": "Exact QP optimum (KKT)", "e_guidance": "Apollo E-guidance (ZEM law)"}
    tol_keys = list(M["neural"]["success"].keys())
    g = CV["generalization"]
    lines = [
        "# OTiS benchmark",
        "",
        f"Generated {r['generated']} by `python main.py --benchmark`. Raw numbers: `otis/outputs/benchmark_results.json`.",
        "",
        f"Model: **{r['model']['size']}**, {r['model']['n_params']:,} parameters, best step {r['model']['step']} of {r['model']['steps_completed']}, trained {_f(r['model']['train_seconds'], 0)} s on `{r['model']['device']}`.",
        f"Problem: {r['problem']['n_steps']} steps × {r['problem']['dt']} s, |u| ≤ {r['problem']['max_thrust']}, λ = {r['problem']['lambda_fuel']}, start/goal in ±{r['problem']['position_limit']}, initial velocity in ±{r['problem']['velocity_limit']}.",
        f"Test set: {g['folds']} disjoint folds × {g['problems_per_fold']} problems drawn from PRNG streams the trainer never touches. Every method, including the convex solver, runs on every problem; Clarabel matches the exact optimum to {CV['convex_vs_exact_on_benchmark']['max_abs_loss_diff']:.1e} in objective.",
        "",
        "Every method is scored by rolling its thrust plan through the same RK4 simulator with the same objective J = ‖p_T − goal‖² + λ Σ‖u‖².",
        "",
        "## Head-to-head (means over all test problems)",
        "",
        "| Method | Objective J | Miss (m) | p99 miss (m) | Control effort Σu² | Δv (m/s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for k in ("neural", "convex", "optimal", "e_guidance"):
        m = M[k]
        lines.append(
            f"| {names[k]} | {_f(m['loss']['mean'], 5)} | {_f(m['miss']['mean'], 4)} | {_f(m['miss'].get('p99'), 4)} | {_f(m['fuel']['mean'], 3)} | {_f(m['delta_v']['mean'], 3)} |"
        )
    lines += ["", "## Numerical deltas", "", "ΔJ % is a ratio of means: (mean J_a − mean J_b) / mean J_b.", "",
              "| Comparison | ΔJ | ΔJ % (of mean) | Δ miss (m) | Δ control effort | Δ control effort % | Δ Δv % |", "| --- | --- | --- | --- | --- | --- | --- |"]
    labels = {"neural_minus_convex": "Neural − convex optimization (status quo)",
              "neural_minus_optimal": "Neural − exact optimum (optimality gap)",
              "neural_minus_e_guidance": "Neural − Apollo E-guidance",
              "e_guidance_minus_optimal": "E-guidance − exact optimum"}
    for k, lab in labels.items():
        d = D[k]
        lines.append(f"| {lab} | {d['loss']['abs']:+.2e} | {d['loss']['pct']:+.3f}% | {d['miss']['abs']:+.2e} | {d['fuel']['abs']:+.2e} | {d['fuel']['pct']:+.3f}% | {d['delta_v']['pct']:+.3f}% |")
    lines += [
        "",
        f"Per problem, the network scores a lower objective than E-guidance on **{100 * r['neural_beats_e_guidance_fraction']:.1f}%** of test cases.",
        f"Mean per-mission optimality gap of the network (each mission's (J_nn − J*)/J*, then averaged): {g['gap_pct_mean']:.3g}% ± {g['gap_pct_std_across_folds']:.3g}% across folds, bootstrap 95% CI [{g['gap_pct_bootstrap_95ci'][0]:.3g}%, {g['gap_pct_bootstrap_95ci'][1]:.3g}%], median {g['gap_pct_median']:.3g}%, p95 {g['gap_pct_p95']:.3g}%, p99 {g.get('gap_pct_p99', float('nan')):.3g}%.",
        "",
        "KKT referee vs training: " + (
            f"same discretization (dt = {CV['kkt_consistency']['kkt_dt']} s, N = {CV['kkt_consistency']['kkt_n_steps']}, checkpoint plant matches); the KKT closed-form terminal map agrees with the training RK4 simulator to {CV['kkt_consistency']['max_abs_terminal_pos_kkt_map_vs_rk4']:.1e} m on {CV['kkt_consistency']['cases']} KKT plans."
            if CV.get("kkt_consistency", {}).get("consistent") else f"MISMATCH: {CV.get('kkt_consistency')}"),
        "",
        "## Compute time per plan",
        "",
        "| Method | ms per problem |",
        "| --- | --- |",
        f"| Neural net, single problem, GPU | {T['neural_single_gpu_latency']:.3f} |",
        f"| Neural net, single problem, CPU (Ryzen) | {T['neural_single_cpu_latency']:.3f} |",
        f"| Neural net, batched | {T['neural_batched_gpu']:.5f} |",
        f"| Convex optimization (Clarabel, median) | {T['convex_clarabel_median']:.3f} |",
        f"| E-guidance, batched closed loop | {T['e_guidance_batched']:.5f} |",
        f"| Exact KKT, vectorized NumPy | {T['optimal_kkt_vectorized']:.5f} |",
        "",
        f"Network vs convex solver: {r['speedup_neural_vs_convex']['single']:.2f}× for one plan at a time (best of GPU/CPU), {r['speedup_neural_vs_convex']['batched']:.0f}× when many plans are batched on the GPU. For this small linear problem a single convex solve is already about a millisecond, so the honest claim is batch throughput and constant cost, not single-plan latency.",
        "",
        "## Cross-validation",
        "",
        "| Check | Result |",
        "| --- | --- |",
        f"| RK4 (float32) vs closed form, {CV['integrator']['cases']} random plans | max abs error {CV['integrator']['max_abs_rk4_vs_closed_form']:.2e} |",
        f"| RK4 (float32) vs SciPy DOP853 (rtol 1e-12) | max abs error {CV['integrator']['max_abs_rk4_vs_scipy_dop853']:.2e} |",
        f"| Closed form vs DOP853 (both float64) | max abs error {CV['integrator']['max_abs_closed_form_vs_dop853']:.2e} |",
        f"| Autodiff vs finite differences, {CV['gradient']['checks']} entries | max abs error {CV['gradient']['max_abs_autodiff_vs_finite_difference']:.2e} |",
        f"| Clarabel vs exact KKT, {CV['solvers']['cases']} problems | max Δu {CV['solvers']['clarabel']['max_abs_control_diff_vs_exact']:.1e}, max ΔJ {CV['solvers']['clarabel']['max_abs_loss_diff_vs_exact']:.1e} |",
        f"| OSQP vs exact KKT, {CV['solvers']['cases']} problems | max Δu {CV['solvers']['osqp']['max_abs_control_diff_vs_exact']:.1e}, max ΔJ {CV['solvers']['osqp']['max_abs_loss_diff_vs_exact']:.1e} |",
        f"| Optimality gap per fold | " + ", ".join(f"{v:.3f}%" for v in g["gap_pct_mean_per_fold"]) + " |",
        "",
        "## Out-of-distribution stress test",
        "",
        f"Initial velocity drawn from ±{r['stress_out_of_distribution']['velocity_limit']} (2× the training range). Network optimality gap {r['stress_out_of_distribution']['neural_gap_pct_mean']:.2f}% (in distribution: {g['gap_pct_mean']:.2f}%), mean miss {r['stress_out_of_distribution']['neural_mean_miss']:.4f} vs optimum {r['stress_out_of_distribution']['optimal_mean_miss']:.4f}. E-guidance gap on the same set: {r['stress_out_of_distribution']['e_guidance_gap_pct_mean']:.2f}%.",
        "",
        "## How to read this honestly",
        "",
        "- Two averages appear above. \"ΔJ %\" compares mean objectives across all missions; the optimality gap averages each mission's own relative gap, which weights short, cheap missions more. Both are reported so neither hides the other.",
        "- The exact KKT solver exists only because this plant is linear. It is the referee, not a competitor: it proves where the true optimum is so the network's gap is measured, not estimated.",
        "- The status-quo comparison is the convex solver (how flight systems such as G-FOLD and Falcon 9 landing guidance solve this class of problem) and Apollo-heritage E-guidance (the classic closed-form law). See SOURCES.md.",
        "- The network's advantage is a fixed, tiny inference cost that does not grow with constraints or nonlinearity, which is where per-problem solvers get expensive. Its cost is a small optimality gap and no hard guarantee.",
        "- Timing was measured on the machine listed in the JSON, so the numbers change across hardware.",
        "",
    ]
    return "\n".join(lines)
