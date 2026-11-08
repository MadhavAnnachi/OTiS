"""Industry-standard baselines for the same guidance problem.

Every baseline is scored by rolling its thrust plan through the *same* RK4
simulator and the *same* loss the network is trained on, so the numbers are
directly comparable.

1. Convex optimization (direct transcription)  — ``qp_solve_cvxpy``
   The status quo for fuel-optimal powered-descent guidance is to transcribe
   the problem into a convex program and solve it on board with an
   interior-point method (Açıkmeşe & Ploen 2007; Blackmore et al. 2010;
   G-FOLD flight tests 2012-2014; Malyuta et al. 2022). Our objective
   ``||p_T - g||^2 + λ||U||^2`` with ``|u_i| <= 1`` is a box-constrained QP,
   solved here with CVXPY + Clarabel (an interior-point conic solver), with
   OSQP as an independent second solver for cross-validation.

2. Exact optimum via KKT  — ``qp_solve_exact``
   Because the plant is linear, the QP separates per axis and its KKT
   conditions reduce to ``u_k = clip(b_k s / λ, -1, 1)`` with a scalar ``s``
   found by bisection. This is the provable global optimum, used to
   cross-check the convex solvers and to compute optimality gaps.

3. Apollo-heritage E-guidance / ZEM feedback law  — ``e_guidance_rollout``
   The closed-form energy-optimal guidance law (Cherry 1964; Klumpp 1974;
   D'Souza 1997) for a free terminal velocity:
       a = 3 * ZEM / t_go^2,   ZEM = r_f - (r + v t_go)
   applied in closed loop at every step and clipped to the thrust limit.
   It aims to hit the target exactly, so it trades fuel for accuracy.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

try:
    from . import config
    from .simulator import rk4_step, simulate, terminal_map
except ImportError:
    import config
    from simulator import rk4_step, simulate, terminal_map


# --------------------------------------------------------------------------
# Exact KKT solver (vectorized over problems, float64)
# --------------------------------------------------------------------------

def _residual_target(x0s, goals, horizon):
    """``r = g - p0 - v0 T``: what the thrust has to make up, per axis."""

    x0s = np.asarray(x0s, dtype=np.float64).reshape(-1, 4)
    goals = np.asarray(goals, dtype=np.float64).reshape(-1, 2)
    return goals - x0s[:, :2] - x0s[:, 2:] * horizon


def qp_solve_exact(
    x0s,
    goals,
    lambda_fuel: float = config.LAMBDA_FUEL,
    max_thrust: float = config.MAX_THRUST,
    n_steps: int = config.N_STEPS,
    dt: float = config.DT,
    iters: int = 100,
):
    """Global optimum of the box-constrained QP for a batch of problems.

    Returns controls of shape ``(B, n_steps, 2)``.

    Per axis:  minimize (b·u - r)^2 + λ||u||^2  s.t. |u_k| <= u_max.
    Stationarity gives u_k = clip(b_k s / λ, ±u_max) with s = r - b·u, and
    h(s) = s + Σ b_k clip(b_k s/λ) - r is strictly increasing, so bisection
    on s ∈ [min(0,r), max(0,r)] converges to the unique optimum.
    """

    b, _c, horizon = terminal_map(n_steps, dt)
    r = _residual_target(x0s, goals, horizon)  # (B, 2)
    lo = np.minimum(0.0, r)
    hi = np.maximum(0.0, r)
    for _ in range(iters):
        s = 0.5 * (lo + hi)
        u = np.clip(b[None, None, :] * s[..., None] / lambda_fuel, -max_thrust, max_thrust)
        h = s + np.einsum("k,bak->ba", b, u) - r
        hi = np.where(h > 0, s, hi)
        lo = np.where(h > 0, lo, s)
    s = 0.5 * (lo + hi)
    u = np.clip(b[None, None, :] * s[..., None] / lambda_fuel, -max_thrust, max_thrust)
    return np.transpose(u, (0, 2, 1))  # (B, n_steps, 2)


# --------------------------------------------------------------------------
# CVXPY / Clarabel (industry-style direct transcription)
# --------------------------------------------------------------------------

class ConvexGuidance:
    """Direct-transcription convex guidance, compiled once and re-solved.

    The problem is built with CVXPY parameters (disciplined parametrized
    programming), so each new start/goal only updates numbers, the same way
    an on-board solver re-solves a fixed problem structure every cycle.
    """

    def __init__(
        self,
        lambda_fuel: float = config.LAMBDA_FUEL,
        max_thrust: float = config.MAX_THRUST,
        n_steps: int = config.N_STEPS,
        dt: float = config.DT,
        solver: str = "CLARABEL",
        b=None,
    ):
        """``b`` overrides the per-step influence of thrust on the terminal
        position (default: the unit plant's closed form). Any linear plant
        (mass, thrust limit, gravity, linear drag) reduces to the same QP
        with its own ``b``; gravity only moves the free-flight target, which
        ``solve_residual`` takes directly."""

        import cvxpy as cp

        b_unit, c, horizon = terminal_map(n_steps, dt)
        b = b_unit if b is None else np.asarray(b, dtype=np.float64)
        self.horizon = horizon
        self.solver = solver
        self.u = cp.Variable((n_steps, 2))
        self.r = cp.Parameter(2)
        p_terminal_offset = b @ self.u  # (2,) exact terminal displacement from thrust
        objective = cp.sum_squares(p_terminal_offset - self.r) + lambda_fuel * cp.sum_squares(self.u)
        constraints = [cp.abs(self.u) <= max_thrust]
        self.problem = cp.Problem(cp.Minimize(objective), constraints)
        self._cp = cp

    def solve(self, x0, goal):
        """Return ``(controls (n_steps, 2), solve_seconds)``."""

        return self.solve_residual(_residual_target(x0, goal, self.horizon)[0])

    def solve_residual(self, residual):
        """Solve for a terminal displacement target ``residual`` (2,)."""

        self.r.value = np.asarray(residual, dtype=np.float64)
        t0 = time.perf_counter()
        kwargs = {"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iter": 200000} if self.solver == "OSQP" else {}
        self.problem.solve(solver=self.solver, warm_start=False, **kwargs)
        elapsed = time.perf_counter() - t0
        if self.u.value is None:
            raise RuntimeError(f"{self.solver} failed: {self.problem.status}")
        return np.asarray(self.u.value, dtype=np.float64), elapsed


# --------------------------------------------------------------------------
# Apollo-heritage E-guidance (ZEM feedback), closed loop in the simulator
# --------------------------------------------------------------------------

@jax.jit
def e_guidance_rollout(x0, goal):
    """Closed-loop E-guidance. Returns the controls it issued, ``(N, 2)``.

    At step k the time-to-go is ``t_go = T - k dt``; the commanded
    acceleration is ``3 (g - p - v t_go) / t_go^2``, clipped to the thrust
    box, held constant over the step, and integrated with the same RK4.
    """

    dt = config.DT
    horizon = config.N_STEPS * dt
    ks = jnp.arange(config.N_STEPS, dtype=jnp.float32)

    def step(state, k):
        t_go = horizon - k * dt
        zem = goal - (state[:2] + state[2:] * t_go)
        u = jnp.clip(3.0 * zem / (t_go * t_go), -config.MAX_THRUST, config.MAX_THRUST)
        return rk4_step(state, u, dt), u

    _, controls = jax.lax.scan(step, x0, ks)
    return controls


e_guidance_batch = jax.jit(jax.vmap(e_guidance_rollout))


# --------------------------------------------------------------------------
# Shared scoring through the project simulator
# --------------------------------------------------------------------------

@jax.jit
def _score_one(controls, x0, goal, lambda_fuel):
    final, traj = simulate(controls, x0)
    miss = jnp.linalg.norm(final[:2] - goal)
    fuel = jnp.sum(controls**2)
    delta_v = jnp.sum(jnp.linalg.norm(controls, axis=-1)) * config.DT
    loss = miss**2 + lambda_fuel * fuel
    path = jnp.concatenate([x0[None, :], traj], axis=0)
    return loss, miss, fuel, delta_v, final, path


score_batch = jax.jit(jax.vmap(_score_one, in_axes=(0, 0, 0, None)))


def score(controls, x0s, goals, lambda_fuel: float = config.LAMBDA_FUEL):
    """Score a batch of plans in the project simulator. Returns a dict of arrays."""

    loss, miss, fuel, dv, final, path = score_batch(
        jnp.asarray(controls, dtype=jnp.float32),
        jnp.asarray(x0s, dtype=jnp.float32),
        jnp.asarray(goals, dtype=jnp.float32),
        lambda_fuel,
    )
    return {
        "loss": np.asarray(loss, dtype=np.float64),
        "miss": np.asarray(miss, dtype=np.float64),
        "fuel": np.asarray(fuel, dtype=np.float64),
        "delta_v": np.asarray(dv, dtype=np.float64),
        "final": np.asarray(final),
        "path": np.asarray(path),
    }
