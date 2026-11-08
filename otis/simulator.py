"""Differentiable 2D spacecraft dynamics.

The state is ``[x, y, vx, vy]`` and the control is a thrust ``[ux, uy]``.
Mass is 1, so thrust is acceleration. Integration uses an exact linear transition, and
``simulate`` rolls a whole thrust sequence out with ``jax.lax.scan`` so the
trajectory stays inside the JAX graph. Gradients of a terminal cost flow
straight back into every control.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp

try:
    from .config import DT, MAX_DECAY_PER_SUBSTEP, MAX_SUBSTEPS, MAX_THRUST, N_STEPS
except ImportError:  # ``python main.py`` from this directory
    from config import DT, MAX_DECAY_PER_SUBSTEP, MAX_SUBSTEPS, MAX_THRUST, N_STEPS

# ``(accel_gain, gravity, drag)``: accel_gain = max thrust / mass [m/s^2 per
# unit throttle], gravity [m/s^2, acting in -y], drag = linear drag
# coefficient / mass [1/s]. The default is the published unit plant.
DEFAULT_PLANT = (1.0, 0.0, 0.0)


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


def rk4_step(state, control, dt, plant=DEFAULT_PLANT):
    """Exact ZOH step of v' = gain*u - drag*v - [0, gravity].

    The legacy name is kept for callers; coefficients are static during JIT.
    """
    gain, gravity, drag = plant
    e, tau, q = point_mass_coefficients(drag, dt)
    acc = gain * control - jnp.array([0.0, gravity])
    return jnp.concatenate((state[:2] + tau * state[2:] + q * acc,
                            e * state[2:] + tau * acc))


def substeps_for(drag: float, dt: float) -> int:
    """Legacy RK4 stiffness-domain guard (0 means unsupported parameters).

    Exact linear propagation no longer needs sub-steps; retain the admission
    limit so extreme parameter combinations are still explicitly rejected.
    """

    z = max(float(drag), 0.0) * float(dt)
    n = max(1, math.ceil(z / MAX_DECAY_PER_SUBSTEP - 1e-12))
    return n if n <= MAX_SUBSTEPS else 0


@partial(jax.jit, static_argnames=("dt", "plant"))
def simulate(controls, x0, dt=DT, plant=DEFAULT_PLANT):
    """Integrate ``controls`` from ``x0``.

    ``controls`` has shape ``(N_steps, 2)`` and ``x0`` has shape ``(4,)``.
    Returns ``(final_state, trajectory)`` where ``trajectory`` stacks the
    state after every step, shape ``(N_steps, 4)``. It does not include
    ``x0``. ``N_steps`` is whatever leading dimension ``controls`` has; the
    trainer always passes the compile-time constant 50.

    ``jax.lax.scan`` is the whole rollout. There is no Python loop, so a
    later ``jax.grad`` can see every exact state transition.
    """

    dt = float(dt)
    plant = tuple(float(v) for v in plant)
    n_sub = substeps_for(plant[2], dt)
    if n_sub == 0:
        raise ValueError(f"drag x time step = {plant[2] * dt:.3g} is too stiff to integrate; "
                         f"lower drag or mission time")
    def step(state, control):
        new_state = rk4_step(state, control, dt, plant)
        return new_state, new_state

    final_state, trajectory = jax.lax.scan(step, x0, controls)
    return final_state, trajectory


def max_travel_distance(
    n_steps: int = N_STEPS,
    dt: float = DT,
    max_thrust: float = MAX_THRUST,
) -> float:
    """How far a craft at rest can get by thrusting in one direction.

    Under constant acceleration ``a`` for time ``T = n_steps * dt``,
    displacement from rest is ``0.5 * a * T**2``. That is the absolute
    reach of the planner; braking is not required because the loss only
    scores terminal position.
    """

    horizon = n_steps * dt
    return 0.5 * max_thrust * horizon * horizon


def terminal_map(n_steps: int = N_STEPS, dt: float = DT):
    """Exact linear map from a thrust sequence to the terminal state.

    For piecewise-constant thrust the double integrator has the closed form

        p_T = p_0 + v_0 T + sum_k b_k u_k,   b_k = dt * (T - (k + 1/2) dt)
        v_T = v_0 + sum_k dt * u_k

    RK4 reproduces this to round-off, which ``--check`` and the benchmark's
    cross-validation both confirm. The convex baseline uses ``b`` to pose
    the same objective as a quadratic program.
    """

    import numpy as np

    horizon = n_steps * dt
    k = np.arange(n_steps, dtype=np.float64)
    b = dt * (horizon - (k + 0.5) * dt)
    c = np.full(n_steps, dt, dtype=np.float64)
    return b, c, horizon
