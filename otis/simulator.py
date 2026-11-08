"""Differentiable 2D spacecraft dynamics.

The state is ``[x, y, vx, vy]`` and the control is a thrust ``[ux, uy]``.
Mass is 1, so thrust is acceleration. Integration is a fixed-step RK4, and
``simulate`` rolls a whole thrust sequence out with ``jax.lax.scan`` so the
trajectory stays inside the JAX graph. Gradients of a terminal cost flow
straight back into every control.
"""

import jax
import jax.numpy as jnp

try:
    from .config import DT, MAX_THRUST, N_STEPS
except ImportError:  # ``python main.py`` from this directory
    from config import DT, MAX_THRUST, N_STEPS

# ``(accel_gain, gravity, drag)``: accel_gain = max thrust / mass [m/s^2 per
# unit throttle], gravity [m/s^2, acting in -y], drag = linear drag
# coefficient / mass [1/s]. The default is the published unit plant.
DEFAULT_PLANT = (1.0, 0.0, 0.0)


def rk4_step(state, control, dt, plant=DEFAULT_PLANT):
    """Advance one RK4 step of the (generalized) double integrator.

    Continuous equations (mass ``m = 1``):

    - ``x_dot = vx``
    - ``y_dot = vy``
    - ``vx_dot = ux / m``
    - ``vy_dot = uy / m``

    RK4 is exact for this piecewise-constant-acceleration system, which keeps
    the backpropagated gradient faithful over the whole horizon. A forward
    Euler step would add a systematic drift that the loss would then fit.

    ``plant`` generalizes this to user parameters:
    ``v_dot = gain * u - drag * v - [0, gravity]``. With the default
    ``(1, 0, 0)`` the arithmetic is identical to the unit plant above.
    With drag the system is no longer polynomial, so RK4 is 4th-order
    accurate rather than exact.
    """

    gain, gravity, drag = plant

    def deriv(s):
        return jnp.array([
            s[2],
            s[3],
            gain * control[0] - drag * s[2],
            gain * control[1] - gravity - drag * s[3],
        ])

    k1 = deriv(state)
    k2 = deriv(state + dt / 2 * k1)
    k3 = deriv(state + dt / 2 * k2)
    k4 = deriv(state + dt * k3)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


@jax.jit
def simulate(controls, x0, dt=DT, plant=DEFAULT_PLANT):
    """Integrate ``controls`` from ``x0``.

    ``controls`` has shape ``(N_steps, 2)`` and ``x0`` has shape ``(4,)``.
    Returns ``(final_state, trajectory)`` where ``trajectory`` stacks the
    state after every step, shape ``(N_steps, 4)``. It does not include
    ``x0``. ``N_steps`` is whatever leading dimension ``controls`` has; the
    trainer always passes the compile-time constant 50.

    ``jax.lax.scan`` is the whole rollout. There is no Python loop, so a
    later ``jax.grad`` can see every RK4 stage.
    """

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
