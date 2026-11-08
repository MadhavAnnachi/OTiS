"""Terminal cost plus fuel.

The scalar used for training is exactly

    loss = ||p_T - goal||^2 + lambda_fuel * ||U||^2

``p_T`` is the position at the end of ``simulate``, so the loss depends on
the controls only through the differentiable rollout. Fuel is the sum of
squared thrust components over the horizon (both axes, every step).

``trajectory_loss`` is the public scalar. The metric helpers next to it
expose the same quantities separately so the trainer can plot terminal error
without a second rollout.
"""

import jax
import jax.numpy as jnp

try:
    from .config import DT
    from .simulator import DEFAULT_PLANT, simulate
except ImportError:
    from config import DT
    from simulator import DEFAULT_PLANT, simulate


def trajectory_loss(controls, x0, goal, lambda_fuel=0.01, dt=DT, plant=DEFAULT_PLANT):
    """Squared miss distance at the horizon plus a fuel penalty."""

    final_state, trajectory = simulate(controls, x0, dt, plant)
    terminal_error = jnp.sum((final_state[:2] - goal) ** 2)
    fuel_cost = jnp.sum(controls**2)
    return terminal_error + lambda_fuel * fuel_cost


def per_example_metrics(controls, x0, goal, lambda_fuel=0.01, dt=DT, plant=DEFAULT_PLANT):
    """Loss pieces for one rollout, plus the path that includes ``x0``."""

    final_state, trajectory = simulate(controls, x0, dt, plant)
    terminal_error = jnp.sum((final_state[:2] - goal) ** 2)
    fuel_cost = jnp.sum(controls**2)
    total = terminal_error + lambda_fuel * fuel_cost
    distance = jnp.linalg.norm(final_state[:2] - goal)
    path = jnp.concatenate([x0[None, :], trajectory], axis=0)
    return total, terminal_error, fuel_cost, distance, final_state, path


def batch_network_loss(params, apply_fn, x0s, goals, lambda_fuel=0.01, dt=DT, plant=DEFAULT_PLANT):
    """Mean ``trajectory_loss`` over a batch.

    ``jax.vmap`` maps the rollout across examples. ``apply_fn(params, x0, goal)``
    must return that example's ``(N_steps, 2)`` controls. The mean is what
    Optax differentiates.
    """

    def single(x0, goal):
        controls = apply_fn(params, x0, goal)
        return trajectory_loss(controls, x0, goal, lambda_fuel, dt, plant)

    return jnp.mean(jax.vmap(single)(x0s, goals))


def batch_network_metrics(params, apply_fn, x0s, goals, lambda_fuel=0.01, dt=DT, plant=DEFAULT_PLANT):
    """Batch means of loss, squared terminal error, fuel, and L2 distance."""

    def single(x0, goal):
        controls = apply_fn(params, x0, goal)
        total, terminal, fuel, distance, _, _ = per_example_metrics(
            controls, x0, goal, lambda_fuel, dt, plant
        )
        return total, terminal, fuel, distance

    totals, terminals, fuels, distances = jax.vmap(single)(x0s, goals)
    return (
        jnp.mean(totals),
        jnp.mean(terminals),
        jnp.mean(fuels),
        jnp.mean(distances),
    )
