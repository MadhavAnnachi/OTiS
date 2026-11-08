"""Open-loop thrust network.

The policy reads the initial state and the goal and writes the entire thrust
sequence in one forward pass:

    input   [x, y, vx, vy, goal_x, goal_y]                    (6,)
    features  normalized position, velocity, goal, goal - position,
              and the zero-effort miss  goal - (position + velocity * T)
    trunk   Dense(width) -> N x [LayerNorm -> Dense -> GELU -> Dense] + skip
    head    LayerNorm -> Dense(n_steps * 2) -> tanh -> * max_thrust

The residual trunk with pre-LayerNorm is what lets the default network go
to ~13 M parameters (width 1024, 6 blocks) without the optimization stalling.
``tanh`` keeps each command inside ``[-1, 1]``; a final clip enforces the
thrust limit even if the scale is edited.
"""

import jax.numpy as jnp
from flax import linen as nn

try:
    from . import config
except ImportError:
    import config


def _featurize(inputs, position_limit, velocity_limit, horizon, gravity=0.0):
    p = inputs[..., 0:2]
    v = inputs[..., 2:4]
    g = inputs[..., 4:6]
    vscale = max(float(velocity_limit), 1e-3)
    # Zero-effort miss; gravity drifts the free path by 0.5 g T^2 downward.
    drift = p + v * horizon
    if gravity:
        drift = drift - jnp.array([0.0, 0.5 * gravity * horizon * horizon])
    zem = g - drift
    return jnp.concatenate(
        [
            p / position_limit,
            v / vscale,
            g / position_limit,
            (g - p) / position_limit,
            zem / position_limit,
        ],
        axis=-1,
    )


class ResidualBlock(nn.Module):
    width: int

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.width)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.width)(h)
        return x + h


class PolicyNet(nn.Module):
    """Maps a 6-vector (or a batch of them) to a ``(n_steps, 2)`` thrust plan."""

    n_steps: int = config.N_STEPS
    max_thrust: float = config.MAX_THRUST
    width: int = config.MODEL_SIZES[config.DEFAULT_SIZE][0]
    depth: int = config.MODEL_SIZES[config.DEFAULT_SIZE][1]
    position_limit: float = config.POSITION_LIMIT
    velocity_limit: float = config.VELOCITY_LIMIT
    horizon: float = config.N_STEPS * config.DT
    gravity: float = 0.0

    @nn.compact
    def __call__(self, inputs):
        x = _featurize(inputs, self.position_limit, self.velocity_limit, self.horizon, self.gravity)
        x = nn.Dense(self.width)(x)
        for _ in range(self.depth):
            x = ResidualBlock(self.width)(x)
        x = nn.LayerNorm()(x)
        raw = nn.Dense(self.n_steps * 2)(x)
        raw = nn.tanh(raw)
        # (6,) -> (n_steps, 2);  (B, 6) -> (B, n_steps, 2)
        controls = jnp.reshape(raw, raw.shape[:-1] + (self.n_steps, 2))
        controls = controls * self.max_thrust
        return jnp.clip(controls, -self.max_thrust, self.max_thrust)


def build_policy(
    size: str | None = None,
    *,
    width: int | None = None,
    depth: int | None = None,
    n_steps: int = config.N_STEPS,
    max_thrust: float = config.MAX_THRUST,
    position_limit: float = config.POSITION_LIMIT,
    velocity_limit: float = config.VELOCITY_LIMIT,
    horizon: float | None = None,
    gravity: float = 0.0,
) -> PolicyNet:
    """Construct the module. ``size`` is a key of ``config.MODEL_SIZES``."""

    w, d = config.MODEL_SIZES[size or config.DEFAULT_SIZE]
    return PolicyNet(
        n_steps=n_steps,
        max_thrust=max_thrust,
        width=width or w,
        depth=d if depth is None else depth,
        position_limit=position_limit,
        velocity_limit=velocity_limit,
        horizon=n_steps * config.DT if horizon is None else float(horizon),
        gravity=float(gravity),
    )


def init_params(model: PolicyNet, key) -> dict:
    """Draw a fresh parameter pytree for a single 6-vector input."""

    dummy = jnp.zeros((6,), dtype=jnp.float32)
    return model.init(key, dummy)["params"]


def count_params(params) -> int:
    import jax

    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)))


def apply_policy(model: PolicyNet, params, x0, goal):
    """Run the network on one example or a batch (last axes 4 and 2)."""

    inputs = jnp.concatenate([x0, goal], axis=-1)
    return model.apply({"params": params}, inputs)
