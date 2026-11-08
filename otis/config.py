"""Shared constants for OTiS.

Plant: a unit-mass 2D double integrator, state ``[x, y, vx, vy]``, thrust
``[ux, uy]`` with each component in ``[-1, 1]``.

Horizon: 50 RK4 steps of ``dt = 0.08`` s = 4.0 s. With unit thrust from rest
the farthest reachable point is ``0.5 * T**2 = 8.0`` length units, which
covers the ``[-2, 2]`` start/goal square (diagonal 5.66) with margin even
when the craft starts moving at up to ``VELOCITY_LIMIT`` in each axis.

Model: sized for an RTX 5070 Ti (16 GB). The default residual MLP is
width 1024, depth 6 (~13 M parameters), trained on batches of 1024 fresh
problems per step. ``--size small`` restores the original ~0.6 M MLP.
"""

N_STEPS = 50
DT = 0.08
MASS = 1.0
MAX_THRUST = 1.0
LAMBDA_FUEL = 0.01

# Network presets: (width, residual blocks). Parameters are approximate.
MODEL_SIZES = {
    "small": (512, 1),    # ~0.6 M, the original Cursor-era network
    "medium": (768, 4),   # ~4.9 M
    "large": (1024, 6),   # ~13 M  (default on the 5070 Ti)
    "xl": (2048, 6),      # ~51 M
}
DEFAULT_SIZE = "large"

BATCH_SIZE = 1024
VAL_BATCH_SIZE = 2048
LEARNING_RATE = 6e-4
WARMUP_STEPS = 300
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
TRAIN_STEPS = 6000
LOG_EVERY = 50
VIZ_EVERY = 10

# Half-width of the square that starts and goals are drawn from.
POSITION_LIMIT = 2.0
# Initial velocity is uniform in [-VELOCITY_LIMIT, VELOCITY_LIMIT]^2.
VELOCITY_LIMIT = 0.5

SEED = 0
DEMO_CASES = 5
HELDOUT_CASES = 256
BENCHMARK_CASES = 1000
BENCHMARK_FOLDS = 5
# Success thresholds on terminal miss distance (length units).
SUCCESS_TOLERANCES = (0.05, 0.1, 0.5)
