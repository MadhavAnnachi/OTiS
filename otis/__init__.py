"""OTiS: train a spacecraft controller by differentiating through physics.

Gradients come from JAX autodiff through an RK4 rollout. There is no
reinforcement-learning estimator.
"""

__version__ = "1.0.0"
