"""Proximal Policy Optimization algorithm.

Re-exports gae_compute and ppo from the ppo module.
"""

from .ppo import ppo, gae_compute
from  .easy_ppo import easy_train_ppo

__all__ = ["ppo", "gae_compute", "easy_train_ppo"]
