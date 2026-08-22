"""Proximal Policy Optimization algorithm.

Re-exports gae_compute and ppo_func from the ppo module.
"""

from .ppo import gae_compute, ppo_func
from  .easy_ppo import easy_train_ppo

__all__ = ["gae_compute",  "ppo_func", "easy_train_ppo"]
