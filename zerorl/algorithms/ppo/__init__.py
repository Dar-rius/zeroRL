"""Proximal Policy Optimization algorithm.

Re-exports gae_compute, ppo_func, and easy_train_ppo.
"""

from .ppo import gae_compute, ppo_func
from  .easy_ppo import easy_train_ppo

__all__ = ["gae_compute",  "ppo_func", "easy_train_ppo"]
