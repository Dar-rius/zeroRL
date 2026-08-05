"""Proximal Policy Optimization algorithm.

Re-exports gae_compute and ppo from the ppo module.
"""

from .ppo import ppo, gae_compute

__all__ = ["ppo", "gae_compute"]
