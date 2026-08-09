"""High-level DX helpers for quick RL prototypes.

Provides make_env() and prototype() so callers can wire a BaseTrain
without hand-rolling Buffer / update_weights for supported algorithms.
"""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
from gymnasium import spaces
from torch import Tensor

from zerorl.env import BaseEnv
from zerorl.vector_env import VectorEnv


class _GymEnv(BaseEnv):
    """Thin BaseEnv wrapper around a Gymnasium env id."""

    def __init__(self, env_id: str):
        super().__init__()
        self._env = gym.make(env_id)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        # BaseTrain batches obs → Discrete actions often arrive as shape (1,)
        if hasattr(action, "shape") and len(getattr(action, "shape", ())) > 0:
            if isinstance(self.action_space, spaces.Discrete):
                action = action.item() if hasattr(action, "item") else action[0]
        if isinstance(action, Tensor):
            action = action.detach().cpu().numpy()
        return self._env.step(action)

    def close(self):
        self._env.close()


def make_env(
    env: BaseEnv | str | Callable[..., BaseEnv] | None,
    *,
    is_vector: bool = False,
    num_envs: int = 1,
) -> BaseEnv:
    """Resolve an environment spec into a BaseEnv (optionally vectorized)."""
    if env is None:
        raise ValueError("env is required (BaseEnv instance, gym id str, or callable env class)")

    if is_vector:
        if isinstance(env, BaseEnv):
            raise ValueError(
                "is_vector=True requires a gym id str or callable env class, not a BaseEnv instance"
            )
        if isinstance(env, str) or callable(env):
            return VectorEnv(env, num_envs=num_envs)
        raise ValueError("env must be a string (Gymnasium ID) or a callable (Env class)")

    if isinstance(env, BaseEnv):
        return env
    if isinstance(env, str):
        return _GymEnv(env)
    if callable(env):
        return env()
    raise ValueError("env must be a BaseEnv, gym id str, or callable env class")
