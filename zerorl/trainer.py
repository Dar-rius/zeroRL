"""High-level DX helpers for quick RL prototypes.

Provides make_env() and prototype() so callers can wire a BaseTrain
without hand-rolling Buffer / update_weights for supported algorithms.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import Tensor, nn

from zerorl.agent import BaseAgent, eval_action
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


def _single_spaces(env: BaseEnv) -> tuple[spaces.Space, spaces.Space]:
    inner = getattr(env, "_env", None)
    if inner is not None and hasattr(inner, "single_observation_space"):
        return inner.single_observation_space, inner.single_action_space
    return env.observation_space, env.action_space


class _DefaultDiscreteAgent(BaseAgent):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.actor = nn.Linear(obs_dim, act_dim)
        self.critic = nn.Linear(obs_dim, 1)

    def forward(self, state: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor]:
        return self.actor(state), self.critic(state)

    @staticmethod
    def build_distribution(logits: Tensor):
        return torch.distributions.Categorical(logits=logits)

    def get_action(self, state: Tensor, action: Tensor | None = None, **kwargs: Any):
        logits, value = self.forward(state, **kwargs)
        dist = self.build_distribution(logits)
        if action is None:
            action = dist.sample()
        log_prob, dist_entropy = eval_action(dist, action)
        return {"action": action, "log_prob": log_prob, "entropy": dist_entropy, "value": value}


class _DefaultContinuousAgent(BaseAgent):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.mean_layer = nn.Linear(obs_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(obs_dim, 1)

    def forward(self, state: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor]:
        mean = self.mean_layer(state)
        std = self.log_std.exp().expand_as(mean)
        logits = torch.cat([mean, std], dim=-1)
        return logits, self.critic(state)

    @staticmethod
    def build_distribution(logits: Tensor):
        act_dim = logits.shape[-1] // 2
        mean = logits[..., :act_dim]
        std = logits[..., act_dim:]
        return torch.distributions.Normal(mean, std)

    def get_action(self, state: Tensor, action: Tensor | None = None, **kwargs: Any):
        logits, value = self.forward(state, **kwargs)
        dist = self.build_distribution(logits)
        if action is None:
            action = dist.sample()
        log_prob, dist_entropy = eval_action(dist, action)
        return {"action": action, "log_prob": log_prob, "entropy": dist_entropy, "value": value}


def _resolve_agent(agent: BaseAgent | None, env: BaseEnv) -> BaseAgent:
    if agent is not None:
        return agent
    obs_space, act_space = _single_spaces(env)
    if obs_space.shape is None:
        raise ValueError("default agent requires an observation space with a defined shape")
    obs_dim = int(np.prod(obs_space.shape))
    if isinstance(act_space, spaces.Discrete):
        return _DefaultDiscreteAgent(obs_dim, int(act_space.n))
    if isinstance(act_space, spaces.Box):
        act_dim = int(np.prod(act_space.shape))
        return _DefaultContinuousAgent(obs_dim, act_dim)
    raise ValueError(f"unsupported action space for default agent: {type(act_space)!r}")
