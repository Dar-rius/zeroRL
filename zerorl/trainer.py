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
from torch.optim.lr_scheduler import LambdaLR

from zerorl.agent import BaseAgent, eval_action
from zerorl.algorithms.ppo.ppo import gae_compute, ppo
from zerorl.common import Buffer
from zerorl.config import AlgoConfig, TrainConfig
from zerorl.env import BaseEnv
from zerorl.function import linear_schedule
from zerorl.train import BaseTrain
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


_TRAIN_KEYS = {"timestamp", "rollout_steps", "num_envs", "model_name", "model_save_path"}
_ALGO_KEYS = set(AlgoConfig.__annotations__)


def _ppo_buffer_shapes(obs_shape: tuple, action_shape: tuple) -> dict[str, tuple]:
    return {
        "state": tuple(obs_shape),
        "actions": tuple(action_shape),
        "old_log_probs": (),
        "reward": (),
        "done": (),
        "entropy": (),
        "value": (),
        "adv": (),
        "returns": (),
    }


def _make_ppo_update(train_config: TrainConfig):
    def update_weights(agent, buffer, optimizer, step, last_output, algo_config):
        all_data = buffer.get_all()
        last_value = last_output["value"]
        if last_value.dim() > 1:
            last_value = last_value.squeeze(-1)
        returns, advantages, _ = gae_compute(
            all_data["reward"],
            all_data["value"],
            last_value,
            all_data["done"],
            algo_config,
        )
        n = buffer.size
        buffer.data["returns"][:n] = returns
        buffer.data["adv"][:n] = advantages
        lr_factor = linear_schedule(step=step, num_update=train_config.num_update)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda _: lr_factor,
        )
        return ppo(
            agent,
            optimizer,
            buffer,
            algo_config,
            scheduler,
            batch_size=algo_config.batch_size,
            epochs=algo_config.epochs,
            device=agent.device,
        )

    return update_weights


_ALGO_REGISTRY = {
    "ppo": {
        "buffer_shapes": _ppo_buffer_shapes,
        "make_update_weights": _make_ppo_update,
    },
}


def prototype(
    *,
    algo: str = "ppo",
    agent: BaseAgent | None = None,
    env: BaseEnv | str | Callable[..., BaseEnv] | None = None,
    is_vector: bool = False,
    num_envs: int = 1,
    timestamp: int = 1_000_000,
    model_name: str = "model",
    model_save_path: str = "./checkpoints",
    rollout_steps: int = 2048,
    **hyper_params: Any,
) -> BaseTrain:
    if algo not in _ALGO_REGISTRY:
        raise ValueError(f"unknown algo {algo!r}; available: {sorted(_ALGO_REGISTRY)}")

    train_kwargs = {
        "model_name": model_name,
        "model_save_path": model_save_path,
        "timestamp": timestamp,
        "rollout_steps": rollout_steps,
        "num_envs": num_envs,
    }
    algo_kwargs: dict[str, Any] = {}
    for key, value in hyper_params.items():
        if key in _TRAIN_KEYS:
            train_kwargs[key] = value
        elif key in _ALGO_KEYS:
            algo_kwargs[key] = value
        else:
            raise TypeError(f"prototype() got an unexpected keyword argument {key!r}")

    resolved_env = make_env(env, is_vector=is_vector, num_envs=train_kwargs["num_envs"])
    resolved_agent = _resolve_agent(agent, resolved_env)
    train_config = TrainConfig(**train_kwargs)
    algo_config = AlgoConfig(**algo_kwargs)

    obs_space, act_space = _single_spaces(resolved_env)
    if obs_space.shape is None:
        raise ValueError("prototype requires an observation space with a defined shape")
    obs_shape = tuple(obs_space.shape)
    if isinstance(act_space, spaces.Discrete):
        action_shape: tuple = ()
    elif isinstance(act_space, spaces.Box):
        action_shape = tuple(act_space.shape)
    else:
        raise ValueError(f"unsupported action space: {type(act_space)!r}")

    spec = _ALGO_REGISTRY[algo]
    buffer = Buffer(
        step=train_config.rollout_steps,
        data=spec["buffer_shapes"](obs_shape, action_shape),
        num_envs=train_config.num_envs,
        device=train_config.device,
    )
    update_weights = spec["make_update_weights"](train_config)
    return BaseTrain(
        resolved_agent,
        resolved_env,
        buffer,
        update_weights,
        train_config,
        algo_config,
    )
