"""Utility functions for RL training.

Provides vectorize_env() for environment wrapping, fast_compile() for
optional torch.compile, get_obs_act() for space extraction, and
get_buffer_params_model() for extracting model parameters.
"""

import shutil
import copy
import sys
import gymnasium as gym
import torch
import numpy as np
from typing import Any, Callable, TypeVar
from gymnasium import spaces
from torch import Tensor
from torch.nn import Parameter
from zerorl.helpers.agent import BaseAgent
from zerorl.helpers.env import BaseEnv
from gymnasium.vector import AutoresetMode, SyncVectorEnv


F = TypeVar("F", bound=Callable[..., Any])

def _cxx_compiler_available() -> bool:
    """True if torch inductor can find a C++ compiler."""
    if sys.platform == "win32":
        return shutil.which("cl") is not None
    return (shutil.which("g++") is not None or
            shutil.which("c++") is not None or
            shutil.which("clang++") is not None)


def fast_compile(fn: F | None = None,  debug: bool = False, **kwargs) -> F | Callable:
    """Like torch.compile; no-op when a C++ compiler is not on PATH."""
    use_compile = _cxx_compiler_available()
    def wrap(f: F) -> F:
        if not use_compile or debug:
            return f
        return torch.compile(f, **kwargs)  # type: ignore[return-value]

    if fn is not None:
        return wrap(fn)
    return wrap


def vectorize_env(env_spec: str | Callable | BaseEnv, num_envs: int = 1, render_mode: str | None = None) -> SyncVectorEnv:
    """Wrap an env spec into a SyncVectorEnv with SAME_STEP autoreset.

    Args:
        env_spec: Gymnasium env ID string, BaseEnv class/instance, or callable.
        num_envs: Number of parallel environments.
        render_mode: Render mode for the environment.

    Returns:
        SyncVectorEnv wrapping num_envs independent copies.
    """
    def make_env_fn(seed:int) -> Callable:
        def _init():
            if isinstance(env_spec, str):
                env = gym.make(env_spec, render_mode = render_mode)
            elif isinstance(env_spec, type):
                env = env_spec()
            elif callable(env_spec):
                env = env_spec()
            else:
                env = copy.deepcopy(env_spec)
            env.reset(seed=seed)
            return env
        return _init
    return gym.vector.SyncVectorEnv([make_env_fn(i) for i in range(num_envs)], autoreset_mode=AutoresetMode.SAME_STEP)


def get_obs_act(env: SyncVectorEnv) -> Any:
    """Extract observation and action spaces from a vectorized environment.

    Args:
        env: A SyncVectorEnv or compatible vectorized environment.

    Returns:
        5-tuple of (obs_shape, act_shape, obs_n, act_n, is_discrete).
        obs_shape/act_shape are raw shape tuples from the spaces.
        obs_n is the observation dim (or the Image space for image obs).
        act_n is the action dim (int for discrete, product-of-shapes for continuous).
        is_discrete is a bool.
    """
    if hasattr(env, "single_observation_space"):
        obs_dim = env.single_observation_space
        act_dim = env.single_action_space
    else:
        obs_dim = env.observation_space
        act_dim = env.action_space

    is_discrete = isinstance(act_dim, spaces.Discrete)

    if is_discrete:
        act_n = act_dim.n #type: ignore
    else:
        act_n = int(np.prod(act_dim.shape)) #type: ignore

    if len(obs_dim.shape) == 3: #type: ignore
        obs_n = obs_dim
    else:
        obs_n = obs_dim.shape[-1] #type: ignore
    return (obs_dim.shape, act_dim.shape, obs_n, act_n, is_discrete)


def get_buffer_params_model(model: BaseAgent) -> tuple[dict[str, Parameter], dict[str, Tensor]]:
    """Extract named parameters and buffers from a model.

    Args:
        model: The neural network model.

    Returns:
        Tuple of (parameters dict, buffers dict).
    """
    return dict(model.named_parameters()), dict(model.named_buffers())
