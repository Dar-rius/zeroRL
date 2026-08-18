"""Utility functions for RL training.

Provides get_buffer_params_model() for extracting model parameters.
"""

import shutil
import copy
import sys
import gymnasium as gym
import torch
from typing import Any, Callable, TypeVar
from  gymnasium.vector import SyncVectorEnv
from torch import Tensor
from torch.nn import Parameter
from zerorl.agent import BaseAgent
from zerorl.helpers import BaseEnv


F = TypeVar("F", bound=Callable[..., Any])

def _cxx_compiler_available() -> bool:
    """True if torch inductor can find a C++ compiler."""
    if sys.platform != "win32":
        return True
    return shutil.which("cl") is not None


def fast_compile(fn: F | None = None, **kwargs) -> F | Callable[[F], F]:
    """Like torch.compile; no-op when a C++ compiler is not on PATH."""
    use_compile = _cxx_compiler_available()
    def wrap(f: F) -> F:
        if not use_compile:
            return f
        return torch.compile(f, **kwargs)  # type: ignore[return-value]

    if fn is not None:
        return wrap(fn)
    return wrap


#Vectorize Env
def vetorize_env(env_spec: str | Callable | BaseEnv, num_envs: int = 1, render_mode: str | None = None) -> SyncVectorEnv:
    def make_env_fn(seed:int) -> Callable:
        def _init():
            if isinstance(env_spec, str):
                env = gym.make(env_spec, render_mode = render_mode)
            elif callable(env_spec):
                env = env_spec()
            elif isinstance(env_spec, type):
                env = env_spec()
            else:
                env = copy.deepcopy(env_spec)
            env.reset(seed=seed)
            return env
        return _init
    return gym.vector.SyncVectorEnv([make_env_fn(i) for i in range(num_envs)])


def get_buffer_params_model(model: BaseAgent) -> tuple[dict[str, Parameter], dict[str, Tensor]]:
    """Extract named parameters and buffers from a model.

    Args:
        model: The neural network model.

    Returns:
        Tuple of (parameters dict, buffers dict).
    """
    return dict(model.named_parameters()), dict(model.named_buffers())
