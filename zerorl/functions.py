"""Utility functions for RL training.

Provides vectorize_env() for environment wrapping, fast_compile() for
optional torch.compile, get_obs_act() for space extraction, and
get_buffer_params_model() for extracting model parameters.
"""

import os
import copy
import gymnasium as gym
import torch
import numpy as np
from typing import Any, Callable
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from torch import Tensor
from torch.nn import Parameter
from zerorl.helpers.agent import BaseAgent
from zerorl.helpers.env import BaseEnv
from zerorl.processing import NormMeanStd


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


#Function help agent to interact with his env
def env_step(env: Any, agent:BaseAgent, state:np.ndarray|Tensor, normalizer:NormMeanStd|None=None, device = torch.device("cpu")) -> dict[str, Tensor]:
    state_tensor = processing_state(state, normalizer, device)
    with torch.inference_mode():
        outputs: dict[str, Tensor] = agent.get_action(state_tensor) #type: ignore[operator]

    action = to_env_action(outputs["action"], env)
    # Gymnasium v1 step() returns: obs, reward, terminated, truncated, info
    # terminated = episode naturally ended; truncated = cut short by time limit
    next_state, reward, terminated, truncated, _ = env.step(action)
    return {"state": state, "next_state": next_state, "reward": reward, "terminated": terminated, "truncated": truncated, **outputs}


def save_checkpoints(agent: BaseAgent, model_path: str, normalizer: NormMeanStd | None = None):
    """Save agent weights and Normalizer state to the path in config.model_path."""
    checkpoints_state = {
        "agent_state_dict": agent.state_dict(),
        "normalizer_state_dict": normalizer.state_dict() if normalizer is not None else None
            }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(checkpoints_state, model_path)


def processing_state(state: np.ndarray | Tensor, normalizer: NormMeanStd | None = None, update: bool = True, device: torch.device = torch.device("cpu")) -> Tensor:
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
    if state_tensor.dim() == 1: state_tensor = state_tensor.unsqueeze(0)
    if normalizer is not None:
        if update: normalizer.update(state_tensor)
        state_tensor = normalizer.normalize(state_tensor)
    return state_tensor


def parse_env_step(output: dict[str, Tensor], device: torch.device = torch.device("cpu")) -> dict[str, Tensor]:
    keys = ["next_state", "reward", "terminated", "truncated"]
    for k in keys:
        value = output[k]
        output[k] = torch.as_tensor(value, dtype=torch.float32, device=device) 
        if output[k].dim() == 0: output.unsqueeze(0)
    return output


def to_env_action(action, env: Any) -> np.ndarray | Tensor:
    device = getattr(env, "device", "cpu")
    if str(device).startswith("cuda"):
        return action
    return action.cpu().numpy()


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
