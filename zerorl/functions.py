"""Utility functions for RL training.

Provides vectorize_env() for environment wrapping, fast_compile() for
optional torch.compile, get_obs_act() for space extraction, and
get_buffer_params_model() for extracting model parameters.
"""

import os
import copy
import random
import gymnasium as gym
import torch
import numpy as np
import imageio
from typing import Any, Callable
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from torch import Tensor
from torch.nn import Parameter
from zerorl.helpers.agent import BaseAgent
from zerorl.helpers.env import BaseEnv
from zerorl.processing import NormMeanStd
from zerorl.config import TrainConfig


def vectorize_env(env_spec: str | Callable | BaseEnv, *,  num_envs: int = 1, render_mode: str | None = None) -> SyncVectorEnv:
    """Wrap an env spec into a SyncVectorEnv with SAME_STEP autoreset.

    Args:
        env_spec: Gymnasium env ID string, BaseEnv class/instance, or callable.
        num_envs: Number of parallel environments.
        render_mode: Render mode for the environment.

    Returns:
        SyncVectorEnv wrapping num_envs independent copies.
    """
    def make_env_fn() -> Callable:
        def _init():
            if isinstance(env_spec, str):
                env = gym.make(env_spec, render_mode = render_mode)
            elif isinstance(env_spec, type):
                env = env_spec()
            elif callable(env_spec):
                env = env_spec()
            else:
                env = copy.deepcopy(env_spec)
            env.reset()
            return env
        return _init
    return gym.vector.SyncVectorEnv([make_env_fn() for _ in range(num_envs)], autoreset_mode=AutoresetMode.SAME_STEP)

#Function help agent to interact with his env
def env_step(env: Any, agent: BaseAgent, state_tensor: Tensor) -> dict[str, Tensor]:
    with torch.inference_mode():
        outputs: dict[str, Tensor] = agent.get_action(state_tensor) #type: ignore[operator]

    action = to_env_action(outputs["action"], env)
    # Gymnasium v1 step() returns: obs, reward, terminated, truncated, info
    # terminated = episode naturally ended; truncated = cut short by time limit
    next_state, reward, terminated, truncated, info = env.step(action)
    return {"state": state_tensor, "next_state": next_state, "reward": reward, "terminated": terminated, "truncated": truncated, "info": info, **outputs}

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

def parse_dict_to_tensor(output: dict[str, Tensor], device: torch.device = torch.device("cpu")) -> dict[str, Tensor]:
    keys = ["next_state", "reward", "terminated", "truncated"]
    for k in keys:
        value = output[k]
        output[k] = torch.as_tensor(value, dtype=torch.float32, device=device) 
        if output[k].dim() == 0: output[k] = output[k].unsqueeze(0)
    return output

def parse_to_tensor(value: int | float, device: torch.device = torch.device("cpu")) -> Tensor:
    output = torch.as_tensor(value, dtype=torch.float32, device=device)
    if output.dim() == 0: output = output.unsqueeze(0)
    return output

def to_env_action(action, env: Any) -> np.ndarray | Tensor:
    device = getattr(env, "device", "cpu")
    if str(device).startswith("cuda"):
        return action
    return action.cpu().numpy()

def set_seed(seed: int, num_envs: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sequence = np.random.SeedSequence(seed)
    return [int(child.generate_state(1)[0]) for child in sequence.spawn(num_envs)]

def try_agent(env: Any, agent: BaseAgent, config: TrainConfig, *, normalizer: NormMeanStd | None = None, iterations: int = 1, gif_path: str | None = None):
    """Evaluate the agent and save a GIF of its behavior.
    
    Args:
        iterations: Number of iterations.
        gif_path: Path to save the GIF.
    """
    env_spec = env
    #check if env has .env or .spec attributs
    if isinstance(env_spec, gym.vector.VectorEnv):
        try:
            env_spec = env_spec.envs[0].spec.id
        except:
            env_spec = env_spec.envs[0]

    env = vectorize_env(env_spec, render_mode = "rgb_array")
    frames: Any = []
    agent.eval()
    for i in range(iterations):
        done_or_trunc = False
        state, _ = env.reset() #type: ignore
        while not done_or_trunc:
            state_tensor = processing_state(state, normalizer, update = False, device = config.device)
            outputs = env_step(env, agent, state_tensor)
            #capture frames
            frame = env.render()
            if frame is not None: frames.append(frame[0])
            done_or_trunc = bool(np.any(outputs["terminated"]) or np.any(outputs["truncated"]))
            state = outputs["next_state"]

        #save to gif
        if gif_path is None:
            gif_path = f"./{config.project_name}_{i}.gif"
        else:
            gif_path = f"./{gif_path}_{i}.gif"
        imageio.mimsave(gif_path, frames, fps=25)
        env.close()

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
