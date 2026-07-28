"""Abstract base environment interface for reinforcement learning.

Provides BaseEnv, a Gymnasium v1 API wrapper that all environment
implementations must extend.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from abc import ABC, abstractmethod
from typing import Any


class BaseEnv(gym.Env, ABC):
    """Abstract base class for all RL environments.

    Follows the Gymnasium v1 API where step() returns 5 values:
    (obs, reward, terminated, truncated, info). Subclasses must set
    observation_space and action_space in their __init__().
    """

    def __init__(self):
        self.observation_space: spaces.Space = None
        self.action_space: spaces.Space = None


    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to its initial state.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Tuple of (observation, info).
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: The action to take.

        Returns:
            5-tuple of (observation, reward, terminated, truncated, info).
        """
        pass

    @abstractmethod
    def close(self):
        """Release environment resources (render windows, connections, etc.)."""
        pass
