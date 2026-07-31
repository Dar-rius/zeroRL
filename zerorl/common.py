"""Pre-allocated numpy rollout buffer for RL training.

Provides Buffer, which stores trajectory data in fixed-size numpy arrays
and converts them to PyTorch tensors for the PPO update step.
"""

import torch
from torch import Tensor


class Buffer:
    """Pre-allocated rollout buffer for collecting RL trajectory data.

    Stores 8 arrays (states, actions, old_log_probs, returns, advantages,
    rewards, values, dones) in pre-allocated numpy arrays with a slice
    pointer for O(1) insertion. After a full rollout, insert_returns()
    fills in GAE-computed returns and advantages, and get_all() converts
    everything to PyTorch tensors for the PPO update step.

    Example:
        buf = Buffer(step=2048, state_shape=(4,), action_shape=())
        for _ in range(2048):
            buf.insert(state, action, log_prob, reward, value, done)
        buf.insert_returns(returns, advantages)
        tensors = buf.get_all()
        buf.clear()
    """

    def __init__(self, 
                 step: int,
                 data: dict[str, tuple] = {}):
        """Initialize pre-allocated arrays.

        Args:
            step: Maximum number of timesteps (capacity).
        """
        self.step = step
        self.slice: int = 0
        self.data = {
                name: torch.zeros((self.step, *shape), dtype=torch.float32)
                for name, shape in data.items()
                }

    @property
    def size(self) -> int:
        """Return the current number of stored elements."""
        return self.slice

    def insert(self, **kwargs):
        if self.slice >= self.step:
            raise ValueError(f"Buffer is full (size={self.step}). Cannot insert more data.")
        for name, val in kwargs.items():
            if name in self.data:
                self.data[name][self.slice] = val
            else:
                raise ValueError("variable {name} don't exist in extras")

        self.slice += 1


    def get_all(self) -> dict[str, Tensor]:
        return {name: val[:self.slice] for name, val in self.data.items()}

    def clear(self):
        """Reset the buffer for reuse.

        Resets the slice pointer to 0. Underlying arrays are not zeroed;
        old data is overwritten by subsequent insert() calls.
        """
        self.slice = 0
