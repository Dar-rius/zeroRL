"""Pre-allocated rollout buffer for RL training.

Provides Buffer, which stores trajectory data in fixed-size PyTorch
tensors and converts them for the PPO update step.
"""

import torch
from zerorl.errors import KeyBufferError


class Buffer:
    """Pre-allocated rollout buffer for collecting RL trajectory data.

    Stores trajectory data in pre-allocated PyTorch tensors with a slice
    pointer for O(1) insertion. gae_compute() writes advantage and return
    directly into the buffer, then use get_all() to retrieve everything as
    PyTorch tensors for the PPO update step.

    Example:
        buf = Buffer(step=2048, data={"state": (4,), "action": ()})
        for _ in range(2048):
            buf.insert(state=..., action=..., reward=..., ...)
        tensors = buf.get_all()
        buf.clear()
    """

    def __init__(self,
                 capacity: int,
                 num_envs: int,
                 schema: dict[str, tuple],
                 device: torch.device = torch.device("cpu")):
        """Initialize pre-allocated arrays.

        Args:
            step: Maximum number of timesteps (capacity).
            data: Dict mapping field names to shape tuples (e.g. {"state": (4,), "action": ()}).
        """
        self.step = capacity
        self.num_envs = num_envs
        self.device_ = device
        self.slice: int = 0
        self.data = {
                name: torch.zeros((self.step, self.num_envs, *shape), dtype = torch.float32, device = device)
                for name, shape in schema.items()
                }

    @property
    def size(self): return self.slice

    @property
    def device(self): return self.device_

    def insert(self, **kwargs):
        """Insert one timestep of data into the buffer.

        Args:
            **kwargs: Keyword arguments matching the keys in self.data.

        Raises:
            ValueError: If the buffer is full.
            KeyBufferError: If a key doesn't exist in the buffer.
        """
        if self.slice >= self.step:
            raise ValueError(f"Buffer is full (size={self.step}). Cannot insert more data.")
        for name, val in kwargs.items():
            if name in self.data:
                self.data[name][self.slice] = val
            else:
                raise KeyBufferError(name, kwargs)

        self.slice += 1


    def get_all(self, reshape: bool = False) -> dict[str, torch.Tensor]:
        """Return all inserted data as a dict of sliced tensors."""
        sliced_data = {name: val[:self.slice] for name, val in self.data.items()}
        if reshape:
            return {k: v.reshape(-1,  *v.shape[2:]) for k, v in sliced_data.items()}
        return sliced_data

    def clear(self):
        """Reset the buffer for reuse.

        Resets the slice pointer to 0. Underlying arrays are not zeroed;
        old data is overwritten by subsequent insert() calls.
        """
        self.slice = 0
