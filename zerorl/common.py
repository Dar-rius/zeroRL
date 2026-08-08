"""Pre-allocated rollout buffer for RL training.

Provides Buffer, which stores trajectory data in fixed-size PyTorch
tensors and converts them for the PPO update step.
"""

import torch
from zerorl.erros import KeyBufferError


class Buffer:
    """Pre-allocated rollout buffer for collecting RL trajectory data.

    Stores trajectory data in pre-allocated PyTorch tensors with a slice
    pointer for O(1) insertion. Callers insert GAE-computed returns and
    advantages directly, then use get_all() to retrieve everything as
    PyTorch tensors for the PPO update step.

    Example:
        buf = Buffer(step=2048, data={"state": (4,), "actions": ()})
        for _ in range(2048):
            buf.insert(state=..., actions=..., reward=..., ...)
        tensors = buf.get_all()
        buf.clear()
    """

    def __init__(self,
                 step: int,
                 data: dict[str, tuple],
                 num_envs: int = 1,
                 device: torch.device = torch.device("cpu")):
        """Initialize pre-allocated arrays.

        Args:
            step: Maximum number of timesteps (capacity).
            data: Dict mapping field names to shape tuples (e.g. {"state": (4,), "action": ()}).
            device: Torch device to allocate tensors on.
        """
        self.step = step
        self.num_envs = num_envs
        self.slice: int = 0
        self.data = {
                name: torch.zeros((self.step, self.num_envs, *shape), dtype = torch.float32, device = device)
                for name, shape in data.items()
                }

    @property
    def size(self): return self.slice

    def insert(self, **kwargs):
        """Insert one timestep of data into the buffer.

        Args:
            **kwargs: Keyword arguments matching the keys in self.data.

        Raises:
            ValueError: If the buffer is full or a key doesn't exist.
        """
        if self.slice >= self.step:
            raise ValueError(f"Buffer is full (size={self.step}). Cannot insert more data.")
        for name, val in kwargs.items():
            if name in self.data:
                self.data[name][self.slice] = val
            else:
                raise KeyBufferError(name, kwargs)

        self.slice += 1


    def get_all(self) -> dict[str, torch.Tensor]:
        """Return all inserted data as a dict of sliced tensors."""
        return {name: val[:self.slice] for name, val in self.data.items()}

    def clear(self):
        """Reset the buffer for reuse.

        Resets the slice pointer to 0. Underlying arrays are not zeroed;
        old data is overwritten by subsequent insert() calls.
        """
        self.slice = 0
