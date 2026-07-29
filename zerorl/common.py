"""Pre-allocated numpy rollout buffer for RL training.

Provides Buffer, which stores trajectory data in fixed-size numpy arrays
and converts them to PyTorch tensors for the PPO update step.
"""

import numpy as np
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
                 state_shape: tuple,
                 action_shape: tuple = (),
                 extra_shapes: dict[str, tuple] = {}):
        """Initialize pre-allocated arrays.

        Args:
            step: Maximum number of timesteps (capacity).
            state_shape: Shape of a single observation (excluding batch dim).
            action_shape: Shape of a single action (excluding batch dim).
                          Use () for scalar discrete actions.
        """
        self.step = step
        self.slice: int = 0
        self.states = torch.zeros((self.step, *state_shape), dtype=torch.float32)
        self.actions = torch.zeros((self.step, *action_shape), dtype=torch.float32)
        self.old_log_probs = torch.zeros(self.step, dtype=torch.float32)
        self.returns = torch.zeros(self.step, dtype=torch.float32)
        self.adv = torch.zeros(self.step, dtype=torch.float32)
        self.rewards = torch.zeros(self.step, dtype=torch.float32)
        self.values = torch.zeros(self.step, dtype=torch.float32)
        self.dones = torch.zeros(self.step, dtype=torch.float32)
        self.extra_shapes = extra_shapes
        self.extras = {
                name: torch.zeros((self.step, *shape), dtype=torch.float32)
                for name, shape in extra_shapes.items()
                }

    @property
    def size(self) -> int:
        """Return the current number of stored elements."""
        return self.slice

    def insert(self,
               state: Tensor,
               action: float | Tensor,
               old_log_prob: Tensor,
               reward: float,
               value: Tensor,
               dones: int,
               **kwargs):
        """Store a single timestep of rollout data.

        Args:
            state: Environment observation.
            action: Action taken by the agent.
            old_log_prob: Log probability of the action under the old policy.
            reward: Reward received from the environment.
            value: Critic value estimate for this state.
            dones: 1 if the episode ended, 0 otherwise.

        Raises:
            ValueError: If the buffer is already at capacity.
        """
        if self.slice >= self.step:
            raise ValueError(f"Buffer is full (size={self.step}). Cannot insert more data.")
        self.states[self.slice] = state
        self.actions[self.slice] = action
        self.old_log_probs[self.slice] = old_log_prob
        self.rewards[self.slice] = reward
        self.values[self.slice] = value
        self.dones[self.slice] = dones

        for name, val in kwargs.items():
            if name in self.extras:
                self.extras[name][self.slice] = val
            else:
                raise ValueError("variable {name} don't exist in extras")

        self.slice += 1

    def insert_returns(self, returns: np.ndarray, adv: np.ndarray):
        """Write GAE-computed returns and advantages into the buffer.

        Called after the rollout phase when GAE has been computed.

        Args:
            returns: Discounted return values for each timestep.
            adv: Generalized Advantage Estimation values for each timestep.
        """
        self.returns[:] = returns
        self.adv[:] = adv

    def get_all(self, device: str = "cpu") -> tuple:
        """Convert buffer data to PyTorch tensors for the PPO update.

        Returns:
            Tuple of 8 tensors: (states, actions, old_log_probs, returns,
            advantages, rewards, values, dones). States are float32,
            dones are long (integer), everything else is float32.
        """
        core_tensor = (self.states[:self.slice],
                self.actions[:self.slice],
                self.old_log_probs[:self.slice],
                self.returns[:self.slice],
                self.adv[:self.slice],
                self.rewards[:self.slice],
                self.values[:self.slice],
                self.dones[:self.slice])
        extra_tensor = {
                name: val[:self.slice]
                for name, val in self.extras.items()
                }
        return *core_tensor, extra_tensor

    def clear(self):
        """Reset the buffer for reuse.

        Resets the slice pointer to 0. Underlying arrays are not zeroed;
        old data is overwritten by subsequent insert() calls.
        """
        self.slice = 0
