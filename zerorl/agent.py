"""Abstract base agent interface for reinforcement learning.

Provides BaseAgent, an ABC + nn.Module hybrid that enforces a consistent
policy/value interface for all RL agent implementations.
"""

import torch
from torch import nn
from torch import Tensor
from torch.distributions import Distribution


# Compute log probability and entropy from a distribution
def eval_action(dist: Distribution, action: Tensor) -> tuple[Tensor, Tensor]:
    log_prob = dist.log_prob(action)
    dist_entropy = dist.entropy()
    if log_prob.dim() > 1:
        log_prob = log_prob.sum(dim=-1)
        dist_entropy = dist_entropy.sum(dim=-1)
    return (log_prob, dist_entropy)


class BaseAgent(nn.Module):
    """Abstract base class for all RL agents.

    Subclasses must implement forward() and build_distribution(). The
    get_action() template method composes those two to sample actions,
    compute log probabilities, and return critic values.
    """

    def __init__(self):
        super().__init__()

    @property
    def device(self) -> torch.device:
        """Return the device where it run."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
