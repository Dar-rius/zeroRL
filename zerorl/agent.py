"""Abstract base agent interface for reinforcement learning.

Provides BaseAgent, an ABC + nn.Module hybrid that enforces a consistent
policy/value interface for all RL agent implementations.
"""

import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Any
from torch import Tensor
from torch.distributions import Distribution


#Function to calcul the action distribution
def eval_action(dist: Distribution, action: Tensor) -> tuple[Tensor, Tensor]:
    log_prob = dist.log_prob(action)
    dist_entropy = dist.entropy()
    if log_prob.dim() > 1:
        log_prob = log_prob.sum(dim=-1)
        dist_entropy = dist_entropy.sum(dim=-1)
    return log_prob, dist_entropy


class BaseAgent(nn.Module, ABC):
    """Abstract base class for all RL agents.

    Subclasses must implement forward() and get_distribution(). The
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

    @abstractmethod
    def forward(self, state: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor]:
        """Compute raw policy logits and value estimate.

        Args:
            state: Environment observation.

        Returns:
            Tuple of (policy_logits, value) tensors.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def build_distribution(logits: Tensor) -> Distribution:
        """Transforme les logits en distribution. Doit être statique."""
        pass

    def get_action(self, state: Tensor, action: Tensor | None = None, **kwargs: Any) -> dict[str, Tensor]:
        """Sample or evaluate an action under the current policy.

        Template method: calls get_distribution(), then samples if no
        action is provided.

        Args:
            state: Environment observation.
            action: Optional pre-selected action. When None, samples from
                    the policy distribution.

        Returns:
            Tuple of (action, log_prob, entropy, value).
        """
        logits, value = self.forward(state, **kwargs)
        dist = self.build_distribution(logits)
        if action is None: action = dist.sample()
        log_prob, dist_entropy = eval_action(dist, action)
        return {"action": action, "log_prob": log_prob,
                "entropy":dist_entropy, "value":value}
