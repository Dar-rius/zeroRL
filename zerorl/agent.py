"""Abstract base agent interface for reinforcement learning.

Provides BaseAgent, an ABC + nn.Module hybrid that enforces a consistent
policy/value interface for all RL agent implementations.
"""

import torch
import torch.nn as nn
import torch.distributions as dists
from torch.distributions import Distribution
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Any


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

    @abstractmethod
    def get_distribution(self, state: Tensor, **kwargs: Any) -> tuple[Distribution, Tensor]:
        """Build a distribution over actions for the given state.

        Args:
            state: Environment observation.

        Returns:
            Tuple of (distribution, value) where distribution is a
            torch.distributions.Distribution instance.
        """
        pass

    def get_action(self, state: Tensor, action: Tensor | None = None, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
        dist, value = self.get_distribution(state, **kwargs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()

        is_categorical = isinstance(dist, dists.Categorical)
        # Gestion des cas particuliers (ex: distributions indépendantes enveloppant du Categorical ou du Normal)
        if isinstance(dist, dists.Independent):
            if isinstance(dist.base_dist, dists.Categorical):
                is_categorical = True

        if not is_categorical:
            # Espace continu ou multi-dimensionnel indépendant : on somme les log_probs et entropies par échantillon
            if log_prob.dim() > 1 and log_prob.shape[-1] > 1:
                log_prob = log_prob.sum(dim=-1)
            if dist_entropy.dim() > 1 and dist_entropy.shape[-1] > 1:
                dist_entropy = dist_entropy.sum(dim=-1)
        return (action, log_prob, dist_entropy, value)
