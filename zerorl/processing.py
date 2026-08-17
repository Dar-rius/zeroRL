"""Observation normalization utilities for RL training.

Provides NormMeanStd (running mean/std) and NormMinMax (min-max scaling)
for preprocessing observations before feeding them to the agent.
"""

import torch
from torch import Tensor
from zerorl.function import fast_compile


class NormMeanStd:
    """Running mean/std normalizer using Welford's online algorithm.

    Tracks mean and variance incrementally as new data arrives.
    Call update() to feed new batches, then normalize() to normalize.
    """

    def __init__(self,
                 shape: tuple[int,...],
                 device: torch.device = torch.device("cpu"),
                 epsilon: float = 1e-4):
        """Initialize normalizer.

        Args:
            shape: Shape of a single observation.
            device: Torch device to allocate tensors on.
            epsilon: Initial count for numerical stability.
        """
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon
    
    def update(self, x: Tensor):
        """Update running statistics with a new batch of observations.

        Args:
            x: Batch of observations, shape (batch_size, *shape).
        """
        if x.dim() == 1: x = x.unsqueeze(0)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, correction=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        new_count = self.count + batch_count
        self.mean += delta * batch_count / new_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + torch.square(delta) * self.count * batch_count / new_count
        self.var = m2 / new_count
        self.count =  new_count

    @fast_compile
    def normalize(self, x: Tensor) -> Tensor:
        """Normalize observations using current running statistics."""
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


class NormMinMax:
    """Min-max normalizer that scales observations to a fixed range."""
    def __init__(self, low: Tensor, high: Tensor, device: str = 'cpu'):
        """Initialize normalizer.

        Args:
            low: Minimum values for each dimension.
            high: Maximum values for each dimension.
            device: Torch device to allocate tensors on.
        """
        self.low = low.to(device)
        self.high = high.to(device)
        self.scale = 1.0 / (self.high - self.low + 1e-8)

    @fast_compile
    def normalize(self, x: Tensor):
        """Normalize observations using min-max scaling."""
        return (x - self.low) * self.scale
