"""Utility functions for RL training.

Provides get_buffer_params_model() for extracting model parameters.
"""

from torch import Tensor
from torch.nn import Parameter
from zerorl.agent import BaseAgent


def get_buffer_params_model(model: BaseAgent) -> tuple[dict[str, Parameter], dict[str, Tensor]]:
    """Extract named parameters and buffers from a model.

    Args:
        model: The neural network model.

    Returns:
        Tuple of (parameters dict, buffers dict).
    """
    return dict(model.named_parameters()), dict(model.named_buffers())
