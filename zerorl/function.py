"""Utility functions for RL training.

Provides linear_schedule() for learning rate decay and
get_buffer_params_model() for extracting model parameters.
"""

from torch import Tensor
from torch.nn import Parameter
from zerorl.agent import BaseAgent


def linear_schedule(step: int, num_update: int) -> float:
    """Compute linearly decaying learning rate factor.

    Args:
        step: Current training step.
        num_update: Total number of training steps.

    Returns:
        Factor in [0, 1] that decays linearly from 1.0 to 0.0.
    """
    return 1.0 - (step / num_update)


def get_buffer_params_model(model: BaseAgent) -> tuple[dict[str, Parameter], dict[str, Tensor]]:
    """Extract named parameters and buffers from a model.

    Args:
        model: The neural network model.

    Returns:
        Tuple of (parameters dict, buffers dict).
    """
    return dict(model.named_parameters()), dict(model.named_buffers())
