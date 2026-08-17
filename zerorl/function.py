"""Utility functions for RL training.

Provides get_buffer_params_model() for extracting model parameters.
"""

import shutil
import sys
from typing import Any, Callable, TypeVar

import torch
from torch import Tensor
from torch.nn import Parameter
from zerorl.agent import BaseAgent


F = TypeVar("F", bound=Callable[..., Any])


def _cxx_compiler_available() -> bool:
    """True if torch inductor can find a C++ compiler."""
    if sys.platform != "win32":
        return True
    return shutil.which("cl") is not None


def fast_compile(fn: F | None = None, **kwargs) -> F | Callable[[F], F]:
    """Like torch.compile; no-op when a C++ compiler is not on PATH."""
    use_compile = _cxx_compiler_available()
    def wrap(f: F) -> F:
        if not use_compile:
            return f
        return torch.compile(f, **kwargs)  # type: ignore[return-value]

    if fn is not None:
        return wrap(fn)
    return wrap


def get_buffer_params_model(model: BaseAgent) -> tuple[dict[str, Parameter], dict[str, Tensor]]:
    """Extract named parameters and buffers from a model.

    Args:
        model: The neural network model.

    Returns:
        Tuple of (parameters dict, buffers dict).
    """
    return dict(model.named_parameters()), dict(model.named_buffers())
