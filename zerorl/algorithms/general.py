#import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer
from ..agent import BaseAgent

def lr_decay(lr: float, optimizer: Optimizer, step: int, num_steps: int):
        """Apply linear learning rate decay from lr to 0.

        Args:
            lr: Base learning rate.
            total_steps: Total number of training steps.
            step: Current training step.
        """
        frac = 1.0 - (step / num_steps)
        current_lr = lr * frac
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr


def get_buffer_params_model(model: BaseAgent) -> tuple[dict[str, Parameter], dict[str, Tensor]]:
    return dict(model.named_parameters()), dict(model.named_buffers())
