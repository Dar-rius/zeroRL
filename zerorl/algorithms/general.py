import torch
from torch import Tensor
from torch.nn import Parameter
from zerorl.agent import BaseAgent


def linear_schedule(step, num_update): 1.0 - (step / num_update)


@torch.compile
def get_buffer_params_model(model: BaseAgent) -> tuple[dict[str, Parameter], dict[str, Tensor]]:
    return dict(model.named_parameters()), dict(model.named_buffers())
