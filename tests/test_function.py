"""Unit tests for utility functions (zerorl.function)."""

import pytest
import torch
import torch.nn as nn
from zerorl.agent import BaseAgent
from zerorl.function import get_buffer_params_model, linear_schedule

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestLinearSchedule:
    def test_step_zero_returns_one(self) -> None:
        assert linear_schedule(0, 1000) == pytest.approx(1.0)

class TestGetBufferParamsModel:
    @pytest.mark.gpu
    def test_returns_tuple_of_dicts(self, device) -> None:
        class SimpleAgent(BaseAgent):
            def __init__(self): super().__init__(); self.actor = nn.Linear(4, 2); self.register_buffer("running_mean", torch.zeros(4))
            def forward(self, state, **kwargs): return self.actor(state), torch.zeros(1)
            @staticmethod
            def build_distribution(logits): return torch.distributions.Categorical(logits=logits)

        agent = SimpleAgent().to(device)
        params, buffers = get_buffer_params_model(agent)
        assert "actor.weight" in params
        assert "running_mean" in buffers
