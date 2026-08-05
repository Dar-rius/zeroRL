"""Unit tests for utility functions (zerorl.function).

Tests linear_schedule() and get_buffer_params_model().
"""

import pytest
import torch
import torch.nn as nn
from zerorl.agent import BaseAgent
from zerorl.function import get_buffer_params_model, linear_schedule


class SimpleAgent(BaseAgent):
    """Minimal agent for testing get_buffer_params_model."""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2):
        super().__init__()
        self.actor = nn.Linear(obs_dim, act_dim)
        self.val = nn.Linear(obs_dim, 1)
        self.register_buffer("running_mean", torch.zeros(obs_dim))

    def forward(self, state: torch.Tensor, **kwargs):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        return self.actor(state_t), self.val(state_t)

    @staticmethod
    def build_distribution(logits: torch.Tensor):
        return torch.distributions.Categorical(logits=logits)


# =============================================================================
# Test linear_schedule()
# =============================================================================


class TestLinearSchedule:
    """Verify linear_schedule() produces correct LR multipliers."""

    def test_step_zero_returns_one(self) -> None:
        assert linear_schedule(0, 1000) == pytest.approx(1.0)

    def test_step_half(self) -> None:
        assert linear_schedule(500, 1000) == pytest.approx(0.5)

    def test_step_end(self) -> None:
        assert linear_schedule(1000, 1000) == pytest.approx(0.0)

    def test_step_near_end(self) -> None:
        result = linear_schedule(999, 1000)
        assert result == pytest.approx(1.0 - 999 / 1000)

    def test_small_total(self) -> None:
        assert linear_schedule(0, 10) == pytest.approx(1.0)
        assert linear_schedule(5, 10) == pytest.approx(0.5)
        assert linear_schedule(10, 10) == pytest.approx(0.0)

    def test_monotonically_decreasing(self) -> None:
        values = [linear_schedule(i, 100) for i in range(101)]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]


# =============================================================================
# Test get_buffer_params_model()
# =============================================================================


class TestGetBufferParamsModel:
    """Verify get_buffer_params_model() extracts parameters and buffers."""

    def test_returns_tuple_of_dicts(self) -> None:
        agent = SimpleAgent()
        result = get_buffer_params_model(agent)
        assert isinstance(result, tuple)
        assert len(result) == 2
        params, buffers = result
        assert isinstance(params, dict)
        assert isinstance(buffers, dict)

    def test_params_keys(self) -> None:
        agent = SimpleAgent()
        params, _ = get_buffer_params_model(agent)
        param_keys = set(params.keys())
        assert "actor.weight" in param_keys
        assert "actor.bias" in param_keys
        assert "val.weight" in param_keys
        assert "val.bias" in param_keys

    def test_buffers_keys(self) -> None:
        agent = SimpleAgent()
        _, buffers = get_buffer_params_model(agent)
        buffer_keys = set(buffers.keys())
        assert "running_mean" in buffer_keys

    def test_params_are_parameters(self) -> None:
        agent = SimpleAgent()
        params, _ = get_buffer_params_model(agent)
        for name, param in params.items():
            assert isinstance(param, torch.nn.Parameter)
            assert param.requires_grad

    def test_buffers_are_tensors(self) -> None:
        agent = SimpleAgent()
        _, buffers = get_buffer_params_model(agent)
        for name, buf in buffers.items():
            assert isinstance(buf, torch.Tensor)

    def test_empty_model(self) -> None:
        class EmptyAgent(BaseAgent):
            def forward(self, state, **kwargs):
                return torch.zeros(1), torch.zeros(1)

            @staticmethod
            def build_distribution(logits):
                return torch.distributions.Categorical(logits=logits)

        agent = EmptyAgent()
        params, buffers = get_buffer_params_model(agent)
        assert len(params) == 0
        assert len(buffers) == 0
