"""Unit tests for BaseAgent abstract interface (zerorl.agent)."""

from abc import ABC

import pytest
import torch
import torch.nn as nn
from zerorl.agent import BaseAgent, eval_action


class ConcreteTestAgent(BaseAgent):
    """Minimal concrete agent for testing the abstract interface."""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(obs_dim, act_dim)
        self.val = nn.Linear(obs_dim, 1)

    def forward(self, state: torch.Tensor, **kwargs):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        return self.linear(state_t), self.val(state_t)

    @staticmethod
    def build_distribution(logits: torch.Tensor):
        return torch.distributions.Categorical(logits=logits)


class TestBaseAgentAbstract:
    """Verify BaseAgent enforces abstract method contracts."""

    def test_is_abstract(self) -> None:
        assert ABC in BaseAgent.__mro__

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseAgent()  # type: ignore[abstract]

    def test_must_implement_forward(self) -> None:
        class IncompleteAgent(BaseAgent):
            @staticmethod
            def build_distribution(logits):
                pass

        with pytest.raises(TypeError):
            IncompleteAgent()  # type: ignore[abstract]

    def test_must_implement_build_distribution(self) -> None:
        class IncompleteAgent(BaseAgent):
            def forward(self, state, **kwargs):
                pass

        with pytest.raises(TypeError):
            IncompleteAgent()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self) -> None:
        agent = ConcreteTestAgent(obs_dim=4, act_dim=2)
        assert isinstance(agent, BaseAgent)
        assert isinstance(agent, nn.Module)


class TestBaseAgentGetAction:
    """Verify get_action() template method behavior."""

    def setup_method(self) -> None:
        self.agent = ConcreteTestAgent(obs_dim=4, act_dim=2)

    def test_returns_dict_with_four_keys(self) -> None:
        state = torch.randn(4)
        result = self.agent.get_action(state)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"action", "log_prob", "entropy", "value"}

    def test_dict_values_are_tensors(self) -> None:
        state = torch.randn(4)
        result = self.agent.get_action(state)
        assert isinstance(result["action"], torch.Tensor)
        assert isinstance(result["log_prob"], torch.Tensor)
        assert isinstance(result["entropy"], torch.Tensor)
        assert isinstance(result["value"], torch.Tensor)

    def test_samples_action_when_none(self) -> None:
        state = torch.randn(4)
        result = self.agent.get_action(state)
        assert result["action"].shape == ()

    def test_uses_provided_action(self) -> None:
        state = torch.randn(4)
        action = torch.tensor(1)
        result = self.agent.get_action(state, action)
        assert result["action"].item() == 1

    def test_log_prob_is_finite(self) -> None:
        state = torch.randn(4)
        result = self.agent.get_action(state)
        assert torch.isfinite(result["log_prob"])

    def test_entropy_is_positive(self) -> None:
        state = torch.randn(4)
        result = self.agent.get_action(state)
        assert result["entropy"].item() > 0.0

    def test_value_is_finite(self) -> None:
        state = torch.randn(4)
        result = self.agent.get_action(state)
        assert torch.isfinite(result["value"])

    def test_batch_state(self) -> None:
        state = torch.randn(8, 4)
        result = self.agent.get_action(state)
        assert result["action"].shape == (8,)
        assert result["log_prob"].shape == (8,)
        assert result["entropy"].shape == (8,)
        # value from nn.Linear(obs_dim, 1) outputs (batch, 1)
        assert result["value"].shape == (8, 1)


class TestEvalAction:
    """Verify eval_action() standalone function."""

    def test_discrete_distribution(self) -> None:
        logits = torch.randn(4, 2)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.tensor([0, 1, 0, 1])
        log_prob, entropy = eval_action(dist, action)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)

    def test_scalar_distribution(self) -> None:
        logits = torch.randn(2)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.tensor(0)
        log_prob, entropy = eval_action(dist, action)
        assert log_prob.shape == ()
        assert entropy.shape == ()

    def test_continuous_sums_last_dim(self) -> None:
        mean = torch.zeros(3, 2)
        std = torch.ones(3, 2)
        dist = torch.distributions.Normal(mean, std)
        action = torch.randn(3, 2)
        log_prob, entropy = eval_action(dist, action)
        assert log_prob.shape == (3,)
        assert entropy.shape == (3,)


class TestDeviceProperty:
    """Verify the device property returns correct device."""

    def test_default_device_is_cpu(self) -> None:
        agent = ConcreteTestAgent()
        assert agent.device == torch.device("cpu")
