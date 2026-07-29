"""Unit tests for BaseAgent abstract interface (rl_template.agent)."""

from abc import ABC

import numpy as np
import pytest
import torch
import torch.nn as nn
from rl_template.agent import BaseAgent


class ConcreteTestAgent(BaseAgent):
    """Minimal concrete agent for testing the abstract interface."""

    def __init__(self, obs_dim=4, act_dim=2):
        super().__init__()
        self.linear = nn.Linear(obs_dim, act_dim)
        self.val = nn.Linear(obs_dim, 1)

    def forward(self, state, **kwargs):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        return self.linear(state_t), self.val(state_t)

    def get_distribution(self, state, **kwargs):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value.squeeze(-1)


class TestBaseAgentAbstract:
    """Verify BaseAgent enforces abstract method contracts."""

    def test_is_abstract(self):
        assert ABC in BaseAgent.__mro__

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseAgent()

    def test_must_implement_forward(self):
        class IncompleteAgent(BaseAgent):
            def get_distribution(self, state):
                pass

        with pytest.raises(TypeError):
            IncompleteAgent()

    def test_must_implement_get_distribution(self):
        class IncompleteAgent(BaseAgent):
            def forward(self, state):
                pass

        with pytest.raises(TypeError):
            IncompleteAgent()

    def test_concrete_subclass_instantiates(self):
        agent = ConcreteTestAgent(obs_dim=4, act_dim=2)
        assert isinstance(agent, BaseAgent)
        assert isinstance(agent, nn.Module)


class TestBaseAgentGetAction:
    """Verify get_action() template method behavior."""

    def setup_method(self):
        self.agent = ConcreteTestAgent(obs_dim=4, act_dim=2)

    def test_returns_four_values(self):
        state = np.random.randn(4).astype(np.float32)
        action, log_prob, entropy, value = self.agent.get_action(state)
        assert isinstance(action, torch.Tensor)
        assert isinstance(log_prob, torch.Tensor)
        assert isinstance(entropy, torch.Tensor)
        assert isinstance(value, torch.Tensor)

    def test_samples_action_when_none(self):
        state = np.random.randn(4).astype(np.float32)
        action, log_prob, entropy, value = self.agent.get_action(state)
        assert action.shape == ()

    def test_uses_provided_action(self):
        state = np.random.randn(4).astype(np.float32)
        action = torch.tensor(1)
        out_action, log_prob, entropy, value = self.agent.get_action(state, action)
        assert out_action.item() == 1

    def test_log_prob_is_finite(self):
        state = np.random.randn(4).astype(np.float32)
        _, log_prob, _, _ = self.agent.get_action(state)
        assert torch.isfinite(log_prob)

    def test_entropy_is_positive(self):
        state = np.random.randn(4).astype(np.float32)
        _, _, entropy, _ = self.agent.get_action(state)
        assert entropy.item() > 0.0

    def test_value_is_finite(self):
        state = np.random.randn(4).astype(np.float32)
        _, _, _, value = self.agent.get_action(state)
        assert torch.isfinite(value)

    def test_batch_state(self):
        state = np.random.randn(8, 4).astype(np.float32)
        action, log_prob, entropy, value = self.agent.get_action(state)
        assert action.shape == (8,)
        assert log_prob.shape == (8,)
        assert entropy.shape == (8,)
        assert value.shape == (8,)
