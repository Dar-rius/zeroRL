"""Unit tests for BaseAgent abstract interface (zerorl.agent)."""

import pytest
import torch
import torch.nn as nn
from zerorl.agent import BaseAgent, eval_action

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConcreteTestAgent(BaseAgent):
    def __init__(self, obs_dim: int = 4, act_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(obs_dim, act_dim)
        self.val = nn.Linear(obs_dim, 1)

    def forward(self, state: torch.Tensor, **kwargs):
        return self.linear(state), self.val(state)

    @staticmethod
    def build_distribution(logits: torch.Tensor):
        return torch.distributions.Categorical(logits=logits)

    def get_action(self, state: torch.Tensor, action: torch.Tensor | None = None, **kwargs):
        logits, value = self.forward(state, **kwargs)
        dist = self.build_distribution(logits)
        if action is None: action = dist.sample()
        log_prob, dist_entropy = eval_action(dist, action)
        return {"action": action, "log_prob": log_prob, "entropy":dist_entropy, "value":value}

class TestBaseAgentGetAction:
    @pytest.mark.gpu
    def test_returns_dict_with_four_keys(self, device) -> None:
        agent = ConcreteTestAgent(obs_dim=4, act_dim=2).to(device)
        state = torch.randn(4, device=device)
        result = agent.get_action(state)
        assert set(result.keys()) == {"action", "log_prob", "entropy", "value"}

    @pytest.mark.gpu
    def test_samples_action_when_none(self, device) -> None:
        agent = ConcreteTestAgent(obs_dim=4, act_dim=2).to(device)
        state = torch.randn(4, device=device)
        result = agent.get_action(state)
        assert result["action"].shape == ()

    @pytest.mark.gpu
    def test_batch_state(self, device) -> None:
        agent = ConcreteTestAgent(obs_dim=4, act_dim=2).to(device)
        state = torch.randn(8, 4, device=device)
        result = agent.get_action(state)
        assert result["action"].shape == (8,)
        assert result["value"].shape == (8, 1)

class TestEvalAction:
    @pytest.mark.gpu
    def test_discrete_distribution(self, device) -> None:
        logits = torch.randn(4, 2, device=device)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.tensor([0, 1, 0, 1], device=device)
        log_prob, entropy = eval_action(dist, action)
        assert log_prob.shape == (4,)

    @pytest.mark.gpu
    def test_continuous_sums_last_dim(self, device) -> None:
        mean = torch.zeros(3, 2, device=device)
        std = torch.ones(3, 2, device=device)
        dist = torch.distributions.Normal(mean, std)
        action = torch.randn(3, 2, device=device)
        log_prob, entropy = eval_action(dist, action)
        assert log_prob.shape == (3,)
