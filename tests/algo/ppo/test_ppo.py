"""Unit tests for PPO standalone functions."""

import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from zerorl.agent import BaseAgent, eval_action
from zerorl.algorithms.ppo.ppo import gae_compute, ppo_loss, ppo
from zerorl.common import Buffer
from zerorl.config import AlgoConfig

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiscreteTestAgent(BaseAgent):
    def __init__(self, obs_dim: int = 4, act_dim: int = 2):
        super().__init__()
        self.actor = nn.Linear(obs_dim, act_dim)
        self.val = nn.Linear(obs_dim, 1)

    def forward(self, state: torch.Tensor, **kwargs):
        return self.actor(state), self.val(state)

    @staticmethod
    def build_distribution(logits: torch.Tensor):
        return torch.distributions.Categorical(logits=logits)
    
    def get_action(self, state: torch.Tensor, action: torch.Tensor | None = None, **kwargs):
        logits, value = self.forward(state, **kwargs)
        dist = self.build_distribution(logits)
        if action is None: action = dist.sample()
        log_prob, dist_entropy = eval_action(dist, action)
        return {"action": action, "log_prob": log_prob, "entropy":dist_entropy, "value":value}

class TestGaeCompute:
    @pytest.mark.gpu
    def test_returns_equals_advantages_plus_values(self, device) -> None:
        cfg = AlgoConfig()
        reward = torch.tensor([1.0, 2.0, 3.0], device=device).unsqueeze(-1)
        values = torch.tensor([0.5, 0.5, 0.5], device=device).unsqueeze(-1)
        dones = torch.tensor([0.0, 0.0, 0.0], device=device).unsqueeze(-1)
        last_value = torch.tensor([1.0], device=device)
        returns, advantages, _ = gae_compute(reward, values, last_value, dones, cfg)
        torch.testing.assert_close(returns, advantages + values)

    @pytest.mark.gpu
    def test_buffer_inside_advantages_(self, device) -> None:
        cfg = AlgoConfig()
        buf = Buffer(step=3, data = {"return":(), "advantage": ()}, device=device)
        buf.slice = 3
        reward = torch.tensor([1.0, 2.0, 3.0], device=device).unsqueeze(-1)
        values = torch.tensor([0.5, 0.5, 0.5], device=device).unsqueeze(-1)
        dones = torch.tensor([0.0, 0.0, 0.0], device=device).unsqueeze(-1)
        last_value = torch.tensor([1.0], device=device)
        returns, advantages, _ = gae_compute(reward, values, last_value, dones, cfg)
        gae_compute(reward, values, last_value, dones, cfg, buf)
        torch.testing.assert_close(returns, buf.data["return"])
        torch.testing.assert_close(advantages, buf.data["advantage"])



class TestPpoLoss:
    @pytest.mark.gpu
    def test_returns_dict(self, device) -> None:
        agent = DiscreteTestAgent().to(device)
        params = dict(agent.named_parameters())
        buffers = dict(agent.named_buffers())
        cfg = AlgoConfig()
        state = torch.randn(16, 4, device=device)
        action = torch.randint(0, 2, (16,), device=device)
        log_prob = torch.randn(16, device=device)
        advantage = torch.randn(16, device=device)
        return_ = torch.randn(16, device=device)
        result = ppo_loss(agent, params, buffers, state, action, log_prob, advantage, return_, cfg)
        assert "loss" in result

class TestPpoFunction:
    def _make_buffer(self, n, obs_dim, device):
        buf = Buffer(step=n, data={"state": (obs_dim,), "action": (), "log_prob": (), "advantage": (), "return": (), "value": ()}, device=device)
        for _ in range(n):
            buf.insert(state=torch.randn(obs_dim, device=device), action=torch.tensor(0, device=device), log_prob=torch.tensor(-0.5, device=device), advantage=torch.tensor(1.0, device=device), **{"return": torch.tensor(2.0, device=device)}, value=torch.tensor(0.5, device=device))
        return buf

    @pytest.mark.gpu
    def test_ppo_runs_without_error(self, device) -> None:
        agent = DiscreteTestAgent().to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        buf = self._make_buffer(64, 4, device)
        cfg = AlgoConfig()
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        result = ppo(agent, optimizer, buf, cfg, scheduler, batch_size=32, epochs=2, device=device)
        assert "loss" in result

    @pytest.mark.gpu
    def test_ppo_updates_weights(self, device) -> None:
        agent = DiscreteTestAgent().to(device)
        w_before = {k: v.clone() for k, v in agent.state_dict().items()}
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
        buf = self._make_buffer(128, 4, device)
        cfg = AlgoConfig()
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        ppo(agent, optimizer, buf, cfg, scheduler, batch_size=32, epochs=5, device=device)
        w_after = agent.state_dict()
        changed = not all(torch.allclose(w_before[k], w_after[k]) for k in w_before)
        assert changed
