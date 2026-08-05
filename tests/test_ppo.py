"""Unit tests for PPO standalone functions (zerorl.algorithms.ppo.ppo).

Tests gae_compute(), ppo_loss(), and ppo() as standalone functions,
plus linear_schedule() from zerorl.function.
"""

import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from zerorl.agent import BaseAgent
from zerorl.algorithms.ppo.ppo import gae_compute, ppo_loss, ppo
from zerorl.common import Buffer
from zerorl.config import AlgoConfig
from zerorl.function import linear_schedule


# =============================================================================
# Test Helper: DiscreteTestAgent
# =============================================================================


class DiscreteTestAgent(BaseAgent):
    """Minimal discrete-policy agent for testing PPO functions."""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2):
        super().__init__()
        self.actor = nn.Linear(obs_dim, act_dim)
        self.val = nn.Linear(obs_dim, 1)

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


# =============================================================================
# Test gae_compute()
# =============================================================================


class TestGaeCompute:
    """Verify gae_compute() produces correct advantages via hand-calculated examples."""

    def test_returns_equals_advantages_plus_values(self) -> None:
        cfg = AlgoConfig()
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 0.5, 0.5])
        dones = torch.tensor([0.0, 0.0, 0.0])
        last_value = torch.tensor([1.0])
        returns, advantages, _ = gae_compute(rewards, values, last_value, dones, cfg)
        torch.testing.assert_close(returns, advantages + values)

    def test_hand_calculated_single_step(self) -> None:
        cfg = AlgoConfig(gamma=0.99, gae_lambda=0.95)
        rewards = torch.tensor([1.0])
        values = torch.tensor([0.0])
        dones = torch.tensor([0.0])
        last_value = torch.tensor([0.0])
        returns, advantages, delta = gae_compute(rewards, values, last_value, dones, cfg)
        expected_delta = 1.0 + 0.99 * 0.0 * 1.0 - 0.0
        expected_gae = expected_delta
        torch.testing.assert_close(delta, torch.tensor([expected_delta]))
        torch.testing.assert_close(advantages, torch.tensor([expected_gae]))
        torch.testing.assert_close(returns, torch.tensor([expected_gae + 0.0]))

    def test_hand_calculated_two_steps(self) -> None:
        cfg = AlgoConfig(gamma=0.99, gae_lambda=0.95)
        rewards = torch.tensor([1.0, 1.0])
        values = torch.tensor([0.0, 0.0])
        dones = torch.tensor([0.0, 0.0])
        last_value = torch.tensor([0.0])
        returns, advantages, delta = gae_compute(rewards, values, last_value, dones, cfg)
        expected_gae_1 = 1.0
        expected_gae_0 = 1.0 + 0.99 * 0.95 * expected_gae_1
        assert advantages[1].item() == pytest.approx(expected_gae_1, abs=1e-6)
        assert advantages[0].item() == pytest.approx(expected_gae_0, abs=1e-6)

    def test_done_resets_gae(self) -> None:
        cfg = AlgoConfig(gamma=0.99, gae_lambda=0.95)
        rewards = torch.tensor([1.0, 1.0])
        values = torch.tensor([0.0, 0.0])
        dones = torch.tensor([1.0, 0.0])
        last_value = torch.tensor([0.0])
        returns, advantages, _ = gae_compute(rewards, values, last_value, dones, cfg)
        assert advantages[1].item() == pytest.approx(1.0, abs=1e-6)
        assert advantages[0].item() == pytest.approx(1.0, abs=1e-6)

    def test_output_shapes(self) -> None:
        cfg = AlgoConfig()
        n = 10
        rewards = torch.ones(n)
        values = torch.zeros(n)
        dones = torch.zeros(n)
        last_value = torch.tensor([0.0])
        returns, advantages, delta = gae_compute(rewards, values, last_value, dones, cfg)
        assert returns.shape == (n,)
        assert advantages.shape == (n,)
        assert delta.shape == (n,)

    def test_with_nonzero_values(self) -> None:
        cfg = AlgoConfig(gamma=0.99, gae_lambda=0.95)
        rewards = torch.tensor([5.0, 3.0, 2.0])
        values = torch.tensor([1.0, 2.0, 3.0])
        dones = torch.tensor([0.0, 0.0, 0.0])
        last_value = torch.tensor([4.0])
        returns, advantages, _ = gae_compute(rewards, values, last_value, dones, cfg)
        torch.testing.assert_close(returns, advantages + values)
        assert returns.shape == (3,)


# =============================================================================
# Test ppo_loss()
# =============================================================================


class TestPpoLoss:
    """Verify ppo_loss() returns correct loss dict with gradients."""

    def test_returns_dict_with_four_keys(self) -> None:
        agent = DiscreteTestAgent(obs_dim=4, act_dim=2)
        params = dict(agent.named_parameters())
        buffers = dict(agent.named_buffers())
        cfg = AlgoConfig()
        states = torch.randn(16, 4)
        actions = torch.randint(0, 2, (16,))
        old_log_probs = torch.randn(16)
        advantages = torch.randn(16)
        returns = torch.randn(16)
        result = ppo_loss(agent, params, buffers, states, actions,
                          old_log_probs, advantages, returns, cfg)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"loss", "policy_loss", "value_loss", "entropy_loss"}

    def test_losses_are_finite(self) -> None:
        agent = DiscreteTestAgent(obs_dim=4, act_dim=2)
        params = dict(agent.named_parameters())
        buffers = dict(agent.named_buffers())
        cfg = AlgoConfig()
        states = torch.randn(16, 4)
        actions = torch.randint(0, 2, (16,))
        old_log_probs = torch.randn(16)
        advantages = torch.randn(16)
        returns = torch.randn(16)
        result = ppo_loss(agent, params, buffers, states, actions,
                          old_log_probs, advantages, returns, cfg)
        for v in result.values():
            assert torch.isfinite(v)

    def test_loss_has_gradient(self) -> None:
        agent = DiscreteTestAgent(obs_dim=4, act_dim=2)
        params = dict(agent.named_parameters())
        buffers = dict(agent.named_buffers())
        cfg = AlgoConfig()
        states = torch.randn(16, 4)
        actions = torch.randint(0, 2, (16,))
        old_log_probs = torch.randn(16)
        advantages = torch.randn(16)
        returns = torch.randn(16)
        result = ppo_loss(agent, params, buffers, states, actions,
                          old_log_probs, advantages, returns, cfg)
        result["loss"].backward()
        for p in agent.parameters():
            assert p.grad is not None


# =============================================================================
# Test ppo() standalone function
# =============================================================================


class TestPpoFunction:
    """Verify ppo() runs a full update cycle with a Buffer."""

    def _make_buffer(self, n: int = 64, obs_dim: int = 4, act_dim: int = 2) -> Buffer:
        """Create and fill a buffer with keys expected by ppo()."""
        buf = Buffer(
            step=n,
            data={
                "state": (obs_dim,),
                "actions": (),
                "old_log_probs": (),
                "adv": (),
                "returns": (),
                "value": (),
            },
            device=torch.device("cpu"),
        )
        for _ in range(n):
            buf.insert(
                state=torch.randn(obs_dim),
                actions=torch.tensor(0),
                old_log_probs=torch.tensor(-0.5),
                adv=torch.tensor(1.0),
                returns=torch.tensor(2.0),
                value=torch.tensor(0.5),
            )
        return buf

    def test_ppo_runs_without_error(self) -> None:
        agent = DiscreteTestAgent(obs_dim=4, act_dim=2)
        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        buf = self._make_buffer(n=64)
        cfg = AlgoConfig()
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        result = ppo(agent, optimizer, buf, cfg, scheduler, batch_size=32, epochs=2,
                     device=torch.device("cpu"))
        assert isinstance(result, dict)
        assert "loss" in result

    def test_ppo_returns_finite_losses(self) -> None:
        agent = DiscreteTestAgent(obs_dim=4, act_dim=2)
        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        buf = self._make_buffer(n=64)
        cfg = AlgoConfig()
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        result = ppo(agent, optimizer, buf, cfg, scheduler, batch_size=32, epochs=2,
                     device=torch.device("cpu"))
        for v in result.values():
            assert torch.isfinite(v)

    def test_ppo_updates_weights(self) -> None:
        agent = DiscreteTestAgent(obs_dim=4, act_dim=2)
        w_before = {k: v.clone() for k, v in agent.state_dict().items()}
        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        buf = self._make_buffer(n=128)
        cfg = AlgoConfig()
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        ppo(agent, optimizer, buf, cfg, scheduler, batch_size=32, epochs=5,
            device=torch.device("cpu"))
        w_after = agent.state_dict()
        changed = False
        for k in w_before:
            if not torch.allclose(w_before[k], w_after[k]):
                changed = True
                break
        assert changed

    def test_ppo_scheduler_steps(self) -> None:
        agent = DiscreteTestAgent(obs_dim=4, act_dim=2)
        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        buf = self._make_buffer(n=64)
        cfg = AlgoConfig()
        # Use a decreasing scheduler
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0 - step * 0.01)
        # LambdaLR may call step() on init in newer PyTorch versions,
        # so record the lr right before calling ppo
        pre_ppo_lr = optimizer.param_groups[0]["lr"]
        ppo(agent, optimizer, buf, cfg, scheduler, batch_size=32, epochs=1,
            device=torch.device("cpu"))
        post_ppo_lr = optimizer.param_groups[0]["lr"]
        # scheduler.step() was called inside ppo(), so lr should have decreased
        assert post_ppo_lr < pre_ppo_lr
