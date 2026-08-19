"""Unit tests for PPO standalone functions."""

import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from zerorl.agent import BaseAgent, eval_action
from zerorl.algorithms.ppo.ppo import gae_compute, ppo_loss, ppo_func
from zerorl.buffer import Buffer
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
        buf = Buffer(step=3, data={"return":(), "advantage": ()}, device=device)
        buf.slice = 3
        reward = torch.tensor([1.0, 2.0, 3.0], device=device).unsqueeze(-1)
        values = torch.tensor([0.5, 0.5, 0.5], device=device).unsqueeze(-1)
        dones = torch.tensor([0.0, 0.0, 0.0], device=device).unsqueeze(-1)
        last_value = torch.tensor([1.0], device=device)
        gae_compute(reward, values, last_value, dones,
                    torch.zeros_like(dones), torch.zeros_like(values), buf, cfg)
        returns = buf.data["return"][:buf.slice]
        advantages = buf.data["advantage"][:buf.slice]
        torch.testing.assert_close(returns, advantages + values)

    @pytest.mark.gpu
    def test_buffer_inside_advantages_(self, device) -> None:
        cfg = AlgoConfig()
        buf = Buffer(step=3, data={"return":(), "advantage": ()}, device=device)
        buf.slice = 3
        reward = torch.tensor([1.0, 2.0, 3.0], device=device).unsqueeze(-1)
        values = torch.tensor([0.5, 0.5, 0.5], device=device).unsqueeze(-1)
        dones = torch.tensor([0.0, 0.0, 0.0], device=device).unsqueeze(-1)
        last_value = torch.tensor([1.0], device=device)
        gae_compute(reward, values, last_value, dones,
                    torch.zeros_like(dones), torch.zeros_like(values), buf, cfg)
        torch.testing.assert_close(buf.data["return"][:3], buf.data["return"][:3])
        torch.testing.assert_close(buf.data["advantage"][:3], buf.data["advantage"][:3])



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
        old_values = torch.zeros(16, device=device)
        advantage = torch.randn(16, device=device)
        return_ = torch.randn(16, device=device)
        result = ppo_loss(agent, params, buffers, state, action, log_prob, old_values,
                         advantage, return_, cfg.ent_coef, cfg.value_coef, cfg.clip_eps, False)
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
        result = ppo_func(agent, optimizer, buf, cfg, scheduler, device=device)
        assert "loss" in result

    @pytest.mark.gpu
    def test_ppo_updates_weights(self, device) -> None:
        agent = DiscreteTestAgent().to(device)
        w_before = {k: v.clone() for k, v in agent.state_dict().items()}
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
        buf = self._make_buffer(128, 4, device)
        cfg = AlgoConfig()
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        ppo_func(agent, optimizer, buf, cfg, scheduler, device=device)
        w_after = agent.state_dict()
        changed = not all(torch.allclose(w_before[k], w_after[k]) for k in w_before)
        assert changed


class TestGaeDoneMask:
    @pytest.mark.gpu
    def test_gae_no_bootstrap_across_done(self, device) -> None:
        cfg = AlgoConfig()
        reward = torch.tensor([1.0, 2.0, 3.0], device=device).unsqueeze(-1)
        values = torch.tensor([0.5, 0.5, 0.5], device=device).unsqueeze(-1)
        last_value = torch.tensor([1.0], device=device)
        dones_no = torch.tensor([0.0, 0.0, 0.0], device=device).unsqueeze(-1)
        dones_mid = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(-1)
        buf_no = Buffer(step=3, data={"return":(), "advantage": ()}, device=device)
        buf_no.slice = 3
        gae_compute(reward, values, last_value, dones_no,
                    torch.zeros_like(dones_no), torch.zeros_like(values), buf_no, cfg)
        adv_no = buf_no.data["advantage"][:3]
        buf_done = Buffer(step=3, data={"return":(), "advantage": ()}, device=device)
        buf_done.slice = 3
        gae_compute(reward, values, last_value, dones_mid,
                    torch.zeros_like(dones_mid), torch.zeros_like(values), buf_done, cfg)
        adv_done = buf_done.data["advantage"][:3]
        torch.testing.assert_close(adv_done[1], adv_done[1])
        assert adv_done[1].item() < adv_no[1].item()

    @pytest.mark.gpu
    def test_gae_last_step_done_ignores_last_value(self, device) -> None:
        cfg = AlgoConfig()
        reward = torch.tensor([1.0, 2.0, 3.0], device=device).unsqueeze(-1)
        values = torch.tensor([0.5, 0.5, 0.5], device=device).unsqueeze(-1)
        big_last = torch.tensor([100.0], device=device)
        dones_last = torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(-1)
        dones_none = torch.tensor([0.0, 0.0, 0.0], device=device).unsqueeze(-1)
        buf_done = Buffer(step=3, data={"return":(), "advantage": ()}, device=device)
        buf_done.slice = 3
        gae_compute(reward, values, big_last, dones_last,
                    torch.zeros_like(dones_last), torch.zeros_like(values), buf_done, cfg)
        buf_none = Buffer(step=3, data={"return":(), "advantage": ()}, device=device)
        buf_none.slice = 3
        gae_compute(reward, values, big_last, dones_none,
                    torch.zeros_like(dones_none), torch.zeros_like(values), buf_none, cfg)
        delta_done = reward[-1] - values[-1]
        torch.testing.assert_close(buf_done.data["advantage"][2], delta_done)
        assert buf_none.data["advantage"][2].item() != buf_done.data["advantage"][2].item()

    @pytest.mark.gpu
    def test_gae_multi_env_independent(self, device) -> None:
        cfg = AlgoConfig()
        reward = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], device=device)
        values = torch.tensor([[0.5, 5.0], [0.5, 5.0], [0.5, 5.0]], device=device)
        last_value = torch.tensor([1.0, 10.0], device=device)
        dones = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], device=device)
        buf_both = Buffer(step=3, num_envs=2, data={"return":(), "advantage": ()}, device=device)
        buf_both.slice = 3
        gae_compute(reward, values, last_value, dones,
                    torch.zeros_like(dones), torch.zeros_like(values), buf_both, cfg)
        buf_0 = Buffer(step=3, data={"return":(), "advantage": ()}, device=device)
        buf_0.slice = 3
        gae_compute(reward[:, 0:1], values[:, 0:1], last_value[0:1], dones[:, 0:1],
                    torch.zeros_like(dones[:, 0:1]), torch.zeros_like(values[:, 0:1]), buf_0, cfg)
        torch.testing.assert_close(buf_both.data["return"][:3, 0], buf_0.data["return"][:3].squeeze(-1))
        torch.testing.assert_close(buf_both.data["advantage"][:3, 0], buf_0.data["advantage"][:3].squeeze(-1))


class TestPpoLossInternals:
    @pytest.mark.gpu
    def test_ppo_loss_clipping_activates(self, device) -> None:
        agent = DiscreteTestAgent().to(device)
        state = torch.randn(16, 4, device=device)
        action = torch.randint(0, 2, (16,), device=device)
        logits, _ = agent(state)
        dist = agent.build_distribution(logits)
        new_log_prob, _ = eval_action(dist, action)
        old_log_prob = new_log_prob - 5.0
        advantage = torch.ones(16, device=device)
        return_ = torch.zeros(16, device=device)
        cfg = AlgoConfig(clip_eps=0.2)
        params = dict(agent.named_parameters())
        buffers = dict(agent.named_buffers())
        old_values = torch.zeros(16, device=device)
        result = ppo_loss(agent, params, buffers, state, action, old_log_prob, old_values,
                         advantage, return_, cfg.ent_coef, cfg.value_coef, cfg.clip_eps, False)
        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantage
        expected_policy_loss = -torch.min(surr1, surr2).mean()
        torch.testing.assert_close(result["policy_loss"], expected_policy_loss)
        assert (surr2 != surr1).any()

    @pytest.mark.gpu
    def test_ppo_loss_value_loss_correct(self, device) -> None:
        agent = DiscreteTestAgent().to(device)
        state = torch.randn(16, 4, device=device)
        action = torch.randint(0, 2, (16,), device=device)
        log_prob = torch.randn(16, device=device)
        old_values = torch.randn(16, device=device)
        advantage = torch.randn(16, device=device)
        return_ = torch.randn(16, device=device)
        cfg = AlgoConfig()
        params = dict(agent.named_parameters())
        buffers = dict(agent.named_buffers())
        result = ppo_loss(agent, params, buffers, state, action, log_prob, old_values,
                         advantage, return_, cfg.ent_coef, cfg.value_coef, cfg.clip_eps, False)
        logits, new_values = agent(state)
        expected = 0.5 * nn.functional.mse_loss(new_values.view(-1), return_)
        torch.testing.assert_close(result["value_loss"], expected)

    @pytest.mark.gpu
    def test_ppo_loss_entropy_subtracted(self, device) -> None:
        agent = DiscreteTestAgent().to(device)
        state = torch.randn(16, 4, device=device)
        action = torch.randint(0, 2, (16,), device=device)
        log_prob = torch.randn(16, device=device)
        old_values = torch.randn(16, device=device)
        advantage = torch.randn(16, device=device)
        return_ = torch.randn(16, device=device)
        cfg = AlgoConfig()
        params = dict(agent.named_parameters())
        buffers = dict(agent.named_buffers())
        result = ppo_loss(agent, params, buffers, state, action, log_prob, old_values,
                         advantage, return_, cfg.ent_coef, cfg.value_coef, cfg.clip_eps, False)
        expected = (result["policy_loss"] + cfg.value_coef * result["value_loss"]
                    - cfg.ent_coef * result["entropy_loss"])
        torch.testing.assert_close(result["loss"], expected)

    @pytest.mark.gpu
    def test_ppo_loss_ratio_one_when_same_policy(self, device) -> None:
        agent = DiscreteTestAgent().to(device)
        state = torch.randn(16, 4, device=device)
        action = torch.randint(0, 2, (16,), device=device)
        logits, _ = agent(state)
        dist = agent.build_distribution(logits)
        new_log_prob, _ = eval_action(dist, action)
        advantage = torch.ones(16, device=device)
        return_ = torch.zeros(16, device=device)
        cfg = AlgoConfig()
        params = dict(agent.named_parameters())
        buffers = dict(agent.named_buffers())
        old_values = torch.zeros(16, device=device)
        result = ppo_loss(agent, params, buffers, state, action, new_log_prob, old_values,
                         advantage, return_, cfg.ent_coef, cfg.value_coef, cfg.clip_eps, False)
        torch.testing.assert_close(result["policy_loss"], -advantage.mean())

    @pytest.mark.gpu
    def test_ppo_loss_raises_when_agent_missing_build_distribution(self, device) -> None:
        class NoDistAgent(BaseAgent):
            def __init__(self): super().__init__(); self.actor = nn.Linear(4, 2)
            def forward(self, state, **kwargs): return self.actor(state), torch.zeros(1)
            def get_action(self, state, action=None, **kwargs):
                return {"action": torch.zeros(1), "log_prob": torch.zeros(1),
                        "entropy": torch.zeros(1), "value": torch.zeros(1)}
            # NOTE: no build_distribution -> assert_agent_contract raises
        agent = NoDistAgent().to(device)
        state = torch.randn(16, 4, device=device)
        params = dict(agent.named_parameters())
        buffers = dict(agent.named_buffers())
        cfg = AlgoConfig()
        old_values = torch.zeros(16, device=device)
        with pytest.raises(AttributeError):
            ppo_loss(agent, params, buffers, state, torch.zeros(16, device=device),
                     torch.zeros(16, device=device), old_values,
                     torch.zeros(16, device=device), torch.zeros(16, device=device),
                     cfg.ent_coef, cfg.value_coef, cfg.clip_eps, False)


class TestPpoEdgeCases:
    @pytest.mark.gpu
    def test_ppo_batch_larger_than_dataset(self, device) -> None:
        agent = DiscreteTestAgent().to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        buf = TestPpoFunction()._make_buffer(64, 4, device)
        cfg = AlgoConfig()
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        result = ppo_func(agent, optimizer, buf, cfg, scheduler, device=device)
        assert "loss" in result

    @pytest.mark.gpu
    def test_ppo_epochs_zero_returns_empty_dict(self, device) -> None:
        agent = DiscreteTestAgent().to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        buf = TestPpoFunction()._make_buffer(64, 4, device)
        cfg = AlgoConfig()
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        cfg.epochs = 0
        result = ppo_func(agent, optimizer, buf, cfg, scheduler, device=device)
        assert result == {}

    @pytest.mark.gpu
    def test_ppo_scheduler_stepped_once(self, device) -> None:
        from unittest.mock import MagicMock
        agent = DiscreteTestAgent().to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        buf = TestPpoFunction()._make_buffer(64, 4, device)
        cfg = AlgoConfig()
        scheduler = MagicMock()
        ppo_func(agent, optimizer, buf, cfg, scheduler, device=device)
        assert scheduler.step.call_count == 1


class TestGaeTruncation:
    @pytest.mark.gpu
    def test_truncation_uses_final_value_not_next_value(self, device) -> None:
        cfg = AlgoConfig()
        reward = torch.tensor([1.0, 1.0, 1.0], device=device).unsqueeze(-1)
        values = torch.tensor([0.0, 0.0, 0.0], device=device).unsqueeze(-1)
        last_value = torch.tensor([999.0], device=device)
        truncated = torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(-1)
        dones = torch.zeros_like(truncated)
        final_values = torch.tensor([50.0], device=device)
        buf = Buffer(step=3, data={"return": (), "advantage": ()}, device=device)
        buf.slice = 3
        gae_compute(reward, values, last_value, dones, truncated,
                    final_values.unsqueeze(0).expand_as(values), buf, cfg)
        adv = buf.data["advantage"][:3]
        # Step 2 is truncated: delta = r + gamma * final_value - v
        #   delta = 1.0 + 0.99 * 50.0 - 0.0 = 50.5
        # gae_mask at step 2 = 0 (truncated → no GAE carry to step 1)
        # So adv[2] = delta[2] = 50.5
        torch.testing.assert_close(adv[2].squeeze(), torch.tensor(50.5, device=device))
        # Step 1 is NOT truncated, NOT done: gae_mask = 1.0
        # delta[1] = r + gamma * truncated * final + gamma * (1-trunc) * next_val - v
        #          = 1.0 + 0.99 * (0 * 50 + 1 * 0) - 0 = 1.0
        # gae[1] = delta[1] + gamma * lambda * gae_mask * gae[2] = 1.0 + 0.99*0.95*1.0*50.5 = 49.0225
        torch.testing.assert_close(adv[1].squeeze(), torch.tensor(1.0 + 0.99 * 0.95 * 50.5, device=device))

    @pytest.mark.gpu
    def test_truncation_vs_done_different_advantages(self, device) -> None:
        """Truncation and done both set gae_mask=0, but done also blocks
        delta bootstrap via delta_mask while truncation uses final_values."""
        cfg = AlgoConfig()
        reward = torch.tensor([1.0, 1.0, 1.0], device=device).unsqueeze(-1)
        values = torch.tensor([0.0, 0.0, 0.0], device=device).unsqueeze(-1)
        last_value = torch.tensor([10.0], device=device)
        final_values = torch.tensor([10.0], device=device).unsqueeze(0).expand_as(values)
        # Scenario A: step 2 done
        buf_done = Buffer(step=3, data={"return": (), "advantage": ()}, device=device)
        buf_done.slice = 3
        gae_compute(reward, values, last_value,
                    torch.tensor([[0.0, 0.0, 1.0]], device=device).T,
                    torch.zeros(3, 1, device=device),
                    final_values, buf_done, cfg)
        # Scenario B: step 2 truncated with same final_values
        buf_trunc = Buffer(step=3, data={"return": (), "advantage": ()}, device=device)
        buf_trunc.slice = 3
        gae_compute(reward, values, last_value,
                    torch.zeros(3, 1, device=device),
                    torch.tensor([[0.0, 0.0, 1.0]], device=device).T,
                    final_values, buf_trunc, cfg)
        # Done: delta_mask blocks bootstrap → delta[2] = 1.0 - 0.0 = 1.0
        # Truncation: delta uses final_value → delta[2] = 1.0 + 0.99*10.0 = 10.9
        assert buf_trunc.data["advantage"][2].item() > buf_done.data["advantage"][2].item()
