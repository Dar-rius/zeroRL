"""Unit tests for continuous action spaces.

Verifies that the PPO functions and Buffer work correctly with continuous
action distributions (Gaussian/Normal) instead of discrete (Categorical).
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from zerorl.agent import BaseAgent
from zerorl.algorithms.ppo.ppo import gae_compute, ppo
from zerorl.common import Buffer
from zerorl.config import AlgoConfig


# =============================================================================
# Test Helper: ContinuousTestAgent
# =============================================================================


class ContinuousTestAgent(BaseAgent):
    """Minimal continuous-policy agent for testing PPO with Gaussian actions.

    forward() returns (logits, value) where logits = concat([mean, std]).
    build_distribution() splits logits back into (mean, std) for Normal.
    """

    def __init__(self, obs_dim: int = 4, act_dim: int = 2):
        super().__init__()
        self.mean_layer = nn.Linear(obs_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.value = nn.Linear(obs_dim, 1)

    def forward(self, state: torch.Tensor, **kwargs):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        action_mean = self.mean_layer(state_t)
        action_std = self.log_std.exp().expand_as(action_mean)
        logits = torch.cat([action_mean, action_std], dim=-1)
        val = self.value(state_t)
        return logits, val

    @staticmethod
    def build_distribution(logits: torch.Tensor):
        act_dim = logits.shape[-1] // 2
        mean = logits[..., :act_dim]
        std = logits[..., act_dim:]
        return torch.distributions.Normal(mean, std)


# =============================================================================
# Test Continuous Action Buffer
# =============================================================================


class TestContinuousBuffer:
    """Verify Buffer handles continuous (float) actions correctly."""

    def test_continuous_action_shape(self) -> None:
        buf = Buffer(step=10, data={"state": (4,), "action": (2,)}, device=torch.device("cpu"))
        assert buf.data["action"].shape == (10, 2)

    def test_insert_continuous_action(self) -> None:
        buf = Buffer(step=5, data={"state": (4,), "action": (2,)}, device=torch.device("cpu"))
        action = torch.tensor([0.5, -0.3])
        buf.insert(state=torch.zeros(4), action=action)
        torch.testing.assert_close(buf.data["action"][0], action)

    def test_get_all_continuous_actions(self) -> None:
        buf = Buffer(step=5, data={"state": (4,), "action": (2,)}, device=torch.device("cpu"))
        for i in range(5):
            action = torch.tensor([float(i), float(i) * -0.5])
            buf.insert(state=torch.ones(4) * i, action=action)
        result = buf.get_all()
        assert result["action"].shape == (5, 2)
        assert result["action"].dtype == torch.float32
        torch.testing.assert_close(result["action"][2], torch.tensor([2.0, -1.0]))


# =============================================================================
# Test Continuous Model Distribution
# =============================================================================


class TestContinuousModel:
    """Verify ContinuousTestAgent produces valid distributions and outputs."""

    def test_get_action_returns_dict(self) -> None:
        agent = ContinuousTestAgent(obs_dim=4, act_dim=2)
        state = torch.randn(4)
        result = agent.get_action(state)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"action", "log_prob", "entropy", "value"}

    def test_get_action_with_provided_action(self) -> None:
        agent = ContinuousTestAgent(obs_dim=4, act_dim=2)
        state = torch.randn(4)
        action = torch.tensor([0.5, -0.5])
        result = agent.get_action(state, action)
        torch.testing.assert_close(result["action"], action)
        # log_prob is a tensor (may be scalar or multi-dim depending on eval_action)
        assert isinstance(result["log_prob"], torch.Tensor)

    def test_log_prob_single_sample(self) -> None:
        """For single-sample multi-dim actions, eval_action does NOT sum (dim==1)."""
        agent = ContinuousTestAgent(obs_dim=4, act_dim=3)
        state = torch.randn(4)
        action = torch.randn(3)
        result = agent.get_action(state, action)
        logits, _ = agent(state)
        dist = agent.build_distribution(logits)
        # eval_action: log_prob.dim() == 1, NOT > 1, so NOT summed
        expected = dist.log_prob(action)
        torch.testing.assert_close(result["log_prob"], expected)

    def test_log_prob_batch(self) -> None:
        """For batch multi-dim actions, eval_action sums over last dim."""
        agent = ContinuousTestAgent(obs_dim=4, act_dim=3)
        states = torch.randn(8, 4)
        actions = torch.randn(8, 3)
        result = agent.get_action(states, actions)
        logits, _ = agent(states)
        dist = agent.build_distribution(logits)
        expected = dist.log_prob(actions).sum(dim=-1)
        torch.testing.assert_close(result["log_prob"], expected)

    def test_entropy_single_sample(self) -> None:
        """For single-sample multi-dim, entropy is NOT summed (dim==1)."""
        agent = ContinuousTestAgent(obs_dim=4, act_dim=3)
        state = torch.randn(4)
        result = agent.get_action(state)
        logits, _ = agent(state)
        dist = agent.build_distribution(logits)
        expected = dist.entropy()
        torch.testing.assert_close(result["entropy"], expected)

    def test_batch_get_action(self) -> None:
        agent = ContinuousTestAgent(obs_dim=4, act_dim=2)
        states = torch.randn(8, 4)
        result = agent.get_action(states)
        assert result["action"].shape == (8, 2)
        assert result["log_prob"].shape == (8,)
        assert result["entropy"].shape == (8,)
        # value from nn.Linear(obs_dim, 1) outputs (batch, 1)
        assert result["value"].shape == (8, 1)


# =============================================================================
# Test PPO Update with Continuous Actions
# =============================================================================


class TestPPOContinuousUpdate:
    """Test the full PPO update pipeline with continuous actions."""

    def _make_buffer(self, n: int = 128, obs_dim: int = 4, act_dim: int = 2) -> Buffer:
        buf = Buffer(
            step=n,
            data={
                "state": (obs_dim,),
                "actions": (act_dim,),
                "old_log_probs": (),
                "adv": (),
                "returns": (),
                "value": (),
            },
            device=torch.device("cpu"),
        )
        agent = ContinuousTestAgent(obs_dim=obs_dim, act_dim=act_dim)
        for _ in range(n):
            state = torch.randn(obs_dim)
            with torch.no_grad():
                out = agent.get_action(state)
            # For single-sample multi-dim actions, eval_action returns
            # log_prob/entropy with shape (act_dim,) because dim()==1.
            # Buffer expects scalar old_log_probs, so sum over action dims.
            log_prob = out["log_prob"]
            if log_prob.dim() > 0:
                log_prob = log_prob.sum()
            buf.insert(
                state=state,
                actions=out["action"],
                old_log_probs=log_prob,
                adv=torch.randn(1).squeeze(),
                returns=torch.randn(1).squeeze(),
                value=out["value"].squeeze(),
            )
        return buf

    def test_ppo_runs_with_continuous_actions(self) -> None:
        agent = ContinuousTestAgent(obs_dim=4, act_dim=2)
        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        buf = self._make_buffer(n=128)
        cfg = AlgoConfig()
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        result = ppo(agent, optimizer, buf, cfg, scheduler, batch_size=32, epochs=2,
                     device=torch.device("cpu"))
        assert isinstance(result, dict)
        assert "loss" in result

    def test_ppo_updates_continuous_weights(self) -> None:
        agent = ContinuousTestAgent(obs_dim=4, act_dim=2)
        w_before = {k: v.clone() for k, v in agent.state_dict().items()}
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
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

    def test_continuous_buffer_full_cycle(self) -> None:
        """Full cycle: fill buffer -> compute GAE -> PPO update."""
        obs_dim, act_dim = 8, 3
        agent = ContinuousTestAgent(obs_dim=obs_dim, act_dim=act_dim)

        # Fill raw buffer
        raw_buf = Buffer(
            step=64,
            data={
                "state": (obs_dim,),
                "actions": (act_dim,),
                "old_log_probs": (),
                "reward": (),
                "done": (),
                "value": (),
            },
            device=torch.device("cpu"),
        )
        for _ in range(64):
            state = torch.randn(obs_dim)
            with torch.no_grad():
                out = agent.get_action(state)
            # Sum multi-dim log_prob to scalar for buffer storage
            log_prob = out["log_prob"]
            if log_prob.dim() > 0:
                log_prob = log_prob.sum()
            raw_buf.insert(
                state=state,
                actions=out["action"],
                old_log_probs=log_prob,
                reward=torch.randn(1).squeeze(),
                done=torch.tensor(0.0),
                value=out["value"].squeeze(),
            )

        # Compute GAE
        all_data = raw_buf.get_all()
        last_value = torch.tensor([0.0])
        returns, advantages, _ = gae_compute(
            all_data["reward"].squeeze(),
            all_data["value"].squeeze(),
            last_value,
            all_data["done"].squeeze(),
            AlgoConfig(),
        )

        # Build ppo-compatible buffer
        ppo_buf = Buffer(
            step=64,
            data={
                "state": (obs_dim,),
                "actions": (act_dim,),
                "old_log_probs": (),
                "adv": (),
                "returns": (),
                "value": (),
            },
            device=torch.device("cpu"),
        )
        for i in range(64):
            ppo_buf.insert(
                state=all_data["state"][i],
                actions=all_data["actions"][i],
                old_log_probs=all_data["old_log_probs"][i],
                adv=advantages[i],
                returns=returns[i],
                value=all_data["value"][i],
            )
        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        cfg = AlgoConfig()
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        result = ppo(agent, optimizer, ppo_buf, cfg, scheduler, batch_size=32, epochs=3,
                     device=torch.device("cpu"))
        assert isinstance(result, dict)
        assert "loss" in result
