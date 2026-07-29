"""Unit tests for continuous action spaces.

Verifies that the PPO trainer and Buffer work correctly with continuous
action distributions (Gaussian/Normal) instead of discrete (Categorical).
"""

import numpy as np
import torch
import torch.nn as nn
from rl_template.algorithms.ppo.ppo import PPOTrainer
from rl_template.common import Buffer
from rl_template.config import PPOConfig

# =============================================================================
# Test Helper: ContinuousTestModel
# =============================================================================


class ContinuousTestModel(nn.Module):
    """Minimal continuous-policy network for testing PPO with Gaussian actions.

    Uses a Normal distribution parameterized by (mean, learnable log_std)
    for continuous action sampling.
    """

    def __init__(self, obs_dim: int = 4, act_dim: int = 2):
        super().__init__()
        self.mean = nn.Linear(obs_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.value = nn.Linear(obs_dim, 1)

    def get_distribution(self, state: torch.Tensor, **kwargs):
        """Build a Normal distribution from the policy output.

        Args:
            state: Observation tensor.

        Returns:
            Tuple of (distribution, value) where distribution is Normal.
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        action_mean = self.mean(state)
        action_std = self.log_std.exp().expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)
        return dist, self.value(state).squeeze(-1)

    def get_action(self, state, action=None, **kwargs):
        """Sample or evaluate an action under the Gaussian policy.

        log_prob and entropy are summed across action dimensions.

        Args:
            state: Observation tensor.
            action: Optional action to evaluate. When None, samples.

        Returns:
            Tuple of (action, log_prob, entropy, value).
        """
        dist, value = self.get_distribution(state)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value


# =============================================================================
# Test Continuous Action Buffer
# =============================================================================


class TestContinuousBuffer:
    """Verify Buffer handles continuous (float) actions correctly."""

    def test_continuous_action_shape(self):
        """Buffer with action_shape=(2,) should store 2D continuous actions."""
        buf = Buffer(step=10, state_shape=(4,), action_shape=(2,))
        assert buf.actions.shape == (10, 2)

    def test_insert_continuous_action(self):
        """Continuous float actions should be stored correctly."""
        buf = Buffer(step=5, state_shape=(4,), action_shape=(2,))
        action = np.array([0.5, -0.3])
        buf.insert(
            state=np.zeros(4),
            action=action,
            old_log_prob=-0.1,
            reward=1.0,
            value=0.5,
            dones=0,
        )
        np.testing.assert_array_almost_equal(buf.actions[0], action)

    def test_get_all_continuous_actions(self):
        """get_all() should return continuous actions as float32 tensors."""
        buf = Buffer(step=5, state_shape=(4,), action_shape=(2,))
        for i in range(5):
            action = np.array([float(i), float(i) * -0.5])
            buf.insert(
                state=np.ones(4) * i,
                action=action,
                old_log_prob=float(i),
                reward=float(i),
                value=float(i),
                dones=0,
            )
        states, actions, _, _, _, _, _, _, _ = buf.get_all()
        assert actions.shape == (5, 2)
        assert actions.dtype == torch.float32
        np.testing.assert_array_almost_equal(actions[2].numpy(), [2.0, -1.0])

    def test_scalar_continuous_action(self):
        """Buffer with action_shape=(1,) for 1D continuous actions."""
        buf = Buffer(step=5, state_shape=(4,), action_shape=(1,))
        for i in range(5):
            buf.insert(
                state=np.ones(4),
                action=np.array([float(i) * 0.1]),
                old_log_prob=0.0,
                reward=0.0,
                value=0.0,
                dones=0,
            )
        _, actions, _, _, _, _, _, _, _ = buf.get_all()
        assert actions.shape == (5, 1)
        assert actions.dtype == torch.float32


# =============================================================================
# Test Continuous Model Distribution
# =============================================================================


class TestContinuousModel:
    """Verify ContinuousTestModel produces valid distributions and outputs."""

    def test_get_action_returns_four_values(self):
        """get_action() should return (action, log_prob, entropy, value)."""
        model = ContinuousTestModel(obs_dim=4, act_dim=2)
        state = torch.randn(4)
        action, log_prob, entropy, value = model.get_action(state)
        assert action.shape == (2,)
        assert log_prob.shape == ()
        assert entropy.shape == ()
        assert value.shape == ()

    def test_get_action_with_provided_action(self):
        """get_action(action=...) should evaluate the given action's log_prob."""
        model = ContinuousTestModel(obs_dim=4, act_dim=2)
        state = torch.randn(4)
        action = torch.tensor([0.5, -0.5])
        out_action, log_prob, entropy, value = model.get_action(state, action)
        torch.testing.assert_close(out_action, action)
        assert log_prob.item() != 0.0

    def test_log_prob_sums_across_dims(self):
        """log_prob should sum across action dimensions."""
        model = ContinuousTestModel(obs_dim=4, act_dim=3)
        state = torch.randn(4)
        action = torch.randn(3)
        _, log_prob, _, _ = model.get_action(state, action)
        dist, _ = model.get_distribution(state)
        expected = dist.log_prob(action).sum(dim=-1)
        torch.testing.assert_close(log_prob, expected)

    def test_entropy_sums_across_dims(self):
        """entropy should sum across action dimensions."""
        model = ContinuousTestModel(obs_dim=4, act_dim=3)
        state = torch.randn(4)
        _, _, entropy, _ = model.get_action(state)
        dist, _ = model.get_distribution(state)
        expected = dist.entropy().sum(dim=-1)
        torch.testing.assert_close(entropy, expected)

    def test_batch_get_action(self):
        """get_action() should work with batched states."""
        model = ContinuousTestModel(obs_dim=4, act_dim=2)
        states = torch.randn(8, 4)
        actions, log_probs, entropies, values = model.get_action(states)
        assert actions.shape == (8, 2)
        assert log_probs.shape == (8,)
        assert entropies.shape == (8,)
        assert values.shape == (8,)


# =============================================================================
# Test PPO Update with Continuous Actions
# =============================================================================


class TestPPOContinuousUpdate:
    """Test the full PPO update pipeline with continuous actions."""

    def _fill_buffer_with_continuous_data(self, buf, model, obs_dim=4):
        """Helper: fill buffer using a continuous model's get_action()."""
        for _ in range(buf.step):
            state = torch.randn(obs_dim)
            with torch.no_grad():
                action, log_prob, _, value = model.get_action(state)
            buf.insert(
                state=state.numpy(),
                action=action.numpy(),
                old_log_prob=log_prob.item(),
                reward=np.random.randn(),
                value=value.item(),
                dones=0,
            )

    def test_update_with_continuous_actions(self):
        """PPO update() should complete without errors with continuous actions."""
        model = ContinuousTestModel(obs_dim=4, act_dim=2)
        trainer = PPOTrainer(model, PPOConfig(lr=3e-4))
        buf = Buffer(step=128, state_shape=(4,), action_shape=(2,))
        self._fill_buffer_with_continuous_data(buf, model, obs_dim=4)
        buf.insert_returns(
            np.random.randn(128).astype(np.float32),
            np.random.randn(128).astype(np.float32),
        )
        total_loss, pi_loss, v_loss, entropy = trainer.update(
            buf, total_steps=1000, step=0, batch_size=32, epochs=2
        )
        assert np.isfinite(total_loss)
        assert np.isfinite(pi_loss)
        assert np.isfinite(v_loss)
        assert np.isfinite(entropy)
        assert isinstance(total_loss, float)

    def test_update_reduces_loss_over_steps(self):
        """PPO should generally reduce loss when trained on fixed data."""
        torch.manual_seed(42)
        model = ContinuousTestModel(obs_dim=4, act_dim=2)
        trainer = PPOTrainer(model, PPOConfig(lr=1e-3))
        buf = Buffer(step=256, state_shape=(4,), action_shape=(2,))
        self._fill_buffer_with_continuous_data(buf, model, obs_dim=4)
        returns = np.ones(256, dtype=np.float32) * 10.0
        adv = np.ones(256, dtype=np.float32) * 1.0
        buf.insert_returns(returns, adv)
        losses = []
        for step in range(5):
            total_loss, _, _, _ = trainer.update(
                buf, total_steps=100, step=step, batch_size=64, epochs=5
            )
            losses.append(total_loss)
        assert losses[-1] < losses[0] + 1.0

    def test_gradient_flows_through_continuous_policy(self):
        """Verify gradients actually update the continuous policy parameters."""
        model = ContinuousTestModel(obs_dim=4, act_dim=2)
        trainer = PPOTrainer(model, PPOConfig(lr=1e-3))
        buf = Buffer(step=64, state_shape=(4,), action_shape=(2,))
        self._fill_buffer_with_continuous_data(buf, model, obs_dim=4)
        buf.insert_returns(
            np.random.randn(64).astype(np.float32),
            np.random.randn(64).astype(np.float32),
        )
        w_before = model.mean.weight.data.clone()
        trainer.update(buf, total_steps=100, step=0, batch_size=32, epochs=5)
        w_after = model.mean.weight.data.clone()
        assert not torch.allclose(w_before, w_after)

    def test_continuous_buffer_full_cycle(self):
        """Full cycle: fill buffer -> compute GAE -> insert returns -> PPO update."""
        model = ContinuousTestModel(obs_dim=8, act_dim=3)
        trainer = PPOTrainer(model, PPOConfig(lr=3e-4))
        buf = Buffer(step=128, state_shape=(8,), action_shape=(3,))
        for _ in range(128):
            state = torch.randn(8)
            with torch.no_grad():
                action, log_prob, _, value = model.get_action(state)
            buf.insert(
                state=state.numpy(),
                action=action.numpy(),
                old_log_prob=log_prob.item(),
                reward=np.random.randn(),
                value=value.item(),
                dones=0,
            )
        rewards = buf.rewards
        values = buf.values
        dones = buf.dones
        returns, advantages, _ = trainer.compute_gae(rewards, values, 0.0, dones)
        buf.insert_returns(returns, advantages)
        total_loss, pi_loss, v_loss, entropy = trainer.update(
            buf, total_steps=1000, step=0, batch_size=32, epochs=3
        )
        assert np.isfinite(total_loss)
        assert buf.size == 128

    def test_high_dim_continuous_actions(self):
        """PPO should work with high-dimensional continuous action spaces."""
        act_dim = 17
        model = ContinuousTestModel(obs_dim=8, act_dim=act_dim)
        trainer = PPOTrainer(model, PPOConfig(lr=3e-4))
        buf = Buffer(step=64, state_shape=(8,), action_shape=(act_dim,))
        for _ in range(64):
            state = torch.randn(8)
            with torch.no_grad():
                action, log_prob, _, value = model.get_action(state)
            buf.insert(
                state=state.numpy(),
                action=action.numpy(),
                old_log_prob=log_prob.item(),
                reward=np.random.randn(),
                value=value.item(),
                dones=0,
            )
        returns = np.random.randn(64).astype(np.float32)
        adv = np.random.randn(64).astype(np.float32)
        buf.insert_returns(returns, adv)
        total_loss, _, _, _ = trainer.update(
            buf, total_steps=100, step=0, batch_size=32, epochs=2
        )
        assert np.isfinite(total_loss)

    def test_single_dim_continuous_action(self):
        """PPO should work with 1D continuous actions (scalar action)."""
        model = ContinuousTestModel(obs_dim=4, act_dim=1)
        trainer = PPOTrainer(model, PPOConfig(lr=3e-4))
        buf = Buffer(step=64, state_shape=(4,), action_shape=(1,))
        for _ in range(64):
            state = torch.randn(4)
            with torch.no_grad():
                action, log_prob, _, value = model.get_action(state)
            buf.insert(
                state=state.numpy(),
                action=action.numpy(),
                old_log_prob=log_prob.item(),
                reward=np.random.randn(),
                value=value.item(),
                dones=0,
            )
        returns = np.random.randn(64).astype(np.float32)
        adv = np.random.randn(64).astype(np.float32)
        buf.insert_returns(returns, adv)
        total_loss, _, _, _ = trainer.update(
            buf, total_steps=100, step=0, batch_size=32, epochs=2
        )
        assert np.isfinite(total_loss)
