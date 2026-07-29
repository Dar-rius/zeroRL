"""Unit tests for PPOTrainer (rl_template.algorithms.ppo.ppo).

Covers initialization, GAE computation (hand-calculated), and linear
learning rate decay. Uses SimpleTestModel as a concrete nn.Module.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from rl_template.algorithms.ppo.ppo import PPOTrainer
from rl_template.config import PPOConfig

# =============================================================================
# Test Helper: SimpleTestModel
# =============================================================================


class SimpleTestModel(nn.Module):
    """Minimal policy+value network for testing PPOTrainer.

    Two-layer network with a Categorical policy head and a scalar
    value head. Implements get_action() to match PPOTrainer's interface.
    """

    def __init__(self, obs_dim=4, act_dim=2):
        super().__init__()
        self.policy = nn.Linear(obs_dim, act_dim)
        self.value = nn.Linear(obs_dim, 1)

    def forward(self, state, **kwargs):
        return self.policy(state), self.value(state)

    def get_action(self, state, action=None, **kwargs):
        """Sample or evaluate an action under the Categorical policy.

        Args:
            state: Observation tensor.
            action: Optional action to evaluate. When None, samples.

        Returns:
            Tuple of (action, log_prob, entropy, value).
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value.squeeze(-1)


# =============================================================================
# Test PPOTrainer Initialization
# =============================================================================


class TestPPOTrainerInit:
    """Verify PPOTrainer.__init__() stores all hyperparameters correctly."""

    def test_stores_hyperparams(self):
        """All constructor arguments should be stored as instance attributes."""
        model = SimpleTestModel()
        trainer = PPOTrainer(
            model,
            PPOConfig(
                lr=1e-3,
                gamma=0.95,
                gae_lambda=0.8,
                clip_eps=0.2,
                value_coef=1.0,
                ent_coef=0.05,
            ),
        )
        assert trainer.ppo_config.lr == 1e-3
        assert trainer.ppo_config.gamma == 0.95
        assert trainer.ppo_config.gae_lambda == 0.8
        assert trainer.ppo_config.clip_eps == 0.2
        assert trainer.ppo_config.value_coef == 1.0
        assert trainer.ppo_config.ent_coef == 0.05

    def test_stores_model(self):
        """The trainer should hold a reference to the model (not a copy)."""
        model = SimpleTestModel()
        trainer = PPOTrainer(model, PPOConfig())
        assert trainer.model is model

    def test_creates_optimizer(self):
        """An Adam optimizer should be created with the specified learning rate."""
        model = SimpleTestModel()
        trainer = PPOTrainer(model, PPOConfig(lr=5e-4))
        assert len(trainer.optimizer.param_groups) == 1
        assert trainer.optimizer.param_groups[0]["lr"] == 5e-4

    def test_default_hyperparams(self):
        """Default values should match the PPOConfig defaults."""
        model = SimpleTestModel()
        trainer = PPOTrainer(model, PPOConfig())
        assert trainer.ppo_config.lr == 3e-5
        assert trainer.ppo_config.gamma == 0.999
        assert trainer.ppo_config.gae_lambda == 0.95
        assert trainer.ppo_config.clip_eps == 0.1
        assert trainer.ppo_config.value_coef == 0.5
        assert trainer.ppo_config.ent_coef == 0.01

    def test_mse_loss_exists(self):
        """The trainer should have an MSELoss instance for value function loss."""
        model = SimpleTestModel()
        trainer = PPOTrainer(model, PPOConfig())
        assert isinstance(trainer.mse_loss, nn.MSELoss)


# =============================================================================
# Test GAE (Generalized Advantage Estimation)
# =============================================================================


class TestComputeGAE:
    """Verify compute_gae() produces correct advantages via hand-calculated examples."""

    def test_returns_equals_advantages_plus_values(self):
        """By definition: returns = advantages + values (always true)."""
        model = SimpleTestModel()
        trainer = PPOTrainer(model, PPOConfig())
        rewards = np.array([1.0, 2.0, 3.0])
        values = np.array([0.5, 0.5, 0.5])
        dones = np.array([0.0, 0.0, 0.0])
        last_value = 1.0
        returns, advantages, _ = trainer.compute_gae(rewards, values, last_value, dones)
        np.testing.assert_allclose(returns, advantages + values)

    def test_hand_calculated_single_step(self):
        """Single-step GAE with known values.

        Setup: r=[1.0], V=[0.0], last_value=0.0, dones=[0.0]
        Expected: delta = 1.0, gae = 1.0
        """
        model = SimpleTestModel()
        trainer = PPOTrainer(model, PPOConfig(gamma=0.99, gae_lambda=0.95))
        rewards = np.array([1.0])
        values = np.array([0.0])
        dones = np.array([0.0])
        last_value = 0.0
        returns, advantages, delta = trainer.compute_gae(
            rewards, values, last_value, dones
        )
        expected_delta = 1.0 + 0.99 * 0.0 * 1.0 - 0.0
        expected_gae = expected_delta
        np.testing.assert_allclose(delta, [expected_delta])
        np.testing.assert_allclose(advantages, [expected_gae])
        np.testing.assert_allclose(returns, [expected_gae + 0.0])

    def test_hand_calculated_two_steps(self):
        """Two-step GAE verifying backward accumulation.

        Setup: r=[1.0, 1.0], V=[0.0, 0.0], last_value=0.0, dones=[0.0, 0.0]
        Expected: gae_1 = 1.0, gae_0 = 1.0 + gamma*lambda = 1.9405
        """
        model = SimpleTestModel()
        trainer = PPOTrainer(model, PPOConfig(gamma=0.99, gae_lambda=0.95))
        rewards = np.array([1.0, 1.0])
        values = np.array([0.0, 0.0])
        dones = np.array([0.0, 0.0])
        last_value = 0.0
        returns, advantages, delta = trainer.compute_gae(
            rewards, values, last_value, dones
        )
        expected_gae_1 = 1.0
        expected_gae_0 = 1.0 + 0.99 * 0.95 * expected_gae_1
        np.testing.assert_allclose(advantages[1], expected_gae_1, atol=1e-6)
        np.testing.assert_allclose(advantages[0], expected_gae_0, atol=1e-6)

    def test_done_resets_gae(self):
        """Episode boundary (done=1) should reset the GAE accumulation.

        Setup: r=[1.0, 1.0], V=[0.0, 0.0], dones=[1.0, 0.0]
        Both advantages should be 1.0 independently (no accumulation across done).
        """
        model = SimpleTestModel()
        trainer = PPOTrainer(model, PPOConfig(gamma=0.99, gae_lambda=0.95))
        rewards = np.array([1.0, 1.0])
        values = np.array([0.0, 0.0])
        dones = np.array([1.0, 0.0])
        last_value = 0.0
        returns, advantages, _ = trainer.compute_gae(rewards, values, last_value, dones)
        np.testing.assert_allclose(advantages[1], 1.0, atol=1e-6)
        np.testing.assert_allclose(advantages[0], 1.0, atol=1e-6)

    def test_output_shapes(self):
        """All output arrays should have the same shape as the input rewards."""
        model = SimpleTestModel()
        trainer = PPOTrainer(model, PPOConfig())
        n = 10
        rewards = np.ones(n)
        values = np.zeros(n)
        dones = np.zeros(n)
        returns, advantages, delta = trainer.compute_gae(rewards, values, 0.0, dones)
        assert returns.shape == (n,)
        assert advantages.shape == (n,)
        assert delta.shape == (n,)

    def test_with_nonzero_values(self):
        """GAE should work correctly with non-zero value estimates."""
        model = SimpleTestModel()
        trainer = PPOTrainer(model, PPOConfig(gamma=0.99, gae_lambda=0.95))
        rewards = np.array([5.0, 3.0, 2.0])
        values = np.array([1.0, 2.0, 3.0])
        dones = np.array([0.0, 0.0, 0.0])
        last_value = 4.0
        returns, advantages, _ = trainer.compute_gae(rewards, values, last_value, dones)
        np.testing.assert_allclose(returns, advantages + values)
        assert returns.shape == (3,)


# =============================================================================
# Test Learning Rate Decay
# =============================================================================


class TestLRDecay:
    """Verify lr_decay() applies linear learning rate scheduling.

    Formula: current_lr = lr * (1.0 - step / total_steps)
    """

    def test_step_zero_no_change(self):
        """At step=0, the learning rate should remain unchanged."""
        model = SimpleTestModel()
        trainer = PPOTrainer(model, PPOConfig(lr=0.01))
        trainer.lr_decay(lr=0.01, total_steps=1000, step=0)
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.01)

    def test_step_half(self):
        """At step=total_steps/2, the learning rate should be halved."""
        model = SimpleTestModel()
        trainer = PPOTrainer(model, PPOConfig(lr=0.01))
        trainer.lr_decay(lr=0.01, total_steps=1000, step=500)
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.005)

    def test_final_step_near_zero(self):
        """At step=total_steps-1, the learning rate should be near zero."""
        model = SimpleTestModel()
        trainer = PPOTrainer(model, PPOConfig(lr=0.01))
        trainer.lr_decay(lr=0.01, total_steps=1000, step=999)
        expected = 0.01 * (1.0 - 999 / 1000)
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(expected)

    def test_lr_zero_at_end(self):
        """At step=total_steps, the learning rate should be exactly zero."""
        model = SimpleTestModel()
        trainer = PPOTrainer(model, PPOConfig(lr=0.01))
        trainer.lr_decay(lr=0.01, total_steps=1000, step=1000)
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0)

    def test_updates_all_param_groups(self):
        """lr_decay should update ALL param groups, not just the first."""
        model = SimpleTestModel()
        trainer = PPOTrainer(model, PPOConfig(lr=0.01))
        trainer.optimizer.add_param_group(
            {"params": [torch.zeros(1, requires_grad=True)], "lr": 0.02}
        )
        trainer.lr_decay(lr=0.01, total_steps=100, step=50)
        for pg in trainer.optimizer.param_groups:
            assert pg["lr"] == pytest.approx(0.005)
