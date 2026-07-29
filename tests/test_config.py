"""Unit tests for configuration dataclasses (rl_template.config).

Tests PPOConfig (frozen hyperparameters) and TrainConfig (computed fields).
"""

import pytest
from rl_template.config import PPOConfig, TrainConfig

# =============================================================================
# Test PPOConfig (Frozen Dataclass)
# =============================================================================


class TestPPOConfig:
    """Tests for the immutable PPOConfig dataclass."""

    def test_defaults(self):
        """Default PPOConfig should use documented hyperparameter values."""
        cfg = PPOConfig()
        assert cfg.lr == 3e-5
        assert cfg.gamma == 0.999
        assert cfg.gae_lambda == 0.95
        assert cfg.clip_eps == 0.1
        assert cfg.ent_coef == 0.01
        assert cfg.value_coef == 0.5

    def test_custom_values(self):
        """PPOConfig should accept and store custom hyperparameter values."""
        cfg = PPOConfig(
            lr=1e-3,
            gamma=0.95,
            gae_lambda=0.8,
            clip_eps=0.2,
            ent_coef=0.05,
            value_coef=1.0,
        )
        assert cfg.lr == 1e-3
        assert cfg.gamma == 0.95
        assert cfg.gae_lambda == 0.8
        assert cfg.clip_eps == 0.2
        assert cfg.ent_coef == 0.05
        assert cfg.value_coef == 1.0

    def test_is_frozen(self):
        """PPOConfig is frozen -- assigning to any field should raise AttributeError."""
        cfg = PPOConfig()
        with pytest.raises(AttributeError):
            cfg.lr = 0.1

    def test_is_frozen_all_fields(self):
        """All PPOConfig fields should be immutable."""
        cfg = PPOConfig()
        with pytest.raises(AttributeError):
            cfg.gamma = 0.5
        with pytest.raises(AttributeError):
            cfg.clip_eps = 0.99


# =============================================================================
# Test TrainConfig (Mutable with Computed Fields)
# =============================================================================


class TestTrainConfig:
    """Tests for TrainConfig with __post_init__ computed fields."""

    def test_model_path_computed(self):
        """model_path should be derived from model_saved_path and model_name."""
        cfg = TrainConfig(model_name="ppo_agent", model_saved_path="/tmp/models")
        assert cfg.model_path == "/tmp/models/ppo_agent.pt"

    def test_num_update_computed(self):
        """num_update should equal timestamp // rollout_steps."""
        cfg = TrainConfig(
            model_name="m",
            model_saved_path="/tmp",
            timestamp=6_000_000,
            rollout_steps=2048,
        )
        assert cfg.num_update == 6_000_000 // 2048

    def test_num_update_custom_values(self):
        """num_update computation should work with any timestamp/rollout_steps."""
        cfg = TrainConfig(
            model_name="m", model_saved_path="/tmp", timestamp=1000, rollout_steps=100
        )
        assert cfg.num_update == 10

    def test_no_print_on_init(self, capsys):
        """TrainConfig.__post_init__ should produce no stdout output."""
        _ = TrainConfig(model_name="m", model_saved_path="/tmp")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_default_values(self):
        """TrainConfig should use documented defaults for optional fields."""
        cfg = TrainConfig(model_name="m", model_saved_path="/tmp")
        assert cfg.batch_size == 64
        assert cfg.rollout_steps == 2048
        assert cfg.timestamp == 1_000_000
        assert cfg.device in ("cpu", "cuda:0")

    def test_model_path_with_nested_dir(self):
        """model_path should correctly handle deeply nested directory paths."""
        cfg = TrainConfig(model_name="agent", model_saved_path="/a/b/c/d")
        assert cfg.model_path == "/a/b/c/d/agent.pt"
