"""Unit tests for configuration dataclasses (zerorl.config).

Tests AlgoConfig (mutable hyperparameters) and TrainConfig (computed fields).
"""

import pytest
import torch
from zerorl.config import AlgoConfig, TrainConfig

# =============================================================================
# Test AlgoConfig (Mutable Dataclass with Custom __init__)
# =============================================================================


class TestAlgoConfig:
    """Tests for the mutable AlgoConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = AlgoConfig()
        assert cfg.lr == 3e-4
        assert cfg.gamma == 0.99
        assert cfg.gae_lambda == 0.95
        assert cfg.clip_eps == 0.2
        assert cfg.ent_coef == 0.01
        assert cfg.value_coef == 0.5
        assert cfg.epochs == 10
        assert cfg.batch_size == 64
        assert cfg.tau == 0.005

    def test_custom_values(self) -> None:
        cfg = AlgoConfig(lr=1e-3, gamma=0.95, gae_lambda=0.8, clip_eps=0.3)
        assert cfg.lr == 1e-3
        assert cfg.gamma == 0.95
        assert cfg.gae_lambda == 0.8
        assert cfg.clip_eps == 0.3

    def test_mutable(self) -> None:
        cfg = AlgoConfig()
        cfg.lr = 0.1
        assert cfg.lr == 0.1

    def test_to_dict(self) -> None:
        cfg = AlgoConfig(lr=0.01)
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["lr"] == 0.01

    def test_partial_override(self) -> None:
        cfg = AlgoConfig(lr=0.5, epochs=20)
        assert cfg.lr == 0.5
        assert cfg.epochs == 20
        # Non-overridden fields keep defaults
        assert cfg.gamma == 0.99
        assert cfg.clip_eps == 0.2

    def test_custom_init_none_values(self) -> None:
        cfg = AlgoConfig()
        # All fields should be set (not None)
        assert cfg.lr is not None
        assert cfg.gamma is not None
        assert cfg.tau is not None

    def test_unknown_kwarg_sets_attr(self) -> None:
        cfg = AlgoConfig(unknown=5)
        assert cfg.unknown == 5


# =============================================================================
# Test TrainConfig (Mutable with Computed Fields)
# =============================================================================


class TestTrainConfig:
    """Tests for TrainConfig with __post_init__ computed fields."""

    def test_model_path_computed(self) -> None:
        cfg = TrainConfig(model_name="ppo_agent", model_save_path="/tmp/models", project_name="test")
        assert cfg.model_path == "./tmp/models/ppo_agent.pt"

    def test_num_update_computed(self) -> None:
        cfg = TrainConfig(
            model_name="m",
            model_save_path="/tmp",
            project_name="test",
            timestamp=6_000_000,
            rollout_steps=2048,
        )
        assert cfg.num_update == 6_000_000 // 2048

    def test_num_update_custom_values(self) -> None:
        cfg = TrainConfig(
            model_name="m", model_save_path="/tmp", project_name="test", timestamp=1000, rollout_steps=100
        )
        assert cfg.num_update == 10

    def test_no_print_on_init(self, capsys: pytest.CaptureFixture[str]) -> None:
        _ = TrainConfig(model_name="m", model_save_path="/tmp", project_name="test")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_default_values(self) -> None:
        cfg = TrainConfig(model_name="m", model_save_path="/tmp", project_name="test")
        assert cfg.rollout_steps == 2048
        assert cfg.timestamp == 1_000_000
        assert isinstance(cfg.device, torch.device)

    def test_model_path_with_nested_dir(self) -> None:
        cfg = TrainConfig(model_name="agent", model_save_path="/a/b/c/d", project_name="test")
        assert cfg.model_path == "./a/b/c/d/agent.pt"

    def test_device_field_exists(self) -> None:
        cfg = TrainConfig(model_name="m", model_save_path="/tmp", project_name="test")
        assert hasattr(cfg, "device")

    def test_no_batch_size_field(self) -> None:
        cfg = TrainConfig(model_name="m", model_save_path="/tmp", project_name="test")
        assert not hasattr(cfg, "batch_size")
