"""Unit tests for BaseTrain (zerorl.train)."""

import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from gymnasium import spaces

from zerorl.agent import BaseAgent
from zerorl.common import Buffer
from zerorl.config import AlgoConfig, TrainConfig
from zerorl.env import BaseEnv
from zerorl.train import BaseTrain


class MockAgent(BaseAgent):
    """Minimal agent for testing BaseTrain."""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2):
        super().__init__()
        self.actor = nn.Linear(obs_dim, act_dim)
        self.val = nn.Linear(obs_dim, 1)

    def forward(self, state: torch.Tensor, **kwargs):
        logits = self.actor(state)
        val = self.val(state)
        return logits, val

    @staticmethod
    def build_distribution(logits: torch.Tensor):
        return torch.distributions.Categorical(logits=logits)


class MockEnv(BaseEnv):
    """Minimal environment for testing BaseTrain."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-4.8, 4.8, shape=(4,))
        self.action_space = spaces.Discrete(2)
        self._env = __import__("gymnasium").make("CartPole-v1")

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        self._env.close()


def _mock_update_weights(
    agent: BaseAgent,
    buffer: Buffer,
    optimizer,
    last_output: dict,
    algo_config: AlgoConfig | None,
) -> dict[str, torch.Tensor]:
    """Mock update_weights callable for testing."""
    return {"loss": torch.tensor(0.0)}


def _make_train_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TrainConfig:
    """Create TrainConfig with CPU device forced."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    return TrainConfig(model_name="test", model_save_path=str(tmp_path))


class TestBaseTrainInit:
    """Verify BaseTrain stores all dependencies."""

    def test_stores_attributes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        obs_dim, act_dim = 4, 2
        agent = MockAgent(obs_dim, act_dim)
        env = MockEnv()
        buf = Buffer(step=10, data={"state": (obs_dim,)}, device=torch.device("cpu"))
        cfg = _make_train_config(tmp_path, monkeypatch)
        algo_cfg = AlgoConfig()

        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, algo_cfg)
        assert trainer.agent is agent
        assert trainer.env is env
        assert trainer.buffer is buf
        assert trainer.train_config is cfg
        assert trainer.algo_config is algo_cfg
        env.close()

    def test_creates_optimizer_from_algo_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = MockAgent()
        env = MockEnv()
        buf = Buffer(step=10, data={"state": (4,)}, device=torch.device("cpu"))
        cfg = _make_train_config(tmp_path, monkeypatch)
        algo_cfg = AlgoConfig(lr=0.001)

        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, algo_cfg)
        assert len(trainer.optimizer.param_groups) == 1
        assert trainer.optimizer.param_groups[0]["lr"] == 0.001
        env.close()

    def test_uses_provided_optimizer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = MockAgent()
        env = MockEnv()
        buf = Buffer(step=10, data={"state": (4,)}, device=torch.device("cpu"))
        cfg = _make_train_config(tmp_path, monkeypatch)
        custom_opt = torch.optim.SGD(agent.parameters(), lr=0.05)

        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, optimizer=custom_opt)
        assert trainer.optimizer is custom_opt
        env.close()

    def test_default_require_buffer_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = MockAgent()
        env = MockEnv()
        buf = Buffer(step=10, data={"state": (4,)}, device=torch.device("cpu"))
        cfg = _make_train_config(tmp_path, monkeypatch)

        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg)
        assert trainer.require_buffer_size == 10
        env.close()

    def test_custom_require_buffer_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = MockAgent()
        env = MockEnv()
        buf = Buffer(step=10, data={"state": (4,)}, device=torch.device("cpu"))
        cfg = _make_train_config(tmp_path, monkeypatch)

        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, require_buffer_size=5)
        assert trainer.require_buffer_size == 5
        env.close()


class TestBaseTrainSaveModel:
    """Verify save_model() writes weights to disk."""

    def test_creates_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = MockAgent()
        env = MockEnv()
        buf = Buffer(step=10, data={"state": (4,)}, device=torch.device("cpu"))
        save_dir = str(tmp_path / "models")
        cfg = TrainConfig(model_name="test", model_save_path=save_dir)
        # Force CPU to avoid CUDA error
        cfg.device = torch.device("cpu")
        cfg.model_path = os.path.join(save_dir, "test.pt")

        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg)
        trainer.save_model()
        assert os.path.exists(cfg.model_path)
        env.close()

    def test_saved_weights_are_loadable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = MockAgent()
        env = MockEnv()
        buf = Buffer(step=10, data={"state": (4,)}, device=torch.device("cpu"))
        save_dir = str(tmp_path / "models")
        cfg = TrainConfig(model_name="test", model_save_path=save_dir)
        # Force CPU to avoid CUDA error
        cfg.device = torch.device("cpu")
        cfg.model_path = os.path.join(save_dir, "test.pt")

        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg)
        trainer.save_model()

        new_agent = MockAgent()
        state_dict = torch.load(cfg.model_path, weights_only=True)
        new_agent.load_state_dict(state_dict)
        for p1, p2 in zip(agent.parameters(), new_agent.parameters()):
            assert torch.allclose(p1, p2)
        env.close()
