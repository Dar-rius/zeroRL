"""Unit tests for BaseTrain (rl_template.train)."""

import os

import pytest
import torch
import torch.nn as nn
from gymnasium import spaces

from zerorl.agent import BaseAgent
from zerorl.algorithms.ppo import ppo
from zerorl.common import Buffer
from zerorl.config import PPOConfig, TrainConfig
from zerorl.env import BaseEnv
from zerorl.errors import EmptyBufferError
from zerorl.train import BaseTrain


class MockAgent(BaseAgent):
    """Minimal agent for testing BaseTrain."""

    def __init__(self, obs_dim=4, act_dim=2):
        super().__init__()
        self.actor = nn.Linear(obs_dim, act_dim)
        self.val = nn.Linear(obs_dim, 1)

    def forward(self, state, **kwargs):
        logits = self.actor(state)
        val = self.val(state)
        return logits, val.squeeze(-1)

    def build_distributions(self, logits):
        return torch.distributions.Categorical(logits=logits)


class MockEnv(BaseEnv):
    """Minimal environment for testing BaseTrain."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-4.8, 4.8, shape=(4,))
        self.action_space = spaces.Discrete(2)
        self._env = __import__("gymnasium").make("CartPole-v1")

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def close(self):
        self._env.close()


class TestBaseTrainInit:
    """Verify BaseTrain stores all dependencies."""

    def test_stores_attributes(self, tmp_path):
        obs_dim, act_dim = 4, 2
        agent = MockAgent(obs_dim, act_dim)
        env = MockEnv()
        buf = Buffer(step=10, state_shape=(obs_dim,))
        cfg = TrainConfig(model_name="test", model_saved_path=str(tmp_path))
        ppo_cfg = PPOConfig()

        # Check type
        trainer = BaseTrain(agent, env, buf, cfg, ppo_cfg)
        assert trainer.agent is agent
        assert trainer.env is env
        assert trainer.buffer is buf
        assert trainer.train_config is cfg
        assert trainer.ppo_config is ppo_cfg

    def test_default_last_value(self, tmp_path):
        agent = MockAgent()
        env = MockEnv()
        buf = Buffer(step=10, state_shape=(4,))
        cfg = TrainConfig(model_name="test", model_saved_path=str(tmp_path))
        ppo_cfg = PPOConfig()

        trainer = BaseTrain(agent, env, buf, cfg, ppo_cfg)
        assert trainer.last_value == torch.tensor(0.0, device=cfg.device)

    def test_default_cumulative_reward(self, tmp_path):
        agent = MockAgent()
        env = MockEnv()
        buf = Buffer(step=10, state_shape=(4,))
        cfg = TrainConfig(model_name="test", model_saved_path=str(tmp_path))
        ppo = PPOTrainer(agent, PPOConfig())

        trainer = BaseTrain(agent, env, buf, cfg, ppo)
        assert trainer.cumulative_reward == 0.0


class TestBaseTrainUpdateWeights:
    """Verify update_weights() raises error on empty buffer."""

    def test_raises_empty_buffer_error(self, tmp_path):
        agent = MockAgent()
        env = MockEnv()
        buf = Buffer(step=10, state_shape=(4,))
        cfg = TrainConfig(model_name="test", model_saved_path=str(tmp_path))
        ppo = PPOTrainer(agent, PPOConfig())

        trainer = BaseTrain(agent, env, buf, cfg, ppo)
        with pytest.raises(EmptyBufferError):
            trainer.update_weights(step=0)

    def test_real(self, tmp_path):
        agent = MockAgent()
        env = MockEnv()
        buf = Buffer(step=2048, state_shape=(4,))
        cfg = TrainConfig(
            device="cpu", model_name="test", model_saved_path=str(tmp_path)
        )
        ppo = PPOTrainer(agent, PPOConfig())
        trainer = BaseTrain(agent, env, buf, cfg, ppo, 1)
        for step in range(5):
            state, _ = env.reset()
            trainer.rollout_phase(state)
            trainer.update_weights(step)
        env.close()


class TestBaseTrainSaveModel:
    """Verify save_model() writes weights to disk."""

    def test_creates_file(self, tmp_path):
        agent = MockAgent()
        env = MockEnv()
        buf = Buffer(step=10, state_shape=(4,))
        save_dir = str(tmp_path / "models")
        cfg = TrainConfig(model_name="test", model_saved_path=save_dir)
        ppo = PPOTrainer(agent, PPOConfig())

        trainer = BaseTrain(agent, env, buf, cfg, ppo)
        trainer.save_model()
        assert os.path.exists(cfg.model_path)

    def test_saved_weights_are_loadable(self, tmp_path):
        agent = MockAgent()
        env = MockEnv()
        buf = Buffer(step=10, state_shape=(4,))
        save_dir = str(tmp_path / "models")
        cfg = TrainConfig(model_name="test", model_saved_path=save_dir)
        ppo = PPOTrainer(agent, PPOConfig())

        trainer = BaseTrain(agent, env, buf, cfg, ppo)
        trainer.save_model()

        new_agent = MockAgent()
        agent.to("cpu")
        state_dict = torch.load(cfg.model_path, weights_only=True)
        new_agent.load_state_dict(state_dict)
        print(agent.device)
        print(new_agent.device)
        for p1, p2 in zip(agent.parameters(), new_agent.parameters()):
            assert torch.allclose(p1, p2)
