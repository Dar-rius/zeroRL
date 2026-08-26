"""Unit tests for easy_train_ppo (zerorl.algorithms.ppo.easy_ppo)."""

import pytest
import torch
import torch.nn as nn
import gymnasium as gym
from zerorl.algorithms.ppo.easy_ppo import easy_train_ppo
from zerorl.train import BaseTrain
from zerorl.helpers.agent import BaseAgent, eval_action
from zerorl.config import TrainConfig, AlgoConfig
from gymnasium.vector import SyncVectorEnv
from zerorl.helpers.env import BaseEnv


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tmp_config(tmp_path, device) -> TrainConfig:
    cfg = TrainConfig(model_name="easy_ppo_test", model_save_path=str(tmp_path), project_name="test")
    cfg.device = device
    cfg.rollout_steps = 8
    cfg.timestamp = 8 * 3
    cfg.num_envs = 1
    cfg.num_update = 3
    return cfg


class CustomCartPole(BaseEnv):
    """Custom env for testing base_env override."""
    def __init__(self):
        super().__init__()
        self._env = gym.make("CartPole-v1")
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed)
    def step(self, action):
        return self._env.step(action)
    def close(self):
        self._env.close()


class CustomAgent(BaseAgent):
    """Custom agent for testing base_agent override."""
    def __init__(self, obs_dim=4, act_dim=2):
        super().__init__()
        self.actor = nn.Linear(obs_dim, act_dim)
        self.critic = nn.Linear(obs_dim, 1)
    def forward(self, state, **kwargs):
        return self.actor(state), self.critic(state)
    @staticmethod
    def build_distribution(logits):
        return torch.distributions.Categorical(logits=logits)
    def get_action(self, state, action=None, **kwargs):
        logits, value = self.forward(state, **kwargs)
        dist = self.build_distribution(logits)
        if action is None: action = dist.sample()
        log_prob, entropy = eval_action(dist, action)
        return {"action": action, "log_prob": log_prob, "entropy": entropy, "value": value}


class TestEasyTrainPpo:
    def test_returns_basetrain(self, tmp_config) -> None:
        algo = AlgoConfig()
        train = easy_train_ppo("CartPole-v1", config=tmp_config, algo_config=algo)
        assert isinstance(train, BaseTrain)
        train.env.close()

    def test_uses_custom_agent(self, tmp_config, device) -> None:
        algo = AlgoConfig()
        agent = CustomAgent(4, 2).to(device)
        train = easy_train_ppo("CartPole-v1", config=tmp_config, algo_config=algo,
                                base_agent=agent)
        assert train.agent is agent
        train.env.close()

    def test_uses_custom_env(self, tmp_config) -> None:
        algo = AlgoConfig()
        env = CustomCartPole()
        train = easy_train_ppo(env, config=tmp_config, algo_config=algo)
        assert isinstance(train.env.envs[0], CustomCartPole)
        assert train.env.envs[0] is not env
        train.env.close()

    def test_default_env_from_id(self, tmp_config) -> None:
        algo = AlgoConfig()
        train = easy_train_ppo("CartPole-v1", config=tmp_config, algo_config=algo)
        assert train.env.observation_space.shape[0] == 1
        train.env.close()

    @pytest.mark.gpu
    def test_train_runs_end_to_end(self, tmp_path, device) -> None:
        cfg = TrainConfig(model_name="easy_e2e", model_save_path=str(tmp_path), project_name="test")
        cfg.device = device
        cfg.rollout_steps = 16
        cfg.timestamp = 16 * 2
        cfg.num_envs = 1
        cfg.num_update = 2
        algo = AlgoConfig(epochs=2)
        train = easy_train_ppo("CartPole-v1", config=cfg, algo_config=algo)
        train.train(use_wandb=False, use_tb=False)
        assert isinstance(train.episode_rewards, list)
        train.env.close()

    def test_discrete_cartpole_agent_get_action_keys(self, tmp_config, device) -> None:
        agent = CustomAgent(4, 2).to(device)
        state = torch.randn(1, 4, device=device)
        out = agent.get_action(state)
        assert set(out.keys()) == {"action", "log_prob", "entropy", "value"}

    def test_easy_ppo_no_double_wrap(self, tmp_config) -> None:
        env = CustomCartPole()
        train = easy_train_ppo(env, config=tmp_config, algo_config=AlgoConfig())
        assert isinstance(train.env, SyncVectorEnv)
        assert isinstance(train.env.envs[0], CustomCartPole)
        assert train.env.envs[0] is not env
        train.env.close()

    def test_discrete_cartpole_uses_categorical(self, tmp_config, device) -> None:
        train = easy_train_ppo("CartPole-v1", config=tmp_config, algo_config=AlgoConfig())
        state = torch.randn(1, 4, device=device)
        logits, _ = train.agent.forward(state)
        dist = train.agent.build_distribution(logits)
        assert isinstance(dist, torch.distributions.Categorical)
        train.env.close()

    def test_continuous_pendulum_uses_normal(self, tmp_config, device) -> None:
        train = easy_train_ppo("Pendulum-v1", config=tmp_config, algo_config=AlgoConfig())
        state = torch.randn(1, 3, device=device)
        logits, _ = train.agent.forward(state)
        dist = train.agent.build_distribution(logits)
        assert isinstance(dist, torch.distributions.Normal)
        train.env.close()