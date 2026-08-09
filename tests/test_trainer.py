"""Tests for zerorl.trainer (make_env + prototype)."""

import pytest
import torch
import torch.nn as nn
import gymnasium as gym
from zerorl.agent import BaseAgent, eval_action
from zerorl.env import BaseEnv
from zerorl.vector_env import VectorEnv
from zerorl import trainer
from zerorl.trainer import _resolve_agent


class _TinyEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self._env = gym.make("CartPole-v1")
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        if hasattr(action, "shape") and len(action.shape) > 0:
            action = action.item()
        return self._env.step(action)

    def close(self):
        self._env.close()


class TestMakeEnv:
    def test_make_env_string(self) -> None:
        env = trainer.make_env("CartPole-v1")
        assert isinstance(env, BaseEnv)
        assert env.observation_space.shape == (4,)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (4,)
        env.close()

    def test_make_env_passthrough_baseenv(self) -> None:
        original = _TinyEnv()
        env = trainer.make_env(original)
        assert env is original
        env.close()

    def test_make_env_vector(self) -> None:
        env = trainer.make_env("CartPole-v1", is_vector=True, num_envs=2)
        assert isinstance(env, VectorEnv)
        assert env.auto_reset is True
        env.close()

    def test_make_env_none_raises(self) -> None:
        with pytest.raises(ValueError):
            trainer.make_env(None)

    def test_make_env_vector_rejects_instance(self) -> None:
        with pytest.raises(ValueError):
            trainer.make_env(_TinyEnv(), is_vector=True, num_envs=2)


class _CustomAgent(BaseAgent):
    """Minimal discrete agent for passthrough tests."""

    def __init__(self) -> None:
        super().__init__()
        self.l = nn.Linear(4, 2)
        self.v = nn.Linear(4, 1)

    def forward(self, state: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return self.l(state), self.v(state)

    @staticmethod
    def build_distribution(logits: torch.Tensor) -> torch.distributions.Distribution:
        return torch.distributions.Categorical(logits=logits)

    def get_action(self, state: torch.Tensor, action: torch.Tensor | None = None, **kwargs):
        logits, value = self.forward(state, **kwargs)
        dist = self.build_distribution(logits)
        if action is None:
            action = dist.sample()
        log_prob, ent = eval_action(dist, action)
        return {"action": action, "log_prob": log_prob, "entropy": ent, "value": value}


class TestResolveAgent:
    def test_default_discrete_agent(self) -> None:
        env = trainer.make_env("CartPole-v1")
        agent = _resolve_agent(None, env)
        assert isinstance(agent, BaseAgent)
        out = agent.get_action(torch.zeros(1, 4))
        assert set(out) >= {"action", "log_prob", "entropy", "value"}
        env.close()

    def test_passthrough_custom_agent(self) -> None:
        env = trainer.make_env("CartPole-v1")
        custom = _CustomAgent()
        assert _resolve_agent(custom, env) is custom
        env.close()
