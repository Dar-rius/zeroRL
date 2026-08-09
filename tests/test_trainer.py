"""Tests for zerorl.trainer (make_env + prototype)."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import gymnasium as gym
from zerorl.agent import BaseAgent, eval_action
from zerorl.env import BaseEnv
from zerorl.train import BaseTrain
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


class TestPrototype:
    @pytest.mark.gpu
    def test_prototype_cartpole_string_default_agent(self, tmp_path: Path) -> None:
        t = trainer.prototype(
            algo="ppo",
            env="CartPole-v1",
            agent=None,
            timestamp=64,
            rollout_steps=8,
            model_name="cartpole",
            model_save_path=str(tmp_path),
            batch_size=8,
            epochs=1,
        )
        assert isinstance(t, BaseTrain)
        state, _ = t.env.reset(seed=0)
        last = t.rollout_phase(state)
        assert t.buffer.size == 8
        assert "value" in last
        t.env.close()

    @pytest.mark.gpu
    def test_prototype_custom_agent_and_env(self, tmp_path: Path) -> None:
        env = _TinyEnv()
        agent = _CustomAgent()
        t = trainer.prototype(
            algo="ppo",
            env=env,
            agent=agent,
            timestamp=32,
            rollout_steps=4,
            model_name="custom",
            model_save_path=str(tmp_path),
            batch_size=4,
            epochs=1,
        )
        state, _ = t.env.reset(seed=1)
        t.rollout_phase(state)
        assert t.buffer.size == 4
        t.env.close()

    def test_prototype_requires_env(self) -> None:
        with pytest.raises(ValueError):
            trainer.prototype(algo="ppo", env=None, agent=None)

    def test_prototype_unknown_algo(self) -> None:
        with pytest.raises(ValueError):
            trainer.prototype(algo="sac", env="CartPole-v1")

    def test_prototype_unknown_kwarg(self) -> None:
        with pytest.raises(TypeError):
            trainer.prototype(algo="ppo", env="CartPole-v1", not_a_param=1)

    @pytest.mark.gpu
    def test_prototype_vector_string_env(self, tmp_path: Path) -> None:
        t = trainer.prototype(
            algo="ppo",
            env="CartPole-v1",
            is_vector=True,
            num_envs=2,
            timestamp=32,
            rollout_steps=4,
            model_name="vec",
            model_save_path=str(tmp_path),
            batch_size=4,
            epochs=1,
        )
        assert isinstance(t.env, VectorEnv)
        assert t.train_config.num_envs == 2
        t.env.close()
