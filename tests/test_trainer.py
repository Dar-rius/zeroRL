"""Tests for zerorl.trainer (make_env + prototype)."""

import pytest
import gymnasium as gym
from zerorl.env import BaseEnv
from zerorl.vector_env import VectorEnv
from zerorl import trainer


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
