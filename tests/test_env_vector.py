"""Integration tests for the VectorEnv wrapper (zerorl.vector_env)."""

import numpy as np
import pytest
import gymnasium as gym
from zerorl.env import BaseEnv
from zerorl.vector_env import VectorEnv


class CustomCartPole(BaseEnv):
    """A custom env class to test the Callable initialization of VectorEnv."""
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


class TestVectorEnvInit:
    """Verify VectorEnv initializes correctly with strings and callables."""

    def test_init_with_string_id(self) -> None:
        env = VectorEnv(env_spec="CartPole-v1", num_envs=4)
        assert isinstance(env, BaseEnv)
        assert env.observation_space.shape == (4, 4) 
        assert env.action_space.shape == (4,)
        env.close()

    def test_init_with_callable_class(self) -> None:
        env = VectorEnv(env_spec=CustomCartPole, num_envs=2)
        assert isinstance(env, BaseEnv)
        assert env.observation_space.shape == (2, 4)
        env.close()
    
    def test_invalid_env_spec_raises_error(self) -> None:
        with pytest.raises(ValueError, match="env_spec must be a string"):
            VectorEnv(env_spec=12345, num_envs=1) # type: ignore[arg-type]


class TestVectorEnvProperties:
    """Verify specific VectorEnv behaviors."""

    def test_auto_reset_is_true(self) -> None:
        env = VectorEnv("CartPole-v1", num_envs=2)
        assert env.auto_reset is True
        env.close()


class TestVectorEnvInteraction:
    """Verify step and reset produce correctly batched data."""

    def setup_method(self) -> None:
        self.num_envs = 4
        self.env = VectorEnv("CartPole-v1", num_envs=self.num_envs)

    def teardown_method(self) -> None:
        self.env.close()

    def test_reset_returns_batched_obs(self) -> None:
        obs, info = self.env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (self.num_envs, 4)
        assert isinstance(info, dict)

    def test_step_returns_batched_data(self) -> None:
        self.env.reset(seed=42)
        actions = np.array([0, 1, 0, 1])
        obs, reward, terminated, truncated, info = self.env.step(actions)
        
        assert obs.shape == (self.num_envs, 4)
        assert reward.shape == (self.num_envs,)
        assert terminated.shape == (self.num_envs,)
        assert truncated.shape == (self.num_envs,)

    def test_auto_reset_on_termination(self) -> None:
        """Ensures env resets automatically and can be stepped indefinitely."""
        self.env.reset(seed=42)
        
        for _ in range(550):
            actions = np.zeros(self.num_envs, dtype=np.int64)
            obs, reward, terminated, truncated, info = self.env.step(actions)
            
            assert obs.shape == (self.num_envs, 4)
            assert reward.shape == (self.num_envs,)
