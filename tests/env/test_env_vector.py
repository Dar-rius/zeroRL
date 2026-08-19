"""Integration tests for vectorize_env (zerorl.functions)."""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AutoresetMode
from zerorl.helpers import BaseEnv
from zerorl.functions import vectorize_env


class CustomCartPole(BaseEnv):
    """A custom env class to test the Callable initialization of vectorize_env."""
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


class TestVectorizeEnvInit:
    """Verify vectorize_env initializes correctly with strings, classes, and instances."""

    def test_init_with_string_id(self) -> None:
        env = vectorize_env("CartPole-v1", num_envs=4)
        assert isinstance(env, SyncVectorEnv)
        assert env.observation_space.shape == (4, 4)
        assert env.single_action_space.n == 2
        env.close()

    def test_init_with_callable_class(self) -> None:
        env = vectorize_env(CustomCartPole, num_envs=2)
        assert isinstance(env, SyncVectorEnv)
        assert env.observation_space.shape == (2, 4)
        env.close()

    def test_init_with_instance_deepcopies_independent_envs(self) -> None:
        """Regression: an env instance must be deep-copied per slot.

        Pins the fix for the shared-instance bug where every slot referenced
        the same env object.
        """
        original = CustomCartPole()
        env = vectorize_env(original, num_envs=3)
        assert all(sub is not original for sub in env.envs)
        assert len({id(sub) for sub in env.envs}) == 3
        env.close()

    def test_invalid_env_spec_raises_error(self) -> None:
        # Contract change: non-str/callable/class specs are deep-copied and
        # fail at the construction-time reset() call.
        with pytest.raises(AttributeError):
            vectorize_env(env_spec=12345, num_envs=1)  # type: ignore[arg-type]


class TestVectorizeEnvProperties:
    """Verify specific vectorize_env behaviors."""

    def test_autoreset_mode_is_same_step(self) -> None:
        env = vectorize_env("CartPole-v1", num_envs=2)
        assert env.metadata["autoreset_mode"] == AutoresetMode.SAME_STEP
        env.close()


class TestVectorizeEnvInteraction:
    """Verify step and reset produce correctly batched data."""

    def setup_method(self) -> None:
        self.num_envs = 4
        self.env = vectorize_env("CartPole-v1", num_envs=self.num_envs)

    def teardown_method(self) -> None:
        self.env.close()

    def test_reset_returns_batched_obs(self) -> None:
        obs, info = self.env.reset(seed=42)  # type: ignore[var-annotated]
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (self.num_envs, 4)
        assert isinstance(info, dict)

    def test_step_returns_batched_data(self) -> None:
        self.env.reset(seed=42)
        actions = np.array([0, 1, 0, 1])
        obs, reward, terminated, truncated, info = self.env.step(actions)  # type: ignore[var-annotated]

        assert obs.shape == (self.num_envs, 4)
        assert reward.shape == (self.num_envs,)
        assert terminated.shape == (self.num_envs,)
        assert truncated.shape == (self.num_envs,)

    def test_auto_reset_on_termination(self) -> None:
        """Ensures env resets automatically and can be stepped indefinitely."""
        self.env.reset(seed=42)

        for _ in range(550):
            actions = np.zeros(self.num_envs, dtype=np.int64)
            obs, reward, terminated, truncated, info = self.env.step(actions)  # type: ignore[var-annotated]

            assert obs.shape == (self.num_envs, 4)
            assert reward.shape == (self.num_envs,)
