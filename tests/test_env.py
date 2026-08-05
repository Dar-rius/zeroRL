"""Unit tests for BaseEnv abstract interface (zerorl.env)."""

import numpy as np
import pytest
from zerorl.env import BaseEnv


class ConcreteTestEnv(BaseEnv):
    """Minimal concrete environment for testing the abstract interface."""

    def __init__(self):
        super().__init__()
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        return np.zeros(4, dtype=np.float32), {"info": "reset"}

    def step(self, action):
        self.steps += 1
        obs = np.ones(4, dtype=np.float32) * self.steps
        reward = float(self.steps)
        terminated = self.steps >= 5
        truncated = False
        return obs, reward, terminated, truncated, {}

    def close(self):
        pass


class TestBaseEnvAbstract:
    """Verify BaseEnv enforces abstract method contracts."""

    def test_is_abstract(self) -> None:
        from abc import ABC

        assert ABC in BaseEnv.__mro__

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseEnv()  # type: ignore[abstract]

    def test_must_implement_reset(self) -> None:
        class IncompleteEnv(BaseEnv):
            def step(self, action):
                pass

            def close(self):
                pass

        with pytest.raises(TypeError):
            IncompleteEnv()  # type: ignore[abstract]

    def test_must_implement_step(self) -> None:
        class IncompleteEnv(BaseEnv):
            def reset(self, *, seed=None, options=None):
                pass

            def close(self):
                pass

        with pytest.raises(TypeError):
            IncompleteEnv()  # type: ignore[abstract]

    def test_must_implement_close(self) -> None:
        class IncompleteEnv(BaseEnv):
            def reset(self, *, seed=None, options=None):
                pass

            def step(self, action):
                pass

        with pytest.raises(TypeError):
            IncompleteEnv()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self) -> None:
        env = ConcreteTestEnv()
        assert isinstance(env, BaseEnv)


class TestConcreteEnvBehavior:
    """Verify the concrete test environment works correctly."""

    def setup_method(self) -> None:
        self.env = ConcreteTestEnv()

    def test_reset_returns_obs_and_info(self) -> None:
        obs, info = self.env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        assert isinstance(info, dict)

    def test_step_returns_five_values(self) -> None:
        self.env.reset()
        result = self.env.step(np.array(0))
        assert len(result) == 5

    def test_step_returns_obs_reward_terminated_truncated_info(self) -> None:
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(np.array(0))
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_terminates_after_5_steps(self) -> None:
        self.env.reset()
        for _ in range(4):
            obs, reward, terminated, truncated, _ = self.env.step(np.array(0))
            assert not terminated
        obs, reward, terminated, truncated, _ = self.env.step(np.array(0))
        assert terminated

    def test_close_does_not_raise(self) -> None:
        self.env.close()
