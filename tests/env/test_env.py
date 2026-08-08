"""Integration tests for BaseEnv abstract interface using real Gymnasium envs."""

import numpy as np
import gymnasium as gym
from zerorl.env import BaseEnv, register_env


class GymTestEnv(BaseEnv):
    """A real ZeroRL-compatible environment wrapping a Gymnasium env."""
    def __init__(self, env_id="CartPole-v1"):
        super().__init__()
        self._env = gym.make(env_id)
        # On récupère les vrais espaces depuis Gymnasium
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        self._env.close()


class TestGymEnvIntegration:
    """Verify that the wrapper correctly interfaces with a real Gymnasium env."""

    def setup_method(self) -> None:
        self.env = GymTestEnv("CartPole-v1")

    def teardown_method(self) -> None:
        self.env.close()

    def test_reset_returns_obs_and_info(self) -> None:
        obs, info = self.env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,) # CartPole state shape
        assert isinstance(info, dict)

    def test_step_returns_five_values(self) -> None:
        self.env.reset(seed=42)
        result = self.env.step(0) # Push cart left
        assert len(result) == 5

    def test_step_returns_correct_types(self) -> None:
        self.env.reset(seed=42)
        obs, reward, terminated, truncated, info = self.env.step(1) # Push cart right
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_can_run_full_episode(self) -> None:
        """Ensure we can step until termination or truncation without errors."""
        self.env.reset(seed=42)
        steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Take random actions from the action space
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            steps += 1
            # CartPole-v1 truncates at 500 steps
            assert steps <= 500 
            
        assert steps > 0


class TestRegisterEnv:
    def test_register_env_registers_id(self) -> None:
        env_id = "ZeroRLTestEnv-v0"
        if env_id in gym.envs.registry:
            del gym.envs.registry[env_id]

        @register_env(env_id)
        class RegEnv(GymTestEnv):
            pass

        assert env_id in gym.envs.registry
        env = gym.make(env_id)
        env.close()
