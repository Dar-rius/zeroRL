"""Unit tests for MujocoEnv."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.vector import SyncVectorEnv

from zerorl.functions import vectorize_env
from zerorl.helpers.mujoco import MujocoEnv

POINT_MASS_XML = """
<mujoco model="point_mass">
  <option timestep="0.01"/>
  <worldbody>
    <body name="particle" pos="0 0 0.05">
      <geom type="sphere" size="0.05" mass="0.1"/>
      <joint name="slide_x" type="slide" axis="1 0 0" limited="false"/>
      <joint name="slide_y" type="slide" axis="0 1 0" limited="false"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide_x" ctrlrange="-1 1" gear="1"/>
    <motor joint="slide_y" ctrlrange="-0.5 0.5" gear="1"/>
  </actuator>
</mujoco>
"""


class RewardPointMass(MujocoEnv):
    def __init__(self):
        super().__init__(model_xml=POINT_MASS_XML, frame_skip=2)

    def _get_reward(self) -> float:
        return -float(np.linalg.norm(self.data.qpos[:2]))


class TestMujocoEnvInit:
    def test_requires_exactly_one_model_source(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            MujocoEnv()
        with pytest.raises(ValueError, match="exactly one"):
            MujocoEnv(model_path="x.xml", model_xml=POINT_MASS_XML)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            MujocoEnv(model_path=tmp_path / "missing.xml")

    def test_frame_skip_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="frame_skip"):
            MujocoEnv(model_xml=POINT_MASS_XML, frame_skip=0)

    def test_xml_sets_spaces_from_model(self) -> None:
        env = MujocoEnv(model_xml=POINT_MASS_XML)
        assert env.action_space.shape == (2,)
        np.testing.assert_allclose(env.action_space.low, [-1.0, -0.5])
        np.testing.assert_allclose(env.action_space.high, [1.0, 0.5])
        assert env.observation_space.shape == (4,)
        assert env.dt == pytest.approx(0.05)
        env.close()

    def test_model_path_loads_file(self, tmp_path: Path) -> None:
        path = tmp_path / "point.xml"
        path.write_text(POINT_MASS_XML, encoding="utf-8")
        env = MujocoEnv(model_path=path)
        assert env.action_space.shape == (2,)
        env.close()


class TestMujocoEnvApi:
    def setup_method(self) -> None:
        self.env = MujocoEnv(model_xml=POINT_MASS_XML, frame_skip=2)

    def teardown_method(self) -> None:
        self.env.close()

    def test_reset_returns_obs_and_info(self) -> None:
        obs, info = self.env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == self.env.observation_space.shape
        assert isinstance(info, dict)
        np.testing.assert_allclose(obs[:2], 0.0, atol=1e-8)

    def test_step_returns_five_values(self) -> None:
        self.env.reset(seed=0)
        obs, reward, terminated, truncated, info = self.env.step(
            self.env.action_space.sample()
        )
        assert obs.shape == (4,)
        assert isinstance(reward, float)
        assert terminated is False
        assert truncated is False
        assert isinstance(info, dict)

    def test_step_accepts_torch_action(self) -> None:
        self.env.reset(seed=0)
        action = torch.zeros(2)
        obs, reward, terminated, truncated, info = self.env.step(action)
        assert obs.shape == (4,)

    def test_step_rejects_wrong_action_dim(self) -> None:
        self.env.reset(seed=0)
        with pytest.raises(ValueError, match="Action dimension"):
            self.env.step(np.zeros(3))

    def test_set_state_updates_qpos_qvel(self) -> None:
        self.env.reset(seed=0)
        qpos = self.env.init_qpos.copy()
        qvel = self.env.init_qvel.copy()
        qpos[0] = 0.3
        qvel[1] = -0.1
        self.env.set_state(qpos, qvel)
        np.testing.assert_allclose(self.env.data.qpos, qpos)
        np.testing.assert_allclose(self.env.data.qvel, qvel)

    def test_subclass_can_override_reward(self) -> None:
        env = RewardPointMass()
        env.reset(seed=0)
        env.set_state(
            np.array([1.0, 0.0], dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        )
        _, reward, *_ = env.step(np.zeros(2))
        assert reward == pytest.approx(-1.0)
        env.close()

    def test_render_returns_rgb_frame(self) -> None:
        self.env.reset(seed=0)
        frame = self.env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[-1] == 3
        assert frame.max() > frame.min(), "render should not be a flat gray frame"


class TestMujocoEnvVectorize:
    def test_vectorize_with_class_factory(self) -> None:
        env = vectorize_env(lambda: MujocoEnv(model_xml=POINT_MASS_XML), num_envs=3)
        assert isinstance(env, SyncVectorEnv)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (3, 4)
        next_obs, rewards, terminated, truncated, _ = env.step(
            np.zeros((3, 2), dtype=np.float32)
        )
        assert next_obs.shape == (3, 4)
        assert rewards.shape == (3,)
        assert terminated.shape == (3,)
        assert truncated.shape == (3,)
        env.close()


class TestGymnasiumMujocoPath:
    def test_inverted_pendulum_via_gym_make(self) -> None:
        env = gym.make("InvertedPendulum-v5")
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        next_obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        assert next_obs.shape == obs.shape
        assert np.ndim(np.asarray(reward)) == 0
        env.close()
