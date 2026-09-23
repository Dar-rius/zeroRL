"""Tests for Hopper2D (MujocoEnv) and its immediate-mode training loop."""

import numpy as np
import pytest
import torch

from examples.hopper_mujoco_immediate import Hopper2D, run_immediate
from zerorl.functions import get_obs_act, set_seed, vectorize_env


class TestHopper2DEnv:
    def test_spaces_and_reset(self) -> None:
        env = Hopper2D()
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert env.action_space.shape == (3,)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)
        # Absolute x is omitted from obs: qpos[1:] (5) + qvel (6) = 11
        assert obs.shape == (11,)
        # Standing pose: torso ~1.25, foot near the floor (not a 1 m drop).
        assert 1.15 < float(env.data.qpos[1]) < 1.35
        foot_id = env.model.geom("foot_geom").id
        assert float(env.data.geom_xpos[foot_id, 2]) < 0.25
        env.close()

    def test_step_clip_and_finite(self) -> None:
        env = Hopper2D()
        env.reset(seed=1)
        action = env.action_space.sample() * 5.0  # outside ctrlrange
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        env.close()

    def test_torch_action_accepted(self) -> None:
        env = Hopper2D()
        env.reset(seed=2)
        action = torch.zeros(env.action_space.shape, dtype=torch.float32)
        obs, reward, *_ = env.step(action)
        assert obs.shape == (11,)
        assert isinstance(reward, float)
        env.close()

    def test_fall_terminates(self) -> None:
        env = Hopper2D(max_steps=500)
        env.reset(seed=3)
        # Collapse the hopper by slamming joints.
        terminated = False
        for _ in range(200):
            _, _, terminated, truncated, _ = env.step(
                np.array([-1.0, -1.0, -1.0], dtype=np.float64)
            )
            if terminated or truncated:
                break
        assert terminated or truncated
        env.close()

    def test_vectorize_class(self) -> None:
        seeds = set_seed(0, 2)
        env = vectorize_env(Hopper2D, num_envs=2)
        obs, _ = env.reset(seed=seeds)
        assert obs.shape[0] == 2
        obs_dim, act_dim, obs_n, act_n, is_discrete = get_obs_act(env)
        assert obs_n == 11
        assert act_n == 3
        assert is_discrete is False
        assert act_dim == (3,)
        actions = np.zeros((2, 3), dtype=np.float32)
        next_obs, rewards, terms, truncs, _ = env.step(actions)
        assert next_obs.shape == obs.shape
        assert rewards.shape == (2,)
        env.close()

    def test_render_rgb(self) -> None:
        env = Hopper2D()
        env.reset(seed=4)
        frame = env.render()
        assert frame.shape == (400, 720, 3)
        assert frame.dtype == np.uint8
        env.close()


@pytest.mark.slow
class TestHopperImmediateSmoke:
    def test_one_update_smoke(self) -> None:
        """One PPO update via immediate mode — validates the full new stack."""
        agent = run_immediate(
            device="cpu",
            timestamp=512,  # 1 update with num_envs=1, rollout=512
            num_envs=1,
            rollout_steps=512,
            seed=0,
            use_wandb=False,
            use_tb=False,
            save_gif=False,
            hidden_layer=64,
        )
        assert agent is not None
        with torch.inference_mode():
            obs = torch.zeros(1, 11, dtype=torch.float32)
            out = agent.get_action(obs)
        assert "action" in out and out["action"].shape[-1] == 3
