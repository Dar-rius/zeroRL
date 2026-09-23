"""Point-mass reach example (MujocoEnv + PPO).

Usage:
  uv run python examples/point_mass.py --device cuda
  CUDA_VISIBLE_DEVICES="" uv run python examples/point_mass.py --device cpu
"""

import argparse

import mujoco
import numpy as np
import torch
from gymnasium import spaces

from zerorl.algorithms.ppo import easy_train_ppo
from zerorl.config import AlgoConfig, TrainConfig
from zerorl.helpers.mujoco import MujocoEnv

POINT_MASS_XML = """
<mujoco model="point_mass">
  <option timestep="0.01"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
  </visual>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.85 0.85 0.85 1"/>
    <site name="target" pos="0 0 0.01" size="0.1" rgba="0.9 0.15 0.15 0.9"/>
    <body name="particle" pos="0 0 0.08">
      <geom type="sphere" size="0.08" mass="0.05" rgba="0.15 0.45 0.95 1"/>
      <joint name="slide_x" type="slide" axis="1 0 0" damping="1.5"/>
      <joint name="slide_y" type="slide" axis="0 1 0" damping="1.5"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide_x" ctrlrange="-1 1" gear="8"/>
    <motor joint="slide_y" ctrlrange="-1 1" gear="8"/>
  </actuator>
</mujoco>
"""


class PointMass(MujocoEnv):
    """2D particle that must reach the origin."""

    def __init__(self, max_steps: int = 150):
        super().__init__(model_xml=POINT_MASS_XML, frame_skip=2)
        self.max_steps = max_steps
        self._steps = 0
        self._prev_dist = 0.0
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64
        )

    def reset_model(self) -> np.ndarray:
        self._steps = 0
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        for _ in range(100):
            xy = self.np_random.uniform(-0.9, 0.9, size=2)
            if float(np.linalg.norm(xy)) >= 0.45:
                break
        else:
            xy = np.array([0.7, 0.0], dtype=np.float64)
        qpos[:2] = xy
        qvel[:2] = 0.0
        self.set_state(qpos, qvel)
        self._prev_dist = float(np.linalg.norm(self.data.qpos[:2]))
        return self._get_obs()

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.clip(
            np.asarray(action, dtype=np.float64),
            self.action_space.low,
            self.action_space.high,
        )
        self._steps += 1
        return super().step(action)

    def _get_obs(self) -> np.ndarray:
        pos = self.data.qpos[:2].astype(np.float64)
        vel = self.data.qvel[:2].astype(np.float64)
        return np.concatenate([pos, vel])

    def _get_reward(self) -> float:
        dist = float(np.linalg.norm(self.data.qpos[:2]))
        progress = self._prev_dist - dist
        self._prev_dist = dist
        ctrl = float(np.square(self.data.ctrl).sum())
        reward = progress * 2.0 - 0.1 * dist - 0.01 * ctrl
        if dist < 0.08:
            reward += 10.0
        return reward

    def _is_truncated(self) -> bool:
        return self._steps >= self.max_steps

    def _is_terminated(self) -> bool:
        return float(np.linalg.norm(self.data.qpos[:2])) < 0.08

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 480)
        if self._camera is None:
            self._camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, self._camera)
            self._camera.lookat[:] = 0.0
            self._camera.distance = 2.8
            self._camera.elevation = -60.0
            self._camera.azimuth = 90.0
        self._renderer.update_scene(self.data, camera=self._camera)
        return np.asarray(self._renderer.render())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--timestamp", type=int, default=500_000)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False")

    config = TrainConfig(
        model_name=f"PointMass-{args.device}",
        project_name="point_mass_mujoco",
        timestamp=args.timestamp,
        rollout_steps=512,
        num_envs=16,
        normalize=False,
        profile=True,
    )
    config.device = torch.device(args.device)
    algo_config = AlgoConfig(
        lr=3e-4,
        ent_coef=0.0,
        epochs=10,
        batch_size=128,
        gae_lambda=0.95,
        clip_eps=0.2,
    )

    trainer = easy_train_ppo(PointMass, config, algo_config, hidden_layer=128)
    torch.nn.init.constant_(trainer.agent.log_std, -2.0)
    trainer.agent.log_std.requires_grad_(False)
    trainer.train(use_wandb=True, save_model=True)
    trainer.try_agent(iterations=2, gif_path=f"point_mass_{args.device}")


if __name__ == "__main__":
    main()
