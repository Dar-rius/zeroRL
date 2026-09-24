"""2-link planar reacher example (MujocoEnv + PPO).

Usage:
  uv run python examples/reacher_mujoco.py --device cuda
  CUDA_VISIBLE_DEVICES="" uv run python examples/reacher_mujoco.py --device cpu
"""

import argparse

import mujoco
import numpy as np
import torch
from gymnasium import spaces

from zerorl.algorithms.ppo import easy_train_ppo
from zerorl.config import AlgoConfig, TrainConfig
from zerorl.helpers.mujoco import MujocoEnv

REACHER_XML = """
<mujoco model="reacher2d">
  <compiler angle="radian"/>
  <option timestep="0.01" gravity="0 0 0"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
  </visual>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="0.4 0.4 0.05" rgba="0.9 0.9 0.9 1"/>
    <body name="link1" pos="0 0 0.05">
      <joint name="shoulder" type="hinge" axis="0 0 1" damping="0.05" limited="true" range="-3.14 3.14"/>
      <geom type="capsule" fromto="0 0 0 0.12 0 0" size="0.02" rgba="0.2 0.45 0.9 1" mass="0.08"/>
      <body name="link2" pos="0.12 0 0">
        <joint name="elbow" type="hinge" axis="0 0 1" damping="0.05" limited="true" range="-2.7 2.7"/>
        <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.018" rgba="0.25 0.55 0.95 1" mass="0.06"/>
        <site name="fingertip" pos="0.1 0 0" size="0.02" rgba="0.1 0.8 0.3 1"/>
      </body>
    </body>
    <body name="target" mocap="true" pos="0.15 0.0 0.05">
      <geom type="sphere" size="0.025" rgba="0.9 0.15 0.15 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="shoulder" ctrlrange="-1 1" gear="1.5"/>
    <motor joint="elbow" ctrlrange="-1 1" gear="1.2"/>
  </actuator>
</mujoco>
"""


class Reacher2D(MujocoEnv):
    """Planar 2-DOF arm; fingertip must reach and hold a random target."""

    SUCCESS_DIST = 0.028  # overlap nette (somme rayons tip+cible = 0.045)
    HOLD_STEPS = 5

    def __init__(self, max_steps: int = 200):
        self.max_steps = max_steps
        self._steps = 0
        self._hold = 0
        self._prev_dist = 0.0
        super().__init__(model_xml=REACHER_XML, frame_skip=2)
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64
        )

    def _tip_pos(self) -> np.ndarray:
        return np.asarray(self.data.site("fingertip").xpos[:2], dtype=np.float64).copy()

    def _target_pos(self) -> np.ndarray:
        return self.data.mocap_pos[0, :2].copy()

    def _set_target(self, xy: np.ndarray) -> None:
        self.data.mocap_pos[0, :2] = xy
        self.data.mocap_pos[0, 2] = 0.05
        mujoco.mj_forward(self.model, self.data)

    def reset_model(self) -> np.ndarray:
        self._steps = 0
        self._hold = 0
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        qpos[:] = self.np_random.uniform(-0.2, 0.2, size=qpos.shape)
        qvel[:] = 0.0
        self.set_state(qpos, qvel)

        for _ in range(100):
            ang = float(self.np_random.uniform(-np.pi, np.pi))
            radius = float(self.np_random.uniform(0.10, 0.20))
            target = np.array([radius * np.cos(ang), radius * np.sin(ang)])
            self._set_target(target)
            dist = float(np.linalg.norm(self._tip_pos() - target))
            if dist >= 0.12:
                break

        self._prev_dist = float(np.linalg.norm(self._tip_pos() - self._target_pos()))
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
        qpos = self.data.qpos.astype(np.float64)
        qvel = self.data.qvel.astype(np.float64)
        tip = self._tip_pos()
        target = self._target_pos()
        return np.concatenate(
            [
                np.cos(qpos),
                np.sin(qpos),
                qvel,
                tip,
                target,
                tip - target,
            ]
        )

    def _get_reward(self) -> float:
        dist = float(np.linalg.norm(self._tip_pos() - self._target_pos()))
        progress = self._prev_dist - dist
        self._prev_dist = dist
        ctrl = float(np.square(self.data.ctrl).sum())
        reward = 2.0 * progress - 0.2 * dist - 0.01 * ctrl
        if dist < self.SUCCESS_DIST:
            self._hold += 1
            reward += 8.0 + 40.0 * (self.SUCCESS_DIST - dist)
        else:
            self._hold = 0
        return reward

    def _is_truncated(self) -> bool:
        return self._steps >= self.max_steps

    def _is_terminated(self) -> bool:
        return self._hold >= self.HOLD_STEPS

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 480)
        if self._camera is None:
            self._camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, self._camera)
            self._camera.lookat[:] = [0.0, 0.0, 0.0]
            self._camera.distance = 0.85
            self._camera.elevation = -90.0
            self._camera.azimuth = 0.0
        self._renderer.update_scene(self.data, camera=self._camera)
        return np.asarray(self._renderer.render())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--timestamp", type=int, default=1_000_000)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False")

    config = TrainConfig(
        model_name=f"Reacher2D-{args.device}",
        project_name="reacher_mujoco",
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
        batch_size=256,
        gae_lambda=0.95,
        clip_eps=0.2,
    )

    trainer = easy_train_ppo(Reacher2D, config, algo_config, hidden_layer=128)
    torch.nn.init.constant_(trainer.agent.log_std, -1.5)
    trainer.agent.log_std.requires_grad_(False)
    trainer.train(use_wandb=True, save_model=True)
    trainer.try_agent(iterations=3, gif_path=f"reacher_{args.device}")


if __name__ == "__main__":
    main()
