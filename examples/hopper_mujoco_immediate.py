"""Hopper2D (MujocoEnv) + immediate-mode PPO.

Usage:
  uv run python examples/hopper_mujoco_immediate.py --device cuda
  CUDA_VISIBLE_DEVICES="" uv run python examples/hopper_mujoco_immediate.py --device cpu
"""

from __future__ import annotations

import argparse

import imageio
import mujoco
import numpy as np
import torch
from gymnasium import spaces
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from zerorl.algorithms.ppo import gae_compute, ppo_func
from zerorl.buffer import Buffer
from zerorl.config import AlgoConfig, TrainConfig
from zerorl.functions import (
    _deterministic_action,
    get_obs_act,
    parse_dict_to_tensor,
    processing_state,
    save_checkpoints,
    set_seed,
    to_env_action,
    vectorize_env,
)
from zerorl.helpers.factory import ActorCriticAgent
from zerorl.helpers.mujoco import MujocoEnv
from zerorl.logger import create_logger
from zerorl.processing import NormMeanStd

# Gymnasium Hopper-v5 geometry; rootz must set ref=1.25 (else spawn floats).
HOPPER_XML = """
<mujoco model="hopper2d">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="1" condim="1" contype="1" margin="0.001"
          solimp=".8 .8 .01" solref=".02 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.45 0.45 0.45"/>
    <rgba haze="0.92 0.94 0.96 1"/>
    <global offwidth="720" offheight="400"/>
  </visual>
  <worldbody>
    <light pos="0 -3 5" dir="0 0.4 -1"/>
    <geom name="floor" type="plane" size="50 4 0.1" rgba="0.82 0.84 0.86 1"
          friction="1.5 0.1 0.1" condim="3"/>
    <!-- Long visual runway (no collision): marks every 1 m, finish at 20 m. -->
    <geom name="start_line" type="box" size="0.04 1.8 0.015" pos="0 0 0.015"
          rgba="0.15 0.55 0.95 1" contype="0" conaffinity="0"/>
    <geom name="mark_1" type="box" size="0.025 1.5 0.01" pos="1 0 0.01"
          rgba="0.32 0.32 0.32 1" contype="0" conaffinity="0"/>
    <geom name="mark_2" type="box" size="0.025 1.5 0.01" pos="2 0 0.01"
          rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
    <geom name="mark_3" type="box" size="0.025 1.5 0.01" pos="3 0 0.01"
          rgba="0.32 0.32 0.32 1" contype="0" conaffinity="0"/>
    <geom name="mark_4" type="box" size="0.025 1.5 0.01" pos="4 0 0.01"
          rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
    <geom name="mark_5" type="box" size="0.025 1.5 0.01" pos="5 0 0.01"
          rgba="0.32 0.32 0.32 1" contype="0" conaffinity="0"/>
    <geom name="mark_6" type="box" size="0.025 1.5 0.01" pos="6 0 0.01"
          rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
    <geom name="mark_7" type="box" size="0.025 1.5 0.01" pos="7 0 0.01"
          rgba="0.32 0.32 0.32 1" contype="0" conaffinity="0"/>
    <geom name="mark_8" type="box" size="0.025 1.5 0.01" pos="8 0 0.01"
          rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
    <geom name="mark_9" type="box" size="0.025 1.5 0.01" pos="9 0 0.01"
          rgba="0.32 0.32 0.32 1" contype="0" conaffinity="0"/>
    <geom name="mark_10" type="box" size="0.025 1.5 0.01" pos="10 0 0.01"
          rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
    <geom name="mark_11" type="box" size="0.025 1.5 0.01" pos="11 0 0.01"
          rgba="0.32 0.32 0.32 1" contype="0" conaffinity="0"/>
    <geom name="mark_12" type="box" size="0.025 1.5 0.01" pos="12 0 0.01"
          rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
    <geom name="mark_13" type="box" size="0.025 1.5 0.01" pos="13 0 0.01"
          rgba="0.32 0.32 0.32 1" contype="0" conaffinity="0"/>
    <geom name="mark_14" type="box" size="0.025 1.5 0.01" pos="14 0 0.01"
          rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
    <geom name="mark_15" type="box" size="0.025 1.5 0.01" pos="15 0 0.01"
          rgba="0.32 0.32 0.32 1" contype="0" conaffinity="0"/>
    <geom name="mark_16" type="box" size="0.025 1.5 0.01" pos="16 0 0.01"
          rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
    <geom name="mark_17" type="box" size="0.025 1.5 0.01" pos="17 0 0.01"
          rgba="0.32 0.32 0.32 1" contype="0" conaffinity="0"/>
    <geom name="mark_18" type="box" size="0.025 1.5 0.01" pos="18 0 0.01"
          rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
    <geom name="mark_19" type="box" size="0.025 1.5 0.01" pos="19 0 0.01"
          rgba="0.32 0.32 0.32 1" contype="0" conaffinity="0"/>
    <geom name="finish_line" type="box" size="0.08 2.0 0.03" pos="20 0 0.03"
          rgba="0.95 0.15 0.1 1" contype="0" conaffinity="0"/>
    <body name="torso" pos="0 0 1.25">
      <joint name="rootx" type="slide" axis="1 0 0" limited="false"
             damping="0" armature="0" stiffness="0"/>
      <joint name="rootz" type="slide" axis="0 0 1" limited="false"
             damping="0" armature="0" stiffness="0" ref="1.25"/>
      <joint name="rooty" type="hinge" axis="0 1 0" limited="false"
             damping="0" armature="0" stiffness="0"/>
      <geom name="torso_geom" type="capsule" size="0.05 0.2"
            friction="0.9" rgba="0.2 0.45 0.9 1"/>
      <body name="thigh" pos="0 0 -0.2">
        <joint name="thigh_joint" type="hinge" axis="0 -1 0" range="-150 0"/>
        <geom name="thigh_geom" type="capsule" pos="0 0 -0.225" size="0.05 0.225"
              friction="0.9" rgba="0.25 0.55 0.95 1"/>
        <body name="leg" pos="0 0 -0.7">
          <joint name="leg_joint" type="hinge" axis="0 -1 0" pos="0 0 0.25"
                 range="-150 0"/>
          <geom name="leg_geom" type="capsule" size="0.04 0.25"
                friction="0.9" rgba="0.3 0.6 1.0 1"/>
          <body name="foot" pos="0.13 0 -0.35">
            <joint name="foot_joint" type="hinge" axis="0 -1 0"
                   pos="-0.13 0 0.1" range="-45 45"/>
            <geom name="foot_geom" type="capsule" pos="-0.065 0 0.1"
                  quat="0.7071 0 -0.7071 0" size="0.06 0.195"
                  friction="2.0" rgba="0.15 0.8 0.4 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="thigh_joint" gear="200"/>
    <motor joint="leg_joint" gear="200"/>
    <motor joint="foot_joint" gear="200"/>
  </actuator>
</mujoco>
"""


class Hopper2D(MujocoEnv):
    """Planar one-legged hopper: stay upright and hop forward (+x)."""

    HEALTHY_Z_MIN = 0.7
    HEALTHY_ANGLE = 0.2  # rad (~11°), Gymnasium Hopper-v5 default
    FORWARD_REWARD_WEIGHT = 1.0
    CTRL_COST_WEIGHT = 1e-3
    HEALTHY_REWARD = 1.0

    def __init__(self, max_steps: int = 1000):
        self.max_steps = max_steps
        self._steps = 0
        self._prev_x = 0.0
        super().__init__(model_xml=HOPPER_XML, frame_skip=4)
        self.init_qpos[:] = np.array([0.0, 1.25, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.init_qvel[:] = 0.0
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64
        )

    def _height(self) -> float:
        return float(self.data.qpos[1])

    def _torso_angle(self) -> float:
        return float(self.data.qpos[2])

    def _is_healthy(self) -> bool:
        return (
            self._height() > self.HEALTHY_Z_MIN
            and abs(self._torso_angle()) < self.HEALTHY_ANGLE
        )

    def reset_model(self) -> np.ndarray:
        self._steps = 0
        noise = 5e-3
        qpos = self.init_qpos + self.np_random.uniform(-noise, noise, size=self.init_qpos.shape)
        qvel = self.init_qvel + self.np_random.uniform(-noise, noise, size=self.init_qvel.shape)
        self.set_state(qpos, qvel)
        self._prev_x = float(self.data.qpos[0])
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
        qvel = np.clip(self.data.qvel.astype(np.float64), -10.0, 10.0)
        return np.concatenate([qpos[1:], qvel])

    def _get_reward(self) -> float:
        x = float(self.data.qpos[0])
        dx = x - self._prev_x
        self._prev_x = x
        forward = self.FORWARD_REWARD_WEIGHT * (dx / self.dt)
        ctrl = self.CTRL_COST_WEIGHT * float(np.square(self.data.ctrl).sum())
        healthy = self.HEALTHY_REWARD if self._is_healthy() else 0.0
        return forward + healthy - ctrl

    def _is_truncated(self) -> bool:
        return self._steps >= self.max_steps

    def _is_terminated(self) -> bool:
        return not self._is_healthy()

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=400, width=720)
        if self._camera is None:
            self._camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, self._camera)
            self._camera.elevation = -16.0
            self._camera.distance = 11.0
            self._camera.azimuth = 90.0
        x = float(self.data.qpos[0])
        self._camera.lookat[:] = [x + 6.0, 0.0, 0.9]
        self._renderer.update_scene(self.data, camera=self._camera)
        return np.asarray(self._renderer.render())


def evaluate_hopper_gif(
    agent: ActorCriticAgent,
    cfg: TrainConfig,
    *,
    gif_path: str,
    normalizer: NormMeanStd | None = None,
    n_trials: int = 8,
    frame_duration: float = 0.04,
    frame_stride: int = 3,
) -> dict[str, float | bool | int]:
    """Multi-seed deterministic eval; keep best by (truncated, dx). Record healthy frames only."""
    agent.eval()
    best: dict[str, object] | None = None

    for trial in range(n_trials):
        env = Hopper2D()
        obs, _ = env.reset(seed=1000 + trial)
        x0 = float(env.data.qpos[0])
        frames: list[np.ndarray] = [env.render()]
        steps = 0
        truncated = False
        terminated = False
        while True:
            state = processing_state(
                obs, normalizer, update=False, device=cfg.device
            )
            with torch.inference_mode():
                action = _deterministic_action(agent, state)
            obs, _, terminated, truncated, _ = env.step(
                action.squeeze(0).detach().cpu().numpy()
            )
            steps += 1
            healthy = env._is_healthy()
            if healthy and (steps % frame_stride == 0 or truncated):
                frames.append(env.render())
            if terminated or truncated:
                break
        dx = float(env.data.qpos[0]) - x0
        env.close()

        score = (1 if truncated and not terminated else 0, dx, steps)
        candidate = {
            "frames": frames,
            "dx": dx,
            "steps": steps,
            "truncated": bool(truncated and not terminated),
            "terminated": bool(terminated),
            "score": score,
            "trial": trial,
        }
        if best is None or score > best["score"]:  # type: ignore[operator]
            best = candidate

    assert best is not None
    frames = best["frames"]  # type: ignore[assignment]
    out0 = f"{gif_path}_0.gif"
    out1 = f"{gif_path}_1.gif"
    if frames:
        durations = [frame_duration] * len(frames)
        durations[-1] = min(0.5, frame_duration * 8)
        imageio.mimsave(out0, frames, duration=durations)
        imageio.mimsave(out1, frames, duration=durations)

    summary = {
        "dx": float(best["dx"]),  # type: ignore[arg-type]
        "steps": int(best["steps"]),  # type: ignore[arg-type]
        "truncated": bool(best["truncated"]),
        "terminated": bool(best["terminated"]),
        "trial": int(best["trial"]),  # type: ignore[arg-type]
        "n_frames": len(frames),
    }
    print(
        "hopper eval GIF: "
        f"dx={summary['dx']:.2f} m steps={summary['steps']} "
        f"truncated={summary['truncated']} terminated={summary['terminated']} "
        f"frames={summary['n_frames']} trial={summary['trial']}"
    )
    return summary


def run_immediate(
    *,
    device: str = "cpu",
    timestamp: int = 5_000_000,
    num_envs: int = 16,
    rollout_steps: int = 512,
    seed: int = 42,
    use_wandb: bool = False,
    use_tb: bool = False,
    save_gif: bool = True,
    gif_path: str = "hopper_immediate",
    hidden_layer: int = 256,
    resume_path: str | None = None,
) -> ActorCriticAgent:
    """Immediate-mode PPO loop on Hopper2D. Returns the trained agent."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but torch.cuda.is_available() is False")

    cfg = TrainConfig(
        model_name=f"Hopper2D-{device}",
        project_name="hopper_mujoco_immediate",
        timestamp=timestamp,
        rollout_steps=rollout_steps,
        num_envs=num_envs,
        normalize=True,
        profile=True,
    )
    cfg.device = torch.device(device)
    algo_cfg = AlgoConfig(
        lr=3e-4,
        ent_coef=0.0,
        epochs=10,
        batch_size=512,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
    )

    seeds = set_seed(seed, cfg.num_envs)
    env = vectorize_env(Hopper2D, num_envs=cfg.num_envs)
    obs_dim, act_dim, obs_n, act_n, is_discrete = get_obs_act(env)
    agent = ActorCriticAgent(obs_n, act_n, is_discrete, hidden_layer).to(cfg.device)
    normalizer = NormMeanStd((int(obs_n),), device=cfg.device)

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=cfg.device, weights_only=False)
        agent.load_state_dict(ckpt["agent_state_dict"])
        if ckpt.get("normalizer_state_dict") is not None:
            normalizer.load_state_dict(ckpt["normalizer_state_dict"])
    else:
        torch.nn.init.constant_(agent.log_std, -0.5)

    buffer = Buffer(
        capacity=cfg.rollout_steps,
        num_envs=cfg.num_envs,
        schema={
            "state": obs_dim,
            "action": act_dim,
            "reward": (),
            "terminated": (),
            "truncated": (),
            "entropy": (),
            "value": (),
            "return": (),
            "log_prob": (),
            "advantage": (),
        },
        device=cfg.device,
    )
    optimizer = optim.Adam(agent.parameters(), lr=algo_cfg.lr, eps=1e-5)
    start_factor = 0.15 if resume_path is not None else 1.0
    scheduler = LambdaLR(
        optimizer,
        lambda step_: start_factor * (1.0 - (step_ / max(cfg.num_update, 1))),
    )
    log = create_logger(cfg, algo_cfg, use_wandb=use_wandb, use_tb=use_tb)

    reward_tensor = torch.zeros(cfg.num_envs, device=cfg.device)
    state, _ = env.reset(seed=seeds)

    for step in tqdm(range(cfg.num_update), desc="hopper-immediate"):
        episodic_reward: list[float] = []
        for _ in range(cfg.rollout_steps):
            state_processed = processing_state(
                state, normalizer, update=True, device=cfg.device
            )
            with torch.inference_mode():
                outputs = agent.get_action(state_processed)
            action = to_env_action(outputs["action"], env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            outputs_final = {
                "next_state": next_state,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                **outputs,
            }
            outputs_final = parse_dict_to_tensor(outputs_final, device=cfg.device)
            next_state_t = outputs_final.pop("next_state")
            buffer.insert(state=state_processed, **outputs_final)
            reward_tensor += outputs_final["reward"]
            episode_done = (outputs_final["terminated"] > 0) | (
                outputs_final["truncated"] > 0
            )
            if episode_done.any():
                episodic_reward.extend(reward_tensor[episode_done].tolist())
                reward_tensor[episode_done] = 0.0
            state = next_state_t

        with torch.inference_mode():
            last_output = agent.get_action(
                processing_state(state, normalizer, update=False, device=cfg.device)
            )

        data = buffer.get_all()
        gae_compute(
            data["reward"],
            data["value"],
            last_output["value"],
            data["terminated"],
            buffer,
            algo_cfg,
        )
        losses = ppo_func(agent, optimizer, buffer, algo_cfg, scheduler)

        mean_reward = float(np.mean(episodic_reward[-10:])) if episodic_reward else 0.0
        metrics = {"train/mean_episode_reward": mean_reward}
        for k, v in losses.items():
            metrics[f"train/{k}"] = v
        log(metrics, step)
        buffer.clear()

    env.close()
    log.close()

    save_checkpoints(agent, cfg.model_path, normalizer=normalizer)

    if save_gif:
        evaluate_hopper_gif(
            agent, cfg, gif_path=gif_path, normalizer=normalizer, n_trials=10
        )
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Hopper2D MujocoEnv + immediate-mode PPO")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--timestamp", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--tb", action="store_true", default=True)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to continue training.",
    )
    args = parser.parse_args()

    run_immediate(
        device=args.device,
        timestamp=args.timestamp,
        num_envs=args.num_envs,
        seed=args.seed,
        use_wandb=args.wandb,
        use_tb=args.tb,
        save_gif=not args.no_gif,
        gif_path=f"hopper_immediate_{args.device}",
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
