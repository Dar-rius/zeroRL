"""MuJoCo physics base for custom robotics environments."""

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from torch import Tensor

from zerorl.helpers.env import BaseEnv


class MujocoEnv(BaseEnv):
    """BaseEnv backed by MuJoCo ``MjModel`` / ``MjData``.

    Provide ``model_path`` or ``model_xml``. Override ``_get_obs``,
    ``_get_reward``, ``_is_terminated``, and ``_is_truncated`` for the task.
    One instance = one simulation; parallelize with ``vectorize_env``.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        model_xml: str | None = None,
        frame_skip: int = 5,
    ):
        super().__init__()
        if (model_path is None) == (model_xml is None):
            raise ValueError("Provide exactly one of model_path or model_xml")
        if frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")

        if model_xml is not None:
            self.model = mujoco.MjModel.from_xml_string(model_xml)
        else:
            path = Path(model_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"MuJoCo model not found: {path}")
            self.model = mujoco.MjModel.from_xml_path(str(path.resolve()))

        self.data = mujoco.MjData(self.model)
        self.frame_skip = int(frame_skip)
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self._renderer: mujoco.Renderer | None = None
        self._camera: mujoco.MjvCamera | None = None

        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)
        obs = np.asarray(self._get_obs(), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float64,
        )

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep * self.frame_skip)

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(self.model.nq)
        qvel = np.asarray(qvel, dtype=np.float64).reshape(self.model.nv)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def do_simulation(self, ctrl: np.ndarray) -> None:
        ctrl = np.asarray(ctrl, dtype=np.float64)
        if ctrl.shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, found {ctrl.shape}"
            )
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        return self.reset_model(), self._get_info()

    def step(
        self, action: np.ndarray | Tensor
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if isinstance(action, Tensor):
            action = action.detach().cpu().numpy()
        self.do_simulation(np.asarray(action, dtype=np.float64))
        obs = np.asarray(self._get_obs(), dtype=np.float64)
        return (
            obs,
            float(self._get_reward()),
            bool(self._is_terminated()),
            bool(self._is_truncated()),
            self._get_info(),
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._camera = None

    def render(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 480)
            self._camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, self._camera)
            self._camera.elevation = -45.0
            self._camera.distance = max(2.5, float(self.model.stat.extent) * 1.5)
        assert self._camera is not None
        if self.model.nbody > 1:
            self._camera.lookat[:] = self.data.xpos[1]
        self._renderer.update_scene(self.data, camera=self._camera)
        return np.asarray(self._renderer.render())

    def reset_model(self) -> np.ndarray:
        self.set_state(self.init_qpos, self.init_qvel)
        return np.asarray(self._get_obs(), dtype=np.float64)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def _get_reward(self) -> float:
        return 0.0

    def _is_terminated(self) -> bool:
        return False

    def _is_truncated(self) -> bool:
        return False

    def _get_info(self) -> dict[str, Any]:
        return {}
