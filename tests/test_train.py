"""Unit tests for BaseTrain (zerorl.train)."""

import math
import pytest
import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from dataclasses import asdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch, MagicMock
from gymnasium import spaces
from zerorl.agent import BaseAgent, eval_action
from zerorl.common import Buffer
from zerorl.config import AlgoConfig, TrainConfig
from zerorl.errors import EmptyBufferError
from zerorl.env import BaseEnv
from zerorl.train import BaseTrain, ProfileMetrics
from torch.optim.lr_scheduler import LambdaLR


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MockAgent(BaseAgent):
    """Minimal agent for testing BaseTrain."""
    def __init__(self, obs_dim: int = 4, act_dim: int = 2) -> None:
        super().__init__()
        self.actor = nn.Linear(obs_dim, act_dim)
        self.val = nn.Linear(obs_dim, 1)

    def forward(self, state: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(state)
        val = self.val(state)
        return logits, val

    @staticmethod
    def build_distribution(logits: torch.Tensor) -> torch.distributions.Distribution:
        return torch.distributions.Categorical(logits=logits)

    def get_action(self, state: torch.Tensor, action: torch.Tensor | None = None, **kwargs):
        logits, value = self.forward(state, **kwargs)
        dist = self.build_distribution(logits)
        if action is None: action = dist.sample()
        log_prob, dist_entropy = eval_action(dist, action)
        return {"action": action, "log_prob": log_prob, "entropy":dist_entropy, "value":value}

class CartPoleEnvWrapper(BaseEnv):
    """Real Gymnasium environment wrapper for integration testing."""
    def __init__(self):
        super().__init__()
        self._env = gym.make("CartPole-v1")
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        # BaseTrain batches obs → action shape (1,); CartPole needs a scalar
        if hasattr(action, "shape") and len(action.shape) > 0:
            action = action.item()
        return self._env.step(action)

    def close(self):
        self._env.close()

def _mock_update_weights(
    agent: BaseAgent,
    buffer: Buffer,
    scheduler: LambdaLR,
    optimizer: torch.optim.Optimizer,
    step: int,
    last_output: dict[str, torch.Tensor],
    algo_config: AlgoConfig | None,
) -> dict[str, torch.Tensor]:
    """Mock update_weights callable for testing."""
    return {"loss": torch.tensor(0.0, device=next(agent.parameters()).device)}

def _make_train_config(tmp_path: Path, device: torch.device) -> TrainConfig:
    """Create TrainConfig with specific device."""
    cfg = TrainConfig(model_name="test", model_save_path=str(tmp_path))
    cfg.device = device
    return cfg

class TestBaseTrainInit:
    @pytest.mark.gpu
    def test_stores_attributes(self, tmp_path: Path, device: torch.device) -> None:
        obs_dim, act_dim = 4, 2
        agent = MockAgent(obs_dim, act_dim)
        # ON UTILISE LE VRAI ENV ICI
        env = CartPoleEnvWrapper()
        buf = Buffer(step=10, data={"state": (obs_dim,)}, device=device)
        cfg = _make_train_config(tmp_path, device)
        algo_cfg = AlgoConfig()

        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, algo_cfg)
        assert trainer.agent is agent
        assert trainer.env is env
        assert trainer.buffer is buf
        assert trainer.train_config is cfg
        assert trainer.algo_config is algo_cfg
        env.close()


class TestBaseTrainRollout:
    @pytest.mark.gpu
    def test_rollout_phase_fills_buffer(self, tmp_path: Path, device: torch.device) -> None:
        obs_dim, act_dim = 4, 2
        rollout_steps = 8
        agent = MockAgent(obs_dim, act_dim)
        env = CartPoleEnvWrapper()
        buf = Buffer(
            step=rollout_steps,
            data={
                "state": (obs_dim,),
                "reward": (),
                "done": (),
                "action": (),
                "log_prob": (),
                "entropy": (),
                "value": (),
            },
            device=device,
        )
        cfg = _make_train_config(tmp_path, device)
        cfg.rollout_steps = rollout_steps
        algo_cfg = AlgoConfig()
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, algo_cfg)

        state, _ = env.reset(seed=42)
        last_output = trainer.rollout_phase(state)

        assert buf.size == rollout_steps
        assert "action" in last_output
        assert "log_prob" in last_output
        assert "entropy" in last_output
        assert "value" in last_output
        assert last_output["value"].shape[0] == 1
        env.close()


class FakeVecEnv(BaseEnv):
    """Deterministic vectorized env for BaseTrain tests.

    Per-env step counters drive termination so different envs can finish at
    different times. observation_space describes a single-env observation
    (obs_dim,) so NormMeanStd works with batched (num_envs, obs_dim) data.
    """
    def __init__(self, num_envs: int = 2, obs_dim: int = 4, act_dim: int = 2,
                 steps_until_done: tuple[int, ...] = (3,), auto_reset: bool = False):
        super().__init__()
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.steps_until_done = list(steps_until_done)
        self._auto_reset = auto_reset
        self.counters = [0] * num_envs
        self.reset_calls = 0
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(act_dim)

    @property
    def device(self): return torch.device("cpu")

    @property
    def auto_reset(self): return self._auto_reset

    def reset(self, *, seed=None, options=None):
        self.reset_calls += 1
        self.counters = [0] * self.num_envs
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        return obs, {}

    def step(self, action):
        terminated = np.zeros(self.num_envs, dtype=bool)
        for i in range(self.num_envs):
            self.counters[i] += 1
            if self.counters[i] >= self.steps_until_done[i]:
                terminated[i] = True
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        for i in range(self.num_envs):
            obs[i, 0] = self.counters[i] if not terminated[i] else 0.0
        if self._auto_reset:
            for i in range(self.num_envs):
                if terminated[i]:
                    self.counters[i] = 0
        reward = np.ones(self.num_envs, dtype=np.float32)
        truncated = np.zeros(self.num_envs, dtype=bool)
        return obs, reward, terminated, truncated, {}

    def close(self): pass


def _make_counting_update_weights(counter: list[int]):
    def _update(agent, buffer, scheduler, optimizer, step, last_output, algo_config):
        counter.append(step)
        return {"loss": torch.tensor(0.0, device=next(agent.parameters()).device)}
    return _update


class TestBaseTrainTrain:
    @pytest.mark.gpu
    def test_train_runs_full_loop(self, tmp_path: Path, device: torch.device) -> None:
        obs_dim, act_dim = 4, 2
        rollout_steps = 16
        agent = MockAgent(obs_dim, act_dim)
        env = FakeVecEnv(num_envs=1, obs_dim=obs_dim, act_dim=act_dim, steps_until_done=(100,))
        buf = Buffer(step=rollout_steps, num_envs=1, data={
            "state": (obs_dim,), "reward": (), "done": (),
            "action": (), "log_prob": (), "entropy": (), "value": (),
        }, device=device)
        cfg = _make_train_config(tmp_path, device)
        cfg.rollout_steps = rollout_steps
        cfg.timestamp = rollout_steps * 3
        cfg.num_envs = 1
        cfg.num_update = 3
        trainer = BaseTrain(agent, env, buf, _make_counting_update_weights([]),
                            cfg, AlgoConfig(), require_buffer_size=4)
        counter: list[int] = []
        trainer.update_weights = _make_counting_update_weights(counter)
        trainer.train(use_wandb=False, use_tb=False)
        assert counter == [0, 1, 2]
        env.close()

    @pytest.mark.gpu
    def test_train_raises_on_empty_buffer(self, tmp_path: Path, device: torch.device) -> None:
        obs_dim, act_dim = 4, 2
        rollout_steps = 8
        agent = MockAgent(obs_dim, act_dim)
        env = FakeVecEnv(num_envs=1, obs_dim=obs_dim, act_dim=act_dim, steps_until_done=(100,))
        buf = Buffer(step=rollout_steps, num_envs=1, data={
            "state": (obs_dim,), "reward": (), "done": (),
            "action": (), "log_prob": (), "entropy": (), "value": (),
        }, device=device)
        cfg = _make_train_config(tmp_path, device)
        cfg.rollout_steps = rollout_steps
        cfg.timestamp = rollout_steps * 2
        cfg.num_envs = 1
        trainer = BaseTrain(agent, env, buf, _make_counting_update_weights([]),
                            cfg, AlgoConfig(), require_buffer_size=100)
        counter: list[int] = []
        trainer.update_weights = _make_counting_update_weights(counter)
        with pytest.raises(EmptyBufferError):
            trainer.train(use_wandb=False, use_tb=False)
        env.close()

    @pytest.mark.gpu
    def test_train_logs_to_wandb(self, tmp_path: Path, device: torch.device) -> None:
        obs_dim, act_dim = 4, 2
        rollout_steps = 16
        agent = MockAgent(obs_dim, act_dim)
        env = FakeVecEnv(num_envs=1, obs_dim=obs_dim, act_dim=act_dim, steps_until_done=(100,))
        buf = Buffer(step=rollout_steps, num_envs=1, data={
            "state": (obs_dim,), "reward": (), "done": (),
            "action": (), "log_prob": (), "entropy": (), "value": (),
        }, device=device)
        cfg = _make_train_config(tmp_path, device)
        cfg.rollout_steps = rollout_steps
        cfg.timestamp = rollout_steps
        cfg.num_envs = 1
        trainer = BaseTrain(agent, env, buf, _make_counting_update_weights([]),
                            cfg, AlgoConfig(), require_buffer_size=4)
        with patch("wandb.log") as mock_log:
            trainer.train(use_wandb=True, use_tb=False)
        assert mock_log.called
        logged = mock_log.call_args.args[0]
        assert "loss" in logged and "mean_episode_reward" in logged and "learning_rate" in logged
        env.close()

    @pytest.mark.gpu
    def test_train_logs_to_tensorboard(self, tmp_path: Path, device: torch.device) -> None:
        obs_dim, act_dim = 4, 2
        rollout_steps = 16
        agent = MockAgent(obs_dim, act_dim)
        env = FakeVecEnv(num_envs=1, obs_dim=obs_dim, act_dim=act_dim, steps_until_done=(100,))
        buf = Buffer(step=rollout_steps, num_envs=1, data={
            "state": (obs_dim,), "reward": (), "done": (),
            "action": (), "log_prob": (), "entropy": (), "value": (),
        }, device=device)
        cfg = _make_train_config(tmp_path, device)
        cfg.rollout_steps = rollout_steps
        cfg.timestamp = rollout_steps
        cfg.num_envs = 1
        trainer = BaseTrain(agent, env, buf, _make_counting_update_weights([]),
                            cfg, AlgoConfig(), require_buffer_size=4)
        trainer.tb_writer = MagicMock()
        trainer.train(use_wandb=False, use_tb=True)
        assert trainer.tb_writer.add_scalar.called
        keys_logged = {call.args[0] for call in trainer.tb_writer.add_scalar.call_args_list}
        assert "loss" in keys_logged
        env.close()


class TestBaseTrainSaveModel:
    @pytest.mark.gpu
    def test_save_model_creates_file(self, tmp_path: Path, device: torch.device) -> None:
        obs_dim, act_dim = 4, 2
        agent = MockAgent(obs_dim, act_dim)
        env = FakeVecEnv(num_envs=1, obs_dim=obs_dim, act_dim=act_dim)
        buf = Buffer(step=8, data={"state": (obs_dim,)}, device=device)
        cfg = _make_train_config(tmp_path, device)
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, AlgoConfig())
        trainer.save_model()
        assert os.path.exists(cfg.model_path)
        loaded = torch.load(cfg.model_path)
        assert "actor.weight" in loaded
        env.close()

    @pytest.mark.gpu
    def test_save_model_creates_nested_dirs(self, tmp_path: Path, device: torch.device) -> None:
        obs_dim, act_dim = 4, 2
        agent = MockAgent(obs_dim, act_dim)
        env = FakeVecEnv(num_envs=1, obs_dim=obs_dim, act_dim=act_dim)
        buf = Buffer(step=8, data={"state": (obs_dim,)}, device=device)
        nested = tmp_path / "nested" / "sub" / "dir"
        cfg = TrainConfig(model_name="m", model_save_path=str(nested))
        cfg.device = device
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, AlgoConfig())
        trainer.save_model()
        assert os.path.exists(cfg.model_path)
        env.close()


class TestBaseTrainLogMetrics:
    @pytest.mark.gpu
    def test_log_metrics_handles_tensors(self, tmp_path: Path, device: torch.device) -> None:
        agent = MockAgent()
        env = FakeVecEnv(num_envs=1, obs_dim=4, act_dim=2)
        buf = Buffer(step=8, data={"state": (4,)}, device=device)
        cfg = _make_train_config(tmp_path, device)
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, AlgoConfig())
        trainer.tb_writer = MagicMock()
        metrics = {"x": torch.tensor(1.5, device=device), "y": 2.5}
        with patch("wandb.log") as mock_log:
            trainer._log_metrics(metrics, step=0, use_wandb=True, use_tb=True)
        logged = mock_log.call_args.args[0]
        assert logged["x"] == 1.5 and isinstance(logged["x"], float)
        assert logged["y"] == 2.5
        env.close()

    @pytest.mark.gpu
    def test_log_metrics_handles_floats(self, tmp_path: Path, device: torch.device) -> None:
        agent = MockAgent()
        env = FakeVecEnv(num_envs=1, obs_dim=4, act_dim=2)
        buf = Buffer(step=8, data={"state": (4,)}, device=device)
        cfg = _make_train_config(tmp_path, device)
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, AlgoConfig())
        trainer.tb_writer = MagicMock()
        metrics = {"y": 1.5}
        with patch("wandb.log") as mock_log:
            trainer._log_metrics(metrics, step=0, use_wandb=True, use_tb=True)
        assert mock_log.call_args.args[0] == {"y": 1.5}
        env.close()


class TestBaseTrainVectorizedRollout:
    @pytest.mark.gpu
    def test_rollout_phase_vectorized(self, tmp_path: Path, device: torch.device) -> None:
        num_envs, obs_dim, act_dim = 2, 4, 2
        rollout_steps = 8
        agent = MockAgent(obs_dim, act_dim)
        env = FakeVecEnv(num_envs=num_envs, obs_dim=obs_dim, act_dim=act_dim, steps_until_done=(100, 100))
        buf = Buffer(step=rollout_steps, num_envs=num_envs, data={
            "state": (obs_dim,), "reward": (), "done": (),
            "action": (), "log_prob": (), "entropy": (), "value": (),
        }, device=device)
        cfg = _make_train_config(tmp_path, device)
        cfg.rollout_steps = rollout_steps
        cfg.num_envs = num_envs
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, AlgoConfig())
        state, _ = env.reset(seed=42)
        trainer.rollout_phase(state)
        assert buf.size == rollout_steps
        assert buf.get_all()["state"].shape == (rollout_steps, num_envs, obs_dim)
        assert trainer.current_episode_reward is not None
        assert trainer.current_episode_reward.shape == (num_envs,)
        env.close()

    @pytest.mark.gpu
    def test_rollout_phase_auto_reset(self, tmp_path: Path, device: torch.device) -> None:
        num_envs, obs_dim, act_dim = 2, 4, 2
        rollout_steps = 8
        agent = MockAgent(obs_dim, act_dim)
        env = FakeVecEnv(num_envs=num_envs, obs_dim=obs_dim, act_dim=act_dim,
                         steps_until_done=(3, 3), auto_reset=True)
        buf = Buffer(step=rollout_steps, num_envs=num_envs, data={
            "state": (obs_dim,), "reward": (), "done": (),
            "action": (), "log_prob": (), "entropy": (), "value": (),
        }, device=device)
        cfg = _make_train_config(tmp_path, device)
        cfg.rollout_steps = rollout_steps
        cfg.num_envs = num_envs
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, AlgoConfig())
        state, _ = env.reset(seed=42)
        trainer.rollout_phase(state)
        assert env.reset_calls == 1
        env.close()

    @pytest.mark.gpu
    @pytest.mark.xfail(reason="vectorized reset bug: train.py resets all envs when any finishes (train.py:141-145)")
    def test_rollout_phase_partial_finish_preserves_survivors(self, tmp_path: Path, device: torch.device) -> None:
        num_envs, obs_dim, act_dim = 2, 4, 2
        rollout_steps = 5
        agent = MockAgent(obs_dim, act_dim)
        env = FakeVecEnv(num_envs=num_envs, obs_dim=obs_dim, act_dim=act_dim,
                         steps_until_done=(3, 100), auto_reset=False)
        buf = Buffer(step=rollout_steps, num_envs=num_envs, data={
            "state": (obs_dim,), "reward": (), "done": (),
            "action": (), "log_prob": (), "entropy": (), "value": (),
        }, device=device)
        cfg = _make_train_config(tmp_path, device)
        cfg.rollout_steps = rollout_steps
        cfg.num_envs = num_envs
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, AlgoConfig())
        state, _ = env.reset(seed=42)
        trainer.rollout_phase(state)
        assert env.counters[1] == 5
        env.close()


# ---------------------------------------------------------------------------
# Helpers for profiler tests
# ---------------------------------------------------------------------------

def _make_profile_config(tmp_path: Path, device: torch.device,
                          profile: bool = True, rollout_steps: int = 8,
                          num_envs: int = 1, num_steps: int = 3) -> TrainConfig:
    cfg = TrainConfig(model_name="profile_test", model_save_path=str(tmp_path))
    cfg.device = device
    cfg.rollout_steps = rollout_steps
    cfg.num_envs = num_envs
    cfg.timestamp = rollout_steps * num_envs * num_steps
    cfg.num_update = num_steps
    cfg.profile = profile
    return cfg


def _capture_profile_metrics(trainer: BaseTrain) -> list[ProfileMetrics]:
    captured: list[ProfileMetrics] = []
    patch.object(trainer, "_log_profile_metrics",
                 side_effect=lambda s, m: captured.append(m)).start()
    return captured


def _cpu_as_tensor_ctx():
    """Patch tensor allocators to coerce the `device` kwarg to CPU.

    Lets us exercise the `is_cuda=True` profiler branch on a CPU-only
    machine: train_config.device is set to cuda (so `is_cuda` is True),
    but all tensor allocations stay on CPU. Patches as_tensor, zeros,
    and ones (rollout_phase uses torch.zeros for current_episode_reward).
    """
    real_as_tensor = torch.as_tensor
    real_zeros = torch.zeros
    real_ones = torch.ones

    def _cpu(dev):
        return torch.device("cpu") if str(dev).startswith("cuda") else dev

    def _as_tensor(data, *args, **kwargs):
        if kwargs.get("device") is not None:
            kwargs["device"] = _cpu(kwargs["device"])
        return real_as_tensor(data, *args, **kwargs)

    def _zeros(*args, **kwargs):
        if kwargs.get("device") is not None:
            kwargs["device"] = _cpu(kwargs["device"])
        return real_zeros(*args, **kwargs)

    def _ones(*args, **kwargs):
        if kwargs.get("device") is not None:
            kwargs["device"] = _cpu(kwargs["device"])
        return real_ones(*args, **kwargs)

    return [patch("torch.as_tensor", side_effect=_as_tensor),
            patch("torch.zeros", side_effect=_zeros),
            patch("torch.ones", side_effect=_ones)]


# ---------------------------------------------------------------------------
# TestProfileMetrics — dataclass shape
# ---------------------------------------------------------------------------

class TestProfileMetrics:
    @pytest.mark.gpu
    def test_profile_metrics_fields_and_asdict(self) -> None:
        pm = ProfileMetrics(
            fps=80.0, rollout_ms=100.0, update_ms=50.0,
            vram_allocated_gb=1.5, vram_peak_gb=2.0, ram_mb=256.0,
        )
        assert isinstance(pm.fps, float) and pm.fps == 80.0
        assert isinstance(pm.rollout_ms, float) and pm.rollout_ms == 100.0
        assert isinstance(pm.update_ms, float) and pm.update_ms == 50.0
        assert isinstance(pm.vram_allocated_gb, float) and pm.vram_allocated_gb == 1.5
        assert isinstance(pm.vram_peak_gb, float) and pm.vram_peak_gb == 2.0
        assert isinstance(pm.ram_mb, float) and pm.ram_mb == 256.0
        d = asdict(pm)
        assert set(d.keys()) == {"fps", "rollout_ms", "update_ms",
                                 "vram_allocated_gb", "vram_peak_gb", "ram_mb"}
        assert d["fps"] == 80.0 and d["rollout_ms"] == 100.0


# ---------------------------------------------------------------------------
# TestBaseTrainProfilerLogMetrics — _log_profile_metrics formatter
# ---------------------------------------------------------------------------

class TestBaseTrainProfilerLogMetrics:
    @pytest.mark.gpu
    def test_log_profile_metrics_format(self, tmp_path: Path,
                                        device: torch.device) -> None:
        agent = MockAgent()
        env = FakeVecEnv(num_envs=1, obs_dim=4, act_dim=2)
        buf = Buffer(step=8, data={"state": (4,)}, device=device)
        cfg = _make_train_config(tmp_path, device)
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, AlgoConfig())
        pm = ProfileMetrics(fps=42.0, rollout_ms=10.5, update_ms=7.3,
                            vram_allocated_gb=0.5, vram_peak_gb=1.0, ram_mb=128.0)
        written: list[str] = []
        with patch("sys.stderr.write", side_effect=lambda s: written.append(s)):
            trainer._log_profile_metrics(5, pm)
        blob = "".join(written)
        assert "\033[94m" in blob and "\033[0m" in blob
        assert "[Profile] Step 5" in blob
        assert "FPS: 42" in blob
        assert "Rollout: 10.5ms" in blob
        assert "Update:" in blob and "ms" in blob
        assert "VRAM:" in blob and "Peak:" in blob
        assert "RAM:" in blob
        env.close()


# ---------------------------------------------------------------------------
# TestBaseTrainProfilerTrain — train() profiler behavior
# ---------------------------------------------------------------------------

class TestBaseTrainProfilerTrain:
    @pytest.mark.gpu
    def test_train_profile_disabled_emits_nothing(self, tmp_path: Path,
                                                  device: torch.device) -> None:
        agent = MockAgent()
        env = FakeVecEnv(num_envs=1, obs_dim=4, act_dim=2, steps_until_done=(100,))
        buf = Buffer(step=8, num_envs=1, data={
            "state": (4,), "reward": (), "done": (),
            "action": (), "log_prob": (), "entropy": (), "value": (),
        }, device=device)
        cfg = _make_profile_config(tmp_path, device, profile=False,
                                   rollout_steps=8, num_envs=1, num_steps=3)
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, AlgoConfig(),
                            require_buffer_size=4)
        written: list[str] = []
        with patch("sys.stderr.write", side_effect=lambda s: written.append(s)), \
             patch("torch.cuda.reset_peak_memory_stats") as mock_reset:
            trainer.train(use_wandb=False, use_tb=False)
        blob = "".join(written)
        assert "[Profile]" not in blob
        assert "ZeroRL Profiler Enabled" not in blob
        assert mock_reset.call_count == 0
        env.close()

    @pytest.mark.gpu
    def test_train_profile_banner_emitted_once(self, tmp_path: Path,
                                               device: torch.device) -> None:
        agent = MockAgent()
        env = FakeVecEnv(num_envs=1, obs_dim=4, act_dim=2, steps_until_done=(100,))
        buf = Buffer(step=8, num_envs=1, data={
            "state": (4,), "reward": (), "done": (),
            "action": (), "log_prob": (), "entropy": (), "value": (),
        }, device=device)
        cfg = _make_profile_config(tmp_path, device, profile=True, num_steps=3)
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, AlgoConfig(),
                            require_buffer_size=4)
        written: list[str] = []
        with patch("sys.stderr.write", side_effect=lambda s: written.append(s)):
            trainer.train(use_wandb=False, use_tb=False)
        banner_count = sum(1 for w in written if "ZeroRL Profiler Enabled" in w)
        assert banner_count == 1
        env.close()

    @pytest.mark.gpu
    def test_train_profile_emits_per_step_on_cpu(self, tmp_path: Path,
                                                 device: torch.device) -> None:
        agent = MockAgent()
        env = FakeVecEnv(num_envs=1, obs_dim=4, act_dim=2, steps_until_done=(100,))
        buf = Buffer(step=8, num_envs=1, data={
            "state": (4,), "reward": (), "done": (),
            "action": (), "log_prob": (), "entropy": (), "value": (),
        }, device=device)
        cfg = _make_profile_config(tmp_path, device, profile=True, num_steps=3)
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, AlgoConfig(),
                            require_buffer_size=4)
        written: list[str] = []
        with patch("sys.stderr.write", side_effect=lambda s: written.append(s)), \
             patch("torch.cuda.synchronize") as mock_sync, \
             patch("torch.cuda.reset_peak_memory_stats") as mock_reset:
            trainer.train(use_wandb=False, use_tb=False)
        profile_lines = [w for w in written if "[Profile] Step" in w]
        assert len(profile_lines) == 3
        blob = "".join(written)
        assert "VRAM: 0.00GB" in blob
        assert "Peak: 0.00GB" in blob
        assert mock_sync.call_count == 0
        assert mock_reset.call_count == 0
        env.close()

    @pytest.mark.gpu
    def test_train_profile_cuda_reset_and_sync_calls(self, tmp_path: Path,
                                                     device: torch.device) -> None:
        agent = MockAgent()
        env = FakeVecEnv(num_envs=1, obs_dim=4, act_dim=2, steps_until_done=(100,))
        buf = Buffer(step=8, num_envs=1, data={
            "state": (4,), "reward": (), "done": (),
            "action": (), "log_prob": (), "entropy": (), "value": (),
        }, device=device)
        # Construct on CPU so agent.to(cpu) works; flip device to cuda after
        # so is_cuda=True exercises the cuda branch without real CUDA.
        cfg = _make_profile_config(tmp_path, device, profile=True, num_steps=3)
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, AlgoConfig(),
                            require_buffer_size=4)
        trainer.train_config.device = torch.device("cuda")
        # Stub the normalizer: its real `normalize` is `@torch.compile`-decorated
        # and dynamo's compile-time probe of triton's CUDA capability would
        # call `torch.cuda.current_device()` -> CUDA init -> RuntimeError on
        # CPU-only. We don't care about normalization in profiler tests.
        trainer.normalizer.update = lambda x: None
        trainer.normalizer.normalize = lambda x: x
        with ExitStack() as stack:
            stack.enter_context(patch("torch.cuda.is_available", return_value=True))
            mock_sync = stack.enter_context(patch("torch.cuda.synchronize"))
            mock_reset = stack.enter_context(patch("torch.cuda.reset_peak_memory_stats"))
            stack.enter_context(patch("torch.cuda.memory_allocated", return_value=0))
            stack.enter_context(patch("torch.cuda.max_memory_allocated", return_value=0))
            for p in _cpu_as_tensor_ctx():
                stack.enter_context(p)
            trainer.train(use_wandb=False, use_tb=False)
        # 3 syncs/step (start, after rollout, after update) * 3 steps = 9
        assert mock_sync.call_count == 9
        # 1 reset_peak/step * 3 steps = 3
        assert mock_reset.call_count == 3
        env.close()

    @pytest.mark.gpu
    def test_train_profile_cuda_vram_uses_memory_allocated(self, tmp_path: Path,
                                                           device: torch.device) -> None:
        agent = MockAgent()
        env = FakeVecEnv(num_envs=1, obs_dim=4, act_dim=2, steps_until_done=(100,))
        buf = Buffer(step=8, num_envs=1, data={
            "state": (4,), "reward": (), "done": (),
            "action": (), "log_prob": (), "entropy": (), "value": (),
        }, device=device)
        cfg = _make_profile_config(tmp_path, device, profile=True, num_steps=1)
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, AlgoConfig(),
                            require_buffer_size=4)
        trainer.train_config.device = torch.device("cuda")
        trainer.normalizer.update = lambda x: None
        trainer.normalizer.normalize = lambda x: x
        written: list[str] = []
        with ExitStack() as stack:
            stack.enter_context(patch("torch.cuda.is_available", return_value=True))
            stack.enter_context(patch("torch.cuda.synchronize"))
            stack.enter_context(patch("torch.cuda.reset_peak_memory_stats"))
            stack.enter_context(patch("torch.cuda.memory_allocated", return_value=1024 ** 3))
            stack.enter_context(patch("torch.cuda.max_memory_allocated", return_value=2 * 1024 ** 3))
            stack.enter_context(patch("sys.stderr.write", side_effect=lambda s: written.append(s)))
            for p in _cpu_as_tensor_ctx():
                stack.enter_context(p)
            trainer.train(use_wandb=False, use_tb=False)
        blob = "".join(written)
        assert "VRAM: 1.00GB (Peak: 2.00GB)" in blob
        env.close()

    @pytest.mark.gpu
    def test_train_profile_timing_with_mocked_clock(self, tmp_path: Path,
                                                    device: torch.device) -> None:
        agent = MockAgent()
        env = FakeVecEnv(num_envs=2, obs_dim=4, act_dim=2, steps_until_done=(100, 100))
        buf = Buffer(step=8, num_envs=2, data={
            "state": (4,), "reward": (), "done": (),
            "action": (), "log_prob": (), "entropy": (), "value": (),
        }, device=device)
        cfg = _make_profile_config(tmp_path, device, profile=True,
                                   rollout_steps=8, num_envs=2, num_steps=1)
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, AlgoConfig(),
                            require_buffer_size=4)
        captured = _capture_profile_metrics(trainer)
        # Replace `time` in zerorl.train's namespace so only train.py's
        # perf_counter calls are controlled; tqdm (which has its own
        # `import time`) is unaffected. Returns 0.0, 0.1, 0.2 for the
        # three train.py calls (t_start, t_rollout, t_end); later calls
        # (none expected for num_steps=1) return the last value.
        import time as _real_time
        class _FakeTime:
            _vals = [0.0, 0.1, 0.2]
            _i = 0
            @staticmethod
            def perf_counter():
                v = _FakeTime._vals[_FakeTime._i] if _FakeTime._i < len(_FakeTime._vals) else _FakeTime._vals[-1]
                _FakeTime._i += 1
                return v
            def __getattr__(self, name):
                return getattr(_real_time, name)
        with patch("zerorl.train.time", new=_FakeTime()):
            trainer.train(use_wandb=False, use_tb=False)
        patch.stopall()
        assert len(captured) == 1
        pm = captured[0]
        assert pm.rollout_ms == pytest.approx(100.0)
        assert pm.update_ms == pytest.approx(100.0)
        # fps = (rollout_steps * num_envs) / (t_end - t_start) = 16 / 0.2 = 80.0
        assert pm.fps == pytest.approx(80.0)
        env.close()

    @pytest.mark.gpu
    def test_train_profile_metrics_values_sane(self, tmp_path: Path,
                                               device: torch.device) -> None:
        agent = MockAgent()
        env = FakeVecEnv(num_envs=1, obs_dim=4, act_dim=2, steps_until_done=(100,))
        buf = Buffer(step=8, num_envs=1, data={
            "state": (4,), "reward": (), "done": (),
            "action": (), "log_prob": (), "entropy": (), "value": (),
        }, device=device)
        cfg = _make_profile_config(tmp_path, device, profile=True, num_steps=3)
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, AlgoConfig(),
                            require_buffer_size=4)
        captured = _capture_profile_metrics(trainer)
        trainer.train(use_wandb=False, use_tb=False)
        patch.stopall()
        assert len(captured) == 3
        for pm in captured:
            assert math.isfinite(pm.fps) and pm.fps > 0
            assert math.isfinite(pm.rollout_ms) and pm.rollout_ms >= 0
            assert math.isfinite(pm.update_ms) and pm.update_ms >= 0
            assert math.isfinite(pm.vram_allocated_gb)
            assert math.isfinite(pm.vram_peak_gb)
            assert math.isfinite(pm.ram_mb) and pm.ram_mb >= 0
        env.close()


# ---------------------------------------------------------------------------
# TestBaseTrainProfilerWandb — wandb integration (includes xfail)
# ---------------------------------------------------------------------------

class TestBaseTrainProfilerWandb:
    @pytest.mark.gpu
    def test_train_profile_wandb_logs_profile_keys(self, tmp_path: Path,
                                                  device: torch.device) -> None:
        agent = MockAgent()
        env = FakeVecEnv(num_envs=1, obs_dim=4, act_dim=2, steps_until_done=(100,))
        buf = Buffer(step=8, num_envs=1, data={
            "state": (4,), "reward": (), "done": (),
            "action": (), "log_prob": (), "entropy": (), "value": (),
        }, device=device)
        cfg = _make_profile_config(tmp_path, device, profile=True, num_steps=1)
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, AlgoConfig(),
                            require_buffer_size=4)
        with patch("wandb.log") as mock_log:
            trainer.train(use_wandb=True, use_tb=False)
        logged_dicts = [c.args[0] for c in mock_log.call_args_list if c.args]
        joined_keys = set()
        for d in logged_dicts:
            joined_keys.update(d.keys())
        assert any(k.startswith("profile/") for k in joined_keys)
        env.close()

    @pytest.mark.gpu
    @pytest.mark.xfail(reason="profiler emits two wandb.log calls per step — "
                              "profile call lacks step= (train.py:279-282), "
                              "metrics misalign on the wandb dashboard")
    def test_train_profile_wandb_step_alignment(self, tmp_path: Path,
                                                device: torch.device) -> None:
        agent = MockAgent()
        env = FakeVecEnv(num_envs=1, obs_dim=4, act_dim=2, steps_until_done=(100,))
        buf = Buffer(step=8, num_envs=1, data={
            "state": (4,), "reward": (), "done": (),
            "action": (), "log_prob": (), "entropy": (), "value": (),
        }, device=device)
        cfg = _make_profile_config(tmp_path, device, profile=True, num_steps=2)
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, AlgoConfig(),
                            require_buffer_size=4)
        with patch("wandb.log") as mock_log:
            trainer.train(use_wandb=True, use_tb=False)
        assert mock_log.call_count == 2  # expect exactly one wandb.log per step
        for c in mock_log.call_args_list:
            assert "step" in c.kwargs and c.kwargs["step"] in (0, 1)
        env.close()

    @pytest.mark.gpu
    def test_train_profile_wandb_excludes_step_key(self, tmp_path: Path,
                                                   device: torch.device) -> None:
        # Contract test: `step` is not a ProfileMetrics field (passed as a
        # separate arg to _log_profile_metrics), so no `profile/step` should
        # be logged to wandb. Locks in the current dataclass shape.
        agent = MockAgent()
        env = FakeVecEnv(num_envs=1, obs_dim=4, act_dim=2, steps_until_done=(100,))
        buf = Buffer(step=8, num_envs=1, data={
            "state": (4,), "reward": (), "done": (),
            "action": (), "log_prob": (), "entropy": (), "value": (),
        }, device=device)
        cfg = _make_profile_config(tmp_path, device, profile=True, num_steps=1)
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, AlgoConfig(),
                            require_buffer_size=4)
        with patch("wandb.log") as mock_log:
            trainer.train(use_wandb=True, use_tb=False)
        logged_dicts = [c.args[0] for c in mock_log.call_args_list if c.args]
        joined_keys = set()
        for d in logged_dicts:
            joined_keys.update(d.keys())
        assert "profile/step" not in joined_keys
        # The 6 ProfileMetrics-derived keys should all be present.
        for k in ("fps", "rollout_ms", "update_ms",
                  "vram_allocated_gb", "vram_peak_gb", "ram_mb"):
            assert f"profile/{k}" in joined_keys
        env.close()
