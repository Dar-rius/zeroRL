"""Unit tests for BaseTrain (zerorl.train)."""

import pytest
import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from pathlib import Path
from unittest.mock import patch, MagicMock
from gymnasium import spaces
from zerorl.agent import BaseAgent, eval_action
from zerorl.common import Buffer
from zerorl.config import AlgoConfig, TrainConfig
from zerorl.errors import EmptyBufferError
from zerorl.env import BaseEnv
from zerorl.train import BaseTrain


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
                "actions": (),
                "old_log_probs": (),
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
    def _update(agent, buffer, optimizer, step, last_output, algo_config):
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
            "actions": (), "old_log_probs": (), "entropy": (), "value": (),
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
            "actions": (), "old_log_probs": (), "entropy": (), "value": (),
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
            "actions": (), "old_log_probs": (), "entropy": (), "value": (),
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
            "actions": (), "old_log_probs": (), "entropy": (), "value": (),
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
            "actions": (), "old_log_probs": (), "entropy": (), "value": (),
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
            "actions": (), "old_log_probs": (), "entropy": (), "value": (),
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
            "actions": (), "old_log_probs": (), "entropy": (), "value": (),
        }, device=device)
        cfg = _make_train_config(tmp_path, device)
        cfg.rollout_steps = rollout_steps
        cfg.num_envs = num_envs
        trainer = BaseTrain(agent, env, buf, _mock_update_weights, cfg, AlgoConfig())
        state, _ = env.reset(seed=42)
        trainer.rollout_phase(state)
        assert env.counters[1] == 5
        env.close()
