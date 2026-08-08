"""Unit tests for BaseTrain (zerorl.train)."""

import pytest
import torch
import torch.nn as nn
import gymnasium as gym
from pathlib import Path
from zerorl.agent import BaseAgent, eval_action
from zerorl.common import Buffer
from zerorl.config import AlgoConfig, TrainConfig
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
        buf = Buffer(step=10, data={"states": (obs_dim,)}, device=device)
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
