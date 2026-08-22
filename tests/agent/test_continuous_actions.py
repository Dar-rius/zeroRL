"""Integration tests for continuous action spaces using real Gymnasium envs."""

import pytest
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from torch.optim.lr_scheduler import LambdaLR

from zerorl.helpers.agent import BaseAgent, eval_action
from zerorl.algorithms.ppo.ppo import gae_compute, ppo_func
from zerorl.buffer import Buffer
from zerorl.config import AlgoConfig
from zerorl.helpers.env import BaseEnv

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Real Continuous Environment (Pendulum-v1)
# =============================================================================

class PendulumEnvWrapper(BaseEnv):
    """Real continuous environment wrapper for integration testing."""
    def __init__(self):
        super().__init__()
        self._env = gym.make("Pendulum-v1")
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        # Gymnasium expects numpy arrays for continuous actions
        return self._env.step(action.cpu().numpy() if isinstance(action, torch.Tensor) else action)

    def close(self):
        self._env.close()


# =============================================================================
# Continuous Agent compatible with Pendulum
# =============================================================================

class ContinuousTestAgent(BaseAgent):
    """Continuous-policy agent for Pendulum (obs_dim=3, act_dim=1)."""

    def __init__(self, obs_dim: int = 3, act_dim: int = 1):
        super().__init__()
        self.mean_layer = nn.Linear(obs_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.value = nn.Linear(obs_dim, 1)

    def forward(self, state: torch.Tensor, **kwargs):
        action_mean = self.mean_layer(state)
        action_std = self.log_std.exp().expand_as(action_mean)
        logits = torch.cat([action_mean, action_std], dim=-1)
        val = self.value(state)
        return logits, val

    @staticmethod
    def build_distribution(logits: torch.Tensor):
        act_dim = logits.shape[-1] // 2
        mean = logits[..., :act_dim]
        std = logits[..., act_dim:]
        return torch.distributions.Normal(mean, std)

    def get_action(self, state: torch.Tensor, action: torch.Tensor | None = None, **kwargs):
        logits, value = self.forward(state, **kwargs)
        dist = self.build_distribution(logits)
        if action is None: action = dist.sample()
        log_prob, dist_entropy = eval_action(dist, action)
        return {"action": action, "log_prob": log_prob, "entropy":dist_entropy, "value":value}



# =============================================================================
# Integration Test: Full Cycle with Real Env
# =============================================================================

class TestPPOContinuousIntegration:
    """Test the full PPO pipeline with a real continuous Gymnasium environment."""

    @pytest.mark.gpu
    def test_full_ppo_cycle_updates_weights(self, device) -> None:
        # 1. Setup Env and Agent
        env = PendulumEnvWrapper()
        assert isinstance(env.observation_space, spaces.Box), "Obs space must be Box"
        assert isinstance(env.action_space, spaces.Box), "Act space must be Box"
        
        obs_shape = env.observation_space.shape
        act_shape = env.action_space.shape
        
        if obs_shape is None or act_shape is None:
            pytest.fail("Environment must have defined shapes for obs and act")
            
        obs_dim = obs_shape[0]
        act_dim = act_shape[0]
        
        agent = ContinuousTestAgent(obs_dim=obs_dim, act_dim=act_dim).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        cfg = AlgoConfig()
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

        # 2. Setup Buffer
        n_steps = 128
        raw_buf = Buffer(
            step=n_steps,
            data={
                "state": (obs_dim,),
                "action": (act_dim,),
                "log_prob": (),
                "reward": (),
                "done": (),
                "value": (),
                "advantage": (),
                "return": (),
            },
            device=device,
        )

        # 3. Collect Rollout (Interact with real env)
        state, _ = env.reset(seed=42)
        for _ in range(n_steps):
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                out = agent.get_action(state_tensor)
            
            # Step the real environment
            next_state, reward, terminated, truncated, _ = env.step(out["action"])
            done = terminated or truncated

            # Sum multi-dim log_prob to scalar for buffer storage
            log_prob = out["log_prob"]
            if log_prob.dim() > 0:
                log_prob = log_prob.sum()

            raw_buf.insert(
                state=state_tensor,
                action=out["action"],
                log_prob=log_prob,
                reward=torch.tensor(reward, dtype=torch.float32, device=device),
                done=torch.tensor(1.0 if done else 0.0, device=device),
                value=out["value"].squeeze(),
            )

            state = next_state if not done else env.reset()[0]

        env.close()

        # 4. Compute GAE (writes advantage/return in place into raw_buf)
        all_data = raw_buf.get_all()
        last_value = torch.zeros(1, device=device) # Assume 0 for simplicity at end of rollout
        gae_compute(
            all_data["reward"],
            all_data["value"],
            last_value,
            all_data["done"],
            raw_buf,
            cfg,
        )

        # 5. Run PPO Update and verify weights change
        w_before = {k: v.clone() for k, v in agent.state_dict().items()}
        
        result = ppo_func(
            agent, optimizer, raw_buf, cfg, scheduler,
            device=device
        )
        
        w_after = agent.state_dict()
        changed = not all(torch.allclose(w_before[k], w_after[k]) for k in w_before)
        
        # Assertions
        assert isinstance(result, dict)
        assert "loss" in result
        assert torch.isfinite(result["loss"])
        assert changed, "PPO update did not change model weights"
