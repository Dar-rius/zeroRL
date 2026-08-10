"""Integration tests for PPO pipeline with Vectorized Environments."""

import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from zerorl.agent import BaseAgent, eval_action
from zerorl.algorithms.ppo.ppo import gae_compute, ppo
from zerorl.buffer import Buffer
from zerorl.config import AlgoConfig
from zerorl.vector_env import VectorEnv

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiscreteTestAgent(BaseAgent):
    """Agent for testing vectorized discrete PPO."""
    def __init__(self, obs_dim: int = 4, act_dim: int = 2):
        super().__init__()
        self.actor = nn.Linear(obs_dim, act_dim)
        self.val = nn.Linear(obs_dim, 1)

    def forward(self, state: torch.Tensor, **kwargs):
        return self.actor(state), self.val(state)

    @staticmethod
    def build_distribution(logits: torch.Tensor):
        return torch.distributions.Categorical(logits=logits)

    def get_action(self, state: torch.Tensor, action: torch.Tensor | None = None, **kwargs):
        logits, value = self.forward(state, **kwargs)
        dist = self.build_distribution(logits)
        if action is None: action = dist.sample()
        log_prob, dist_entropy = eval_action(dist, action)
        return {"action": action, "log_prob": log_prob, "entropy": dist_entropy, "value": value}


class TestPPOVectorizedIntegration:
    """Test the full PPO pipeline with a real vectorized environment."""

    @pytest.mark.gpu
    def test_full_vectorized_ppo_cycle(self, device) -> None:
        num_envs = 4
        obs_dim = 4
        T = 32  # Rollout steps

        # 1. Setup Vectorized Env and Agent
        env = VectorEnv("CartPole-v1", num_envs=num_envs)
        agent = DiscreteTestAgent(obs_dim=obs_dim, act_dim=2).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        cfg = AlgoConfig()
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

        # 2. Setup Buffer with num_envs to match vectorized outputs
        buf = Buffer(
            step=T,
            num_envs=num_envs, 
            data={
                "state": (obs_dim, ),
                "action": (),
                "log_prob": (),
                "reward": (),
                "done": (),
                "value": (),
                "advantage": (),
                "return": (),
            },
            device=device,
        )

        # 3. Simulate Vectorized Rollout
        state, _ = env.reset(seed=42)
        
        for _ in range(T):
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                out = agent.get_action(state_tensor)
                
            # Step the real vectorized environment
            next_state, reward, terminated, _, _ = env.step(out["action"].cpu().numpy())
            
            # Convert to tensors for the buffer
            reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=device)
            done_tensor = torch.as_tensor(terminated, dtype=torch.float32, device=device)
            
            # Insert batched data into buffer
            buf.insert(
                state=state_tensor,
                action=out["action"],
                log_prob=out["log_prob"],
                reward=reward_tensor,
                done=done_tensor,
                value=out["value"].squeeze(-1),
            )
            state = next_state

        env.close()

        # 4. Compute GAE on 3D data (T, num_envs, ...)
        all_data = buf.get_all()
        last_value = torch.zeros(num_envs, 1, device=device)  # Mock last value (N, 1)
        
        returns, advantages, _ = gae_compute(
            all_data["reward"],      # (T, N)
            all_data["value"],       # (T, N, 1)
            last_value,              # (N, 1)
            all_data["done"],        # (T, N)
            cfg,
        )
        
        buf.data["return"][:buf.size] = returns
        buf.data["advantage"][:buf.size] = advantages

        # 5. Run PPO Update (which will flatten (T, N) -> (T*N) internally)
        w_before = {k: v.clone() for k, v in agent.state_dict().items()}
        
        result = ppo(
            agent, optimizer, buf, cfg, scheduler,
            batch_size=16,
            epochs=2,
            device=device,
        )

        # 6. Assertions
        w_after = agent.state_dict()
        
        assert isinstance(result, dict)
        assert "loss" in result
        assert torch.isfinite(result["loss"])
        
        changed = not all(torch.allclose(w_before[k], w_after[k]) for k in w_before)
        assert changed, "PPO update did not change model weights in vectorized env"
