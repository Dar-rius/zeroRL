import numpy as np
import torch
import gymnasium as gym
import torch.nn as nn
from gymnasium import spaces
from torch.optim.lr_scheduler import LambdaLR

from zerorl.agent import BaseAgent, eval_action
from zerorl.env import BaseEnv
from zerorl.train import BaseTrain
from zerorl.common import Buffer
from zerorl.config import TrainConfig, AlgoConfig
from zerorl.algorithms.ppo import gae_compute, ppo
from zerorl.function import linear_schedule

# 1. Define your agent
class CartPoleAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(4, 2)
        self.critic = nn.Linear(4, 1)

    def forward(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        return self.actor(state_t), self.critic(state_t)

    @staticmethod
    def build_distribution(logits):
        return torch.distributions.Categorical(logits=logits)
    
    def get_action(self, state: torch.Tensor, action: torch.Tensor | None = None, **kwargs):
        logits, value = self.forward(state, **kwargs)
        dist = self.build_distribution(logits)
        if action is None: action = dist.sample()
        log_prob, dist_entropy = eval_action(dist, action)
        return {"action": action, "log_prob": log_prob, "entropy":dist_entropy, "value":value}


# 2. Define your environment (or use a Gymnasium wrapper)
class CartPoleEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self._env = gym.make("CartPole-v1")
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, seed=None, options=None):
        return self._env.reset(seed=seed)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        self._env.close()


# 3. Configure and train
config = TrainConfig(model_name="cartpole", model_save_path="./checkpoints")
algo_config = AlgoConfig(lr=3e-4, gamma=0.99, clip_eps=0.2, ent_coef=0.01)

agent = CartPoleAgent()
env = CartPoleEnv()
buffer = Buffer(
    step=config.rollout_steps,
    data={
        "state": (4,),
        "old_log_probs": (),
        "reward": (),
        "done": (),
        "entropy": (),
        "value": (),
        "adv": (),
        "returns": (),
        "actions": (),
    },
    device=config.device,
)

def update_weights(agent, buffer, optimizer, step, last_output, algo_config):
    """Compute GAE advantages then run PPO update."""
    all_data = buffer.get_all()
    # Compute GAE from rollout data
    returns, advantages, _ = gae_compute(
        all_data["reward"],
        all_data["value"],
        last_output["value"],
        all_data["done"],
        algo_config,
    )
    buffer.insert(returns=returns, adv=advantages)
    scheduler = LambdaLR(optimizer, lr_lambda=linear_schedule(step=step, num_update=config.num_update))
    return ppo(
        agent, optimizer, buffer, algo_config, scheduler,
        batch_size=algo_config.batch_size,
        epochs=algo_config.epochs,
        device=agent.device,
    )

trainer = BaseTrain(agent, env, buffer, update_weights, config, algo_config)
trainer.train(use_wandb=False, use_tb=False)