import numpy as np
import torch
import torch.nn as nn
from zerorl.helpers.agent import BaseAgent, eval_action
from zerorl.train import BaseTrain
from zerorl.buffer import Buffer
from zerorl.config import TrainConfig, AlgoConfig
from zerorl.algorithms.ppo import gae_compute, ppo_func
from zerorl.helpers.factory import get_env
from zerorl.functions import get_obs_act


# 1. Define your agent
class Agent(BaseAgent):
    def __init__(self, input_layer, output_layer):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(output_layer))
        self.actor = nn.Sequential(
                nn.Linear(input_layer, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, output_layer)
                )
        
        self.critic = nn.Sequential(
                nn.Linear(input_layer, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64,  1)
                )

    def _orthogonal_init(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            if module.out_features == self.hidden_dim:
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            elif module.out_features == 1:
                nn.init.orthogonal_(module.weight, gain=1.0)
            else:
                nn.init.orthogonal_(module.weight, gain=1.0)
                
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor):
        logits = self.actor(state)
        value = self.critic(state)
        return logits, value

    def build_distribution(self, logits: torch.Tensor):
        log_std_clamped = torch.clamp(self.log_std, min=-3.0, max=1.0)
        std = log_std_clamped.exp().expand_as(logits)
        return torch.distributions.Normal(logits, std)
    
    def get_action(self, state: torch.Tensor, action: torch.Tensor | None = None):
        logits, value = self.forward(state)
        dist = self.build_distribution(logits)
        if action is None: action = dist.sample()
        log_prob, dist_entropy = eval_action(dist, action)
        return {"action": action, "log_prob": log_prob, "entropy":dist_entropy, "value":value}


# 3. Configure and train
config = TrainConfig(project_name="acrobot_example", model_name="agent_1", timestamp=2_000_000, num_envs=4, profile=True)
config.device = torch.device("cpu")
algo_config = AlgoConfig(ent_coef=0.0)
env = get_env("BipedalWalker-v3", config.num_envs)

obs_dim, act_dim, obs_n, act_n, _ = get_obs_act(env)
agent = Agent(obs_n, act_n) #type: ignore
buffer = Buffer(data={"state": obs_dim, "action": act_dim, #type: ignore
                      "reward": (), "done": (), "entropy": (), "value": (),
                      "return": (), "log_prob": (), "advantage": (), "truncated": ()},
                config=config)

def update_weights(agent, buffer, scheduler, optimizer, last_output, algo_config):
    """Compute GAE advantages then run PPO update."""
    all_data = buffer.get_all()
    # Compute GAE from rollout data
    gae_compute(all_data["reward"], all_data["value"], last_output["value"], all_data["done"], buffer, algo_config)
    return ppo_func(agent, optimizer, buffer, algo_config, scheduler)

trainer = BaseTrain(agent, env, buffer, update_weights, config, algo_config, render_mode="human")
trainer.train(use_wandb=True)
trainer.test(iterations=2)
