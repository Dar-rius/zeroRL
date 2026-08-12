import torch
from zerorl.vector_env import VectorEnv
from zerorl.agent import BaseAgent, eval_action
from zerorl.buffer import Buffer 
from zerorl.config import TrainConfig 
from torch import nn
from torch import Tensor


#Auto create new BaseEnv
def get_env(env_id: str, num_envs: int, render_mode: str | None): return VectorEnv(env_id, num_envs, render_mode)

#Auto create Actor-Critic buffer
def get_actor_critic_buffer(state_space: tuple, action_space: tuple, config: TrainConfig): 
    buffer = Buffer(step = config.rollout_steps,
             data = {"state": (state_space, ), "action": action_space,
                "reward": (), "done": (), "entropy": (), "value": (),
                "return": (), "log_prob": (), "advantage": ()},
                device=config.device)
    return buffer

#Auto create new Actor-Critic BaseAgent
class ActorCriticAgent(BaseAgent):
    def __init__(self, input_dim: int, output_dim: int, is_discrete: bool, hidden_dim: int = 64):
        self.is_discrete = is_discrete
        output_dim = output_dim if self.is_discrete else output_dim * 2
        
        super().__init__()
        # Feature Extractor
        self.extract_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                )
        # Actor
        self.actor = nn.Linear(hidden_dim, output_dim)
        # Critic
        self.critic = nn.Linear(hidden_dim,  1)

    def forward(self, state: Tensor):
        x = self.extract_layer(state)
        logits = self.actor(x)
        value = self.critic(x)
        return (logits, value)

    def build_distribution(self, logits: torch.Tensor):
        if self.is_discrete:
            return torch.distributions.Categorical(logits=logits)
        mean, log_std = logits.chunk(2, dim=-1)
        std = log_std.exp()
        return torch.distributions.Normal(mean, std)
    
    def get_action(self, state: torch.Tensor, action: torch.Tensor | None = None):
        logits, value = self.forward(state)
        dist = self.build_distribution(logits)
        if action is None: action = dist.sample()
        log_prob, dist_entropy = eval_action(dist, action)
        return {"action": action, "log_prob": log_prob, "entropy":dist_entropy, "value":value}
