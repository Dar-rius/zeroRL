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
def get_actor_critic_buffer(state_space: int, action_space: tuple, config: TrainConfig): 
    buffer = Buffer(step = config.rollout_steps,
             data = {"state": (state_space, ), "action": action_space,
                "reward": (), "done": (), "entropy": (), "value": (),
                "return": (), "log_prob": (), "advantage": ()},
                device=config.device)
    return buffer

#Auto create new Actor-Critic BaseAgent
class ActorCriticAgent(BaseAgent):
    def __init__(self, input_dim: int, output_dim: int, is_discrete: bool, hidden_dim: int = 64):
        super().__init__()

        self.is_discrete = is_discrete
        
        # Feature Extractor
        self.extract_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
                )
        # Actor
        self.actor = nn.Linear(hidden_dim, output_dim)
        # Critic
        self.critic = nn.Linear(hidden_dim,  1)

        if not is_discrete:
            self.log_std = nn.Parameter(torch.zeros(output_dim))

        self.apply(self._orthogonal_init)

    def _orthogonal_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: Tensor):
        x = self.extract_layer(state)
        logits = self.actor(x)
        value = self.critic(x)
        return (logits, value)

    def build_distribution(self, logits: torch.Tensor):
        if self.is_discrete:
            return torch.distributions.Categorical(logits=logits)
        std = self.log_std.exp().expand_as(logits)
        return torch.distributions.Normal(logits, std)
    
    def get_action(self, state: torch.Tensor, action: torch.Tensor | None = None):
        logits, value = self.forward(state)
        dist = self.build_distribution(logits)
        if action is None: action = dist.sample()
        log_prob, dist_entropy = eval_action(dist, action)
        return {"action": action, "log_prob": log_prob, "entropy":dist_entropy, "value":value}
