import torch
from zerorl.vector_env import VectorEnv
from zerorl.agent import BaseAgent, eval_action
from zerorl.buffer import Buffer 
from zerorl.config import TrainConfig 
from torch import nn
from torch import Tensor


#Auto create new BaseEnv
def get_env(env_id: str, num_envs: int, render_mode: str): return VectorEnv(env_id, num_envs, render_mode)

#Auto create Actor-Critic buffer
def get_actor_critic_buffer(state_space: tuple, action_space: tuple, config: TrainConfig): 
    buffer = Buffer(step = config.rollout_steps,
             data = {"state": state_space, "action": action_space,
                "reward": (), "done": (), "entropy": (), "value": (),
                "adv": (), "return": (), "log_prob": (), "advantage": ()},
                device=config.device)
    return buffer

#Auto create new Actor-Critic BaseAgent
class ActorCriticAgent(BaseAgent):
    def __init__(self, input_layer: int, output_layer: int, hidden_layer: int = 64):
        super().__init__()
        # Feature Extractor
        self.extract_layer = nn.Sequential(
                nn.Linear(input_layer, hidden_layer),
                nn.Tanh(),
                nn.Linear(hidden_layer, hidden_layer),
                )
        # Actor
        self.actor = nn.Linear(hidden_layer, output_layer)
        # Critic
        self.critic = nn.Linear(hidden_layer,  1)

    def forward(self, state: Tensor):
        x = self.extract_layer(state)
        logits = self.actor(x)
        value = self.critic(x)
        return (logits, value)

    @staticmethod
    def build_distribution(logits: torch.Tensor):
        return torch.distributions.Categorical(logits=logits)
    
    def get_action(self, state: torch.Tensor, action: torch.Tensor | None = None):
        logits, value = self.forward(state)
        dist = self.build_distribution(logits)
        if action is None: action = dist.sample()
        log_prob, dist_entropy = eval_action(dist, action)
        return {"action": action, "log_prob": log_prob, "entropy":dist_entropy, "value":value}
