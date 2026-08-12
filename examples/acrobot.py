import torch
import gymnasium as gym
import torch.nn as nn
from zerorl.agent import BaseAgent, eval_action
from zerorl.env import BaseEnv
from zerorl.train import BaseTrain
from zerorl.buffer import Buffer
from zerorl.config import TrainConfig, AlgoConfig
from zerorl.algorithms.ppo import gae_compute, ppo


# 1. Define your agent
class Agent(BaseAgent):
    def __init__(self, input_layer, output_layer):
        super().__init__()
        self.extract_layer = nn.Sequential(
                nn.Linear(input_layer, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                )

        self.actor = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, output_layer)
                )
        
        self.critic = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64,  1)
                )


    def forward(self, state: torch.Tensor):
        x = self.extract_layer(state)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    @staticmethod
    def build_distribution(logits: torch.Tensor):
        return torch.distributions.Categorical(logits=logits)
    
    def get_action(self, state: torch.Tensor, action: torch.Tensor | None = None):
        logits, value = self.forward(state)
        dist = self.build_distribution(logits)
        if action is None: action = dist.sample()
        log_prob, dist_entropy = eval_action(dist, action)
        return {"action": action, "log_prob": log_prob, "entropy":dist_entropy, "value":value}


# 2. Define your environment (or use a Gymnasium wrapper)
class AcrobotEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self._env = gym.make("Acrobot-v1")
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, seed=None, options=None):
        return self._env.reset(seed=seed)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        self._env.close()


# 3. Configure and train
config = TrainConfig(project_name="acrobot_example", model_name="agent_1", model_save_path=".checkpoints")
#config.device = torch.device("cpu")
algo_config = AlgoConfig(lr=3e-4, gamma=0.99, clip_eps=0.2, ent_coef=0.01)

env = AcrobotEnv()
input_layer = env.observation_space.shape
output_layer = env.action_space.n
agent = Agent(input_layer[-1], output_layer)
buffer = Buffer(
    step=config.rollout_steps,
    data={
        "state": input_layer, "action": output_layer.shape, 
        "reward": (), "done": (), "entropy": (), "value": (),
        "return": (), "log_prob": (), "advantage": ()},
    device=config.device)


def update_weights(agent, buffer, scheduler, optimizer, last_output, algo_config):
    """Compute GAE advantages then run PPO update."""
    all_data = buffer.get_all()
    # Compute GAE from rollout data
    gae_compute(all_data["reward"], all_data["value"], last_output["value"], all_data["done"], algo_config, buffer)
    return ppo(agent, optimizer, buffer, algo_config, scheduler, device=agent.device)

trainer = BaseTrain(agent, env, buffer, update_weights, config, algo_config, render_mode="human")
trainer.train(use_wandb=True)
