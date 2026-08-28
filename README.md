[![PyPI version](https://img.shields.io/pypi/v/zerorl)](https://pypi.org/project/zerorl/)
[![Python](https://img.shields.io/pypi/pyversions/zerorl)](https://pypi.org/project/zerorl/)
[![License](https://img.shields.io/pypi/l/zerorl)](https://pypi.org/project/zerorl/)

<div align="center">
  <h1> zeroRL </h1>
</div>

Reinforcement learning is a demanding field that requires significant time and focus to train agents properly. Existing solutions such as SB3, RLlib, and Tianshou reduce this friction by providing ready to use implementations. However, some experiments and deeper modifications can be difficult to implement and may require understanding their abstractions and navigating a more complex codebase.

zeroRL takes a different approach: a simple, explicit, and modular architecture designed to facilitate experimentation in reinforcement learning.

The framework allows you to:

- Easily implement algorithms that are not included in the framework;
- Easily integrate new environments;
- Run experiments without modifying the training pipeline;
- Replace or modify components of algorithm implementations;
- Maintain full control over the training pipeline.

zeroRL is designed to make reinforcement learning experimentation easier without imposing heavy abstractions.

The framework development follow this [roadmap](https://github.com/Dar-rius/zeroRL/issues/43).
  
## Installation

Before installing zeroRL, ensure Python `3.11+` is available.

Install zeroRL with uv or pip: 

```bash
uv pip install zerorl
```

The package depends on `torch`, `numpy`, `gymnasium`, `tqdm`, and `imageio`. 

## Quick Start

The fastest way to train an agent — one function call:

```python
from zerorl.algorithms.ppo import easy_train_ppo
from zerorl.config import TrainConfig, AlgoConfig

config = TrainConfig(model_name="Pendulum", project_name="my_experiment")
algo_config = AlgoConfig(ent_coef=0.0)

trainer = easy_train_ppo("Pendulum-v1", config, algo_config)
trainer.train(use_tb=True)
trainer.test()
```

This creates an `ActorCriticAgent`, vectorized environments, a rollout buffer, and runs PPO — all wired together automatically. Override any component:

```python
# Custom environment (BaseAgent subclass)
trainer = easy_train_ppo("Pendulum-v1", config, algo_config, agent=my_agent)

# Custom environment (BaseEnv subclass)
trainer = easy_train_ppo(my_env, config, algo_config)

# Multiple environments
config.num_envs = 4
trainer = easy_train_ppo("CartPole-v1", config, algo_config)
```

## Advanced Usage

For full control over agent, environment, and the training loop:

```python
import torch
import torch.nn as nn
import numpy as np
from zerorl.helpers.agent import BaseAgent
from zerorl.train import BaseTrain
from zerorl.buffer import Buffer
from zerorl.config import TrainConfig, AlgoConfig
from zerorl.algorithms.ppo import gae_compute, ppo_func
from zerorl.factory import get_env
from zerorl.functions import get_obs_act


# 1. Define your agent
class Agent(BaseAgent):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

    def build_distribution(self, logits):
        return torch.distributions.Categorical(logits=logits)

    def get_action(self, state, action=None):
        logits, value = self.forward(state)
        dist = self.build_distribution(logits)
        if action is None:
            action = dist.sample()
        log_prob, entropy = eval_action(dist, action)
        return {"action": action, "log_prob": log_prob, "entropy": entropy, "value": value}


# 2. Set up environment and buffer
config = TrainConfig(project_name="cartpole_example", model_name="agent", timestamp=1_000_000, num_envs=2)
algo_config = AlgoConfig()

env = get_env("CartPole-v1", config.num_envs)
obs_shape, act_shape, obs_n, act_n, _ = get_obs_act(env)

agent = Agent(obs_n, act_.n)
buffer = Buffer(
    data={
        "state": obs_shape, "action": act_shape,
        "reward": (), "done": (), "truncated": (),
        "entropy": (), "value": (), "return": (),
        "log_prob": (), "advantage": (), "truncated": ()
    },
    config=config,
)


# 3. Define the update weights function
def update_weights(agent, buffer, scheduler, optimizer, last_output, algo_config):
    all_data = buffer.get_all()
    gae_compute(all_data["reward"], all_data["value"], last_output["value"],
                all_data["done"], buffer, algo_config)
    return ppo_func(agent, optimizer, buffer, algo_config, scheduler, device=agent.device)


# 4. Train
trainer = BaseTrain(agent, env, buffer, update_weights, config, algo_config)
trainer.train(use_wandb=True, model_save=True)
```

### Custom Environment

Implement `BaseEnv` to use your own environment with `easy_train_ppo` or `BaseTrain`:

```python
import numpy as np
from gymnasium import spaces
from zerorl.helpers.env import BaseEnv


class GridWorld(BaseEnv):
    """Simple 4x4 grid world — agent starts at (0,0), goal at (3,3)."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=3.0, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self._pos = None

    def reset(self, *, seed=None, options=None):
        self._pos = np.array([0, 0], dtype=np.float32)
        return self._pos.copy(), {}

    def step(self, action):
        direction = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]])[action]
        self._pos = np.clip(self._pos + direction, 0, 3)
        terminated = np.array_equal(self._pos, [3, 3])
        reward = 1.0 if terminated else -0.01
        return self._pos.copy(), reward, terminated, False, {}

    def close(self):
        pass
```

Then pass it directly:

```python
from zerorl.algorithms.ppo import easy_train_ppo
from zerorl.config import TrainConfig, AlgoConfig

config = TrainConfig(model_name="gridworld", project_name="gridworld_exp", timestamp=500_000)
algo_config = AlgoConfig()

trainer = easy_train_ppo(GridWorld(), config, algo_config)
trainer.train()
```

### Modular function

All RL algorithm are modular function where you can change some components:

```python
from torch import Tensor
from zerorl.algorithms.ppo import ppo_func, gae_compute
from zerorl.algorithms.helpers.agent import BaseAgent

def custom_ppo_loss(agent: BaseAgent,
                    params: dict,
                    buffers: dict,
                    states: Tensor,
                    actions: Tensor,
                    old_log_prob: Tensor,
                    old_values: Tensor,
                    advantages: Tensor,
                    returns: Tensor,
                    ent_coef: float,
                    value_coef: float,
                    clip_eps: float,
                    clip_vf: float,
                    ) -> dict[str, Tensor]:
    #write your own PPO loss
    ...

def update_weights(agent, buffer, scheduler, optimizer, last_output, algo_config):
    all_data = buffer.get_all()
    gae_compute(all_data["reward"], all_data["value"], last_output["value"],
                all_data["done"], buffer, algo_config)
    return ppo_func(agent, optimizer, buffer, algo_config, scheduler, ppo_loss_func = custom_ppo_loss, device=agent.device)
```

### Implemented your own algorithm

You can implemented your algorithm using the update_weights function:

```python
import torch
from zerorl import BaseTrain

# 1. Define your pure PyTorch update function
def update_weights(agent, buffer, optimizer, algo_config, scheduler=None, last_output=None):
    #Reinforce algorithm implementation
    data = buffer.get_all()
    rewards = data["reward"].squeeze()
    dones = data["done"].squeeze()
    returns = []
    R = 0.0
    for r, d in zip(reversed(rewards.tolist()), reversed(dones.tolist())):
        if d: R = 0.0
        R = r + algo_config.gamma * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns, device=agent.device)
    logits, _ = agent(data["state"])
    dist = agent.build_distribution(logits)
    log_probs = dist.log_prob(data["action"]).sum(dim=-1)
    loss = -(log_probs * returns).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5) # Max grad norm
    optimizer.step()
    return {"loss": loss}

# 2. Plug it in. BaseTrain handles rollouts.
trainer = BaseTrain(
    agent=agent, 
    env=env, 
    buffer=buffer, 
    update_weights=reinforce_update, 
    config=config, 
    algo_config=algo_config
)
trainer.train()
```

## What's Included

| Component | Description |
| --- | --- |
| `BaseAgent` | Plain `nn.Module` base class. Agents must define `get_action()` returning a dict and `build_distributions()` returning a Distribution from torch. |
| `BaseEnv` | Abstract Gymnasium environment. Implement `reset()`, `step()`, and `close()`. |
| `BaseTrain` | Training orchestrator: rollout collection, observation normalization, updates weights, model saving ans profiling. |
| `Buffer` | Inspired from TorchDict is a dictionary-like data container for tensors that lets you manipulate a collection of tensors. |
| `AlgoConfig` | Mutable hyperparameters for RL algorithms: `lr`, `gamma`, `gae_lambda`, `clip_eps`, `ent_coef`, `value_coef`, `batch_size`, `epochs`, `tau`. |
| `TrainConfig` | Training settings with auto-computed `model_path`, `num_update`, and `device`. |
| `easy_train_ppo` | One-call setup: creates agent, env, buffer, and returns a ready-to-train `BaseTrain`. |
| `ActorCriticAgent` | Built-in agent with orthogonal init, supports discrete and continuous action spaces. |
| `vectorize_env` | Wraps env specs into `SyncVectorEnv` with `SAME_STEP` autoreset. |

| Algorithm | Included |
| --- | --- |
| `PPO` | ✅ |
| `SAC` | ❌ |
| `DQN` | ❌ |
| `TD3` | ❌ |
| `DDPG` | ❌ |

The algorithms not yet included will be added soon, along with their derivatives.

## Configuration

```python
from zerorl.config import AlgoConfig, TrainConfig

algo = AlgoConfig(
    lr=3e-4,          
    gamma=0.99,       
    gae_lambda=0.95,  
    clip_eps=0.2,     
    ent_coef=0.01,    
    value_coef=0.5,   
    batch_size=64,    
    epochs=10,       
    tau: float = 0.005
)

train = TrainConfig(
    model_name="my_agent",                               # Required, used for save model in specific path
    project_name="my_experiment",                        # Required,  used for wandb/tensorboard
    model_save_path=".checkpoints",                      # Default
    timestamp=1_000_000,                                 # Total timesteps
    rollout_steps=2048,                                  # Steps per rollout
    num_envs=1,                                          # Parallel environments
    normalize=False,                                     # Normalize observations of environment
    profile=False,                                       # Profile steps of training
    device=torch.device("cuda"),                         # Tensor device, check if the device has a GPU 
    num_update=timestamp // (rollout_steps * num_envs),  # Number of weights update 
    model_path=".checkpoints/my_agent.pt"                # Path for saving agent weights 

)
```

For any questions and features requests, please create an [issue](https://github.com/Dar-rius/zeroRL/issues/new)

## License

MIT License - see [LICENSE](LICENSE) for details.
