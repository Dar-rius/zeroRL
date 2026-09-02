[![PyPI version](https://img.shields.io/pypi/v/zerorl)](https://pypi.org/project/zerorl/)
[![Python](https://img.shields.io/pypi/pyversions/zerorl)](https://pypi.org/project/zerorl/)
[![License](https://img.shields.io/pypi/l/zerorl)](https://pypi.org/project/zerorl/)

<div align="center">
  <h1> zeroRL </h1>
</div>

Reinforcement learning is demanding. Existing solutions are excellent for standard baselines, but when your research requires custom algorithms, novel buffer structures, or specific multi-agent setups, you often end up fighting the framework instead of focusing on the science.

**zeroRL takes a different approach.** It's a simple, explicit, and modular architecture designed to reduce the friction between your research idea and its implementation.

The core principle: **If you can write it in PyTorch, you can use it in zeroRL.**

The framework allows you to:

- Implement custom algorithms that are not included in the framework;
- Integrate new environments without unnecessary wrappers
- Replace or modify individual components without rewriting the training pipeline
- Maintain full control and visibility over the training pipeline
- Debug and understand what's happening at every step

zeroRL is designed to make reinforcement learning experimentation easier without imposing heavy abstractions or hiding the details that matter.
  
## Installation

Before installing zeroRL, ensure Python `3.11+` is available.

Install zeroRL with uv or pip: 

```bash
uv pip install zerorl

or 

pip install zerorl
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
# Custom agent (BaseAgent subclass)
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
from zerorl.helpers.factory import get_env
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
        # Note: eval_action must be imported or defined in your module
        log_prob, entropy = eval_action(dist, action)
        return {"action": action, "log_prob": log_prob, "entropy": entropy, "value": value}


# 2. Set up environment and buffer
config = TrainConfig(project_name="cartpole_example", model_name="agent", total_timesteps=1_000_000, num_envs=2)
algo_config = AlgoConfig()

env = get_env("CartPole-v1", config.num_envs)
obs_shape, act_shape, obs_n, act_n, _ = get_obs_act(env)

agent = Agent(obs_n, act_n)
buffer = Buffer(
    data={
        "state": obs_shape, "action": act_shape,
        "reward": (), "done": (), "truncated": (),
        "entropy": (), "value": (), "return": (),
        "log_prob": (), "advantage": () 
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

config = TrainConfig(model_name="gridworld", project_name="gridworld_exp", total_timesteps=500_000)
algo_config = AlgoConfig()

trainer = easy_train_ppo(GridWorld(), config, algo_config)
trainer.train()
```

### Modular function

All RL algorithms are modular functions where you can change some components:

```python
from torch import Tensor
from zerorl.algorithms.ppo import ppo_func, gae_compute
from zerorl.helpers.agent import BaseAgent # Fixed import path

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
    # Write your own PPO loss here
    ...

def update_weights(agent, buffer, scheduler, optimizer, last_output, algo_config):
    all_data = buffer.get_all()
    gae_compute(all_data["reward"], all_data["value"], last_output["value"],
                all_data["done"], buffer, algo_config)
    return ppo_func(agent, optimizer, buffer, algo_config, scheduler, ppo_loss_func=custom_ppo_loss, device=agent.device)
```

### Implement your own algorithm

```python
# This is an excerpt from examples/reinforce.py

import torch
from zerorl.train import BaseTrain

# Define your pure PyTorch update function
def reinforce_update(agent, buffer, optimizer, algo_config, scheduler=None, last_output=None):
    data = buffer.get_all(reshape=True)
    rewards = data["reward"]
    total_size = rewards.shape[0]
    dones = data["done"]
    returns = torch.empty_like(rewards)
    mask = 1.0 - dones
    R = 0.0
    for step in reversed(range(total_size)):
        R = rewards[step] + algo_config.gamma * mask[step] * R 
        returns[step] = R
    
    global_losses = agent.get_action(data["state"], data["action"])
    loss = -(global_losses["log_prob"] * returns).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5) # Max grad norm
    optimizer.step()
    return {"loss": loss.detach()}

# Plug it in. BaseTrain handles rollouts.
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

*Go to the [examples](https://github.com/Dar-rius/zeroRL/tree/main/examples) folder to see some examples of how to use the framework.*

## What's Included

zeroRL provides a minimal set of composable components, each designed to be transparent, extensible, and easy to understand.

| Component | Description |
| --- | --- |
| `BaseAgent` | Plain `nn.Module` base class that allows you to define `get_action()` and `build_distribution()` in pure PyTorch — no custom abstractions to learn. |
| `BaseEnv` | Abstract Gymnasium environment where you implement `reset()`, `step()`, and `close()` for zero-friction integration with the ecosystem. |
| `BaseTrain` | Transparent training orchestrator handling rollout collection, observation normalization, weight updates, and profiling, keeping everything visible and debuggable. |
| `Buffer` | Dictionary-like tensor container inspired by TorchDict, allowing you to store and manipulate trajectories with a clean, flexible interface. |
| `AlgoConfig` | Centralized hyperparameters (`lr`, `gamma`, `gae_lambda`, `clip_eps`, `ent_coef`, `value_coef`, `batch_size`, `epochs`, `tau`) that are mutable at runtime for fast experimentation. |
| `TrainConfig` | Training settings with auto-computed `model_path`, `num_update`, and device detection, providing sensible defaults while remaining easy to override. |
| `easy_train_ppo` | One-call setup that wires agent, env, and buffer into a ready-to-train `BaseTrain` — perfect for baselines, trivial to extend. |
| `ActorCriticAgent` | Built-in agent with orthogonal initialization, supporting both discrete and continuous action spaces out of the box. |

| Algorithm | Status |
| --- | --- |
| **PPO** | ✅ Implemented & Tested |
| **SAC, DQN, PPO Recurrent, DDPG** | 🚧 Planned / Contributions Welcome |

*These algorithms are the next priorities on our [roadmap](https://github.com/Dar-rius/zeroRL/issues/43). If you are familiar with any of these implementations, we would be thrilled to welcome your PRs to integrate them!*

## Configuration

```python
from zerorl.config import AlgoConfig, TrainConfig
import torch

algo = AlgoConfig(
    lr=3e-4,          
    gamma=0.99,       
    gae_lambda=0.95,  
    clip_eps=0.2,     
    ent_coef=0.01,    
    value_coef=0.5,   
    batch_size=64,    
    epochs=10,       
    tau=0.005
)

train = TrainConfig(
    model_name="my_agent",                 # Required, used to save model in a specific path
    project_name="my_experiment",          # Required, used for wandb/tensorboard
    model_save_path=".checkpoints",        # Default
    total_timesteps=1_000_000,             # Total training steps (renamed from 'timestamp' for clarity)
    rollout_steps=2048,                    # Steps per rollout
    num_envs=1,                            # Parallel environments
    normalize=False,                       # Normalize observations of environment
    profile=False,                         # Profile steps of training
    debug=False,                           # Enable training-pipeline validation and anomaly detection
    device=torch.device("cuda"),           # Tensor device, checks if the device has a GPU 
    num_update=1_000_000 // (2048 * 1),    # Number of weight updates (total_timesteps // (rollout_steps * num_envs))
    model_path=".checkpoints/my_agent.pt"  # Path for saving agent weights 
)
```

## Contributing

zeroRL is actively developed with a focus on modularity and research-grade flexibility, you take a look at our [roadmap](https://github.com/Dar-rius/zeroRL/issues/43). 
Contributions are welcome in the following areas:

To propose a feature, report a bug, or discuss an idea, please [open an issue](https://github.com/Dar-rius/zeroRL/issues). Pull Requests are encouraged.

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.
