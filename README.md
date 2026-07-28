[![PyPI version](https://img.shields.io/pypi/v/rl-template.svg)](https://pypi.org/project/rl-template/)
[![Python](https://img.shields.io/pypi/pyversions/rl-template.svg)](https://pypi.org/project/rl-template/)
[![License](https://img.shields.io/pypi/l/rl-template.svg)](https://pypi.org/project/rl-template/)

# zeroRL

`rl-template` provides a structured foundation for building and training reinforcement learning agents. Instead of writing boilerplate code for training loops, rollouts, and PPO optimization, you extend the provided abstract base classes and focus on defining your agent and environment.

The package includes:

- **Abstract base classes** — `BaseAgent`, `BaseEnv`, `BaseTrain` that enforce a clean interface via the template method pattern. Subclasses implement the methods that matter; the framework handles the rest.
- **PPO implementation** — A production-ready Proximal Policy Optimization trainer with GAE, clipped surrogate loss, linear learning rate decay, entropy bonus, and gradient clipping.
- **Pre-allocated rollout buffer** — A zero-allocation `Buffer` that stores trajectory data in pre-allocated NumPy arrays, supporting both discrete and continuous action spaces.
- **Typed configurations** — Immutable `PPOConfig` and computed `TrainConfig` dataclasses with auto GPU detection.
- **107 tests** — Full test suite covering unit, integration, and continuous action scenarios.

## Installation

Requires **Python 3.11+**.

```bash
pip install rl-template
```

The package depends on `torch` and `numpy`. Make sure PyTorch is installed with the appropriate CUDA version for your system if you plan to train on GPU.

## Quick Start

```python
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_template.agent import BaseAgent
from rl_template.env import BaseEnv
from rl_template.train import BaseTrain
from rl_template.common import Buffer
from rl_template.config import TrainConfig, PPOConfig
from rl_template.algorithms.ppo.ppo import PPOTrainer


# 1. Define your agent
class CartPoleAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(4, 2)
        self.critic = nn.Linear(4, 1)

    def forward(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        return self.actor(state_t), self.critic(state_t)

    def get_distribution(self, state):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value.squeeze(-1)


# 2. Define your environment (or use a Gymnasium wrapper)
class CartPoleEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-4.8, 4.8, shape=(4,))
        self.action_space = spaces.Discrete(2)
        self._env = __import__("gymnasium").make("CartPole-v1")

    def reset(self, seed=None):
        return self._env.reset(seed=seed)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        self._env.close()


# 3. Configure and train
config = TrainConfig(model_name="cartpole", model_saved_path="./checkpoints")
ppo_config = PPOConfig(lr=3e-4, gamma=0.99, clip_eps=0.2, ent_coef=0.01)

agent = CartPoleAgent()
env = CartPoleEnv()
buffer = Buffer(step=config.rollout_steps, state_shape=(4,))
ppo_trainer = PPOTrainer(agent, lr=ppo_config.lr, gamma=ppo_config.gamma, clip_eps=ppo_config.clip_eps)


class MyTrain(BaseTrain):
    def rollout_phase(self, state):
        super().rollout_phase(state)

    def update_weights(self, step):
        super().update_weights(step)

    def save_model(self):
        super().save_model()


trainer = MyTrain(agent, env, buffer, config, ppo_trainer)
state, _ = env.reset()

for step in range(config.num_update):
    trainer.rollout_phase(state)
    trainer.update_weights(step)
    trainer.save_model()

env.close()
```

## What's Included

| Component     | Description                                                                                                                                                                            |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `BaseAgent`   | Abstract agent (ABC + nn.Module). Implement `forward()` and `get_distribution()`. The `get_action()` template method handles sampling, log-probability, entropy, and value estimation. |
| `BaseEnv`     | Abstract Gymnasium v1 environment wrapper. Implement `reset()`, `step()`, and `close()`.                                                                                               |
| `BaseTrain`   | Training loop orchestrator. Handles rollout collection, GAE computation, PPO updates, and model saving.                                                                                |
| `Buffer`      | Pre-allocated NumPy rollout buffer with bounds checking, size tracking, and tensor conversion for PPO.                                                                                 |
| `PPOTrainer`  | PPO algorithm with GAE, clipped surrogate loss, value loss, entropy bonus, linear LR decay, and gradient clipping (max norm 1.0).                                                      |
| `PPOConfig`   | Frozen (immutable) PPO hyperparameters: `lr`, `gamma`, `gae_lambda`, `clip_eps`, `ent_coef`, `value_coef`.                                                                             |
| `TrainConfig` | Training settings with auto-computed `model_path` and `num_update`. Automatically detects CUDA.                                                                                        |
| `WandbConfig` | Weights & Biases logging configuration.                                                                                                                                                |

## PPO Algorithm Details

The `PPOTrainer` implements:

- **Generalized Advantage Estimation (GAE)** — Blended n-step returns controlled by `gae_lambda` for low-variance advantage estimates
- **Clipped surrogate loss** — Restricts policy updates to prevent destructive changes (`clip_eps` controls the range)
- **Value function loss** — MSE loss for the critic network
- **Entropy bonus** — Encourages exploration via `ent_coef`
- **Linear LR decay** — Learning rate anneals to zero over training
- **Gradient clipping** — Max norm of 1.0 for training stability
- **Advantage & return normalization** — Reduces variance across minibatches

## Configuration

```python
from rl_template.config import PPOConfig, TrainConfig

# PPO hyperparameters (immutable)
ppo = PPOConfig(
    lr=3e-4,          # Adam learning rate
    gamma=0.99,       # Discount factor
    gae_lambda=0.95,  # GAE lambda
    clip_eps=0.2,     # PPO clipping range
    ent_coef=0.01,    # Entropy bonus coefficient
    value_coef=0.5,   # Value loss coefficient
)

# Training settings (computed fields: model_path, num_update)
train = TrainConfig(
    model_name="my_agent",
    model_saved_path="./checkpoints",
    timestamp=1_000_000,    # Total env timesteps
    rollout_steps=2048,     # Steps per rollout
    batch_size=64,          # Minibatch size
)
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.0
- NumPy >= 1.24
- Gymnasium

## License

MIT License - see [LICENSE](https://github.com/Dar-rius/rl_template/blob/main/LICENSE) for details.
