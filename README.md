[![PyPI version](https://img.shields.io/pypi/v/zerorl.svg)](https://pypi.org/project/zerorl/)
[![Python](https://img.shields.io/pypi/pyversions/zerorl.svg)](https://pypi.org/project/zerorl/)
[![License](https://img.shields.io/pypi/l/zerorl.svg)](https://pypi.org/project/zerorl/)

# zeroRL

`zeroRL` provides a structured foundation for building and training reinforcement learning agents. Instead of writing boilerplate code for training loops, rollouts, and PPO optimization, you extend the provided abstract base classes and focus on defining your agent and environment.

The package includes:

- **Abstract base classes** — `BaseAgent`, `BaseEnv` that enforce a clean interface via the template method pattern. Subclasses implement the methods that matter; the framework handles the rest.
- **Training loop** — `BaseTrain` orchestrates rollout collection, PPO updates, and model saving. You pass a standalone `update_weights` function.
- **PPO implementation** — Standalone functions (`gae_compute`, `ppo_loss`, `ppo`) implementing Generalized Advantage Estimation, clipped surrogate loss, and minibatch SGD with linear LR decay.
- **Pre-allocated rollout buffer** — A zero-allocation `Buffer` that stores trajectory data in pre-allocated PyTorch tensors, supporting both discrete and continuous action spaces.
- **Typed configurations** — Mutable `AlgoConfig` and computed `TrainConfig` dataclasses with auto GPU detection.
- **Observation normalization** — `NormMeanStd` (running mean/std) and `NormMinMax` (min/max scaling) for observation preprocessing.
- **143 tests** — Full test suite covering unit, integration, and continuous action scenarios.

## Installation

Requires **Python 3.11+**.

```bash
pip install zerorl
```

The package depends on `torch`, `numpy`, `gymnasium`, `tensorboard`, and `wandb`. Make sure PyTorch is installed with the appropriate CUDA version for your system if you plan to train on GPU.

## Quick Start

```python
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from zerorl.agent import BaseAgent
from zerorl.env import BaseEnv
from zerorl.train import BaseTrain
from zerorl.common import Buffer
from zerorl.config import TrainConfig, AlgoConfig
from zerorl.algorithms.ppo.ppo import ppo


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


# 2. Define your environment (or use a Gymnasium wrapper)
class CartPoleEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-4.8, 4.8, shape=(4,))
        self.action_space = spaces.Discrete(2)
        self._env = __import__("gymnasium").make("CartPole-v1")

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
        "action": (),
        "log_prob": (),
        "reward": (),
        "done": (),
        "entropy": (),
        "value": (),
        "adv": (),
        "returns": (),
        "actions": (),
        "old_log_probs": (),
    },
    device=config.device,
)


def update_weights(agent, buffer, optimizer, last_output, algo_config):
    """Compute GAE advantages then run PPO update."""
    from torch.optim.lr_scheduler import LambdaLR

    from zerorl.algorithms.ppo.ppo import gae_compute, ppo

    all_data = buffer.get_all()

    # Compute GAE from rollout data
    returns, advantages, _ = gae_compute(
        all_data["reward"],
        all_data["value"],
        last_output["value"],
        all_data["done"],
        algo_config,
    )

    # Store GAE results into buffer
    buffer.data["adv"][:buffer.size] = advantages
    buffer.data["returns"][:buffer.size] = returns

    # Copy keys to match ppo() expectations (action → actions, log_prob → old_log_probs)
    buffer.data["actions"][:buffer.size] = all_data["action"]
    buffer.data["old_log_probs"][:buffer.size] = all_data["log_prob"]

    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    return ppo(
        agent, optimizer, buffer, algo_config, scheduler,
        batch_size=algo_config.batch_size,
        epochs=algo_config.epochs,
        device=agent.device,
    )


trainer = BaseTrain(agent, env, buffer, update_weights, config, algo_config)
trainer.train(use_wandb=False, use_tb=False)
```

## What's Included

| Component     | Description |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `BaseAgent`   | Abstract agent (ABC + nn.Module). Implement `forward()` and `build_distribution()`. The `get_action()` template method handles sampling, log-probability, entropy, and value estimation. Returns a `dict[str, Tensor]`. |
| `BaseEnv`     | Abstract Gymnasium v1 environment wrapper. Implement `reset()`, `step()`, and `close()`.                                                                                               |
| `BaseTrain`   | Training loop orchestrator. Handles rollout collection, observation normalization, PPO updates, and model saving. Accepts a `Callable` for the weight update strategy.                                                                                |
| `Buffer`      | Pre-allocated PyTorch tensor rollout buffer with bounds checking, size tracking, and dict-based field management.                                                                                 |
| `AlgoConfig`  | Mutable hyperparameters: `lr`, `gamma`, `gae_lambda`, `clip_eps`, `ent_coef`, `value_coef`, `batch_size`, `epochs`, `tau`.                                                                             |
| `TrainConfig` | Training settings with auto-computed `model_path`, `num_update`, and `device`. Automatically detects CUDA.                                                                                        |
| `NormMeanStd` | Online running mean/std normalization for observations.                                                                                                                                                |
| `NormMinMax`  | Min-max normalization for observations.                                                                                                                                                |
| `linear_schedule` | Linear decay function for learning rate scheduling.                                                                                                                                                |
| `ppo` | Standalone PPO update function — GAE, clipped surrogate loss, minibatch SGD, gradient clipping.                                                      |

## PPO Algorithm Details

The PPO implementation consists of three standalone functions:

- **`gae_compute()`** — Generalized Advantage Estimation. Blended n-step returns controlled by `gae_lambda` for low-variance advantage estimates. Works backwards through the trajectory accumulating TD errors.
- **`ppo_loss()`** — Clipped surrogate loss for policy updates, MSE value loss, and entropy bonus. Uses `torch.func.functional_call` for functional-style forward passes.
- **`ppo()`** — Full update cycle: normalizes advantages/returns, runs `epochs` passes of minibatch SGD with gradient clipping (max norm 1.0), and steps the LR scheduler.

## Configuration

```python
from zerorl.config import AlgoConfig, TrainConfig

# Algorithm hyperparameters (mutable)
algo = AlgoConfig(
    lr=3e-4,          # Adam learning rate
    gamma=0.99,       # Discount factor
    gae_lambda=0.95,  # GAE lambda
    clip_eps=0.2,     # PPO clipping range
    ent_coef=0.01,    # Entropy bonus coefficient
    value_coef=0.5,   # Value loss coefficient
    batch_size=64,    # Minibatch size
    epochs=10,        # PPO epochs per update
)

# Training settings (computed fields: model_path, num_update, device)
train = TrainConfig(
    model_name="my_agent",
    model_save_path="./checkpoints",
    timestamp=1_000_000,    # Total env timesteps
    rollout_steps=2048,     # Steps per rollout
)
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.0
- NumPy >= 1.24
- Gymnasium >= 1.3
- TensorBoard >= 2.21
- Weights & Biases >= 0.28

## License

MIT License - see [LICENSE](LICENSE) for details.
