[![PyPI version](https://img.shields.io/pypi/v/zerorl.svg)](https://pypi.org/project/zerorl/)
[![Python](https://img.shields.io/pypi/pyversions/zerorl.svg)](https://pypi.org/project/zerorl/)
[![License](https://img.shields.io/pypi/l/zerorl.svg)](https://pypi.org/project/zerorl/)

# zeroRL

Most reinforcement learning libraries are black boxes. If you want to modify a specific function, you often run into unexpected errors. If you want to implement and test a new RL algorithm, you usually have to spend hours reading documentation and digging through the codebase just to understand how everything works.

**ZeroRL** was built to solve this problem. It is an RL library designed to help researchers and labs prototype new environments and algorithms quickly. Its modular architecture makes it easy to customize the components you need without having to understand a massive codebase or fight against the library's abstractions.

Built on PyTorch 2, ZeroRL is designed with scalability, flexibility, and research productivity in mind.

The goal is to make RL research faster, simpler, and more enjoyable.

## Installation

Requires **Python 3.11+**.

```bash
pip install zerorl
```

The package depends on `torch`, `numpy`, `gymnasium`, `tensorboard`, `wandb`, `tqdm`, and `imageio`. Make sure PyTorch is installed with the appropriate CUDA version for your system if you plan to train on GPU.

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
trainer = easy_train_ppo("Pendulum-v1", config, algo_config, base_agent=my_agent)

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
from zerorl.agent import BaseAgent, eval_action
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
obs_dim, act_dim = get_obs_act(env)

agent = Agent(obs_dim.shape[-1], act_dim.n)
buffer = Buffer(
    step=config.rollout_steps,
    num_envs=config.num_envs,
    data={
        "state": (obs_dim.shape[-1],), "action": (),
        "reward": (), "done": (), "truncated": (),
        "entropy": (), "value": (), "return": (),
        "log_prob": (), "advantage": (),
    },
    device=config.device,
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
from zerorl.helpers import BaseEnv


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

## RL Algorithm included

| Algorithm | Included |
| --- | --- |
| `PPO` | ✅ |
| `SAC` | ❌ |
| `DQN` | ❌ |
| `TD3` | ❌ |
| `DDPM` | ❌ |

All RL algorithm is a modular function where you can change some components:

```python
from torch import Tensor
from zerorl.algorithms.ppo import ppo_func, gae_compute
from zerorl.algorithms.agent import BaseAgent

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
    timestamp=1_000_000,                                 # Total env timesteps
    rollout_steps=2048,                                  # Steps per rollout
    num_envs=1,                                          # Parallel environments
    normalize=False,                                     # Normalize obs env
    device=torch.device("cuda"),                         # Tensor device, check if the device has a GPU 
    num_update=timestamp // (rollout_steps * num_envs),  # Number of weights update 
    model_path=".checkpoints/my_agent.pt"                # Path for saving agent weights 

)
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.0
- NumPy >= 1.24
- Gymnasium >= 1.3
- TensorBoard >= 2.21
- Weights & Biases >= 0.28
- tqdm >= 4.70.0
- imageio >= 2.31.0

## License

MIT License - see [LICENSE](LICENSE) for details.
