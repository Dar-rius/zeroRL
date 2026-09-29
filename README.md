[![PyPI version](https://img.shields.io/pypi/v/zerorl)](https://pypi.org/project/zerorl/)
[![Python](https://img.shields.io/pypi/pyversions/zerorl)](https://pypi.org/project/zerorl/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Dar-rius/zeroRL/blob/main/LICENSE)

<div align="center">
  <h1> zeroRL </h1>
</div>
   
Reinforcement learning research often requires modifying the training pipeline:
changing rollout collection, experimenting with new losses, introducing custom
buffers and RL algorithms or integrating non-standard environments.

Many RL frameworks optimize for standard workflows. zeroRL instead focuses on
giving researchers control over how experiments are built.

**zeroRL is a modular PyTorch reinforcement learning framework that lets you
work at the level of abstraction your experiment requires.**

You can:

- Train RL agents quickly with a high-level API.
- Customize agents, environments, buffers, algorithms, and update functions through `BaseTrain`.
- Build complete training loops from low-level zeroRL primitives.
- Integrate Gymnasium environments and custom MuJoCo simulations.
- Keep the training pipeline explicit, inspectable, and easy to modify.

The core principle:

> **Stay close to your training pipeline.**
  
## Installation

Before installing zeroRL, ensure Python `3.11+` is available.

Install zeroRL with `uv` or `pip`: 

```bash
# With uv
uv pip install zerorl

# With pip
pip install zerorl
```

The package depends on `torch`, `numpy`, `gymnasium`, `mujoco`, `tqdm`, and `imageio`. 

## Choose Your Level of Control

zeroRL exposes the same training stack at different levels of abstraction.

### High-level — train quickly

The fastest way to train an agent — one function call:

```python
from zerorl.algorithms.ppo import easy_train_ppo
from zerorl.config import TrainConfig, AlgoConfig

config = TrainConfig(model_name="pendulum_agent", project_name="my_experiment")
algo_config = AlgoConfig(ent_coef=0.0)

trainer = easy_train_ppo("Pendulum-v1", config, algo_config)
trainer.train(use_tb=True)
trainer.try_agent()
```

This creates an `ActorCriticAgent`, vectorized environments, a rollout buffer, and a PPO training pipeline, with TensorBoard or W&B tracking wired automatically.

Override any component:

```python
# Custom agent (BaseAgent subclass)
trainer = easy_train_ppo("Pendulum-v1", config, algo_config, agent=my_agent)

# Custom environment (BaseEnv subclass)
trainer = easy_train_ppo(my_env, config, algo_config)

# Multiple environments
config.num_envs = 4
trainer = easy_train_ppo("CartPole-v1", config, algo_config)
```
### Mid-level — customize the experiment

Use `BaseTrain` when you want to provide your own agent, environment, buffer, RL algorithm, or optimization logic while letting zeroRL handle rollout orchestration.

```python
# Set up environment and buffer
config = TrainConfig(project_name="cartpole_example", model_name="agent", timestamp=1_000_000, num_envs=2)
algo_config = AlgoConfig()
env = get_env("CartPole-v1", config.num_envs)
obs_shape, act_shape, obs_n, act_n, _ = get_obs_act(env)
agent = Agent(obs_n, act_n)
buffer = Buffer(
        capacity=config.rollout_steps,
        num_envs=config.num_envs,
        schema={
              "state": obs_shape, "action": act_shape,
              "reward": (), "terminated": (), "truncated": (),
              "entropy": (), "value": (), "return": (),
              "log_prob": (), "advantage": () 
          },
        device=config.device)

# Define the update weights function
def update_weights(agent, buffer, scheduler, optimizer, last_output, algo_config):
    all_data = buffer.get_all()
    gae_compute(all_data["reward"], all_data["value"], last_output["value"], all_data["terminated"], buffer, algo_config)
    return ppo_func(agent, optimizer, buffer, algo_config, scheduler, device=agent.device)

# Train
trainer = BaseTrain(agent, env, buffer, update_weights, config, algo_config)
trainer.train(use_wandb=True, save_model=True)
```

See [examples/bipedal.py](https://github.com/Dar-rius/zeroRL/blob/main/examples/bipedal.py) and [examples/reinforce.py](https://github.com/Dar-rius/zeroRL/blob/main/examples/reinforce.py) for complete examples.

### Low-level — build the training loop yourself

Use zeroRL primitives when the training loop itself is part of the experiment. An abridged training loop looks like this:

```python
from zerorl.logger import create_logger
from zerorl.functions import (processing_state, parse_dict_to_tensor, to_env_action, try_agent, get_obs_act, vectorize_env, set_seed)

cfg = TrainConfig(model_name="Lunar-model", project_name="Lunar-example", num_envs=4)
algo_cfg = AlgoConfig()
seed = set_seed(42, cfg.num_envs)
env = vectorize_env("LunarLander-v3", num_envs = cfg.num_envs)
obs_dim, act_dim, obs_n, act_n, is_discrete = get_obs_act(env)
agent = ActorCriticAgent(obs_n, act_n, is_discrete).to(cfg.device)
buffer = Buffer(capacity = cfg.rollout_steps,
                num_envs = cfg.num_envs,
                schema = {"state": obs_dim, "action": act_dim,
                          "reward": (), "terminated": (), "entropy": (), "value": (),
                          "return": (), "log_prob": (), "advantage": (), "truncated": ()},
                device = cfg.device)
log = create_logger(cfg, algo_cfg, use_tb=True)
state, _ = env.reset(seed = seed)

for step in tqdm(range(cfg.num_update)):
    metrics = {}
    for _ in range(cfg.rollout_steps):
        state_processed = processing_state(state)
        with torch.inference_mode():
            outputs = agent.get_action(state_processed)
        action = to_env_action(outputs["action"], env)
        next_state, reward, terminated, truncated, _ = env.step(action)
        outputs_final = {"next_state": next_state, "reward": reward,
                   "terminated": terminated, "truncated": truncated, **outputs} 
        outputs_final = parse_dict_to_tensor(outputs_final)
        next_state = outputs_final.pop("next_state")
        buffer.insert(state = state_processed, **outputs_final)
         ...

    metrics = {"train/mean_episodic_reward": mean_reward}
    for k, v in losses.items(): metrics[f"train/{k}"] = v
    log(metrics, step)
    buffer.clear()

env.close()
log.close()
try_agent("LunarLander-v3", agent, cfg)
```

At this level, zeroRL provides reusable building blocks without owning the training loop. You decide how transitions are collected, processed, stored, optimized, and logged.

## Core Building Blocks

zeroRL provides a small set of composable components designed to remain explicit, extensible, and easy to inspect.

| Layer | Components | Purpose |
| --- | --- | --- |
| High-level API | `easy_train_ppo` | Build and train a standard PPO experiment quickly |
| Training orchestration | `BaseTrain` | Manage rollout collection while keeping components replaceable |
| Agents | `BaseAgent`, `ActorCriticAgent`, `PolicyAgent` | Define agent architectures in PyTorch |
| Storage | `Buffer` | Pre-allocated trajectory storage with a customizable schema |
| Normalization | `NormMeanStd` | Normalize observations using running statistics |
| Algorithms | `ppo_func`, `gae_compute` | Reusable optimization and return-estimation primitives |
| Environment | `BaseEnv`, `MujocoEnv`, `vectorize_env` | Gymnasium and custom MuJoCo integration |
| Pipeline primitives | `processing_state`, `to_env_action`, `parse_dict_to_tensor`, `env_step`, ... | Assemble custom training loops |
| Experiment tools | logging, profiling, debugging | Observe and validate experiments |

| Algorithm | Status |
| --- | --- |
| **PPO** | ✅ Implemented & Tested |
| **SAC** | 🚧 Planned / Contributions Welcome |

*See the [roadmap](https://github.com/Dar-rius/zeroRL/issues/43) for planned algorithms and upcoming features. Contributions are welcome.*

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
    timestamp=1_000_000,                   # Total training steps (renamed from 'timestamp' for clarity)
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

## Examples

See the examples below for different ways to use zeroRL, from high-level training APIs to fully customized training loops.

| Example | Description |
| --- | --- |
| [Bipedal (Gymnasium)](https://github.com/Dar-rius/zeroRL/blob/main/examples/bipedal.py) | Train a custom agent with a custom buffer and PPO update function |
| [Hopper (MuJoCo)](https://github.com/Dar-rius/zeroRL/blob/main/examples/hopper_mujoco_immediate.py) | Build a custom MuJoCo environment and training loop |
| [Hopper (Gymnasium)](https://github.com/Dar-rius/zeroRL/blob/main/examples/hopper_v5_baseline.py) | Train a Hopper agent quickly with `easy_train_ppo` |
| [Humanoid Standup (Gymnasium)](https://github.com/Dar-rius/zeroRL/blob/main/examples/humanoid_standup.py) | Train a HumanoidStandup agent with `easy_train_ppo` |
| [Immediate Mode](https://github.com/Dar-rius/zeroRL/blob/main/examples/immediate_mode.py) | Build a fully customized training loop with zeroRL primitives |
| [REINFORCE](https://github.com/Dar-rius/zeroRL/blob/main/examples/reinforce.py) | Implement REINFORCE and plug it into `BaseTrain` |
| [Reacher (MuJoCo)](https://github.com/Dar-rius/zeroRL/blob/main/examples/reacher_mujoco.py) | Build a custom Reacher environment and train it with `easy_train_ppo` |
| [Point Mass (MuJoCo)](https://github.com/Dar-rius/zeroRL/blob/main/examples/point_mass.py) | Build a custom Point Mass environment and train it with `easy_train_ppo` |

## Contributing

zeroRL is actively developed with a focus on modularity and research-grade flexibility.

Take a look at the [roadmap](https://github.com/Dar-rius/zeroRL/issues/43) for planned features and open tasks. 

To propose a feature, report a bug, or discuss an idea, please [open an issue](https://github.com/Dar-rius/zeroRL/issues). 

Pull requests are welcome.

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.
