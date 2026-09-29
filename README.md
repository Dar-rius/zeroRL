[![PyPI version](https://img.shields.io/pypi/v/zerorl)](https://pypi.org/project/zerorl/)
[![Python](https://img.shields.io/pypi/pyversions/zerorl)](https://pypi.org/project/zerorl/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Dar-rius/zeroRL/blob/main/LICENSE)

<div align="center">
  <h1> zeroRL </h1>
</div>
   
Reinforcement learning research often requires modifying the training pipeline:
changing rollout collection, experimenting with new losses, introducing custom
buffers, or integrating non-standard environments.

Many RL frameworks optimize for standard workflows. zeroRL instead focuses on
giving researchers control over how experiments are built.

**zeroRL is a modular PyTorch reinforcement learning framework that lets you
work at the level of abstraction your experiment requires.**

You can:

- Train a RL agent quickly with a high-level API.
- Customize agents, environments, buffers, and update functions through `BaseTrain`.
- Build complete training loops from low-level zeroRL primitives.
- Integrate Gymnasium environments and custom MuJoCo simulations.
- Keep the training pipeline explicit, inspectable, and easy to modify.

The core principle:

> **Stay close to your training pipeline.**
  
## Installation

Before installing zeroRL, ensure Python `3.11+` is available.

Install zeroRL with uv or pip: 

```bash
uv pip install zerorl

or 

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
### Mid-level — customize the experiment

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
| **SAC** | 🚧 Planned / Contributions Welcome |

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

## Advanced Usage

For full control over training pipeline (High Level):


### Custom Environment

Implement with `MujocoEnv` to use your own MuJoCo environment with `easy_train_ppo` or `BaseTrain`

MJCF content:

```Python
REACHER_XML = """
<mujoco model="reacher2d">
  <compiler angle="radian"/>
  <option timestep="0.01" gravity="0 0 0"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
  </visual>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="0.4 0.4 0.05" rgba="0.9 0.9 0.9 1"/>
    <body name="link1" pos="0 0 0.05">
      <joint name="shoulder" type="hinge" axis="0 0 1" damping="0.05" limited="true" range="-3.14 3.14"/>
      <geom type="capsule" fromto="0 0 0 0.12 0 0" size="0.02" rgba="0.2 0.45 0.9 1" mass="0.08"/>
      <body name="link2" pos="0.12 0 0">
        <joint name="elbow" type="hinge" axis="0 0 1" damping="0.05" limited="true" range="-2.7 2.7"/>
        <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.018" rgba="0.25 0.55 0.95 1" mass="0.06"/>
        <site name="fingertip" pos="0.1 0 0" size="0.02" rgba="0.1 0.8 0.3 1"/>
      </body>
    </body>
    <body name="target" mocap="true" pos="0.15 0.0 0.05">
      <geom type="sphere" size="0.025" rgba="0.9 0.15 0.15 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="shoulder" ctrlrange="-1 1" gear="1.5"/>
    <motor joint="elbow" ctrlrange="-1 1" gear="1.2"/>
  </actuator>
</mujoco>
"""
```

Integrate this MJCF content to your own environment:

```python
class Reacher2D(MujocoEnv):
    """Planar 2-DOF arm; fingertip must reach and hold a random target."""

    SUCCESS_DIST = 0.028  # overlap nette (somme rayons tip+cible = 0.045)
    HOLD_STEPS = 5

    def __init__(self, max_steps: int = 200):
        self.max_steps = max_steps
        self._steps = 0
        self._hold = 0
        self._prev_dist = 0.0
        super().__init__(model_xml=REACHER_XML, frame_skip=2)
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64
        )

    def _tip_pos(self) -> np.ndarray:
        return np.asarray(self.data.site("fingertip").xpos[:2], dtype=np.float64).copy()

    def _target_pos(self) -> np.ndarray:
        return self.data.mocap_pos[0, :2].copy()

    def _set_target(self, xy: np.ndarray) -> None:
        self.data.mocap_pos[0, :2] = xy
        self.data.mocap_pos[0, 2] = 0.05
        mujoco.mj_forward(self.model, self.data)

    def reset_model(self) -> np.ndarray:
        self._steps = 0
        self._hold = 0
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        qpos[:] = self.np_random.uniform(-0.2, 0.2, size=qpos.shape)
        qvel[:] = 0.0
        self.set_state(qpos, qvel)

        for _ in range(100):
            ang = float(self.np_random.uniform(-np.pi, np.pi))
            radius = float(self.np_random.uniform(0.10, 0.20))
            target = np.array([radius * np.cos(ang), radius * np.sin(ang)])
            self._set_target(target)
            dist = float(np.linalg.norm(self._tip_pos() - target))
            if dist >= 0.12:
                break

        self._prev_dist = float(np.linalg.norm(self._tip_pos() - self._target_pos()))
        return self._get_obs()

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.clip(
            np.asarray(action, dtype=np.float64),
            self.action_space.low,
            self.action_space.high,
        )
        self._steps += 1
        return super().step(action)

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos.astype(np.float64)
        qvel = self.data.qvel.astype(np.float64)
        tip = self._tip_pos()
        target = self._target_pos()
        return np.concatenate(
            [
                np.cos(qpos),
                np.sin(qpos),
                qvel,
                tip,
                target,
                tip - target,
            ]
        )

    def _get_reward(self) -> float:
        dist = float(np.linalg.norm(self._tip_pos() - self._target_pos()))
        progress = self._prev_dist - dist
        self._prev_dist = dist
        ctrl = float(np.square(self.data.ctrl).sum())
        reward = 2.0 * progress - 0.2 * dist - 0.01 * ctrl
        if dist < self.SUCCESS_DIST:
            self._hold += 1
            reward += 8.0 + 40.0 * (self.SUCCESS_DIST - dist)
        else:
            self._hold = 0
        return reward

    def _is_truncated(self) -> bool:
        return self._steps >= self.max_steps

    def _is_terminated(self) -> bool:
        return self._hold >= self.HOLD_STEPS

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 480)
        if self._camera is None:
            self._camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, self._camera)
            self._camera.lookat[:] = [0.0, 0.0, 0.0]
            self._camera.distance = 0.85
            self._camera.elevation = -90.0
            self._camera.azimuth = 0.0
        self._renderer.update_scene(self.data, camera=self._camera)
        return np.asarray(self._renderer.render())

```

Then pass it directly:

```python
from zerorl.algorithms.ppo import easy_train_ppo
from zerorl.config import TrainConfig, AlgoConfig

config = TrainConfig(
        model_name=f"Reacher2D-{args.device}",
        project_name="reacher_mujoco",
        timestamp=args.timestamp,
        rollout_steps=512,
        num_envs=16,
        normalize=False,
        profile=True)

algo_config = AlgoConfig(
        lr=3e-4,
        ent_coef=0.0,
        epochs=10,
        batch_size=256,
        gae_lambda=0.95,
        clip_eps=0.2)

trainer = easy_train_ppo(Reacher2D, config, algo_config, hidden_layer=128)
trainer.agent.log_std.requires_grad_(False)
trainer.train(use_wandb=True, save_model=True)
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

### Create your own pipeline training
Implement your own pipeline training without any abstractions (BaseTrain)

```python
from zerorl.logger import create_logger
from zerorl.functions import (processing_state,
                              parse_dict_to_tensor,
                              to_env_action,
                              try_agent,
                              get_obs_act,
                              vectorize_env,
                              set_seed)


cfg = TrainConfig(model_name="Lunar-model", project_name="Lunar-example", num_envs=4)
cfg.device = torch.device("cpu")
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
optimizer = optim.Adam(agent.parameters(), lr=algo_cfg.lr, eps=1e-5)
scheduler = LambdaLR(optimizer, lambda step_: 1.0 - (step_ / cfg.num_update))
log = create_logger(cfg, algo_cfg, use_tb=True)
reward_tensor = torch.zeros(cfg.num_envs, device=cfg.device)
state, _ = env.reset(seed = seed)

for step in tqdm(range(cfg.num_update)):
    episodic_reward = []
    metrics = {}
    for _ in range(cfg.rollout_steps):
        state_processed = processing_state(state)
        with torch.inference_mode():
            outputs = agent.get_action(state_processed)
        action  = to_env_action(outputs["action"], env)
        next_state, reward, terminated, truncated, _ = env.step(action)
        outputs_final = {"next_state": next_state, "reward": reward,
                   "terminated": terminated, "truncated": truncated, **outputs} 
        outputs_final = parse_dict_to_tensor(outputs_final)
        next_state = outputs_final.pop("next_state")
        buffer.insert(state = state_processed, **outputs_final)
        reward_tensor += outputs_final["reward"]
        finished = (outputs_final["terminated"] > 0) | (outputs_final["truncated"] > 0)

        if finished.any():
            episodic_reward.extend(reward_tensor[finished].tolist())
            reward_tensor[finished] = 0.0
        state = next_state

    with torch.inference_mode():
        state_processed = processing_state(state)
        last_output = agent.get_action(state_processed)

    data = buffer.get_all()
    gae_compute(data["reward"], data["value"], last_output["value"], data["terminated"], buffer, algo_cfg)
    losses = ppo_func(agent, optimizer, buffer, algo_cfg, scheduler)
    if len(episodic_reward) > 0:
        recent = episodic_reward[-10:]
        mean_reward = float(np.mean(recent))
    else:
        mean_reward = 0.0
    metrics = {"train/mean_episodic_reward": mean_reward}
    for k, v in losses.items(): metrics[f"train/{k}"] = v
    log(metrics, step)
    buffer.clear()

env.close()
log.close()
try_agent("LunarLander-v3", agent, cfg)
```

*Go to the [examples](https://github.com/Dar-rius/zeroRL/tree/main/examples) folder to see some examples of how to use the framework.*

## Contributing

zeroRL is actively developed with a focus on modularity and research-grade flexibility, you take a look at our [roadmap](https://github.com/Dar-rius/zeroRL/issues/43). 
Contributions are welcome in the following areas:

To propose a feature, report a bug, or discuss an idea, please [open an issue](https://github.com/Dar-rius/zeroRL/issues). Pull Requests are encouraged.

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.
