"""Abstract training loop for RL agents.

Provides BaseTrain, which orchestrates the rollout-update cycle:
collect experience, compute GAE, run PPO, and repeat.
"""

import os
import numpy as np
import torch
import wandb
import tqdm
from typing import Callable
from torch import Tensor
from torch import optim
from zerorl.agent import BaseAgent
from zerorl.common import Buffer
from zerorl.config import TrainConfig, AlgoConfig
from zerorl.env import BaseEnv
from zerorl.processing import NormMeanStd
from zerorl.errors import EmptyBufferError
from torch.utils.tensorboard import SummaryWriter


class BaseTrain:
    """Abstract training loop coordinating agent, environment, and PPO.

    Subclasses must implement rollout_phase(), update_weights(), and
    save_model(). The provided implementations are template methods;
    subclasses may override or call super() to reuse them.
    """

    def __init__(self,
                 agent: BaseAgent,
                 env: BaseEnv,
                 buffer: Buffer,
                 update_weights: Callable[[BaseAgent, Buffer,
                                           optim.Optimizer, dict[str,Tensor],
                                           AlgoConfig | None], dict[str, Tensor]],
                 train_config: TrainConfig,
                 algo_config: AlgoConfig | None = None,
                 optimizer: optim.Optimizer | None = None,
                 require_buffer_size: int = 10):
        """Initialize the training loop.

        Args:
            agent: Neural network policy to train.
            env: Environment to collect experience from.
            buffer: Pre-allocated buffer for rollout data.
            train_config: Training configuration (device, paths, hyperparams).
            ppo_trainer: PPO trainer handling GAE and weight updates.
        """
        super().__init__()
        self.train_config = train_config
        self.agent = agent.to(self.train_config.device)
        self.env = env
        self.buffer = buffer
        self.update_weights = update_weights
        self.algo_config = algo_config
        self.optimizer: optim.Optimizer
        if optimizer is None:
            lr = getattr(self.algo_config, 'lr', 3e-4)
            self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
        obs_shape = env.observation_space.shape
        assert obs_shape is not None, "NormMeanStd requires environment with a defined observation shape"
        self.normalizer = NormMeanStd(obs_shape, train_config.device)
        tb_log_dir = os.path.join(self.train_config.model_save_path, "tensobaord", self.train_config.model_name)
        self.tb_writer = SummaryWriter(tb_log_dir)
        self.current_episode_reward = 0.0
        self.episode_rewards: list[float] = []
        self.require_buffer_size = require_buffer_size


    def rollout_phase(self, state: np.ndarray):
        """Collect experience by running the agent in the environment.

        Stores each transition in the buffer and resets on episode end.
        After the rollout, computes the bootstrap value for GAE.

        Args:
            state: Initial observation to start the rollout from.
        """
        for _ in range(self.train_config.rollout_steps):
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.train_config.device)
            self.normalizer.update(state_tensor)
            state_normalized = self.normalizer.normalize(state_tensor)
            with torch.inference_mode():
                outputs = self.agent.get_action(state_normalized)
                action_np = outputs["action"].cpu().numpy()

            # Convention: truncate = terminated (episode naturally ended)
            #             done = truncated (episode cut short by time limit)
            next_state, reward, done, truncate, _ = self.env.step(action_np)
            done_casted = 1.0 if done else 0.0

            self.buffer.insert(
                state = state_normalized,
                reward = reward,
                done = done_casted,
                **outputs
            )
            self.current_episode_reward += reward

            if done or truncate:
                self.episode_rewards.append(self.current_episode_reward)
                self.current_episode_reward = 0.0
                state, _ = self.env.reset()
            else:
                state = next_state

        with torch.inference_mode():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.train_config.device)
            self.normalizer.update(state_tensor)
            state_normalized = self.normalizer.normalize(state_tensor)
            next_output = self.agent.get_action(state_normalized)
        return next_output


    def _log_metrics(self, metrics: dict, step: int, use_wandb: bool, use_tb: bool):
        clean_metrics: dict[str, float] = {}
        tensor_keys: list[str]  = []
        tensor_vals: list[Tensor] = []

        for k, v in metrics.items():
            if isinstance(v, Tensor):
                tensor_keys.append(k)
                tensor_vals.append(v.detach())
            else:
                clean_metrics[k] = float(v)
            
        if tensor_vals:
            cpu_vals = torch.stack(tensor_vals).cpu().numpy()
            for k, v in zip(tensor_keys, cpu_vals):
                clean_metrics[k] = float(v)        

        if use_wandb: wandb.log(clean_metrics, step=step)

        if use_tb:
            for key, value in clean_metrics.items():
                self.tb_writer.add_scalar(key, value, step)



    def train(self, use_wandb: bool = False, use_tb: bool = False):
        state, _ = self.env.reset()
        for step in tqdm(range(self.train_config.num_update)):
            last_output = self.rollout_phase(state)

            if self.buffer.size < self.require_buffer_size:
                raise EmptyBufferError(self.buffer.size, self.require_buffer_size)

            losses = self.update_weights(self.agent,
                        self.buffer,
                        self.optimizer,
                        last_output,
                        self.algo_config
                    )

            if len(self.episode_rewards) > 0:
                recent = self.episode_rewards[-10:]
                mean_reward = sum(recent) / len(recent)
            else:
                mean_reward = 0.0

            metrics = {
                "mean_episode_reward": mean_reward,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }
            for k, v in losses.items(): metrics[k] = v
            self._log_metrics(metrics, step, use_wandb, use_tb)

            self.buffer.clear()


    def save_model(self):
        """Save agent weights to the path in train_config.model_path."""
        os.makedirs(os.path.dirname(self.train_config.model_path), exist_ok=True)
        torch.save(self.agent.state_dict(), self.train_config.model_path)
