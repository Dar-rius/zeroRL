"""Abstract training loop for RL agents.

Provides BaseTrain, which orchestrates the rollout-update cycle:
collect experience, compute GAE, run PPO, and repeat.
"""

import os
import numpy as np
import torch
import tqdm
from typing import Callable
from torch import optim
from .agent import BaseAgent
from .common import Buffer
from .config import TrainConfig, PPOConfig
from .env import BaseEnv
from .errors import EmptyBufferError


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
                 update_weights: Callable,
                 param_config: PPOConfig,
                 train_config: TrainConfig,
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
        self.agent = agent
        self.env = env
        self.buffer = buffer
        self.update_weights = update_weights()
        self.train_config = train_config
        self.param_config = param_config
        self.optimizer: optim.Optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.agent.parameters(), lr=self.param_config.lr)
        else:
            self.optimizer = optimizer
        self.cumulative_reward = 0.0
        self.require_buffer_size = require_buffer_size

    @torch.compile
    def rollout_phase(self, state: np.ndarray):
        """Collect experience by running the agent in the environment.

        Stores each transition in the buffer and resets on episode end.
        After the rollout, computes the bootstrap value for GAE.

        Args:
            state: Initial observation to start the rollout from.
        """
        for _ in range(self.train_config.rollout_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.train_config.device)
            with torch.inference_mode():
                outputs = self.agent.get_action(state_tensor)
                action_np = outputs["action"].cpu().numpy()

            # Convention: truncate = terminated (episode naturally ended)
            #             done = truncated (episode cut short by time limit)
            next_state, reward, done, truncate, _ = self.env.step(action_np)
            done_casted = 1.0 if done else 0.0

            self.buffer.insert(
                state = state_tensor,
                reward = reward,
                done = done_casted,
                **outputs
            )
            self.cumulative_reward += reward

            if done or truncate:
                state, _ = self.env.reset()
            else:
                state = next_state

        with torch.inference_mode():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.train_config.device)
            next_output = self.agent.get_action(state_tensor)
        return next_output

    def train(self):
        state, _ = self.env.reset()
        for step in tqdm(range(self.train_config.num_update)):
            last_output = self.rollout_phase(state)

            if self.buffer.size < self.require_buffer_size:
                raise EmptyBufferError(self.buffer.size, self.require_buffer_size)

            self.update_weights(
                        agent = self.agent,
                        buffer = self.buffer,
                        optimizer = self.optimizer,
                        last_output = last_output,
                        config = self.param_config
                    )
            self.buffer.clear()


    def save_model(self):
        """Save agent weights to the path in train_config.model_path."""
        os.makedirs(os.path.dirname(self.train_config.model_path), exist_ok=True)
        torch.save(self.agent.state_dict(), self.train_config.model_path)
