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
from zerorl.agent import BaseAgent
from zerorl.common import Buffer
from zerorl.config import TrainConfig, AlgoConfig
from zerorl.env import BaseEnv
from zerorl.algorithms.processing import NormMeanStd
from zerorl.errors import EmptyBufferError


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
        self.cumulative_reward = 0.0
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
            self.cumulative_reward += reward

            if done or truncate:
                state, _ = self.env.reset()
            else:
                state = next_state

        with torch.inference_mode():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.train_config.device)
            self.normalizer.update(state_tensor)
            state_normalized = self.normalizer.normalize(state_tensor)
            next_output = self.agent.get_action(state_normalized)
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
                        config = self.algo_config
                    )
            self.buffer.clear()


    def save_model(self):
        """Save agent weights to the path in train_config.model_path."""
        os.makedirs(os.path.dirname(self.train_config.model_path), exist_ok=True)
        torch.save(self.agent.state_dict(), self.train_config.model_path)
