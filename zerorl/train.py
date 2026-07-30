"""Abstract training loop for RL agents.

Provides BaseTrain, which orchestrates the rollout-update cycle:
collect experience, compute GAE, run PPO, and repeat.
"""

import os
import numpy as np
import torch
from torch import Tensor
from torch import optim
from .agent import BaseAgent
from .algorithms.ppo import ppo, gae
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
                 train_config: TrainConfig,
                 ppo_config = PPOConfig,
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
        self.ppo_config = ppo_config
        self.agent = agent
        self.env = env
        self.buffer = buffer
        self.optimizer: optim.Optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.agent.parameters(), lr=self.ppo_config.lr)
        else:
            self.optimizer = optimizer
        self.last_value = torch.zeros(1, dtype=torch.float32, device=self.train_config.device)
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
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.train_config.device)
            with torch.inference_mode():
                action, log_prob, _, value = self.agent.get_action(state_tensor)
                action_np = action.cpu().numpy()

            # Convention: truncate = terminated (episode naturally ended)
            #             done = truncated (episode cut short by time limit)
            next_state, reward, done, truncate, _ = self.env.step(action_np)
            done_casted = 1 if done else 0

            self.buffer.insert(
                state=state_tensor,
                action=action,
                old_log_prob=log_prob,
                reward=reward,
                value=value,
                dones=done_casted,
            )
            self.cumulative_reward += reward

            if done or truncate:
                state, _ = self.env.reset()
            else:
                state = next_state

        with torch.inference_mode():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.train_config.device)
            _, _, _, next_value = self.agent.get_action(state_tensor)
        self.last_value[0] = next_value


    def update_weights(self, step: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute GAE and update agent weights via PPO.

        Args:
            step: Current training step (used for learning rate decay).

        Raises:
            EmptyBufferError: If the buffer has fewer entries than
                             require_buffer_size.
        """
        if self.buffer.size < self.require_buffer_size:
            raise EmptyBufferError(self.buffer.size, self.require_buffer_size)

        rewards_list = self.buffer.rewards
        values_list = self.buffer.values
        dones_list = self.buffer.dones

        with torch.inference_mode():
            returns, adv, _ = gae(rewards_list,
                            values_list,
                            self.last_value,
                            dones_list,
                            self.ppo_config)
            self.buffer.insert_returns(returns, adv)

        loss, policy_loss, value_loss, entropy_loss = ppo(
                model = self.agent,
                ppo_config = self.ppo_config,
                optimizer = self.optimizer,
                buffer = self.buffer,
                step = step,
                num_update = self.train_config.num_update,
                batch_size = self.train_config.batch_size,
                device=self.train_config.device
        )
        self.buffer.clear()
        return (loss, policy_loss, value_loss, entropy_loss)


    def save_model(self):
        """Save agent weights to the path in train_config.model_path."""
        os.makedirs(os.path.dirname(self.train_config.model_path), exist_ok=True)
        torch.save(self.agent.state_dict(), self.train_config.model_path)
