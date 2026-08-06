"""Training loop for RL agents.

Provides BaseTrain, which orchestrates the rollout-update cycle:
collect experience, compute GAE, run the update_weights callable,
and repeat.
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
    """Training loop coordinating agent, environment, and weight updates.

    Orchestrates the rollout-update cycle: collect experience via
    rollout_phase(), compute GAE, run the update_weights callable,
    and repeat. Handles observation normalization, logging, and
    model saving.
    """

    def __init__(self,
                 agent: BaseAgent,
                 env: BaseEnv,
                 buffer: Buffer,
                 update_weights: Callable[[BaseAgent, Buffer,
                                           optim.Optimizer, int, dict[str,Tensor],
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
            update_weights: Callable that computes the weight update from
                collected rollout data. Signature:
                (agent, buffer, optimizer, last_output, algo_config) -> dict[str, Tensor].
            train_config: Training configuration (device, paths, hyperparams).
            algo_config: Algorithm hyperparameters (passed to update_weights).
            optimizer: Optional optimizer. If None, creates Adam with algo_config.lr.
            require_buffer_size: Minimum buffer size before an update is allowed.
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
        self.current_episode_reward: Tensor | None = None
        self.episode_rewards: list[float] = []
        self.require_buffer_size = require_buffer_size


    def rollout_phase(self, state: np.ndarray):
        """Collect experience by running the agent in the environment.

        Stores each transition in the buffer and resets on episode end.
        After the rollout, computes the bootstrap value for GAE.

        Args:
            state: Initial observation to start the rollout from.
        """
        dev = self.train_config.device
        t_args = (torch.float32, dev)
        env_device = getattr(self.env, "device", "cpu")
        state_tensor = torch.as_tensor(state, *t_args)
        if state_tensor.dim() == 1: state_tensor.unsqueeze(0)

        num_envs = state_tensor.shape[0]
        if self.current_episode_reward is None:
            self.current_episode_reward = torch.zeros(num_envs, device=dev)

        for _ in range(self.train_config.rollout_steps):
            self.normalizer.update(state_tensor)
            state_normalized = self.normalizer.normalize(state_tensor)
            with torch.inference_mode():
                outputs = self.agent.get_action(state_normalized)
                if str(env_device).startswith("cuda"):
                    action_input: np.ndarray | Tensor = outputs["action"]
                else:
                    action_input = outputs["action"].cpu().numpy()

            # Convention: truncate = terminated (episode naturally ended)
            #done = truncated (episode cut short by time limit)
            next_state, reward, done, truncate, _ = self.env.step(action_input)
            done_tensor = torch.as_tensor(done, *t_args)
            trunc_tensor = torch.as_tensor(truncate, *t_args)
            reward_tensor = torch.as_tensor(reward, *t_args)
            next_state = torch.as_tensor(next_state, *t_args)

            if reward_tensor.dim() == 0:
                next_state = next_state.unsqueeze(0)
                reward_tensor = reward_tensor.unsqueeze(0)
                done_tensor = done_tensor.unsqueeze(0)
                trunc_tensor = trunc_tensor.unsqueeze(0)

            self.buffer.insert(
                state = state_normalized,
                reward = reward_tensor,
                done = done_tensor,
                **outputs
            )
            self.current_episode_reward += reward_tensor
            finished = (done_tensor > 0) | (trunc_tensor > 0)

            if finished.any():
                finished_rewards = self.current_episode_reward[finished]
                self.episode_rewards.extend(finished_rewards.tolist())
                self.current_episode_reward[finished] = 0.0

            if finished.any() and not self.env.auto_reset:
                state, _ = self.env.reset()
                state_tensor = torch.as_tensor(state, *t_args)
            else:
                state_tensor = next_state

            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)

        with torch.inference_mode():
            self.normalizer.update(state_tensor)
            state_normalized = self.normalizer.normalize(state_tensor)
            next_output = self.agent.get_action(state_normalized)
        return next_output


    def _log_metrics(self, metrics: dict, step: int, use_wandb: bool, use_tb: bool):
        """Log training metrics to wandb and/or TensorBoard.

        Args:
            metrics: Dict of metric names to values (floats or Tensors).
            step: Current training step.
            use_wandb: Whether to log to Weights & Biases.
            use_tb: Whether to log to TensorBoard.
        """
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
        """Run the full training loop.

        Repeats rollout -> update_weights -> log -> clear for num_update steps.

        Args:
            use_wandb: Whether to log to Weights & Biases.
            use_tb: Whether to log to TensorBoard.
        """
        state, _ = self.env.reset()
        for step in tqdm(range(self.train_config.num_update)):
            last_output = self.rollout_phase(state)

            if self.buffer.size < self.require_buffer_size:
                raise EmptyBufferError(self.buffer.size, self.require_buffer_size)

            losses = self.update_weights(
                    self.agent,
                    self.buffer,
                    self.optimizer,
                    step,
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
