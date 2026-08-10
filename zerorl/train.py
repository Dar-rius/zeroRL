"""Training loop for RL agents.

Provides BaseTrain, which orchestrates the rollout-update cycle:
collect experience, copute GAE, run the update_weights callable,
and repeat.
"""

import os
import sys
import time
import resource
import numpy as np
import torch
from dataclasses import dataclass, asdict
import wandb
from tqdm import tqdm
from typing import Callable
from torch import Tensor
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from zerorl.agent import BaseAgent
from zerorl.common import Buffer
from zerorl.config import TrainConfig, AlgoConfig
from zerorl.env import BaseEnv
from zerorl.processing import NormMeanStd
from zerorl.errors import EmptyBufferError, assert_agent_contract


#Profiler Metric
@dataclass
class ProfileMetrics:
    fps: float
    rollout_ms: float
    update_ms: float
    vram_allocated_gb: float
    vram_peak_gb: float
    ram_mb: float


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
                 update_weights: Callable[[BaseAgent, Buffer, LambdaLR,
                                           optim.Optimizer, int, dict[str,Tensor],
                                           AlgoConfig | None], dict[str, Tensor]],
                 train_config: TrainConfig,
                 algo_config: AlgoConfig | None = None,
                 optimizer: optim.Optimizer | None = None,
                 schedule_func: Callable[[int], float] | None = None,
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
        assert_agent_contract(self.agent,
                        {"get_action": "Your agent should have the method `get_action`"})
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
        if obs_shape is None:
            raise ValueError("NormMeanStd requires environment with a defined observation shape")

        if schedule_func is None:
            schedule_func = lambda current_step: 1.0 - (current_step / self.train_config.num_update)
        self.scheduler = LambdaLR(self.optimizer, schedule_func)

        self.require_buffer_size = require_buffer_size
        self.normalizer = NormMeanStd(obs_shape, train_config.device)
        tb_log_dir = os.path.join(self.train_config.model_save_path, "tensorboard", self.train_config.model_name)
        self.tb_writer = SummaryWriter(tb_log_dir)
        self.current_episode_reward: Tensor | None = None
        self.episode_rewards: list[float] = []


    def rollout_phase(self, state: np.ndarray):
        """Collect experience by running the agent in the environment.

        Stores each transition in the buffer and resets on episode end.
        After the rollout, computes the bootstrap value for GAE.

        Args:
            state: Initial observation to start the rollout from.
        """
        dev = self.train_config.device
        env_device = getattr(self.env, "device", "cpu")
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=dev)
        if state_tensor.dim() == 1: state_tensor = state_tensor.unsqueeze(0)

        num_envs = state_tensor.shape[0]
        if self.current_episode_reward is None:
            self.current_episode_reward = torch.zeros(num_envs, device=dev)

        for _ in range(self.train_config.rollout_steps):
            self.normalizer.update(state_tensor)
            state_normalized = self.normalizer.normalize(state_tensor)
            with torch.inference_mode():
                outputs = self.agent.get_action(state_normalized) #type: ignore[operator]
                outputs["value"] = outputs["value"].squeeze(-1)
                if str(env_device).startswith("cuda"):
                    action_input: np.ndarray | Tensor = outputs["action"]
                else:
                    action_input = outputs["action"].cpu().numpy()

            # Convention: truncate = terminated (episode naturally ended)
            #done = truncated (episode cut short by time limit)
            next_state, reward, done, truncate, _ = self.env.step(action_input)
            done_tensor = torch.as_tensor(done, dtype=torch.float32, device=dev)
            trunc_tensor = torch.as_tensor(truncate, dtype=torch.float32, device=dev)
            reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=dev)
            next_state = torch.as_tensor(next_state, dtype=torch.float32, device=dev)

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
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=dev)
            else:
                state_tensor = next_state

            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)

        with torch.inference_mode():
            self.normalizer.update(state_tensor)
            state_normalized = self.normalizer.normalize(state_tensor)
            next_output = self.agent.get_action(state_normalized) #type: ignore[operator]
        return next_output

    
    #Profiler display
    def _log_profile_metrics(self, step: int, metrics: ProfileMetrics):
        sys.stderr.write(
                f"\033[94m[Profile] Step {step} | FPS: {metrics.fps:.0f} | "
                f"Rollout: {metrics.rollout_ms:.1f}ms | Update: {metrics.update_ms:1f}ms |"
                f"VRAM: {metrics.vram_allocated_gb:.2f}GB (Peak: {metrics.vram_peak_gb:.2f}GB) | "
                f"RAM: {metrics.ram_mb:.0f}MB\033[0m\n"
                )


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



    def train(self, *, use_wandb: bool = False, use_tb: bool = False):
        """Run the full training loop.

        Repeats rollout -> update_weights -> log -> clear for num_update steps.

        Args:
            use_wandb: Whether to log to Weights & Biases.
            use_tb: Whether to log to TensorBoard.
        """
        is_profile = self.train_config.profile
        is_cuda = self.train_config.device ==  torch.device("cuda") and torch.cuda.is_available()
        sync = torch.cuda.synchronize
        if is_profile: sys.stderr.write("\033[96mZeroRL Profiler Enabled (TIME & VRAM).\033[0m\n")
        state, _ = self.env.reset()
        for step in tqdm(range(self.train_config.num_update)):
            if is_profile:
                if is_cuda :
                    sync()
                    torch.cuda.reset_peak_memory_stats()
                t_start = time.perf_counter()

            last_output = self.rollout_phase(state)

            if is_profile:
                if is_cuda: sync()
                t_rollout = time.perf_counter()

            if self.buffer.size < self.require_buffer_size:
                raise EmptyBufferError(self.buffer.size, self.require_buffer_size)
            
            losses = self.update_weights(
                    self.agent,
                    self.buffer,
                    self.scheduler,
                    self.optimizer,
                    step,
                    last_output,
                    self.algo_config
                    )
               
            if is_profile:
                if is_cuda: sync()
                t_end = time.perf_counter()
                ram_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ram_mb = ram_kb /1024 if ram_kb > 0 else 0.0
                profile_data = ProfileMetrics(
                    fps= (self.train_config.rollout_steps * self.train_config.num_envs) / (t_end - t_start),
                    rollout_ms = (t_rollout - t_start) * 1000,
                    update_ms = (t_end - t_rollout) * 1000,
                    vram_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3) if is_cuda else 0.0,
                    vram_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if is_cuda else 0.0,
                    ram_mb = ram_mb
                    )
                self._log_profile_metrics(step, profile_data)

            if len(self.episode_rewards) > 0:
                recent = self.episode_rewards[-10:]
                mean_reward = float(np.mean(recent))
            else:
                mean_reward = 0.0

            metrics = {
                "mean_episode_reward": mean_reward,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }
            if use_wandb and is_profile:
                wandb.log({f"profile/{k}": v for k, v in asdict(profile_data).items()})
            for k, v in losses.items(): metrics[k] = v
            self._log_metrics(metrics, step, use_wandb, use_tb)
            self.buffer.clear()


    def save_model(self):
        """Save agent weights to the path in train_config.model_path."""
        os.makedirs(os.path.dirname(self.train_config.model_path), exist_ok=True)
        torch.save(self.agent.state_dict(), self.train_config.model_path)
