"""Training loop for RL agents.

Provides BaseTrain, which orchestrates the rollout-update cycle:
collect experience, compute GAE, run the update_weights callable,
and repeat.
"""

import sys
import contextlib
import numpy as np
import torch
import gymnasium as gym
from dataclasses import asdict
from tqdm import tqdm
from typing import Callable, Any
from torch import Tensor
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from zerorl.helpers.agent import BaseAgent
from zerorl.buffer import Buffer
from zerorl.config import TrainConfig, AlgoConfig
from zerorl.processing import NormMeanStd
from zerorl.errors import EmptyBufferError, assert_agent_contract
from zerorl.logger import PhaseProfiler, PhaseMetrics, create_logger
from zerorl.functions import (
        vectorize_env,
        processing_state,
        parse_dict_to_tensor,
        env_step,
        save_checkpoints,
        try_agent,
        set_seed)


class BaseTrain:
    """Training loop coordinating agent, environment, and weight updates.

    Orchestrates the rollout-update cycle: collect experience via
    rollout_phase(), compute GAE, run the update_weights callable,
    and repeat. Handles observation normalization, logging, and
    model saving.
    """
    def __init__(self,
                 agent: BaseAgent,
                 env: Any,
                 buffer: Buffer,
                 update_weights: Callable,
                 config: TrainConfig,
                 algo_config: AlgoConfig,
                 optimizer: optim.Optimizer | None = None,
                 schedule_func: Callable[[int], float] | None = None,
                 seed: int = 22,
                 render_mode: str | None = None,
                 require_buffer_size: int = 10):
        """Initialize the training loop.

        Args:
            agent: Neural network policy to train.
            env: Environment to collect experience from.
            buffer: Pre-allocated buffer for rollout data.
            update_weights: Callable that computes the weight update from
                collected rollout data. Signature:
                (agent, buffer, scheduler, optimizer, last_output, algo_config) -> dict[str, Tensor].
            config: Training configuration (device, paths, hyperparams).
            algo_config: Algorithm hyperparameters (passed to update_weights).
            optimizer: Optional optimizer. If None, creates Adam with algo_config.lr.
            schedule_func: Custom LR schedule function (overrides default linear decay).
                Takes current_step and returns a learning rate multiplier.
            render_mode: Render mode for the environment.
            require_buffer_size: Minimum buffer size before an update is allowed.
        """
        super().__init__()
        torch.set_float32_matmul_precision('high')
        self.config = config
        self.num_envs = self.config.num_envs
        self.agent = agent.to(self.config.device)
        assert_agent_contract(self.agent,
                        {"get_action": "Your agent should have the method `get_action`"})
        self.env = env
        if not isinstance(self.env, gym.vector.VectorEnv) and not getattr(self.env, "auto_reset", False):
            self.env = vectorize_env(self.env, num_envs = self.num_envs, render_mode = render_mode)
        self.state = Tensor()
        self.buffer = buffer
        self.update_weights = update_weights
        self.algo_config = algo_config
        self.optimizer: optim.Optimizer

        if optimizer is None:
            lr = getattr(self.algo_config, 'lr', 3e-4)
            self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)
        else:
            self.optimizer = optimizer

        obs_ = getattr(env, "single_observation_space", env.observation_space)
        obs_shape = obs_.shape

        if obs_shape is None:
            raise ValueError("NormMeanStd requires environment with a defined observation shape")

        if schedule_func is None: schedule_func = lambda current_step: 1.0 - (current_step / self.config.num_update)
        self.scheduler = LambdaLR(self.optimizer, schedule_func)
        self.seed = set_seed(seed, self.num_envs)
        self.require_buffer_size = require_buffer_size
        self.normalizer = NormMeanStd(obs_shape, config.device) if self.config.normalize else None
        self.current_episode_reward: Tensor | None = None
        self.episode_rewards: list[float] = []
        self.env_device = getattr(self.env, "device", "cpu")
        self.debug = self.config.debug
        self.device = self.config.device

        if self.debug:
            sys.stderr.write("\033[96mzeroRL DEBUG MODE ENABLED. torch.compile is disabled.\033[0m\n")
            torch.autograd.set_detect_anomaly(True)
            obs_space = getattr(self.env, "single_observation_space", self.env.observation_space)
            act_space = getattr(self.env, "single_action_space", self.env.action_space)
                        
            from zerorl.debug import check_tensor, check_shape, check_reward_scale
            def _env_hook(data_: dict[str, Tensor], step: int):
                for k, v in data_.items():
                    check_tensor(v, k , step)
                    if k in ["state", "next_state"]:
                        check_shape(v, (self.num_envs, *obs_space.shape), k, step)
                    elif k == "action":
                        check_shape(v, (self.num_envs, *act_space.shape), k, step)
                    else:
                        check_shape(v, (self.num_envs,), k, step)

                check_reward_scale(data_["reward"], step)

            self._hook_env_check_ = _env_hook
        else:
            self._hook_env_check_ = lambda data_, step : None

    def rollout_phase(self) -> dict[str, Tensor] | None:
        """Collect experience by running the agent in the environment.

        Stores each transition in the buffer and resets on episode end.
        After the rollout, computes the bootstrap value for GAE.

        Uses self.state internally as the starting observation.
        """
        state = self.state
        if self.current_episode_reward is None:
            self.current_episode_reward = torch.zeros(self.num_envs, device=self.device)

        for i in range(self.config.rollout_steps):
            state_processed = processing_state(state, self.normalizer, device = self.device)
            outputs = env_step(self.env, self.agent, state_processed)
            done = outputs["terminated"] | outputs["truncated"]
            outputs["terminated"] = done
            outputs = parse_dict_to_tensor(outputs, self.device)
            outputs.pop("info")
            self._hook_env_check_(outputs, i)
            next_state = outputs.pop("next_state")
            self.buffer.insert(**outputs)
            self.current_episode_reward += outputs["reward"]
            finished = (outputs["terminated"] > 0) | (outputs["truncated"] > 0)

            if finished.any():
                finished_rewards = self.current_episode_reward[finished]
                self.episode_rewards.extend(finished_rewards.tolist())
                self.current_episode_reward[finished] = 0.0

            state = next_state

        if "value" in self.buffer.data:
            with torch.inference_mode():
                state_processed = processing_state(state, self.normalizer, update=False, device = self.device)
                next_output = self.agent.get_action(state_processed) #type: ignore[operator]
        else:
            next_output = None
        self.state = state
        return next_output

    def _log_profile_metrics(self, step: int, metrics: PhaseMetrics):
        """Print profiling metrics for the current training step to stderr."""
        sys.stderr.write(
                f"\n\033[94m[Profile] Step {step} | FPS: {metrics.fps:.0f} | "
                f"Rollout: {metrics.rollout_ms:.1f}ms | Update: {metrics.update_ms:1f}ms |"
                f"VRAM: {metrics.vram_allocated_gb:.2f}GB (Peak: {metrics.vram_peak_gb:.2f}GB) | "
                f"RAM: {metrics.ram_mb:.0f}MB\033[0m\n"
                )

    def train(self, *, save_model: bool = False, use_wandb: bool = False, use_tb: bool = False):
        """Run the full training loop.

        Repeats rollout -> update_weights -> log -> clear for num_update steps.

        Args:
            save_model: Whether to save the agent weights after training.
            use_wandb: Whether to log to Weights & Biases.
            use_tb: Whether to log to TensorBoard.
        """
        is_profile = self.config.profile
        is_cuda = True if str(self.device).startswith("cuda") else False
        profiler = PhaseProfiler(self.config, is_cuda = is_cuda)
        log = create_logger(self.config, self.algo_config, use_wandb=use_wandb, use_tb=use_tb)
        state, _ = self.env.reset(seed = self.seed)
        self.state = torch.as_tensor(state, dtype=torch.float32, device=self.config.device)

        for step in tqdm(range(self.config.num_update)):
            if is_profile: profiler.start_phase()

            with profiler.track("rollout") if is_profile else contextlib.nullcontext():
                last_output = self.rollout_phase()

            if self.buffer.size < self.require_buffer_size: raise EmptyBufferError(self.buffer.size, self.require_buffer_size)

            if self.debug:
                self.algo_config._debug_mode = True #type: ignore
                weights_before = {k: v.clone() for k, v in self.agent.state_dict().items()}

            with profiler.track("update") if is_profile else contextlib.nullcontext():
                losses = self.update_weights(
                                agent = self.agent,
                                buffer = self.buffer,
                                scheduler = self.scheduler,
                                optimizer = self.optimizer,
                                last_output = last_output,
                                algo_config = self.algo_config)


            if self.debug:
                weights_after = self.agent.state_dict()
                changed = any(not torch.allclose(weights_before[k], weights_after[k]) for k in weights_before)
                if not changed:
                    sys.stderr.write(
                            "\n\033[93m [DEBUG ALERT] Model weights did not change after update_weights()!\n"
                            "Did you forget to call `optimizer.step()` in your update function ?\033[0m\n"
                            )
               
            if is_profile:
                profile_data = profiler.end_phase()
                self._log_profile_metrics(step, profile_data)

            if len(self.episode_rewards) > 0:
                recent = self.episode_rewards[-10:]
                mean_reward = float(np.mean(recent))
            else:
                mean_reward = 0.0

            metrics = {"train/mean_episode_reward": mean_reward,
                        "train/learning_rate": self.optimizer.param_groups[0]['lr']}
            if use_wandb and is_profile:
                for k, v in asdict(profile_data).items(): metrics[f"profile/{k}"] = v
            for k, v in losses.items(): metrics[f"train/{k}"] = v
            log(metrics, step)
            self.buffer.clear()

        self.env.close()
        log.close()
        if save_model: self.save()

    def try_agent(self, iterations: int = 1, gif_path: str | None = None):
        """Evaluate the agent and save a GIF."""
        try_agent(self.env, self.agent, self.config, normalizer = self.normalizer, iterations= iterations, gif_path = gif_path)

    def save(self):
        """Save agent weights and normalizer state to disk."""
        save_checkpoints(self.agent, self.config.model_path, self.normalizer)
