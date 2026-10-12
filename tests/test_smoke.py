"""Minimum-train smoke tests: tiny end-to-end trains per integration path.

Modeled on examples/pendulum.py (T1), the BaseTrain contract (T2), and
examples/immediate_mode.py (T3). Each smoke runs the smallest useful train
(few env steps, 1-2 PPO updates) and asserts the full pipeline — rollout,
GAE, PPO update, logging, checkpointing — produces finite metrics and
updated weights. CPU always; CUDA legs run on the Jetson CI GPU job.
"""

import math
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import pytest
import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from zerorl.algorithms.ppo import easy_train_ppo, gae_compute, ppo_func
from zerorl.buffer import Buffer
from zerorl.config import AlgoConfig, TrainConfig
from zerorl.functions import (
    deterministic_action,
    parse_dict_to_tensor,
    processing_state,
    set_seed,
    to_env_action,
    vectorize_env,
)
from zerorl.helpers.env import BaseEnv
from zerorl.helpers.factory import ActorCriticAgent, get_actor_critic_buffer
from zerorl.helpers.train import BaseTrain
from zerorl.logger import create_logger


_DEVICE_PARAMS = [
    pytest.param("cpu", id="cpu"),
    pytest.param("cuda", marks=pytest.mark.gpu, id="cuda"),
]


@pytest.fixture(params=_DEVICE_PARAMS)
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine")
    return torch.device(request.param)


class _CartPole(BaseEnv):
    """Single CartPole; batches actions down to the scalar CartPole expects."""

    def __init__(self):
        super().__init__()
        self._env = gym.make("CartPole-v1")
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        if hasattr(action, "shape") and len(action.shape) > 0:
            action = action.item()
        return self._env.step(action)

    def close(self):
        self._env.close()


class TestSmokeEasyTrainPpo:
    """T1: one-call easy_train_ppo — train/eval metrics + checkpoint."""

    def test_min_train_logs_finite_metrics_and_saves(self, tmp_path: Path, device: torch.device) -> None:
        cfg = TrainConfig(model_name="smoke", model_save_path=str(tmp_path),
                          project_name="smoke-easy")
        cfg.device = device
        cfg.rollout_steps = 32
        cfg.num_envs = 2
        cfg.timestamp = 128  # -> num_update = 2
        trainer = easy_train_ppo("Pendulum-v1", cfg, AlgoConfig(epochs=1),
                                 evals=1, seed=7)
        with patch("wandb.init"), patch("wandb.log") as mock_log:
            trainer.train(save_model=True, use_wandb=True, use_tb=False)

        logged = [c.args[0] for c in mock_log.call_args_list]
        keys_seen: set[str] = set()
        for metrics in logged:
            keys_seen.update(metrics)
        for key in ("train/loss", "train/mean_episode_reward",
                    "evals/mean_return", "evals/std_return"):
            assert key in keys_seen, f"missing {key} in {keys_seen}"
        for metrics in logged:
            for k, v in metrics.items():
                assert math.isfinite(float(v)), f"{k} not finite: {v}"

        ckpt_path = Path(cfg.model_path)
        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert "agent_state_dict" in ckpt
        fresh = ActorCriticAgent(3, 1, is_discrete=False)
        fresh.load_state_dict(ckpt["agent_state_dict"])


class TestSmokeBaseTrainRealPpo:
    """T2: BaseTrain wiring with real gae_compute + ppo_func (no mocks)."""

    def test_real_gae_and_ppo_update(self, tmp_path: Path, device: torch.device) -> None:
        rollout, num_envs = 32, 1
        env = _CartPole()
        cfg = TrainConfig(model_name="smoke-bt", model_save_path=str(tmp_path),
                          project_name="smoke-bt")
        cfg.device = device
        cfg.rollout_steps = rollout
        cfg.num_envs = num_envs
        cfg.timestamp = rollout * 3  # -> num_update = 3
        algo = AlgoConfig(epochs=1)
        agent = ActorCriticAgent(4, 2, is_discrete=True).to(device)
        buf = get_actor_critic_buffer((4,), (), rollout, num_envs, device)

        captured: list[dict[str, torch.Tensor]] = []

        def real_update(agent, buffer, scheduler, optimizer, algo_config):
            data = buffer.get_all()
            gae_compute(data["reward"], data["value"], data["final_value"],
                        data["terminated"], data["truncated"], buffer, algo_config)
            out = ppo_func(agent, optimizer, buffer, algo_config, scheduler)
            captured.append(out)
            return out

        trainer = BaseTrain(agent, env, buf, real_update, cfg, algo,
                            require_buffer_size=16)
        before = {k: v.clone() for k, v in agent.state_dict().items()}
        trainer.train(use_wandb=False, use_tb=False)

        assert len(captured) == cfg.num_update
        for losses in captured:
            assert set(losses) == {"loss", "policy_loss", "value_loss", "entropy_loss"}
            for v in losses.values():
                assert torch.isfinite(v).all(), f"non-finite loss: {losses}"
        after = agent.state_dict()
        assert any(not torch.allclose(before[k], after[k]) for k in before)
        assert len(trainer.episode_rewards) > 0


class TestSmokeImmediatePrimitives:
    """T3: manual immediate-mode loop straight from examples/immediate_mode.py."""

    def test_manual_loop_runs_and_updates(self, tmp_path: Path, device: torch.device) -> None:
        num_envs, rollout = 2, 32
        cfg = TrainConfig(model_name="smoke-imm", model_save_path=str(tmp_path),
                          project_name="smoke-imm")
        cfg.device = device
        cfg.num_envs = num_envs
        cfg.rollout_steps = rollout
        cfg.timestamp = rollout * num_envs * 2  # -> num_update = 2
        algo = AlgoConfig(epochs=1)
        seeds = set_seed(7, num_envs)
        env = vectorize_env("CartPole-v1", num_envs=num_envs)
        obs_shape = env.single_observation_space.shape
        act_n = env.single_action_space.n
        agent = ActorCriticAgent(obs_shape[-1], act_n, is_discrete=True).to(device)
        buffer = Buffer(
            capacity=rollout, num_envs=num_envs,
            schema={"state": obs_shape, "action": (), "reward": (),
                    "terminated": (), "truncated": (), "entropy": (),
                    "value": (), "log_prob": (), "final_value": (),
                    "return": (), "advantage": ()},
            device=device)
        optimizer = optim.Adam(agent.parameters(), lr=algo.lr, eps=1e-5)
        scheduler = LambdaLR(optimizer, lambda s: 1.0 - s / max(cfg.num_update, 1))
        log = create_logger(cfg, algo, use_wandb=False, use_tb=False)

        reward_tensor = torch.zeros(num_envs, device=device)
        state, _ = env.reset(seed=seeds)
        episode_rewards: list[float] = []
        all_losses: dict[str, torch.Tensor] = {}

        for step in range(cfg.num_update):
            for t in range(rollout):
                state_processed = processing_state(state, device=device)
                with torch.inference_mode():
                    outputs = agent.get_action(state_processed)
                action = to_env_action(outputs["action"], env)
                next_state, reward, terminated, truncated, _ = env.step(action)
                outputs_final = parse_dict_to_tensor(
                    {"next_state": next_state, "reward": reward,
                     "terminated": terminated, "truncated": truncated, **outputs},
                    device)
                next_state = outputs_final.pop("next_state")
                is_done = (outputs_final["terminated"] > 0) | (outputs_final["truncated"] > 0)
                if t == rollout - 1:
                    with torch.inference_mode():
                        last_val = agent.get_action(
                            processing_state(next_state, update=False, device=device)
                        )["value"]
                    final_value = torch.where(is_done, torch.zeros_like(last_val), last_val)
                else:
                    final_value = torch.zeros(num_envs, device=device)
                buffer.insert(state=state_processed, final_value=final_value,
                              **outputs_final)
                reward_tensor += outputs_final["reward"]
                if is_done.any():
                    episode_rewards.extend(reward_tensor[is_done].tolist())
                    reward_tensor[is_done] = 0.0
                state = next_state

            data = buffer.get_all()
            gae_compute(data["reward"], data["value"], data["final_value"],
                        data["terminated"], data["truncated"], buffer, algo)
            all_losses = ppo_func(agent, optimizer, buffer, algo, scheduler)
            log({f"train/{k}": float(v) for k, v in all_losses.items()}, step)
            buffer.clear()

        env.close()
        log.close()

        assert set(all_losses) == {"loss", "policy_loss", "value_loss", "entropy_loss"}
        for v in all_losses.values():
            assert torch.isfinite(v).all()
        assert buffer.size == 0
        assert len(episode_rewards) > 0
        with torch.inference_mode():
            greedy = deterministic_action(agent, torch.zeros(1, obs_shape[-1], device=device))
        assert greedy.shape == (1,)
        assert int(greedy.item()) in range(act_n)
