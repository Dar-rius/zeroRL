"""Training-protocol experiments C and D (PPO, exact transition budget).

Protocols (spec-mandated, never merged):

- **C** — fixed ``rollout_steps = 2048`` per env per update, so the global
  batch grows with N (``2048 * N``) and the number of updates falls
  (``total / (2048 * N)``).
- **D** — fixed *global* rollout of 2048 transitions per update, so
  ``rollout_steps = 2048 / N`` and the update count is constant
  (``total / 2048``).

Both run ``PPOProtocolConfig.total = 262_144`` measured transitions
(warmup is a collection-only concept; training starts from scratch so the
whole ``train()``/``learn()``/collector loop is the measured phase).

Alignment across the three frameworks:

- shared 64-64 Tanh trunk + heads, initialized from the same reference
  state dict (``policy_reference``);
- lr 3e-4 (Adam, eps 1e-5), LINEAR decay to 0 across the run: zeroRL's
  official default schedule, ``learning_rate`` callable for SB3, LambdaLR
  for TorchRL;
- gamma .99, gae_lambda .95, clip .2, entropy .01, value .5, epochs 10,
  minibatch 64, grad-clip 0.5 (zeroRL hardcodes 0.5 in ``ppo_func``).

Known micro-diffs (documented, not "fixed" — core algorithms stay
untouched): zeroRL's value term is ``0.5 * MSE`` (SB3/TorchRL use
``1.0 * MSE`` with coef .5), and zeroRL normalizes advantages once over
the full batch while SB3 normalizes per minibatch (TorchRL is configured
full-batch like zeroRL).

Phase semantics: ``init`` = env + policy/model construction + weight
load; ``measure`` = each framework's official training entry point
(``BaseTrain.train()``, ``PPO.learn()``, Collector construction + its
iteration loop). Each entry point performs its own first env reset inside
the measured window (ms-scale, recorded here as the one asymmetry).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from benchmarks.scalability.collection import (
    FRAMEWORKS_A,
    lifecycle,
    seeded_env_factory,
    seeded_env_fns,
)
from benchmarks.scalability.policy_reference import (
    ACT_DIM,
    HIDDEN,
    OBS_DIM,
    TorchRLSharedTrunk,
    build_reference_state_dict,
    load_into_sb3,
    load_into_zerorl,
    shared_trunk_policy_cls,
)

EXPERIMENTS_CD = ("C", "D")
DEFAULT_PPO_TOTAL = 262_144


@dataclass(frozen=True)
class PPOProtocolConfig:
    """Workload + shared hyperparameters for Experiments C/D."""

    env_id: str = "LunarLander-v3"
    total: int = DEFAULT_PPO_TOTAL
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    epochs: int = 10
    batch_size: int = 64
    grad_clip: float = 0.5
    reference_seed: int = 4242
    rollout_fixed: int = 2048      # C: steps/env/update; D: global batch

    def rollout_steps(self, experiment: str, num_envs: int) -> int:
        """Per-env steps collected per update for this protocol."""
        if experiment not in EXPERIMENTS_CD:
            raise ValueError(f"experiment must be one of {EXPERIMENTS_CD}, "
                             f"got {experiment!r}")
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if experiment == "C":
            return self.rollout_fixed
        if self.rollout_fixed % num_envs != 0:
            raise ValueError(
                f"experiment D needs rollout_fixed {self.rollout_fixed} "
                f"divisible by num_envs {num_envs}"
            )
        return self.rollout_fixed // num_envs

    def updates(self, experiment: str, num_envs: int) -> int:
        """Optimizer updates implied by the exact total budget."""
        rollout = self.rollout_steps(experiment, num_envs)
        per_update = rollout * num_envs
        if self.total % per_update != 0:
            raise ValueError(
                f"total {self.total} not divisible by the per-update budget "
                f"{per_update} (rollout {rollout} x N={num_envs})"
            )
        n = self.total // per_update
        if n < 1:
            raise ValueError(
                f"total {self.total} too small for one update of "
                f"{per_update} transitions"
            )
        return n


def _finalize_train_phases(init_s: float, meas_s: float) -> dict[str, float]:
    if min(init_s, meas_s) <= 0:
        raise RuntimeError(
            f"non-positive phase time: init={init_s} measure={meas_s}"
        )
    return {"init_seconds": init_s, "measure_seconds": meas_s}


def _return_stats(returns: list[float]) -> tuple[float | None, float | None]:
    if not returns:
        return None, None
    return float(np.mean(returns)), float(np.mean(returns[-100:]))


def _transitions_block(total: int) -> dict[str, Any]:
    return {
        "warmup_target": 0, "warmup_actual": 0,
        "measure_target": total, "measured": total, "exact": True,
    }


def _algo_config_from(cfg: PPOProtocolConfig) -> Any:
    """P5: explicit protocol -> zeroRL ``AlgoConfig`` mapping. Every PPO
    field is set from the protocol (no silent defaults); ``tau`` is
    off-policy-only and intentionally not mapped."""
    from zerorl.config import AlgoConfig

    return AlgoConfig(
        lr=cfg.lr, gamma=cfg.gamma, batch_size=cfg.batch_size,
        gae_lambda=cfg.gae_lambda, clip_eps=cfg.clip_eps,
        ent_coef=cfg.ent_coef, value_coef=cfg.vf_coef, epochs=cfg.epochs,
    )


def _lr_and_optimizer(optimizer: torch.optim.Optimizer) -> tuple[float, dict[str, Any]]:
    """P5 evidence: current lr of param group 0 + optimizer identity."""
    pg = optimizer.param_groups[0]
    return float(pg["lr"]), {
        "type": type(optimizer).__name__,
        "eps": float(pg["eps"]),
        "schedule": "linear_decay_to_zero",
    }


def _hyper_detail(cfg: PPOProtocolConfig, experiment: str, num_envs: int,
                  seeds: list[int], *, lr_start: float, lr_end: float,
                  optimizer: dict[str, Any]) -> dict[str, Any]:
    rollout = cfg.rollout_steps(experiment, num_envs)
    return {
        "rollout_steps_per_env": rollout,
        "rollout_global": rollout * num_envs,
        "updates": cfg.updates(experiment, num_envs),
        "total": cfg.total,
        "lr": cfg.lr, "gamma": cfg.gamma, "gae_lambda": cfg.gae_lambda,
        "clip_eps": cfg.clip_eps, "ent_coef": cfg.ent_coef,
        "vf_coef": cfg.vf_coef, "epochs": cfg.epochs,
        "batch_size": cfg.batch_size, "grad_clip": cfg.grad_clip,
        "lr_start": lr_start, "lr_end": lr_end, "optimizer": optimizer,
        "net_arch": [HIDDEN, HIDDEN], "reference_seed": cfg.reference_seed,
        "seeds": seeds,
    }


# ---------------------------------------------------------------------------
# zeroRL
# ---------------------------------------------------------------------------

def run_zerorl_ppo(cfg: PPOProtocolConfig, experiment: str, num_envs: int,
                   seed: int) -> dict:
    from zerorl.algorithms.ppo.easy_ppo import easy_train_ppo
    from zerorl.config import TrainConfig
    from zerorl.functions import derive_env_seeds
    from zerorl.helpers.factory import ActorCriticAgent

    rollout = cfg.rollout_steps(experiment, num_envs)
    updates = cfg.updates(experiment, num_envs)
    seeds = derive_env_seeds(seed, num_envs)

    with lifecycle() as lc:
        t0 = time.perf_counter()
        agent = ActorCriticAgent(OBS_DIM, ACT_DIM, is_discrete=True,
                                 hidden_dim=HIDDEN)
        load_into_zerorl(agent, build_reference_state_dict(cfg.reference_seed))
        tconfig = TrainConfig(
            model_name=f"bench_ppo_{experiment}",
            project_name="benchmarks",
            timestamp=cfg.total,
            rollout_steps=rollout,
            num_envs=num_envs,
        )
        tconfig.device = torch.device("cpu")
        trainer = easy_train_ppo(
            seeded_env_factory(cfg.env_id, seeds), tconfig,
            _algo_config_from(cfg),
            base_agent=agent, seed=seed,
        )
        # P6: train() closes env only on success (train.py:279); this closer
        # covers the failure path (double close is guarded by gymnasium).
        lc.add("env", trainer.env.close)
        init_s = time.perf_counter() - t0
        lr_start, opt_ev = _lr_and_optimizer(trainer.optimizer)

        t = time.perf_counter()
        trainer.train()
        meas_s = time.perf_counter() - t
        lr_end, _ = _lr_and_optimizer(trainer.optimizer)

        if trainer.config.num_update != updates:
            raise RuntimeError(
                f"trainer ran {trainer.config.num_update} updates, expected {updates}"
            )
        measured = updates * rollout * num_envs
        if measured != cfg.total:
            raise RuntimeError(f"budget mismatch: {measured} != {cfg.total}")
        returns = [float(r) for r in trainer.episode_rewards]
        mean_r, last100 = _return_stats(returns)
        art = {
            "phases": _finalize_train_phases(init_s, meas_s),
            "transitions": _transitions_block(cfg.total),
            "metrics": {
                "updates": updates, "episodes": len(returns),
                "mean_return": mean_r, "mean_return_last100": last100,
                "throughput": cfg.total / meas_s,
            },
            "episode_returns": returns,
            "config_detail": _hyper_detail(cfg, experiment, num_envs, seeds,
                                           lr_start=lr_start, lr_end=lr_end,
                                           optimizer=opt_ev),
        }
    lc.attach(art)
    return art


# ---------------------------------------------------------------------------
# SB3
# ---------------------------------------------------------------------------

def run_sb3_ppo(cfg: PPOProtocolConfig, experiment: str, num_envs: int,
                seed: int) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv
    from zerorl.functions import derive_env_seeds

    rollout = cfg.rollout_steps(experiment, num_envs)
    updates = cfg.updates(experiment, num_envs)
    seeds = derive_env_seeds(seed, num_envs)

    with lifecycle() as lc:
        t0 = time.perf_counter()
        set_random_seed(seed)
        vec = DummyVecEnv(seeded_env_fns(cfg.env_id, seeds))
        lc.add("vec_env", vec.close)
        model = PPO(
            policy=shared_trunk_policy_cls(),
            env=vec,
            learning_rate=lambda progress_remaining: cfg.lr * progress_remaining,
            n_steps=rollout,
            batch_size=cfg.batch_size,
            n_epochs=cfg.epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_eps,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.grad_clip,
            seed=seed,
            verbose=0,
            device="cpu",
        )
        load_into_sb3(model.policy, build_reference_state_dict(cfg.reference_seed))
        init_s = time.perf_counter() - t0
        lr_start, opt_ev = _lr_and_optimizer(model.policy.optimizer)

        state: dict[str, Any] = {
            "steps": 0, "ep_ret": np.zeros(num_envs), "episodes": [],
        }

        def callback(locals_: dict, _globals: dict) -> bool:
            dones = locals_["dones"]
            rews = locals_["rewards"]
            state["steps"] += num_envs
            state["ep_ret"] += rews
            for i, done in enumerate(dones):
                if done:
                    state["episodes"].append(float(state["ep_ret"][i]))
                    state["ep_ret"][i] = 0.0
            return True

        t = time.perf_counter()
        model.learn(
            total_timesteps=cfg.total, reset_num_timesteps=True,
            callback=callback, progress_bar=False,
        )
        meas_s = time.perf_counter() - t
        lr_end, _ = _lr_and_optimizer(model.policy.optimizer)

        if state["steps"] != cfg.total:
            raise RuntimeError(
                f"callback counted {state['steps']} transitions, expected {cfg.total}"
            )
        ran_updates = model.num_timesteps // (rollout * num_envs)
        if ran_updates != updates:
            raise RuntimeError(f"SB3 ran {ran_updates} updates, expected {updates}")
        returns = state["episodes"]
        mean_r, last100 = _return_stats(returns)
        art = {
            "phases": _finalize_train_phases(init_s, meas_s),
            "transitions": _transitions_block(cfg.total),
            "metrics": {
                "updates": updates, "episodes": len(returns),
                "mean_return": mean_r, "mean_return_last100": last100,
                "throughput": cfg.total / meas_s,
            },
            "episode_returns": returns,
            "config_detail": _hyper_detail(cfg, experiment, num_envs, seeds,
                                           lr_start=lr_start, lr_end=lr_end,
                                           optimizer=opt_ev),
        }
    lc.attach(art)
    return art


# ---------------------------------------------------------------------------
# TorchRL
# ---------------------------------------------------------------------------

def _dedup_params(*modules: torch.nn.Module) -> list[torch.nn.Parameter]:
    seen: set[int] = set()
    out: list[torch.nn.Parameter] = []
    for module in modules:
        for p in module.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
    return out


def build_torchrl_ppo_wiring(trunk: TorchRLSharedTrunk,
                             cfg: PPOProtocolConfig,
                             updates: int) -> dict[str, Any]:
    """Single-shared-trunk PPO wiring for TorchRL (shared-trunk fix).

    One trunk execution per actor-critic pass:

    - collector recomputes ``td["features"]`` once per step (no grad);
    - the loss actor and loss critic are HEAD-ONLY consumers of
      ``td["features"]``; ``_torchrl_train_minibatch`` recomputes the
      features WITH GRAD exactly once per minibatch, so actor and critic
      share the same fresh tensor (zeroRL semantics: one forward per
      minibatch), instead of each module re-running the trunk.
    - GAE still evaluates the critic from ``td["features"]`` /
      ``td["next","features"]`` (targets, no grad; next-features injected
      per batch in the runner).
    """
    from tensordict.nn import TensorDictModule, TensorDictSequential, composite_lp_aggregate
    from torchrl.modules import ProbabilisticActor, ValueOperator
    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value import GAE

    lp_key = ("sample_log_prob" if composite_lp_aggregate()
              else "action_log_prob")
    feat_td = TensorDictModule(trunk.trunk, in_keys=["observation"],
                               out_keys=["features"])
    collect_logits = TensorDictModule(trunk.actor_head,
                                      in_keys=["features"],
                                      out_keys=["logits"])
    loss_logits = TensorDictModule(trunk.actor_head,
                                   in_keys=["features"],
                                   out_keys=["logits"])
    collector_policy = ProbabilisticActor(
        module=TensorDictSequential(feat_td, collect_logits),
        in_keys=["logits"], out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True, log_prob_key=lp_key,
    )
    loss_actor = ProbabilisticActor(
        module=loss_logits,
        in_keys=["logits"], out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True, log_prob_key=lp_key,
    )
    critic = ValueOperator(module=trunk.critic_head,
                           in_keys=["features"],
                           out_keys=["state_value"])
    advantage_module = GAE(gamma=cfg.gamma, lmbda=cfg.gae_lambda,
                           value_network=critic)
    loss_module = ClipPPOLoss(
        actor_network=loss_actor, critic_network=critic,
        clip_epsilon=cfg.clip_eps, entropy_coeff=cfg.ent_coef,
        critic_coeff=cfg.vf_coef, loss_critic_type="l2",
    )
    params = _dedup_params(trunk, collector_policy, loss_actor, critic)
    optimizer = torch.optim.Adam(params, lr=cfg.lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0 - step / updates
    )
    return {
        "lp_key": lp_key,
        "collector_policy": collector_policy,
        "loss_actor": loss_actor,
        "critic": critic,
        "advantage_module": advantage_module,
        "loss_module": loss_module,
        "params": params,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


def _torchrl_train_minibatch(trunk: TorchRLSharedTrunk, loss_module: Any,
                             td: Any, optimizer: torch.optim.Optimizer,
                             params: list[torch.Parameter],
                             grad_clip: float) -> None:
    """One PPO minibatch = ONE shared-trunk pass (with grad) + loss step."""
    td["features"] = trunk.trunk(td["observation"])
    out = loss_module(td)
    loss = (out["loss_objective"] + out["loss_critic"]
            + out["loss_entropy"])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, grad_clip)
    optimizer.step()


def run_torchrl_ppo(cfg: PPOProtocolConfig, experiment: str, num_envs: int,
                    seed: int) -> dict:
    from gymnasium.vector import AutoresetMode, SyncVectorEnv
    from torchrl.collectors import Collector
    from torchrl.envs.libs.gym import GymWrapper
    from zerorl.functions import set_seed

    rollout = cfg.rollout_steps(experiment, num_envs)
    updates = cfg.updates(experiment, num_envs)
    fpb = rollout * num_envs
    seeds = set_seed(seed, num_envs)

    with lifecycle() as lc:
        t0 = time.perf_counter()
        venv = SyncVectorEnv(
            seeded_env_fns(cfg.env_id, seeds),
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
        lc.add("venv", venv.close)
        wrapper = GymWrapper(venv, categorical_action_encoding=True)
        lc.add("wrapper", wrapper.close)
        # GymWrapper's info-reader check consumed the canonical first reset;
        # re-arm so the Collector's carrier reset (built below) IS canonical.
        from benchmarks.scalability.collection import rearm_first_reset

        rearm_first_reset(venv)
        trunk = TorchRLSharedTrunk(build_reference_state_dict(cfg.reference_seed))
        wiring = build_torchrl_ppo_wiring(trunk, cfg, updates)
        lr_start, opt_ev = _lr_and_optimizer(wiring["optimizer"])
        # P5: collector construction (env carrier reset + spec wiring) is
        # setup work — timed under init, never inside the measure timer.
        collector = Collector(
            wrapper, wiring["collector_policy"], backend="direct",
            trust_policy=True,
            auto_register_policy_transforms=False,
            frames_per_batch=fpb, total_frames=cfg.total,
        )
        lc.add("collector", collector.shutdown)
        init_s = time.perf_counter() - t0

        t = time.perf_counter()
        n_updates = 0
        collected = 0
        ep_ret = torch.zeros(num_envs)
        returns: list[float] = []
        for td in collector:
            collected += int(td.batch_size.numel())
            rews = td["next", "reward"]
            done = td["next", "done"]
            if done.dim() == rews.dim() - 1:
                done = done.unsqueeze(-1)
            for step in range(rews.shape[1]):
                ep_ret += rews[:, step].squeeze(-1)
                finished = done[:, step].squeeze(-1).bool()
                if finished.any():
                    returns.extend(ep_ret[finished].tolist())
                    ep_ret[finished] = 0.0

            with torch.no_grad():
                # GAE bootstraps from td["next"], but the collector only
                # computed features for the CURRENT observation; one extra
                # no-grad trunk call per batch supplies the next-state
                # features the head-only critic needs.
                td["next", "features"] = trunk.trunk(
                    td["next", "observation"])
                wiring["advantage_module"](td)
            adv = td["advantage"]
            td["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)

            rows = td.reshape(-1)
            n_rows = rows.shape[0]
            for _ in range(cfg.epochs):
                perm = torch.randperm(n_rows)
                for start in range(0, n_rows, cfg.batch_size):
                    _torchrl_train_minibatch(
                        trunk, wiring["loss_module"],
                        rows[perm[start:start + cfg.batch_size]],
                        wiring["optimizer"], wiring["params"], cfg.grad_clip,
                    )
            wiring["scheduler"].step()
            n_updates += 1
        meas_s = time.perf_counter() - t
        lr_end, _ = _lr_and_optimizer(wiring["optimizer"])

        if collected != cfg.total:
            raise RuntimeError(f"collected {collected} transitions, "
                               f"expected {cfg.total}")
        if n_updates != updates:
            raise RuntimeError(f"TorchRL ran {n_updates} updates, "
                               f"expected {updates}")
        mean_r, last100 = _return_stats(returns)
        art = {
            "phases": _finalize_train_phases(init_s, meas_s),
            "transitions": _transitions_block(cfg.total),
            "metrics": {
                "updates": updates, "episodes": len(returns),
                "mean_return": mean_r, "mean_return_last100": last100,
                "throughput": cfg.total / meas_s,
            },
            "episode_returns": returns,
            "config_detail": _hyper_detail(cfg, experiment, num_envs, seeds,
                                           lr_start=lr_start, lr_end=lr_end,
                                           optimizer=opt_ev),
        }
    lc.attach(art)
    return art


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

def run_ppo_training(framework: str, experiment: str, num_envs: int,
                     seed: int, cfg: PPOProtocolConfig | None = None) -> dict:
    """Run one C/D training pass and return the artifact dict."""
    if experiment not in EXPERIMENTS_CD:
        raise ValueError(f"experiment must be one of {EXPERIMENTS_CD}, "
                         f"got {experiment!r}")
    if framework not in FRAMEWORKS_A:
        raise ValueError(f"unknown framework {framework!r} "
                         f"(one of {FRAMEWORKS_A})")
    cfg = cfg if cfg is not None else PPOProtocolConfig()
    if framework == "zeroRL":
        return run_zerorl_ppo(cfg, experiment, num_envs, seed)
    if framework == "SB3":
        return run_sb3_ppo(cfg, experiment, num_envs, seed)
    return run_torchrl_ppo(cfg, experiment, num_envs, seed)
