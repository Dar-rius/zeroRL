"""Experiment B: worker scaling (env count fixed, worker processes varied).

Worker (W) = an env-executing worker process.

- W == 1: everything runs in the main process (sync path, reuses the
  Experiment A runners; mode="sync").
- zeroRL: only W == 1 exists — ``vectorize_env`` builds a synchronous
  ``SyncVectorEnv`` and zeroRL has no multi-process path. W >= 2 rows are
  recorded as ``status="unsupported"`` (never faked).
- SB3: W == 1 (DummyVecEnv) or W == N (SubprocVecEnv, one process per env).
  Intermediate W (2, 4 with N=8) has no SB3 representation -> unsupported.
- TorchRL: W in {1, 2, 4, 8} — W >= 2 uses ``MultiSyncCollector`` with W
  worker processes, each running a vector env of N/W single envs (verified:
  yields have batch (W, N/W, T), prod == frames_per_batch, exact stop at
  total_frames).

Workload (budgets, policy, seeds) is identical to Experiment A.
"""

from __future__ import annotations

import time
from functools import partial

from gymnasium.vector import AutoresetMode, SyncVectorEnv

from benchmarks.scalability.collection import (
    CollectionConfig,
    _verify_transition_count,
    lifecycle,
    make_one_env,
    run_sb3_collection,
    run_torchrl_collection,
    run_zerorl_collection,
)
from benchmarks.scalability.policy_reference import (
    TorchRLSharedTrunk,
    build_reference_state_dict,
)

NUM_ENVS_B = 8
FRAMEWORKS_B = ("zeroRL", "SB3", "TorchRL")
WORKERS_B = (1, 2, 4, 8)


class UnsupportedConfiguration(RuntimeError):
    """The requested (framework, W) has no representation in that framework
    and must be recorded as ``status="unsupported"`` — never simulated."""


def supported_workers(framework: str, num_envs: int = NUM_ENVS_B) -> set[int]:
    """Worker counts with a real implementation in each framework."""
    if framework == "zeroRL":
        return {1}
    if framework == "SB3":
        return {1, num_envs}          # DummyVecEnv / SubprocVecEnv(1 proc per env)
    if framework == "TorchRL":
        return {w for w in WORKERS_B if num_envs % w == 0}
    raise ValueError(f"unknown framework {framework!r}")


# ---------------------------------------------------------------------------
# TorchRL multi-process env factory (module-level => picklable under spawn)
# ---------------------------------------------------------------------------

def make_torchrl_worker_env(env_id: str, worker_seeds: list[int]):
    """Build one worker's env: a SyncVectorEnv of len(worker_seeds) single
    envs, wrapped for torchrl (returned object must be an EnvBase)."""
    from torchrl.envs.libs.gym import GymWrapper

    from benchmarks.scalability.collection import rearm_first_reset

    venv = SyncVectorEnv(
        [partial(make_one_env, env_id, s) for s in worker_seeds],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    wrapper = GymWrapper(venv, categorical_action_encoding=True)
    # GymWrapper's info-reader check consumed the canonical first reset;
    # re-arm here so the worker's inner collector carrier reset (created
    # after this factory returns) IS the canonical reset.
    rearm_first_reset(venv)
    return wrapper


def split_seeds(seeds: list[int], num_workers: int) -> list[list[int]]:
    """Contiguous per-worker seed groups (env i keeps seeds[i] regardless of
    the split, so trajectories do not depend on W)."""
    if len(seeds) % num_workers != 0:
        raise ValueError(
            f"num_envs {len(seeds)} not divisible by num_workers {num_workers}"
        )
    size = len(seeds) // num_workers
    return [seeds[i * size:(i + 1) * size] for i in range(num_workers)]


def run_torchrl_mp_collection(cfg: CollectionConfig, num_envs: int,
                              num_workers: int, seed: int) -> dict:
    from tensordict.nn import TensorDictModule
    from torchrl.collectors import MultiSyncCollector
    from zerorl.functions import derive_env_seeds

    from benchmarks.scalability.collection import (
        _TorchRLCollectPolicy, _absorb_actions, _finalize_phases,
        _make_fingerprint_part, choose_collector_fpb,
    )
    from benchmarks.scalability.budget import BudgetTracker

    seeds = derive_env_seeds(seed, num_envs)
    splits = split_seeds(seeds, num_workers)

    with lifecycle() as lc:
        t0 = time.perf_counter()
        trunk = TorchRLSharedTrunk(build_reference_state_dict(cfg.reference_seed))
        policy = _TorchRLCollectPolicy(trunk)
        tdm = TensorDictModule(policy, in_keys=["observation"],
                               out_keys=["action", "log_prob", "entropy", "value"])
        fpb_w = choose_collector_fpb(cfg.warmup, num_envs, cfg.max_frames_per_batch)
        collector1 = MultiSyncCollector(
            [partial(make_torchrl_worker_env, cfg.env_id, s) for s in splits],
            tdm,
            num_workers=num_workers,
            frames_per_batch=fpb_w,
            total_frames=cfg.warmup,
            trust_policy=True,
            auto_register_policy_transforms=False,
        )
        lc.add("collector1", collector1.shutdown)
        init_s = time.perf_counter() - t0   # spawn + worker startup + spec checks

        fb = _make_fingerprint_part()
        fb["_warm_target"] = cfg.warmup
        warm_tracker = BudgetTracker(cfg.warmup)
        first_obs: list[float] | None = None
        t = time.perf_counter()
        for td in collector1:
            n = int(td.batch_size.numel())
            if first_obs is None:
                first_obs = [round(float(x), 6)
                             for x in td["observation"][0, 0].reshape(-1)[:4]]
            _absorb_actions(fb, td["action"].movedim(-1, 0).reshape(-1).tolist(),
                            warm_tracker)
            warm_tracker.record(n)
        warm_s = time.perf_counter() - t
        _verify_transition_count(warm_tracker.actual, cfg.warmup, "warmup")

        # boundary: fresh collector (worker respawn cost -> boundary_seconds)
        tb = time.perf_counter()
        fpb_m = choose_collector_fpb(cfg.measure, num_envs, cfg.max_frames_per_batch)
        collector2 = MultiSyncCollector(
            [partial(make_torchrl_worker_env, cfg.env_id, s) for s in splits],
            tdm,
            num_workers=num_workers,
            frames_per_batch=fpb_m,
            total_frames=cfg.measure,
            trust_policy=True,
            auto_register_policy_transforms=False,
        )
        lc.add("collector2", collector2.shutdown)
        boundary_s = time.perf_counter() - tb
        del fb["_warm_target"]
        meas_tracker = BudgetTracker(cfg.measure)
        t = time.perf_counter()
        for td in collector2:
            n = int(td.batch_size.numel())
            _absorb_actions(fb, td["action"].movedim(-1, 0).reshape(-1).tolist(),
                            meas_tracker)
            meas_tracker.record(n)
        meas_s = time.perf_counter() - t
        _verify_transition_count(meas_tracker.actual, cfg.measure, "measure")

        fb["first_obs_head"] = first_obs or []
        art = {
            "phases": _finalize_phases(init_s, warm_s, meas_s, boundary_s),
            "transitions": {
                "warmup_target": cfg.warmup, "warmup_actual": warm_tracker.actual,
                "measure_target": cfg.measure, "measured": meas_tracker.actual,
                "exact": warm_tracker.exact and meas_tracker.exact,
            },
            "fingerprint": {k: v for k, v in fb.items() if not k.startswith("_")},
            "config_detail": {
                "collector": "MultiSyncCollector",
                "num_workers": num_workers,
                "envs_per_worker": num_envs // num_workers,
                "frames_per_batch_warmup": fpb_w,
                "frames_per_batch_measure": fpb_m,
                "seeds": seeds,
            },
        }
    lc.attach(art)
    return art


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

def run_b_collection(framework: str, cfg: CollectionConfig, num_workers: int,
                     seed: int, num_envs: int = NUM_ENVS_B) -> dict:
    """One Experiment B run; raises UnsupportedConfiguration when the
    framework cannot express this worker count (caller records it)."""
    if framework not in FRAMEWORKS_B:
        raise ValueError(f"unknown framework {framework!r}")
    if num_workers not in supported_workers(framework, num_envs):
        if framework == "zeroRL":
            raise UnsupportedConfiguration(
                "zeroRL has no multi-process environment path "
                "(vectorize_env -> SyncVectorEnv only); W>=2 unsupported"
            )
        if framework == "SB3":
            raise UnsupportedConfiguration(
                "SB3 SubprocVecEnv runs exactly one process per env, so only "
                f"W=1 (DummyVecEnv) and W={num_envs} (SubprocVecEnv) exist; "
                f"W={num_workers} unsupported"
            )
        raise UnsupportedConfiguration(
            f"{framework} has no worker count W={num_workers} "
            f"(supported: {sorted(supported_workers(framework, num_envs))})"
        )

    if framework == "zeroRL":
        return run_zerorl_collection(cfg, num_envs, seed)          # W=1
    if framework == "SB3":
        return run_sb3_collection(cfg, num_envs, seed,
                                  subproc=num_workers > 1)         # W=1 or N
    if num_workers == 1:
        return run_torchrl_collection(cfg, num_envs, seed)
    return run_torchrl_mp_collection(cfg, num_envs, num_workers, seed)
