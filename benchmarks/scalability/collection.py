"""Experiment A runners: synchronous, single-process collection benchmark.

One timed run per (framework, num_envs, seed, repetition):

    init      -> env + policy + weight load + initial reset (no transitions)
    warmup    -> warmup transitions, EXCLUDED from throughput
    (boundary reset outside both timers, identical in all frameworks)
    measure   -> measured transitions -> throughput = measured / measure_s

Framework paths (official collection paths, see INSPECTION.md):

- zeroRL: ``BaseTrain.rollout_phase`` driven externally (env, inference,
  episode bookkeeping, buffer insert, final_value computation all included).
- SB3:    ``DummyVecEnv`` + ``SharedTrunkPolicy.forward_collection`` — the
  single-pass (action, value, log_prob, entropy) contract of §4.1 in
  INSPECTION.md (SB3's own ``collect_rollouts`` samples stochastically and
  ``predict`` computes no value/log_prob/entropy, so neither is used as-is).
- TorchRL: ``GymWrapper(SyncVectorEnv[SAME_STEP])`` + ``Collector``
  (backend="direct"); collector storage is part of TorchRL's official path.

Cross-framework trajectory parity: all three frameworks start from the
canonical FirstResetSeeded observation and produce IDENTICAL action
trajectories (asserted in tests). TorchRL's GymWrapper consumes the first
reset during its info-reader check (official behavior, timed under init),
so ``rearm_first_reset`` restores the canonical reset before the Collector
is built.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, reduce
from math import gcd
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch

from benchmarks.scalability.budget import BudgetTracker, remaining_batches
from benchmarks.scalability.policy_reference import (
    TorchRLSharedTrunk,
    build_deterministic_zerorl_agent,
    build_reference_state_dict,
    build_sb3_shared_policy,
    load_into_sb3,
)

FINGERPRINT_HEAD = 20
FRAMEWORKS_A = ("zeroRL", "SB3", "TorchRL")


# ---------------------------------------------------------------------------
# config + seeding
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CollectionConfig:
    """Workload definition for Experiment A (shared by all frameworks)."""

    env_id: str = "LunarLander-v3"
    warmup: int = 16_000
    measure: int = 100_000
    reference_seed: int = 4242
    max_rollout_chunk: int = 250      # zeroRL rollout_phase chunk (rows/env)
    max_frames_per_batch: int = 8_000  # TorchRL Collector frames_per_batch cap


class FirstResetSeeded(gym.Wrapper):
    """Canonical seeding: the FIRST reset always uses ``canonical``.

    Later resets (including ones where a framework passes its own seed) are
    forced to seed=None so every framework consumes env RNG identically by
    construction. This is what makes the zeroRL/SB3 trajectories bit-equal.
    """

    def __init__(self, env: gym.Env, canonical: int) -> None:
        super().__init__(env)
        self._canonical = canonical
        self._first = True

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self._first:
            self._first = False
            return self.env.reset(seed=self._canonical, options=options)
        return self.env.reset(seed=None, options=options)

    def rearm(self) -> None:
        """Re-arm so the NEXT reset consumes ``canonical`` again (once)."""
        self._first = True


def rearm_first_reset(env_or_venv: Any) -> None:
    """Re-arm every ``FirstResetSeeded`` reachable from an env / vector env.

    TorchRL's ``GymWrapper`` construction runs an info-reader check
    (``reset() + rand_step() + reset() + check_env_specs()``), consuming the
    canonical first reset before the Collector exists. Re-arming right before
    the Collector is built makes the collector's own carrier reset the
    canonical one, so TorchRL collection starts from the same initial state
    as zeroRL and SB3 (verified by
    ``test_three_framework_initial_state_and_action_parity``).
    """
    sub_envs = getattr(env_or_venv, "envs", None)
    targets = list(sub_envs) if sub_envs is not None else [env_or_venv]
    for env in targets:
        rearm = getattr(env, "rearm", None)
        if callable(rearm):
            rearm()


def make_one_env(env_id: str, seed: int) -> gym.Env:
    """Single canonically-seeded env. Module-level (and partial()-friendly)
    so multi-process collectors can pickle it under the spawn method."""
    return FirstResetSeeded(gym.make(env_id), seed)


def seeded_env_fns(env_id: str, seeds: list[int]) -> list[Callable[[], gym.Env]]:
    """One factory per env, each with its canonical seed (picklable)."""
    return [partial(make_one_env, env_id, s) for s in seeds]


def seeded_env_factory(env_id: str, seeds: list[int]) -> Callable[[], gym.Env]:
    """Stateful counter factory for ``vectorize_env`` (which builds N envs
    from a single callable): env i receives seeds[i] in construction order."""
    state = {"i": 0}

    def factory() -> gym.Env:
        i = state["i"]
        state["i"] = i + 1
        return FirstResetSeeded(gym.make(env_id), seeds[i])

    return factory


# ---------------------------------------------------------------------------
# budget helpers
# ---------------------------------------------------------------------------

def largest_divisor_at_most(n: int, cap: int) -> int:
    """Largest divisor of ``n`` that is <= ``cap`` (>= 1)."""
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    d = min(cap, n)
    while n % d != 0:
        d -= 1
    return d


def choose_collector_fpb(budget: int, num_envs: int, cap: int) -> int:
    """TorchRL ``frames_per_batch``: divides ``budget``, <= ``cap``, and is a
    multiple of ``num_envs`` (collector batches are shaped (N, T), so
    prod(batch_size) == fpb requires fpb % N == 0)."""
    if budget % num_envs != 0:
        raise ValueError(f"budget {budget} not divisible by num_envs {num_envs}")
    per_env = budget // num_envs
    k = largest_divisor_at_most(per_env, max(1, cap // num_envs))
    return num_envs * k


def choose_zerorl_chunk(cfg: CollectionConfig, num_envs: int) -> int:
    """Rollout chunk (rows/env) dividing both budgets for this env count."""
    for budget in (cfg.warmup, cfg.measure):
        if budget % num_envs != 0:
            raise ValueError(
                f"budget {budget} not divisible by num_envs {num_envs}"
            )
    rows_w, rows_m = cfg.warmup // num_envs, cfg.measure // num_envs
    g = reduce(gcd, (rows_w, rows_m))
    return largest_divisor_at_most(g, cfg.max_rollout_chunk)


def _make_fingerprint_part() -> dict[str, Any]:
    return {"actions_head": [], "action_sum_warmup": 0, "action_sum_measure": 0}


def _absorb_actions(fb: dict[str, Any], actions: list[int],
                    tracker: BudgetTracker) -> None:
    for a in actions:
        if len(fb["actions_head"]) < FINGERPRINT_HEAD:
            fb["actions_head"].append(int(a))
    total = int(sum(actions))
    if tracker.target == fb.get("_warm_target"):
        fb["action_sum_warmup"] += total
    else:
        fb["action_sum_measure"] += total


def _verify_transition_count(actual: int, target: int, phase: str) -> None:
    """P2: fail loudly when a phase's ACTUAL transition count differs from
    the requested budget. Used uniformly by all three Exp A runners (and
    torchrl workers) so a mis-counted run can never be recorded as valid."""
    if actual != target:
        raise RuntimeError(
            f"{phase} budget mismatch: actual={actual} target={target}"
        )


class _Lifecycle:
    """Resource registry for :func:`lifecycle`; see that docstring."""

    def __init__(self) -> None:
        self._closers: list[tuple[str, Callable[[], Any]]] = []
        self.cleanup_warnings: list[str] = []

    def add(self, name: str, closer: Callable[[], Any]) -> None:
        self._closers.append((name, closer))

    def attach(self, artifact: dict[str, Any]) -> None:
        """Success path: copy collected cleanup warnings into the artifact."""
        if self.cleanup_warnings:
            artifact["cleanup_warnings"] = self.cleanup_warnings


@contextmanager
def lifecycle() -> Iterator[_Lifecycle]:
    """P6: reliable shutdown with preserved error details.

    Runners register each resource with ``lc.add(name, closer)`` as it is
    created. Closers always run, in REVERSE creation order:

    * body raised -> each failing closer is attached to the ORIGINAL
      exception via ``exc.add_note(f"cleanup [{name}] failed: ...")`` and
      the original exception is re-raised untouched;
    * body succeeded -> failures land in ``lc.cleanup_warnings`` (merged
      into the artifact via ``lc.attach``) and are surfaced with
      ``warnings.warn``. Closers are never silently swallowed.
    """
    lc = _Lifecycle()
    try:
        yield lc
    except BaseException as exc:
        for name, closer in reversed(lc._closers):
            try:
                closer()
            except Exception as err:
                exc.add_note(f"cleanup [{name}] failed: {err!r}")
        raise
    else:
        for name, closer in reversed(lc._closers):
            try:
                closer()
            except Exception as err:
                lc.cleanup_warnings.append(
                    f"cleanup [{name}] failed: {err!r}"
                )
        if lc.cleanup_warnings:
            warnings.warn(
                "benchmark cleanup issues: "
                + "; ".join(lc.cleanup_warnings),
                RuntimeWarning,
                stacklevel=2,
            )


def _finalize_phases(init_s: float, warm_s: float, meas_s: float,
                     boundary_s: float = 0.0) -> dict[str, float]:
    if min(init_s, warm_s, meas_s) <= 0:
        raise RuntimeError(
            f"non-positive phase time: init={init_s} warmup={warm_s} measure={meas_s}"
        )
    if boundary_s < 0:
        raise RuntimeError(f"negative boundary time: {boundary_s}")
    phases = {
        "init_seconds": init_s,
        "warmup_seconds": warm_s,
        "measure_seconds": meas_s,
    }
    if boundary_s > 0:
        # P5 phase taxonomy: boundary = phase reset / collector-worker
        # restart between warmup and measure (A/B artifacts only).
        phases["boundary_seconds"] = boundary_s
    return phases


# ---------------------------------------------------------------------------
# zeroRL
# ---------------------------------------------------------------------------

def _never_update(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("update_weights must not run in collection-only mode")


def run_zerorl_collection(cfg: CollectionConfig, num_envs: int, seed: int) -> dict:
    from zerorl.config import AlgoConfig, TrainConfig
    from zerorl.functions import derive_env_seeds, get_obs_act, vectorize_env
    from zerorl.helpers.factory import get_actor_critic_buffer
    from zerorl.helpers.train import BaseTrain

    seeds = derive_env_seeds(seed, num_envs)
    chunk = choose_zerorl_chunk(cfg, num_envs)
    steps_w = remaining_batches(cfg.warmup, num_envs * chunk)
    steps_m = remaining_batches(cfg.measure, num_envs * chunk)
    # Buffer capacity counts slices (one slice = one vectorized step = N
    # transitions); rollout_phase inserts `chunk` slices per call.
    capacity = (cfg.warmup + cfg.measure) // num_envs
    if capacity % chunk != 0:
        raise RuntimeError(f"capacity {capacity} not divisible by chunk {chunk}")

    with lifecycle() as lc:
        t0 = time.perf_counter()
        factory = seeded_env_factory(cfg.env_id, seeds)
        env = vectorize_env(factory, num_envs=num_envs)
        lc.add("env", env.close)                 # P6: never skip env close
        obs_shape, act_shape, _, _, _ = get_obs_act(env)
        agent = build_deterministic_zerorl_agent(
            build_reference_state_dict(cfg.reference_seed)
        )
        buffer = get_actor_critic_buffer(
            obs_shape, act_shape, capacity, num_envs, torch.device("cpu")
        )
        tconfig = TrainConfig(
            model_name="bench_collect",
            project_name="benchmarks",
            rollout_steps=chunk,
            num_envs=num_envs,
            timestamp=cfg.warmup + cfg.measure,
        )
        tconfig.device = torch.device("cpu")
        trainer = BaseTrain(
            agent=agent, env=env, buffer=buffer, update_weights=_never_update,
            config=tconfig, algo_config=AlgoConfig(), seeds=seeds,
        )
        state, _ = env.reset(seed=seeds)          # initial reset (init phase)
        trainer.state = torch.as_tensor(state, dtype=torch.float32)
        first_obs = np.array(state, copy=True)
        init_s = time.perf_counter() - t0

        t = time.perf_counter()
        for _ in range(steps_w):
            trainer.rollout_phase()
        warm_s = time.perf_counter() - t
        # P2: counts come from the BUFFER (slices × N), never from the target.
        warm_actual = buffer.size * num_envs
        _verify_transition_count(warm_actual, cfg.warmup, "warmup")
        slices_b = buffer.size

        tb = time.perf_counter()
        state, _ = env.reset()                    # boundary reset
        trainer.state = torch.as_tensor(state, dtype=torch.float32)
        boundary_s = time.perf_counter() - tb

        t = time.perf_counter()
        for _ in range(steps_m):
            trainer.rollout_phase()
        meas_s = time.perf_counter() - t
        meas_actual = (buffer.size - slices_b) * num_envs
        _verify_transition_count(meas_actual, cfg.measure, "measure")

        # fingerprints from the (official-path) buffer
        data = buffer.get_all()
        actions = data["action"].reshape(-1).tolist()
        warm_n = cfg.warmup
        fb = {
            "first_obs_head": [round(float(x), 6) for x in first_obs.reshape(-1)[:4]],
            "actions_head": [int(a) for a in actions[:FINGERPRINT_HEAD]],
            "action_sum_warmup": float(np.sum(actions[:warm_n])),
            "action_sum_measure": float(np.sum(actions[warm_n:])),
        }

        warm_t = BudgetTracker(cfg.warmup)
        warm_t.record(warm_actual)                # verified == target above
        meas_t = BudgetTracker(cfg.measure)
        meas_t.record(meas_actual)
        # Measure-phase fields from actual buffer storage (rows after the
        # measured warmup slice count); all four Exp A contract fields in schema.
        fields = {k: int(data[k].shape[0] - slices_b) * num_envs
                  for k in ("action", "log_prob", "entropy", "value")}
        art = {
            "phases": _finalize_phases(init_s, warm_s, meas_s, boundary_s),
            "transitions": {
                "warmup_target": cfg.warmup, "warmup_actual": warm_t.actual,
                "measure_target": cfg.measure, "measured": meas_t.actual,
                "exact": warm_t.exact and meas_t.exact,
            },
            "fingerprint": fb,
            "config_detail": {
                "rollout_chunk_rows_per_env": chunk,
                "rollout_calls_warmup": steps_w, "rollout_calls_measure": steps_m,
                "buffer_capacity_rows_per_env": capacity,
                "fields": fields,
                "seeds": seeds,
            },
        }
    lc.attach(art)
    return art


# ---------------------------------------------------------------------------
# SB3
# ---------------------------------------------------------------------------

def run_sb3_collection(cfg: CollectionConfig, num_envs: int, seed: int,
                       *, subproc: bool = False) -> dict:
    """SB3 collection. ``subproc=True`` uses SubprocVecEnv (one worker
    process per env -> W == N; used by Experiment B)."""
    from stable_baselines3.common.buffers import RolloutBuffer
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from zerorl.functions import derive_env_seeds

    seeds = derive_env_seeds(seed, num_envs)
    warm_steps = remaining_batches(cfg.warmup, num_envs)
    meas_steps = remaining_batches(cfg.measure, num_envs)

    with lifecycle() as lc:
        t0 = time.perf_counter()
        set_random_seed(seed)
        vec_cls = SubprocVecEnv if subproc else DummyVecEnv
        vec = vec_cls(seeded_env_fns(cfg.env_id, seeds))
        lc.add("vec_env", vec.close)   # P6: terminates SubprocVecEnv workers
        policy = build_sb3_shared_policy(vec.observation_space, vec.action_space)
        load_into_sb3(policy, build_reference_state_dict(cfg.reference_seed))
        # Native rollout storage: rows are env-steps; +1 slack so SB3's
        # pos-wrap at exact capacity can never eat a counted row.
        rows = max(cfg.warmup, cfg.measure) // num_envs + 1
        rollout_buf = RolloutBuffer(rows, vec.observation_space,
                                    vec.action_space, torch.device("cpu"),
                                    n_envs=num_envs)
        entropy_side = np.zeros((rows, num_envs), dtype=np.float32)
        obs = vec.reset()                            # initial reset (init phase)
        first_obs = np.array(obs, copy=True)
        init_s = time.perf_counter() - t0

        fb = _make_fingerprint_part()
        warm_tracker = BudgetTracker(cfg.warmup)
        fb["_warm_target"] = cfg.warmup

        rollout_buf.reset()
        t = time.perf_counter()
        obs = _sb3_collect_phase(vec, policy, rollout_buf, entropy_side, obs,
                                 warm_steps, num_envs, warm_tracker, fb)
        warm_s = time.perf_counter() - t
        _verify_transition_count(warm_tracker.actual, cfg.warmup, "warmup")

        tb = time.perf_counter()
        obs = vec.reset()                            # boundary reset
        boundary_s = time.perf_counter() - tb

        meas_tracker = BudgetTracker(cfg.measure)
        del fb["_warm_target"]
        rollout_buf.reset()
        t = time.perf_counter()
        obs = _sb3_collect_phase(vec, policy, rollout_buf, entropy_side, obs,
                                 meas_steps, num_envs, meas_tracker, fb)
        meas_s = time.perf_counter() - t
        _verify_transition_count(meas_tracker.actual, cfg.measure, "measure")
        # Measure-phase fields come from what the buffer actually stored
        # (rollout_buf.pos monotonic: no wrap, capacity > meas_steps).
        fields = {k: int(rollout_buf.pos) * num_envs
                  for k in ("action", "log_prob", "entropy", "value")}

        fb["first_obs_head"] = [round(float(x), 6)
                                for x in first_obs.reshape(-1)[:4]]
        art = {
            "phases": _finalize_phases(init_s, warm_s, meas_s, boundary_s),
            "transitions": {
                "warmup_target": cfg.warmup, "warmup_actual": warm_tracker.actual,
                "measure_target": cfg.measure, "measured": meas_tracker.actual,
                "exact": warm_tracker.exact and meas_tracker.exact,
            },
            "fingerprint": {k: v for k, v in fb.items() if not k.startswith("_")},
            "config_detail": {
                "vec_env": "SubprocVecEnv" if subproc else "DummyVecEnv",
                "collection": "SharedTrunkPolicy.forward_collection "
                              "(action, value, log_prob, entropy), deterministic",
                "storage": "RolloutBuffer + entropy side array",
                "fields": fields,
                "seeds": seeds,
            },
        }
    lc.attach(art)
    return art


def _sb3_collect_phase(vec, policy, rollout_buf, entropy_side, obs,
                       steps: int, num_envs: int, tracker: BudgetTracker,
                       fb: dict[str, Any]):
    """One SB3 collection phase: single-pass forward producing all four Exp A
    fields, stored in SB3's native RolloutBuffer (+ entropy side array).
    Returns the final observation (pre-boundary)."""
    prev_dones = np.zeros(num_envs, dtype=np.bool_)
    for _ in range(steps):
        with torch.no_grad():
            obs_t, _ = policy.obs_to_tensor(obs)
            actions_t, values, log_probs, entropy = \
                policy.forward_collection(obs_t)
        actions = actions_t.cpu().numpy().reshape(-1)
        _absorb_actions(fb, np.asarray(actions).reshape(-1).tolist(), tracker)
        next_obs, rewards, dones, _ = vec.step(actions)
        row = rollout_buf.pos
        entropy_side[row] = entropy.cpu().numpy().reshape(num_envs)
        rollout_buf.add(obs, actions, rewards, prev_dones, values, log_probs)
        tracker.record(num_envs)
        prev_dones = np.asarray(dones, dtype=np.bool_)
        obs = next_obs
    return obs


# ---------------------------------------------------------------------------
# TorchRL
# ---------------------------------------------------------------------------

def run_torchrl_collection(cfg: CollectionConfig, num_envs: int, seed: int) -> dict:
    from gymnasium.vector import AutoresetMode, SyncVectorEnv
    from tensordict.nn import TensorDictModule
    from torchrl.collectors import Collector
    from torchrl.envs.libs.gym import GymWrapper
    from zerorl.functions import derive_env_seeds

    seeds = derive_env_seeds(seed, num_envs)

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
        # re-arm so the collector's carrier reset below IS the canonical reset.
        rearm_first_reset(venv)
        trunk = TorchRLSharedTrunk(build_reference_state_dict(cfg.reference_seed))
        policy = _TorchRLCollectPolicy(trunk)
        tdm = TensorDictModule(policy, in_keys=["observation"],
                               out_keys=["action", "log_prob", "entropy", "value"])
        fpb_w = choose_collector_fpb(cfg.warmup, num_envs, cfg.max_frames_per_batch)
        collector1 = Collector(
            wrapper, tdm, backend="direct", trust_policy=True,
            auto_register_policy_transforms=False,
            frames_per_batch=fpb_w, total_frames=cfg.warmup,
        )
        lc.add("collector1", collector1.shutdown)
        init_s = time.perf_counter() - t0   # includes GymWrapper's official
        # check_env_specs dummy rollout + collector carrier reset.

        fb = _make_fingerprint_part()
        warm_tracker = BudgetTracker(cfg.warmup)
        fb["_warm_target"] = cfg.warmup
        first_obs: list[float] | None = None

        t = time.perf_counter()
        for td in collector1:
            n = int(td.batch_size.numel())
            if first_obs is None:
                first_obs = [round(float(x), 6)
                             for x in td["observation"][0].reshape(-1)[:4]]
            # td batch dims are (env, time); fingerprint order must be
            # time-major (step0_env0, step0_env1, ...) to match zeroRL/SB3.
            _absorb_actions(fb, td["action"].movedim(-1, 0).reshape(-1).tolist(),
                            warm_tracker)
            warm_tracker.record(n)
        warm_s = time.perf_counter() - t
        _verify_transition_count(warm_tracker.actual, cfg.warmup, "warmup")

        # boundary: constructing collector2 resets the shared env
        tb = time.perf_counter()
        fpb_m = choose_collector_fpb(cfg.measure, num_envs, cfg.max_frames_per_batch)
        collector2 = Collector(
            wrapper, tdm, backend="direct", trust_policy=True,
            auto_register_policy_transforms=False,
            frames_per_batch=fpb_m, total_frames=cfg.measure,
        )
        lc.add("collector2", collector2.shutdown)
        boundary_s = time.perf_counter() - tb

        meas_tracker = BudgetTracker(cfg.measure)
        meas_fields = {"action": 0, "log_prob": 0, "entropy": 0, "value": 0}
        del fb["_warm_target"]
        t = time.perf_counter()
        for td in collector2:
            n = int(td.batch_size.numel())
            _absorb_actions(fb, td["action"].movedim(-1, 0).reshape(-1).tolist(),
                            meas_tracker)
            for key in meas_fields:
                meas_fields[key] += n
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
                "collector": "Collector(backend='direct')",
                "frames_per_batch_warmup": fpb_w,
                "frames_per_batch_measure": fpb_m,
                "autoreset": "SyncVectorEnv SAME_STEP",
                "fields": meas_fields,
                "seeds": seeds,
            },
        }
    lc.attach(art)
    return art


class _TorchRLCollectPolicy(torch.nn.Module):
    """Deterministic collection policy producing the full Exp A field set in
    one trunk pass: (action, log_prob, entropy, value).

    Mirrors what zeroRL's ``get_action`` computes per step so all three
    frameworks do equivalent work; action is the greedy argmax so
    trajectories stay identical across frameworks.
    """

    def __init__(self, trunk: TorchRLSharedTrunk) -> None:
        super().__init__()
        self.trunk = trunk

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # ONE shared-trunk execution per step: actor_path/critic_path would
        # each re-run the (parameter-shared) trunk for the same observation.
        features = self.trunk.trunk(obs)
        logits = self.trunk.actor_head(features)
        value = self.trunk.critic_head(features).squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.probs.argmax(dim=-1)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, log_prob, entropy, value


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

def run_collection(framework: str, cfg: CollectionConfig, num_envs: int,
                   seed: int) -> dict:
    """Run one Experiment A collection pass and return the artifact dict."""
    if framework == "zeroRL":
        return run_zerorl_collection(cfg, num_envs, seed)
    if framework == "SB3":
        return run_sb3_collection(cfg, num_envs, seed)
    if framework == "TorchRL":
        return run_torchrl_collection(cfg, num_envs, seed)
    raise ValueError(f"unknown framework {framework!r} (one of {FRAMEWORKS_A})")
