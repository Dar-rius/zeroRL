# Implementation note — framework inspection findings

Inspection performed before writing the scalability benchmark (2026-10-27, this
repository). All claims below were verified against installed source or live
probes, not assumed.

## 1. zeroRL (this repository)

- **Environment handling is synchronous and single-process only.**
  `vectorize_env` / `get_env` build `gym.vector.SyncVectorEnv` with
  `AutoresetMode.SAME_STEP` (`zerorl/functions.py:50`). There is no
  `AsyncVectorEnv`, `Subproc*`, or any multiprocessing path anywhere in
  `zerorl/`.
  → **Worker-scaling (Experiment B): zeroRL is `unsupported` for W ≥ 2.**
    We do not invent a feature the framework does not have.
- **Collection**: `BaseTrain.rollout_phase()` (`zerorl/helpers/train.py:141`)
  is a Python loop of `rollout_steps` iterations; each iteration does
  batched `get_action` over N envs via `env_step`, episode bookkeeping,
  `final_value` computation at the horizon, and `buffer.insert`.
- **PPO**: single `Adam` over all agent parameters (lr from `AlgoConfig`,
  `eps=1e-5`), `LambdaLR` schedule, grad-clip `0.5`, advantage normalized
  **once over the full flattened batch** (`ppo.py:163`), value loss
  `0.5 × MSE` (`ppo.py:116`), `rollout_steps` is **per environment**,
  `num_update = timestamp // (rollout_steps * num_envs)`.
- **Architecture** (`factory.ActorCriticAgent`): ONE shared 64-64 Tanh trunk
  (`extract_layer`) + separate actor/critic heads; orthogonal init with gains
  √2 (trunk) / 0.01 (actor) / 1.0 (critic), zero biases.

## 2. Stable-Baselines3 2.9.2 (installed)

- VecEnv types: `DummyVecEnv` (sync, single process) and `SubprocVecEnv`
  (**exactly one process per env** — no chunking). W ≠ N is therefore not
  expressible with SB3 → rows with W ∉ {1, N} are recorded `unsupported`.
- `n_steps` is **per environment** (rollout size = `n_steps × n_envs`).
- `learn(total_timesteps=..., reset_num_timesteps=False)` uses an
  **absolute cumulative** target — benchmark loops must accumulate targets
  or they silently train ~0 steps (bug found and fixed in earlier work).
- **`net_arch=[64, 64]` is NOT a shared trunk since SB3 1.8.0.** In
  SB3 ≤ 1.7 a plain int list built shared layers (docstring: *"A simple
  shared network topology … specified as [128, 128]"*); shared layers were
  removed in v1.8.0 and the same config now builds **two separate 64-64
  trunks** (verified live: 0 shared parameters, 9797 params vs zeroRL's 5061
  on LunarLander). The benchmark therefore uses a `SharedTrunkPolicy`
  subclass that re-unifies `mlp_extractor.value_net = mlp_extractor.policy_net`
  after `_build`, restoring shared-trunk topology and matching zeroRL's
  parameter count and init scheme (extractor gain √2, action head 0.01,
  value head 1.0).
- Loss deviations vs zeroRL (documented, not hidden): value loss is
  `1.0 × MSE` (zeroRL: `0.5 × MSE` → effective value weight differs ×2 at
  `vf_coef=0.5`); advantages are re-normalized **per minibatch** (zeroRL:
  once per rollout). Gradient clip 0.5, Adam `eps=1e-5`, and all scalar
  hyperparameters match.
- Deterministic batched inference: `policy.predict(obs_batch,
  deterministic=True)`.

## 3. TorchRL 0.14.1 / tensordict 0.14.3 (installed)

- **`SyncDataCollector` no longer exists.** The main entry is now
  `torchrl.collectors.Collector(create_env_fn, policy, backend="direct" |
  "process", num_collectors, frames_per_batch, total_frames, ...)`;
  `MultiSyncCollector(seq_of_env_fns, num_workers=...)` for multi-process
  collection; `AsyncCollector` / `AsyncBatchedCollector` also exist.
- **`GymEnv(env_id, num_envs=N)` builds a gymnasium `AsyncVectorEnv`
  (multiprocessing!)** — unsuitable for Experiment A's single-process
  requirement. The official **synchronous** path is
  `GymWrapper(gym.vector.SyncVectorEnv(...))` (docstring: *"Batched
  environments (VecEnv or gym.VectorEnv) are supported"*). Validated live:
  wrapper batch size `[N]`, deterministic policy runs end to end.
- Collector yields tensors shaped `(N, T)` where `prod(batch_size) =
  frames_per_batch`; iteration stops exactly at `total_frames`
  (verified: exact budget hit). `frames_per_batch` counts **total
  transitions across envs** (unlike SB3's per-env `n_steps`).
- `GAE` moved: `torchrl.objectives.value.GAE`. `PPOLoss`, `ClipPPOLoss`,
  `ProbabilisticActor`, `ValueOperator`, `MLP` available in
  `torchrl.objectives` / `torchrl.modules`.
- Multiprocessing uses the `spawn` start method → all worker entry points
  must be importable module-level functions with `__main__` guards.
- Default `GymEnv`/collector auto-reset: torchrl handles resets itself when
  the vector env is created with `AutoresetMode.SAME_STEP` (accepted by
  `_validate_env`) or `DISABLED`; benchmark keeps zeroRL-style
  `SAME_STEP` for the sync path.
- Loss deviations vs zeroRL: critic loss is plain MSE via
  `loss_critic_type="l2"` (no 0.5 factor); advantage normalization follows
  `PPOLoss` defaults (checked at runtime, documented in REPORT.md).
- TorchRL has **no one-call trainer**: Protocol C/D use the official
  component path — `Collector` → `GAE` → `PPOLoss` → single `Adam`
  (shared trunk ⇒ one optimizer) with the benchmark's own epoch/minibatch
  loop matching zeroRL/SB3 (epochs=10, batch=64, shuffle, grad-clip 0.5).

## 4. Cross-framework equivalence decisions

| Concern | Decision |
|---|---|
| Sync single-process collection (Exp A) | zeroRL `rollout_phase` · SB3 `DummyVecEnv` + `SharedTrunkPolicy.forward_collection` (§4.1) · TorchRL `GymWrapper(SyncVectorEnv)` + `Collector(backend="direct")` |
| Same policy + weights (collection) | `policy_reference.py`: one seeded shared-trunk MLP (argmax, discrete); per-framework adapters load identical tensors; equality asserted in tests |
| Architecture (protocols C/D) | shared 64-64 Tanh trunk everywhere: zeroRL default agent; SB3 `SharedTrunkPolicy`; TorchRL custom shared module — ortho gains √2/0.01/1.0, one Adam each |
| Worker mapping (Exp B) | W = env-executing worker processes; W=1 = in-process (sync); TorchRL sweeps W ∈ {1,2,4,8} at N=8 via `MultiSyncCollector`; SB3 only W=1 (Dummy) and W=N (Subproc); zeroRL only W=1 |
| Transition counting | individual env transitions: every vectorized step counts N; TorchRL counts `prod(batch_size)` per yield; runtime `BudgetTracker` verifies actual == target or records the actual count + deviation flag (never relabels) |
| Phases | `init` (env + policy + weight load + canonical restore) → `warmup` (16,000 transitions, excluded) → `boundary` (phase reset / collector construction, timed as `boundary_seconds`) → `measure` (100,000 transitions) |
| What "collection" includes | **Uniform Exp A workload contract** (all three): one policy forward per transition computing BOTH heads + distribution → deterministic action → `log_prob(action)` → `entropy` → `value`, recorded with reward/dones in native storage (see §4.1). Adapter/storage overhead differences are documented, not hidden. |

### 4.1 Exp A workload contract (per transition, all frameworks)

| | zeroRL | SB3 | TorchRL |
|---|---|---|---|
| forward (both heads) | `agent.get_action` via `env_step` | `SharedTrunkPolicy.forward_collection` (single trunk pass; NOT `predict`, which stops after dist+action) | `_TorchRLCollectPolicy.forward` (single trunk pass) |
| distribution | `Categorical` (inside agent) | `_get_action_dist_from_latent` | `Categorical(logits=…)` |
| action | greedy argmax | `distribution.get_actions(deterministic=True)` | `probs.argmax(-1)` |
| log_prob / entropy / value | returned per step | returned per step | returned per step |
| storage (measured-phase evidence) | `Buffer` rows (`action,log_prob,entropy,value` schema) | SB3 `RolloutBuffer.add(...)` + entropy side array (RolloutBuffer has no entropy slot) | collector TensorDict keys `action,log_prob,entropy,value` |

Evidence: every A artifact carries `config_detail["fields"]` with all four
counts == measured transitions (asserted by
`test_collection_fields_recorded_for_all_frameworks`); single-pass property
asserted by `test_sb3_single_pass_collection_forward`.

**Shared-trunk single-execution contract** (asserted by
`TestSharedTrunkSinglePass`): the parameter-shared trunk executes **once**
per actor-critic pass in every framework, collection AND training:

- zeroRL: `forward` runs `extract_layer` once, then both heads (reference).
- SB3: `_SharedOnceMlpExtractor` (`policy_reference.py`) replaces stock
  `MlpExtractor.forward` — which would call the identity-shared
  `value_net is policy_net` module twice — so `forward_collection`, the
  stock `forward` (used by `collect_rollouts`) and `evaluate_actions`
  (training) each run the trunk once.
- TorchRL collection: `_TorchRLCollectPolicy.forward` computes
  `trunk.trunk(obs)` once and feeds both heads (the `actor_path` /
  `critic_path` Sequential variants each re-run the trunk and are avoided
  on hot paths).
- TorchRL training: collector writes `td["features"]`; the ClipPPOLoss
  actor and critic are head-only consumers of `td["features"]`, which
  `_torchrl_train_minibatch` recomputes **with grad exactly once per
  minibatch**. GAE evaluates the critic from those stored features, plus
  one no-grad trunk call per td yield to supply `td["next","features"]`
  (the collector only computes features for current observations).
  E2E trunk counts decompose exactly
  (collector calls + GAE injections + minibatches) in the tests.

**Exp B structural note**: TorchRL W≥2 uses `MultiSyncCollector` — its
measure-phase collector starts FRESH worker envs (fresh canonical reset),
while zeroRL/SB3 (and TorchRL W=1) continue from the post-warmup RNG state.
The per-transition workload contract above is identical; cross-framework
trajectory parity is therefore asserted only for Exp A (where all three
frameworks run in one process over the same env family).

**Known residual micro-diff (documented, not "fixed"):** zeroRL's official
`rollout_phase` additionally performs ONE bootstrap critic forward per
`rollout_phase` call (`train.py:179-183`) plus one per truncated step
(`train.py:172-178`) to build the GAE `final_value` column. At chunk=250
that is ≤0.4 % extra critic forwards (LunarLander truncations are rare);
removing it would require editing the production core, which the benchmark
forbids. SB3/TorchRL do not compute a collection-time bootstrap value in A
(they have no GAE phase there).
| Seeds | recorded per repetition; deterministic policy + seeded env resets make trajectories comparable; cross-framework trajectory parity asserted in smoke tests |

## 5. Installed versions (recorded into every result row)

- zerorl (repo, git sha recorded), torch (see run), gymnasium, numpy
- stable-baselines3 2.9.2
- torchrl 0.14.1, tensordict 0.14.3
- Device: CPU everywhere (`DEVICE="cpu"`); the dev-box GTX 1050 Ti (sm_61)
  cannot run this PyTorch build's CUDA kernels, so all experiments are CPU
  benchmarks by construction (`CUDA_VISIBLE_DEVICES=""` prefix).

## 6. Training protocols C/D — implementation details

- **Budgets**: `total = 262,144` measured transitions per run (warmup is a
  collection-only concept). Protocol C fixes `rollout_steps = 2048` per env
  (global batch `2048*N`, updates `128/N`); protocol D fixes the global batch
  at 2048 (`rollout_steps = 2048/N`, updates `128`). Non-divisible totals or
  `N` that does not divide 2048 are rejected before any run starts.
- **Phases**: `init` = env + policy/model construction + weight load +
  canonical first-reset restore + TorchRL `Collector` construction (the
  collector's carrier reset is setup, never inside the measure timer);
  `measure` = each framework's official steady-state training entry point
  (`BaseTrain.train()` loop body, `PPO.learn()` iterations, Collector
  iteration + updates). Each entry point performs its own first env reset /
  rollout start inside the measured window (ms-scale; recorded as the one
  deliberate asymmetry). zeroRL's measure also includes its one-off
  `create_logger` call — same order of magnitude, official path. C/D runs are
  continuous (no warmup, so no `boundary` phase; A/B artifacts carry
  `phases["boundary_seconds"]` for the warmup→measure reset/collector
  restart).
- **LR schedule**: all three decay linearly to 0 over the run — zeroRL's
  official default `LambdaLR(1 - step/num_update)`, SB3 via a
  `learning_rate=lambda p: lr * p` schedule (SB3's own default for a float lr
  is *constant*), TorchRL via `LambdaLR(1 - step/updates)`. Adam `eps=1e-5`
  everywhere (zeroRL hardcodes it; SB3's default; configured for TorchRL).
  **Verified evidence**: every C/D artifact records `config_detail.lr_start`
  (== 3e-4), `config_detail.lr_end` (== 0), and
  `config_detail.optimizer = {"type": "Adam", "eps": 1e-5, "schedule":
  "linear_decay_to_zero"}` — asserted equal across all three frameworks by
  `test_lr_and_optimizer_recorded_aligned`. zeroRL receives its parameters
  through the explicit `_algo_config_from()` mapping (every field set from
  `PPOProtocolConfig`; `tau` off-policy-only, not mapped) instead of silent
  `AlgoConfig()` defaults.
- **TorchRL loss wiring**: `ProbabilisticActor(..., return_log_prob=True,
  log_prob_key=lp_key)` with `lp_key` following
  `tensordict.nn.composite_lp_aggregate()` (default: `action_log_prob`);
  `GAE(value_network=critic)` precomputes advantage + value_target once per
  batch; advantages are normalized once over the full batch (zeroRL-style);
  `ClipPPOLoss(loss_critic_type="l2")` (its default `smooth_l1` would differ).
  `ClipPPOLoss` shares the collector's parameters (verified in tests: grads
  land on all 8 trunk params and `optimizer.step()` changes all 8) — the
  optimizer is built over `_dedup_params(actor, critic)`.
- **Episode returns**: zeroRL uses its official `episode_rewards`;
  SB3/TorchRL track returns manually from per-step rewards + done flags
  (time-limit truncations included), matching zeroRL's semantics.
- **Known micro-diffs (documented, not "fixed")**: zeroRL's value term is
  `0.5 * MSE` where SB3/TorchRL use `1.0 * MSE` (all with `vf_coef=0.5`);
  zeroRL normalizes advantages once per full batch while SB3 normalizes per
  minibatch (TorchRL configured full-batch like zeroRL). Core algorithms are
  not modified (spec constraint).
