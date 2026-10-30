# Review fixes — scalability benchmark (zeroRL vs SB3 vs TorchRL)

Code-review priorities 1–6 implemented before any full campaign. Working tree
on branch `new-bench`, base commit `1680036`; no `zerorl/` core changes; no
methodology change that favors any framework (workload contract is uniform
and asserted for all three).

## 1. Per-priority fixes

### P1 — Equal-work collection contract (Exp A)
Every transition does one policy forward (both heads) + distribution +
deterministic action + `log_prob` + `entropy` + `value`, stored in native
storage:

- SB3: `SharedTrunkPolicy.forward_collection` (`policy_reference.py:142`);
  collection loop `_sb3_collect_phase` (`collection.py:480`) uses SB3's
  `RolloutBuffer.add` + entropy side array (RolloutBuffer has no entropy
  slot). `predict` is NOT used (it computes no value/log_prob/entropy).
- TorchRL: `_TorchRLCollectPolicy.forward` returns
  `(action, log_prob, entropy, value)` (`collection.py:609`); tdm out_keys
  in `collection.py` and `workers.py`.
- zeroRL: official `rollout_phase` already produces all four fields
  (Buffer schema).
- Evidence: every A artifact carries `config_detail["fields"]` with all four
  counts == measured transitions (`collection.py:371,454,574`);
  single-pass property asserted by call-count test.

### P2 — Verified actual transition counts
- `_verify_transition_count(actual, target, phase)` (`collection.py:193`)
  used by all three A runners and the TorchRL worker runner.
- zeroRL counts derived from the BUFFER (`buffer.size × N`), never from the
  target (fabricated trackers removed).

### P3 — Canonical initial state + trajectory parity
- `FirstResetSeeded.rearm()` / `rearm_first_reset()` (`collection.py:92,97`)
  restore the canonical seeded reset after TorchRL `GymWrapper`'s
  info-reader check consumes it; applied at all three TorchRL construction
  sites (A sync, B worker env, C/D protocol).
- Cross-framework parity asserted: identical first obs, first action,
  action sums (warmup + measure) for zeroRL ↔ SB3 ↔ TorchRL.
- Bug fixed en route: TorchRL td batch dims are env-major `(N, T)` —
  fingerprint absorption now time-major via `.movedim(-1, 0)`
  (`collection.py:556,579`, `workers.py:140,165`).

### P4 — Per-repetition speedup/efficiency with real dispersion
- `paired_curve()` (`aggregate.py:89`): per-rep speedup
  `t_baseline(rep,seed)/t(point)`, efficiency `/ (point/baseline)`, mean +
  sample std + normal-approx CI95; missing-baseline reps skipped and
  counted (`n_pairs`, `n_skipped`).
- Plots rewired (`plots.py`): y-errors are paired stds (was throughput_std
  misuse for efficiency, hardcoded 0.0 for speedup).
- Report tables/findings show `mean ± std (n=…)` from paired curve.
- `scaling_curve` kept for back-compat.

### P5 — PPO parameter mapping + lr/optimizer verification + phase taxonomy
- Explicit `_algo_config_from()` (`ppo_protocols.py:142`) maps every PPO
  field from `PPOProtocolConfig` into `AlgoConfig` (tau off-policy-only).
- Empirically verified (probe + tests): all three frameworks
  **lr_start = 3e-4 → lr_end = 0**, Adam **eps = 1e-5**, schedule
  `linear_decay_to_zero`. Recorded in every C/D artifact as
  `config_detail.lr_start/lr_end/optimizer{type,eps,schedule}`
  (`_lr_and_optimizer`, `ppo_protocols.py:155`).
  *Note:* the pre-implementation plan guessed `lr_end ≈ 3e-4·(1-1/updates)`;
  probes disproved it (after `updates` scheduler steps the linear decay
  reaches 0 in all three frameworks) — tests assert the measured reality.
- TorchRL C/D `Collector` construction moved inside init (was timed as
  measure).
- A/B artifacts carry `phases["boundary_seconds"] > 0` (phase reset /
  collector-consumer restart between warmup and measure; 4 timed sites:
  `collection.py:345,442,571`, `workers.py:159`). Phase taxonomy documented
  in INSPECTION.md §4/§6 and report methodology.

### P6 — Reliable shutdown + preserved error details
- `lifecycle()` (`collection.py:220`) wraps all seven runners: closers run
  in reverse creation order on every exit; body failure → failing closers
  attached via `exc.add_note`, ORIGINAL exception re-raised; body success →
  failures become `cleanup_warnings` artifact key + `warnings.warn`.
  No bare `except: pass` remains. zeroRL A/C/D env close added (collection
  never called `train()`, which is what closes envs on success).
- `run.py` failed rows persist the full traceback (asserted by CLI test).

## 2. Tests and results (exact commands)

- `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/ -q` →
  **447 passed, 3 skipped** (3 skips = cuda-leg smoke runtime skips).
- `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_benchmark_scaling.py -q`
  → **103 passed** (84 pre-fix + 19 new: parity, fields, single-pass,
  four-field policy, count verification ×3, failure-injection ×3, PPO
  alignment ×7, paired curve ×3).
- `.venv/bin/ruff check zerorl/ tests/ benchmarks/ scripts/` → clean.
- `.venv/bin/mypy zerorl/` → success, no issues.
- CLI smokes (tiny budgets): A → 3 ok; B → 4 ok + 2 correctly-recorded
  unsupported; C → 3 ok; D → 3 ok; 0 failed.
  `plots` generated 10 figures; `report` wrote REPORT.md with paired
  columns from the partial set.

## 3. Unsupported configurations (recorded as `status=unsupported`, never run)

- Experiment B: SB3 only W ∈ {1, N} (Dummy/SubprocVecEnv); zeroRL is
  sync-only (W=1). TorchRL covers W ∈ {1,2,4,8} via `MultiSyncCollector`.
- Experiment B TorchRL W≥2 structural note (INSPECTION.md §4.1): the
  measure-phase collector starts fresh worker envs; per-transition workload
  is identical, trajectory parity is asserted only in Exp A.

## 4. Experiments executed; discarded prior results; neutrality

- Pre-fix campaign rows (collected under the old unequal workload) were
  **discarded**: expA=75, expB=60, expC=45, expD=45 rows, total 225 —
  `results/*.jsonl` deleted (Task 0).
- At the time of writing, only tiny-budget CLI smokes had been re-run
  (section 2). The full spec campaign (A/B: 5 reps, C/D: 3 reps, default
  budgets 16k+100k and 262,144) is launched after this file, results in
  `results/*.jsonl` (gitignored).
- No `zerorl/` core source was modified. No timing window, budget, seed
  schedule, or workload element was adjusted to benefit any framework;
  differences that could not be equalized (official-path overheads, TorchRL
  spec checks, Exp B worker restarts) are timed and documented instead of
  hidden (INSPECTION.md).

## 5. Reproducibility record

- Base commit: `1680036` (branch `new-bench`); benchmark files untracked in
  working tree (no commits made unless requested).
- Device: CPU only, `CUDA_VISIBLE_DEVICES=""` (dev-box GTX 1050 Ti sm_61
  lacks kernels for torch 2.13+cu130).
- Frameworks: zeroRL (this repo) / stable-baselines3 / torchrl+tensordict
  from the repo venv; versions recorded into every result row.
- Budgets: A/B warmup 16,000 (excluded) + measure 100,000; C/D exactly
  262,144; seeds base 1000 + rep; reference weights seed 4242.
