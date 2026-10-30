# Benchmark Review-Fix Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 6 mandatory code-review priorities in `benchmarks/scalability/` — workload comparability, verified transition counts, canonical initial states, per-rep uncertainty, PPO parameter alignment, and lifecycle/error handling — then deliver `benchmarks/scalability/REVIEW_FIXES.md` and a pre-run report before any full campaign.

**Architecture:** All changes are benchmark-side (no `zerorl/` core edits). Exp A collection gets a uniform per-transition contract (forward → dist → argmax → log_prob → entropy → value → native storage), a re-arm-after-setup seeding procedure proven by three-way parity tests, and buffer/loop-derived budget verification. C/D get explicit `AlgoConfig` mapping, lr/optimizer evidence, and a phase taxonomy with `boundary_seconds`. Metrics get per-repetition paired speedup/efficiency.

**Tech Stack:** Python 3.11, pytest/ruff/mypy via `.venv/bin/*`, SB3 2.9.2, torchrl 0.14.1, gymnasium v1.

**Spec:** the user's 6-priority code-review message (requirements copied verbatim into Global Constraints / tasks below).

## Global Constraints
- Prefix every test/smoke run: `CUDA_VISIBLE_DEVICES=""` (GTX 1050 Ti sm_61 crashes on CUDA).
- **No changes to `zerorl/` core algorithms** (esp. `encoder.py`, `ppo.py`, `train.py`); no methodology change that favors zeroRL — equalize work, don't remove it.
- Budgets/seeds/supported-configs unchanged: A warmup 16 000 (excluded) + measure 100 000; C/D 262 144 exact; N∈{1,2,4,8,16}; zeroRL W≥2 and intermediate SB3 W remain `unsupported`.
- Stats from ALL reps; failed/unsupported rows keep error details.
- Verification commands: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/ -q` (baseline 420 passed), `.venv/bin/ruff check zerorl/ tests/ benchmarks/`, `.venv/bin/mypy zerorl/`.
- **No git commits** (user must ask explicitly).
- Deliverable `.md`: `benchmarks/scalability/REVIEW_FIXES.md`.

## Review Focus (most likely failure modes + pinning tests)
1. **Re-arm placement misses where torchrl consumes the first reset** (empirically `GymWrapper` → `auto_register_info_dict` → `set_info_dict_reader` does `reset(); rand_step(); reset(); check_env_specs()` — `gym_like.py:649-658,735`) → pinned by Task 1's three-way `first_obs_head` **and** `actions_head` **and** `action_sum_*` parity test.
2. **SB3 field parity accidentally costs two trunk passes** (get_distribution + predict_values) → pinned by Task 2 ops test asserting single-pass `forward_collection` call count == steps.
3. **zeroRL count verification reads the wrong quantity** (buffer `size` is slices; transitions = slices × N) → pinned by Task 3 tiny-run asserts + unit test of `_verify_transition_count`.
4. **Cleanup helper masks the original exception or double-closes** → pinned by Task 4 failure-injection test (original message + `add_note` cleanup text + closers invoked exactly once).
5. **Paired speedup mispairs reps or silently drops missing baseline reps** → pinned by Task 6 synthetic test with distinct per-rep times + missing-baseline case + `n_pairs` reporting.

---

### Task 0: Stop the running campaign & archive discarded results

**Files:** none created; process/state only.
- [ ] Step 1: `pkill -f run_all.sh; pkill -f "benchmarks/scalability/run.py"`; verify `pgrep -f "benchmarks/scalability/run.py"` empty.
- [ ] Step 2: Record `wc -l benchmarks/scalability/results/*.jsonl` output (for REVIEW_FIXES.md "discarded" section), then `rm benchmarks/scalability/results/*.jsonl` (rows invalid under fixed methodology; dir gitignored).
- [ ] Step 3: Confirm no runner processes remain.

### Task 1: P3 — Canonical initial state after setup (all frameworks)

**Files:**
- Modify: `benchmarks/scalability/collection.py` (`FirstResetSeeded` @ :70-87, `run_torchrl_collection` @ :351-377)
- Modify: `benchmarks/scalability/workers.py` (`make_torchrl_worker_env` @ :64-73)
- Modify: `benchmarks/scalability/ppo_protocols.py` (`run_torchrl_ppo` @ :337-374)
- Test: `tests/test_benchmark_scaling.py` (extend parity test ~:433-445)

**Interfaces:**
- Produces: `FirstResetSeeded.rearm(self) -> None`; `rearm_first_reset(env_or_venv) -> None` in `collection.py` (walks `venv.envs` or single env; safe no-op on objects without `.envs`/`.rearm`).

- [ ] Step 1: Failing test — extend the zeroRL↔SB3 parity test into `test_three_framework_initial_state_and_action_parity` (tiny budgets, seed=777): assert `first_obs_head`, `actions_head`, `action_sum_warmup`, `action_sum_measure` equal for `run_zerorl_collection`, `run_sb3_collection`, `run_torchrl_collection`. Run → FAIL (torchrl differs; verified empirically: `[-0.006253…]` vs `[-0.004799…]`).
- [ ] Step 2: Implement `rearm()` + `rearm_first_reset()`.
- [ ] Step 3: Placement: `run_torchrl_collection` — `rearm_first_reset(venv)` immediately after `wrapper = GymWrapper(...)` (line 365), before `Collector(...)` (carrier reset becomes the armed canonical reset). `make_torchrl_worker_env` — re-arm after `GymWrapper(venv)` before return (factory completes inside construction chain; info-reader check already consumed first reset). `run_torchrl_ppo` — same as A-path, before `Collector` (moves into init in Task 5).
- [ ] Step 4: Run parity test → PASS; run `test_torchrl_exact_budget_and_determinism` and `test_torchrl_mp_exact_budget_and_determinism` → PASS (no process leaks).

### Task 2: P1 — Equal-work collection contract + field/ops evidence (Exp A)

**Files:**
- Modify: `benchmarks/scalability/policy_reference.py` (`SharedTrunkPolicy` @ :131 — add method)
- Modify: `benchmarks/scalability/collection.py` (SB3 loop @ :305-323; TorchRL `_TorchRLCollectPolicy` + tdm @ :367-368,397-401; zeroRL fields @ :243-257)
- Modify: `benchmarks/scalability/INSPECTION.md` (new "Exp A workload contract" table)
- Test: `tests/test_benchmark_scaling.py`

**Interfaces:**
- Produces: `SharedTrunkPolicy.forward_collection(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]` = (actions, values, log_prob, entropy), single trunk pass, body mirrors SB3 `policies.py:636-658` + `distribution.entropy()` (SB3 `_predict` at :709 only builds dist + action — value/log_prob/entropy NOT computed today).
- Produces: `_TorchRLCollectPolicy.forward(obs) -> tuple[Tensor, Tensor, Tensor, Tensor]` (action=argmax, log_prob, entropy, value); tdm `out_keys=["action", "log_prob", "entropy", "value"]`.
- Produces: every A artifact carries `config_detail["fields"] = {"action": T, "log_prob": T, "entropy": T, "value": T}` with T == measured transitions.

Contract (document in INSPECTION): per transition each path computes policy forward (both heads) → distribution → deterministic action → log_prob(action) → entropy → value, and records all four + reward/dones in native storage: zeroRL `Buffer` (schema has `entropy,value,log_prob` — `factory.py:50-52`), SB3 `RolloutBuffer.add(obs, action, reward, done, value_pred, log_prob)` + entropy side `np.ndarray` (RolloutBuffer has no entropy slot — documented), TorchRL TensorDict keys.

- [ ] Step 1: Failing tests: (a) `test_collection_fields_recorded_for_all_frameworks` — tiny runs; assert `fields[k] == measured` per framework; (b) `test_sb3_single_pass_collection_forward` — spy counting `forward_collection` calls == `warm_steps + meas_steps`; (c) `test_torchrl_policy_outputs_four_fields` — one forward returns 4 tensors shape `(N,)`.
- [ ] Step 2: Implement `forward_collection` on `SharedTrunkPolicy`.
- [ ] Step 3: Rewrite SB3 warm/measure loops: `obs_t = obs_as_tensor(obs, device)` under `no_grad`; `actions, values, log_probs, entropy = policy.forward_collection(obs_t)`; action→numpy; `_absorb_actions`; `vec.step(actions)`; tracker; RolloutBuffer `add` per vec-step + entropy side array; `buffer.reset()` between phases; capacity `max(warmup, measure)` with `n_envs=num_envs` (verify constructor signature at implementation). Replace `config_detail["predict"]` with `"collection"` + `"fields"`.
- [ ] Step 4: Extend `_TorchRLCollectPolicy` to 4 outputs; update both tdm constructions (A sync path; B via `workers.py:104` — verify mp fingerprints `td["action"]`).
- [ ] Step 5: zeroRL: populate `fields` from `buffer.get_all()` shapes × N (real counts); add `config_detail["fields"]`.
- [ ] Step 6: Document zeroRL residual extra work: one bootstrap critic forward per `rollout_phase` call (`train.py:179-183`) + per-truncation forwards ≤0.4 % at chunk=250 — INSPECTION "known micro-diffs" bullet (do NOT alter core).
- [ ] Step 7: Run new tests + full parity test + `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_benchmark_scaling.py -q` → PASS.

### Task 3: P2 — Verified actual transition counts (all frameworks)

**Files:**
- Modify: `benchmarks/scalability/collection.py` (zeroRL fabricated trackers @ :254-257; SB3 hard assert @ :310,322; torchrl raises @ :392,411 routed through helper)
- Test: `tests/test_benchmark_scaling.py`

**Interfaces:**
- Produces: `_verify_transition_count(actual: int, target: int, phase: str) -> None` raising `RuntimeError(f"{phase} budget mismatch: actual={actual} target={target}")`.

- [ ] Step 1: Failing tests: (a) `test_zerorl_counts_come_from_buffer` — config whose budgets cannot match buffer accounting → runner raises, does not report target; (b) `test_verify_transition_count_raises` unit test (actual≠target raises with both numbers; equal passes).
- [ ] Step 2: zeroRL: after warmup `warm_actual = buffer.size * num_envs`; verify; boundary `slices_b = buffer.size`; after measure `(buffer.size - slices_b) * num_envs`; verify; feed real values into `BudgetTracker.record(actual)` (exactness computed, not assumed).
- [ ] Step 3: SB3: `_verify_transition_count(warm_tracker.actual, cfg.warmup, "warmup")` after each phase (and measure). TorchRL: route existing raises through helper.
- [ ] Step 4: Run tests → PASS; existing exact-budget tests all three → PASS.

### Task 4: P6 — Reliable shutdown + preserved error details

**Files:**
- Modify: `benchmarks/scalability/collection.py`, `workers.py`, `ppo_protocols.py` (swallows: `collection.py:325-328,414-418`; `workers.py:156-160`; `ppo_protocols.py:413-418`; missing finally: `run_zerorl_collection` never closes env, `run_sb3_collection`/`run_torchrl_mp_collection` close only on success)
- Test: `tests/test_benchmark_scaling.py`

**Interfaces:**
- Produces: `lifecycle()` contextmanager in `collection.py`: caller appends `(name, closer)` as resources are created; body success → run all closers, collect failures into `cleanup_warnings: list[str]` (artifact key) + `warnings.warn`; body exception → run all closers, attach each via `exc.add_note(f"cleanup [{name}] failed: {e!r}")`, re-raise ORIGINAL exception.

- [ ] Step 1: Failing test `test_failure_mid_collection_preserves_error_and_closes` — monkeypatch `trainer.rollout_phase` to raise `RuntimeError("boom-marker")` after first call; assert raised message contains `boom-marker`, contains a cleanup note, env close invoked (spy), no new active children.
- [ ] Step 2: Implement `lifecycle()`; wrap all seven runners. zeroRL collection closes `trainer.env` (collection never calls `train()`, which closes at `train.py:279`). Success path: `cleanup_warnings` into artifact (omit when empty); never bare `except: pass`.
- [ ] Step 3: Verify `run.py` failed rows store traceback string — add assertion if uncovered.
- [ ] Step 4: Run failure-injection test + `active_children` leak tests → PASS.

### Task 5: P5 — PPO parameter alignment + phase taxonomy

**Files:**
- Modify: `benchmarks/scalability/ppo_protocols.py` (`AlgoConfig()` @ :187; `_finalize_train_phases` @ :120; torchrl collector @ :370-374 → before :367; lr/optimizer capture all three)
- Modify: `benchmarks/scalability/collection.py::_finalize_phases` @ :167-176 (add `boundary_s` → `boundary_seconds`); boundary timing :235 (zeroRL), :313 (SB3), :395-401 (torchrl A); `workers.py:134-144` (torchrl B collector2)
- Modify: `benchmarks/scalability/INSPECTION.md` §6, `benchmarks/scalability/report.py` methodology text
- Test: `tests/test_benchmark_scaling.py`

**Interfaces:**
- Produces: every C/D `config_detail` gains `lr_start`, `lr_end`, `optimizer: {"type": "Adam", "eps": float, "schedule": str}`.
- Produces: A/B artifacts gain `phases["boundary_seconds"] > 0`; C/D keep `{init_seconds, measure_seconds}`.

- [ ] Step 1: Failing tests: (a) `test_zerorl_algo_config_mapped_from_protocol` — mapping helper fields equal `PPOProtocolConfig` values; (b) `test_lr_schedule_aligned_across_frameworks` — tiny C-run per framework: `lr_start ≈ 3e-4`, `lr_end ≈ 3e-4 * (1 - 1/updates)`, `eps == 1e-5`; (c) `test_boundary_seconds_present_and_positive` parametrized over frameworks (A-run).
- [ ] Step 2: Replace `AlgoConfig()` in `run_zerorl_ppo` with explicit `AlgoConfig(lr=cfg.lr, gamma=cfg.gamma, batch_size=cfg.batch_size, gae_lambda=cfg.gae_lambda, clip_eps=cfg.clip_eps, ent_coef=cfg.ent_coef, value_coef=cfg.vf_coef, epochs=cfg.epochs)` (tau N/A — docstring note).
- [ ] Step 3: Capture lr/optimizer: zeroRL `trainer.optimizer.param_groups[0]` before/after `train()`; SB3 `model.optimizer.param_groups[0]`; torchrl local. Record in `config_detail`.
- [ ] Step 4: Boundary timing: `boundary_s` param on `_finalize_phases` (default 0.0; A/B always pass real values; keep `min(init,warm,meas) > 0`). Time four boundary sites. Move torchrl C/D `Collector(...)` above `init_s`.
- [ ] Step 5: INSPECTION §6: phase definitions (init = construction + weight load + canonical restore + worker spawn; warmup; boundary = phase reset/collector-worker restart; measure = steady state); zeroRL C/D `train()` internals remain inside measure (official-path, no core changes); Exp B structural note — TorchRL W≥2 measure phase starts fresh worker envs (fresh canonical) while zeroRL/SB3 continue post-warmup RNG (workload identical; trajectory parity asserted only in Exp A).
- [ ] Step 6: New tests + C/D CLI smokes tiny → PASS.

### Task 6: P4 — Per-repetition speedup/efficiency with real dispersion

**Files:**
- Modify: `benchmarks/scalability/aggregate.py`, `benchmarks/scalability/plots.py` (`_curve_series` :122-132 misuses `throughput_std` for efficiency; `_speedup_series` :135-148 hardcodes `0.0`), `benchmarks/scalability/report.py` (tables :109-130,144-165, findings :263-278)
- Test: `tests/test_benchmark_scaling.py`

**Interfaces:**
- Produces: `paired_curve(rows, experiment, framework, *, kind, point_key, baseline) -> list[dict]` in `aggregate.py`; per point `{point, speedup_mean, speedup_std, speedup_ci95_lo, speedup_ci95_hi, efficiency_mean, efficiency_std, n_pairs}`. Pairing key = `rep` (+`seed`); rep missing at baseline skipped + counted. Per-rep: `speedup_r = t_base_r / t_r`; env efficiency `= speedup_r / N`; worker efficiency `= speedup_r / W`. Aggregation via `summary_stats`.
- Keeps: `scaling_curve` (mean-time formula) for schema/back-compat; plots+report switch to `paired_curve`.

- [ ] Step 1: Failing tests: (a) `test_paired_curve_known_values` — synthetic rows, 3 reps, hand-computed means/stds; (b) `test_paired_curve_missing_baseline_rep` — skipped + `n_pairs`; (c) `test_plot_series_use_paired_dispersion` — `plot_experiment_a` series yerr == paired stds (not `throughput_std`, not `0.0`).
- [ ] Step 2: Implement `paired_curve`.
- [ ] Step 3: Rewire `_speedup_series` → `speedup_mean/speedup_std`; efficiency → `efficiency_mean/efficiency_std`.
- [ ] Step 4: Report: speedup/efficiency columns + findings print `mean ± std (n=…)` from `paired_curve`.
- [ ] Step 5: Tests + figure generation from synthetic rows → PASS.

### Task 7: Full validation + `REVIEW_FIXES.md` + pre-run report

**Files:**
- Create: `benchmarks/scalability/REVIEW_FIXES.md`
- Modify: `benchmarks/scalability/README.md`, `AGENTS.md` (only if commands/structure changed), `INSPECTION.md` (final pass)

- [ ] Step 1: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/ -q` → all green (420 baseline + new; exact counts); `.venv/bin/ruff check zerorl/ tests/ benchmarks/`; `.venv/bin/mypy zerorl/`.
- [ ] Step 2: CLI smokes `run.py --experiment A|B|C|D` tiny → ok rows; `python -m benchmarks.scalability.plots` + `report` on synthetic/partial set → figures + REPORT.md.
- [ ] Step 3: Write `REVIEW_FIXES.md`: (1) per-priority fixes with file:line refs; (2) tests run + results (exact commands, pass counts); (3) unsupported configs; (4) experiments actually executed (smokes only; campaign paused) + discarded-results note (Task 0 counts) + statement: no core zerorl changes, no methodology change favoring any framework.
- [ ] Step 4: Present pre-run report in chat and **stop for user go/no-go** before relaunching the full A→B→C→D campaign.
