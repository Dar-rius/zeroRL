# AGENTS.md

## Project
zeroRL — RL training framework on PyTorch 2 + Gymnasium v1. Base classes
(`BaseAgent`, `BaseEnv`, `BaseTrain`), standalone PPO functions, factory
helpers (`factory.py`, `easy_train_ppo`). 128 tests.

## Commands
- `uv sync --all-extras --dev`          # uv-managed, hatchling build; no .python-version (requires-python >=3.11)
- `ruff check zerorl/ tests/`           # Pyflakes rules only (select=["F"])
- `mypy zerorl/`                        # NOT strict: disables operator/attr-defined/call-arg/override; library only
- `CUDA_VISIBLE_DEVICES="" uv run pytest tests/`          # full suite (128) — prefix mandatory, see below
- `CUDA_VISIBLE_DEVICES="" uv run pytest tests/test_train.py::TestClass::test_name`  # single test
- `.pre-commit-config.yaml` exists: ruff + mypy via `uv run` (always_run)
- CI: `.github/workflows/test.yml` — CPU matrix py3.11–3.13 (`pytest -m "not gpu" --cov=zerorl`),
  GPU job on a self-hosted **Jetson Thor** runner (`pytest -m "gpu"`); plus `security.yml` (SBOM)

## GPU on this dev box
- This box has a GTX 1050 Ti (sm_61) + torch 2.13+cu130: `torch.cuda.is_available()` returns
  True but sm_61 kernels are missing → CUDA ops crash. ALWAYS prefix test/benchmark/example
  runs with `CUDA_VISIBLE_DEVICES=""`. GPU CI runs on the separate Jetson Thor runner.
- `@pytest.mark.gpu` does NOT auto-skip; the `device` fixture falls back to CPU when CUDA
  is unavailable, so the whole suite runs CPU-only with the prefix.

## Architecture contracts (not obvious from filenames)
- Duck-typed agent contract, not ABC: `BaseAgent` is a plain nn.Module (+`device` property).
  Agents must define `get_action()` themselves (see `factory.ActorCriticAgent`);
  `BaseTrain.__init__` asserts it exists, `ppo_func` asserts `forward`/`build_distribution`
  (`assert_agent_contract` in errors.py). `BaseEnv` keeps real @abstractmethods (reset/step/close).
- Buffer keys: rollout inserts `state, reward, done` + `get_action()` output keys
  (`action, log_prob, entropy, value`); `gae_compute` writes `"advantage"` and `"return"`
  (NOT "returns"/"adv") directly into `buffer.data` and returns None. Unknown key → `KeyBufferError`.
- `BaseTrain` wiring: wraps envs lacking `auto_reset=True` in `VectorEnv`; creates Adam(eps=1e-5)
  and a linear-decay LambdaLR unless given; `rollout_phase()` takes no args (uses `self.state`),
  returns the bootstrap-output dict; `update_weights(agent, buffer, scheduler, optimizer,
  last_output, algo_config)` — scheduler BEFORE optimizer; `train(save_model=, use_wandb=, use_tb=)`
  is keyword-only.
- `TrainConfig` requires `project_name` (no default); `num_update` is computed ONCE in
  `__post_init__` (`timestamp // (rollout_steps * num_envs)`) — mutating fields later does not
  recompute it. `model_path` = `.{model_save_path}/{model_name}.pt` (hardcoded leading dot →
  hidden dir). `normalize=True` enables NormMeanStd; `profile=True` enables the stderr profiler.
- PPO: `ppo_func` (not `ppo`), with injectable `ppo_loss_func`; `ppo_backward` is wrapped in
  `fast_compile` (torch.compile, no-op without a C++ compiler). `easy_train_ppo(config,
  algo_config, env_id)` is the one-call quickstart. `algorithms/ppo/__init__` exports
  `gae_compute, ppo_func, easy_train_ppo`; `zerorl/__init__.py` and `algorithms/__init__.py`
  are empty — import from full module paths.

## Trust hierarchy
- README quickstart is STALE (old `update_weights`/`gae_compute` signatures, missing required
  `project_name`). Copy working code from `examples/` (cartpole.py, acrobot.py) or tests.
- `benchmarks/` run on THIS dev box (SB3-style comparison scripts with `__main__` blocks).
  They auto-set `DEVICE = cuda if available` → same sm_61 crash without `CUDA_VISIBLE_DEVICES=""`.
  Their deps (stable-baselines3, tianshou, pettingzoo, pygame, matplotlib) are installed in
  `.venv` but NOT in pyproject/uv.lock — `uv sync` will EVICT them; reinstall from
  `requirements.txt` (skip its broken last line: `-e file:///home/research/projects/zeroRL`).
  Don't `pip install -r requirements.txt` as-is.
- `.opencode/agents/` (reviewer/senior/test-programer) workflow expects this file kept current.

## Testing notes
- tests/ is grouped by area: `agent/`, `algo/ppo/`, `buffer/`, `env/` + flat
  `test_config|test_errors|test_factory|test_function|test_processing|test_train.py`
  (test_train.py is the largest, 28 tests; profiler tests mock CUDA).
- 1 known xfail: vectorized-reset bug — `rollout_phase` resets ALL sub-envs when any finishes
  on a non-auto_reset env (train.py, `if finished.any() and not self.env.auto_reset`).
