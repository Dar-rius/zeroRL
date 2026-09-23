# AGENTS.md

## Project
zeroRL — RL training framework on PyTorch 2 + Gymnasium v1. Base classes
(`BaseAgent`, `BaseEnv`, `BaseTrain` in `helpers/train.py`), standalone PPO
functions, factory helpers (`factory.py`, `easy_train_ppo`). Immediate mode =
manual loop with `functions` + `logger` (see `examples/immediate_mode.py`).
`MujocoEnv` in `helpers/mujoco.py` for custom XML; Gymnasium MuJoCo ids still
via `gym.make`. `mujoco` is a core dep.

## Commands
- `uv sync --all-extras --dev`          # uv-managed, hatchling build; no .python-version (requires-python >=3.11)
- `ruff check zerorl/ tests/`           # Pyflakes rules only (select=["F"])
- `mypy zerorl/`                        # NOT strict: disables operator/attr-defined/call-arg/override; library only
- `CUDA_VISIBLE_DEVICES="" uv run pytest tests/`          # full suite — prefix mandatory, see below
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
- Buffer: construct with `capacity, num_envs, schema, device` (not `data=`+`config=`).
  Rollout keys: `state, reward, terminated, truncated` + `get_action()` outputs
  (`action, log_prob, entropy, value`). Pass **true** `data["terminated"]` to
  `gae_compute` (do **not** OR with `truncated` — time-limit must still bootstrap).
  Episode-end bookkeeping uses `terminated | truncated`. GAE writes `"advantage"`
  and `"return"`.
- `BaseTrain` lives in `zerorl.helpers.train` (NOT `zerorl.train`). Wraps envs lacking
  `auto_reset=True` via `vectorize_env` (SAME_STEP). Logging via `create_logger`.
  Eval/GIF: `trainer.try_agent()` → `functions.try_agent` (deterministic mean/argmax,
  AutoresetMode.DISABLED). `train(save_model=, use_wandb=, use_tb=)` is keyword-only.
- `TrainConfig` requires `project_name` (no default); `num_update` is computed ONCE in
  `__post_init__` (`timestamp // (rollout_steps * num_envs)`). `normalize=True` enables
  NormMeanStd (also saved by `save_checkpoints`). `profile=True` → `PhaseProfiler`.
- PPO: `ppo_func` with injectable `ppo_loss_func`; `fast_compile` in `zerorl.compiler`
  (no-op without C++ compiler or without Triton on CUDA). `easy_train_ppo(env_spec,
  config, algo_config, *, seed=22, ...)` — env first. Import from full module paths.

## Trust hierarchy
- README quickstart may be stale. Copy working code from `examples/` (pendulum.py,
  immediate_mode.py, point_mass.py, reacher_mujoco.py) or tests.
- `benchmarks/` auto-set `DEVICE = cuda if available` → same sm_61 crash without
  `CUDA_VISIBLE_DEVICES=""`. Their deps are NOT in pyproject — `uv sync` may evict them.
- `.opencode/agents/` workflow expects this file kept current.

## Testing notes
- tests/ is grouped by area: `agent/`, `algo/ppo/`, `buffer/`, `env/` + flat
  `test_config|test_errors|test_factory|test_function|test_processing|test_train.py`
  plus `tests/compiler.py`.
- Vectorized train uses Gymnasium SAME_STEP autoreset (no manual multi-env reset in
  `rollout_phase`).
