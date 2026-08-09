# AGENTS.md

## Project

RL training framework built on PyTorch + Gymnasium. Provides abstract base classes (`BaseAgent`, `BaseEnv`, `BaseTrain`) and standalone PPO functions (`gae_compute`, `ppo_loss`, `ppo`). Includes 111 tests, all passing (1 xfailed documenting a known vectorized-reset bug).

## Package management

- Managed with **uv** (lockfile: `uv.lock`, Python version: `.python-version` → 3.11)
- Build system: **hatchling** (declared in `pyproject.toml`)
- Dependencies pinned in `requirements.txt`; runtime deps declared in `pyproject.toml` (`torch`, `numpy`, `gymnasium`, `tensorboard`, `wandb`)

## Lint / typecheck

```
ruff check zerorl/ tests/   # linter (Pyflakes only, pyproject.toml)
mypy zerorl/                # strict: disallow_untyped_defs, warn_unreachable (pyproject.toml)
```

No pre-commit hooks, no CI, no formatter config beyond ruff defaults.

## Running tests on this machine

- GPU: GTX 1050 Ti (sm_61); installed torch cu130 has no sm_61 kernels → CUDA ops fail
- Run CPU-only: `CUDA_VISIBLE_DEVICES="" uv run python -m pytest tests/ -v`
- Tests are marked `@pytest.mark.gpu`; the marker is informational only (no auto-deselect)

## Import style

- `train.py` uses **absolute imports** (`from zerorl.agent import BaseAgent`)
- `ppo.py` uses **relative imports** (`from ...common import Buffer`, `from ...config import AlgoConfig`)
- `algorithms/__init__.py` is empty; `algorithms/ppo/__init__.py` re-exports `ppo` and `gae_compute`

## Structure

```
zerorl/
  __init__.py              # empty package marker
  agent.py                 # BaseAgent (ABC, nn.Module), eval_action()
  env.py                   # BaseEnv (ABC, gym.Env) — Gymnasium v1 API wrapper; register_env() decorator
  vector_env.py             # VectorEnv(BaseEnv) — vectorized env wrapper (env_id string or callable class)
  train.py                 # BaseTrain — rollout/update/save training loop
  trainer.py               # make_env(), prototype() — DX factory over BaseTrain
  common.py                # Buffer — dict-based pre-allocated torch buffer
  config.py                # TrainConfig (computed fields), AlgoConfig (mutable dataclass)
  errors.py                # EmptyBufferError
  function.py              # linear_schedule(), get_buffer_params_model()
  processing.py            # NormMeanStd, NormMinMax
  algorithms/
    __init__.py            # empty
    ppo/
      __init__.py          # re-exports ppo, gae_compute
      ppo.py               # gae_compute(), ppo_loss(), ppo() — standalone functions
tests/
  __init__.py
  test_config.py             # 15 tests
  test_errors.py             # 12 tests
  test_function.py           # 4 tests
  test_processing.py         # 3 tests
  test_train.py              # 13 tests
  test_trainer.py            # 14 tests
  agent/
    test_agent.py            # 6 tests
    test_continuous_actions.py # 1 test
  algo/ppo/
    test_ppo.py              # 16 tests
    test_ppo_vector.py       # 1 test
  buffer/
    test_buffer_integration.py # 4 tests
    test_common.py           # 10 tests
  env/
    test_env.py              # 5 tests
    test_env_vector.py       # 7 tests
```

## Key APIs

### Buffer

```python
Buffer(step: int, data: dict[str, tuple], num_envs: int = 1, device: torch.device)
```

Constructor takes a dict mapping field names to shapes (e.g., `{"state": (4,), "actions": ()}`). Internally pre-allocates torch tensors of shape `(step, num_envs, *shape)` on the given device. `get_all()` returns a dict of sliced tensors (same keys). No `insert_returns()` method — callers insert GAE-computed returns/advantages directly.

### BaseAgent

Abstract methods: `forward(state) -> tuple[Tensor, Tensor]` and `build_distribution(logits) -> Distribution` (static).

`get_action(state, action=None)` is a template method that calls `forward()` → `build_distribution()` → samples or evaluates → returns `dict[str, Tensor]` with keys `"action"`, `"log_prob"`, `"entropy"`, `"value"`.

### PPO (standalone functions)

- `gae_compute(rewards, values, last_value, dones, hyper_params) -> (returns, advantages, deltas)`
- `ppo_loss(agent, params, buffers, states, actions, old_log_probs, advantages, returns, hyper_params) -> dict[str, Tensor]`
- `ppo(agent, optimizer, buffer, hyper_params, scheduler, batch_size, epochs, device) -> dict[str, Tensor]`

No `PPOTrainer` class — PPO is composed as standalone functions that accept an `AlgoConfig` and a `Buffer`.

### Config

- `AlgoConfig`: Mutable dataclass with custom `__init__` (init=False). Fields: `lr`, `gamma`, `batch_size`, `gae_lambda`, `clip_eps`, `ent_coef`, `value_coef`, `epochs`, `tau`.
- `TrainConfig`: Mutable dataclass with `__post_init__` for computed fields (`model_path`, `num_update`, `device`). Fields: `model_name`, `model_save_path`, `timestamp`, `rollout_steps`, `num_envs`. `num_update` is computed as `timestamp // (rollout_steps * num_envs)` — overriding `rollout_steps`/`timestamp`/`num_envs` after construction does NOT recompute `num_update`; set it manually if needed.

No `PPOConfig` (replaced by `AlgoConfig`). No `WandbConfig`.

### BaseTrain

```python
BaseTrain(agent, env, buffer, update_weights, train_config, algo_config=None, optimizer=None, require_buffer_size=10)
```

`update_weights` is a `Callable[[BaseAgent, Buffer, Optimizer, int, dict[str,Tensor], AlgoConfig | None], dict[str, Tensor]]`. Methods: `rollout_phase(state)`, `_log_metrics(metrics, step, use_wandb, use_tb)`, `train(use_wandb, use_tb)`, `save_model()`.

### trainer (`make_env` / `prototype`)

Import as `from zerorl import trainer`.

- `make_env(env, *, is_vector=False, num_envs=1) -> BaseEnv` — resolves a gym id `str`, callable env class, or existing `BaseEnv`; optionally wraps in `VectorEnv` when `is_vector=True`.
- `prototype(*, algo="ppo", agent=None, env=None, is_vector=False, num_envs=1, timestamp=1_000_000, model_name="model", model_save_path="./checkpoints", rollout_steps=2048, **hyper_params) -> BaseTrain` — wires default MLP agent (when `agent=None`), PPO buffer shapes, and `update_weights`; splits kwargs into `TrainConfig` / `AlgoConfig` fields (unknown keys → `TypeError`). Returns a ready-to-run `BaseTrain`; call `.train(use_wandb=..., use_tb=...)`.

### Utility modules

- `function.py`: `linear_schedule(step, num_update)`, `get_buffer_params_model(model) -> (params, buffers)`
- `processing.py`: `NormMeanStd(shape, device, epsilon)`, `NormMinMax(low, high, device)` — both have `normalize(x)` and `NormMeanStd` has `update(x)`
- `env.py`: `register_env(env_id)` decorator registers a class with Gymnasium
- `vector_env.py`: `VectorEnv(env_spec, num_envs)` — wraps `env_spec` (string env_id or callable class) into a vectorized env

## Conventions

- Abstract base classes use `@abstractmethod` with concrete bodies (template method pattern) — subclasses override and optionally call `super()`
- `Buffer` pre-allocates torch tensors and uses a slice pointer with a `size` property and bounds checking in `insert()`
- GPU detection is automatic in `TrainConfig.device`
- Tests use pytest, import zerorl as a package (`from zerorl.common import Buffer`)
- `train.py` includes wandb and tensorboard logging via `_log_metrics()`

## Testing

```
CUDA_VISIBLE_DEVICES="" uv run python -m pytest tests/ -v    # run all 111 tests (CPU-only on this machine)
```

- 110 pass, 1 xfailed (documents vectorized reset bug at `train.py:141-145`: all envs reset when any finishes)
- `test_train.py::test_rollout_phase_partial_finish_preserves_survivors` is the xfail test
