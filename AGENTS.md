# AGENTS.md

## Project

RL training framework built on PyTorch + Gymnasium. Provides abstract base classes (`BaseAgent`, `BaseEnv`, `BaseTrain`) and standalone PPO functions (`gae_compute`, `ppo_loss`, `ppo`). Includes 143 tests, all passing.

## Package management

- Managed with **uv** (lockfile: `uv.lock`, Python version: `.python-version` → 3.11)
- Build system: **hatchling** (declared in `pyproject.toml`)
- Dependencies pinned in `requirements.txt`; runtime deps declared in `pyproject.toml` (`torch`, `numpy`, `gymnasium`, `tensorboard`, `wandb`)

## Lint / typecheck

```
ruff check zerorl/       # linter (Pyflakes only, pyproject.toml)
mypy zerorl/             # strict: disallow_untyped_defs, warn_unreachable (pyproject.toml)
```

No pre-commit hooks, no CI, no formatter config beyond ruff defaults.

## Import style

- `train.py` uses **absolute imports** (`from zerorl.agent import BaseAgent`)
- `ppo.py` uses **relative imports** (`from ...common import Buffer`, `from ...config import AlgoConfig`)
- `algorithms/__init__.py` is empty; `algorithms/ppo/__init__.py` re-exports `ppo` and `gae_compute`

## Structure

```
zerorl/
  __init__.py              # empty package marker
  agent.py                 # BaseAgent (ABC, nn.Module), eval_action()
  env.py                   # BaseEnv (ABC) — Gymnasium v1 API wrapper
  train.py                 # BaseTrain — rollout/update/save training loop
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
  test_agent.py              # 17 tests
  test_buffer_integration.py # 7 tests
  test_common.py             # 26 tests
  test_config.py             # 14 tests
  test_continuous_actions.py # 12 tests
  test_env.py                # 11 tests
  test_errors.py             # 7 tests
  test_function.py           # 12 tests
  test_ppo.py                # 17 tests
  test_processing.py         # 13 tests
  test_train.py              # 7 tests
```

## Key APIs

### Buffer

```python
Buffer(step: int, data: dict[str, tuple], device: torch.device)
```

Constructor takes a dict mapping field names to shapes (e.g., `{"state": (4,), "actions": ()}`). Internally pre-allocates torch tensors on the given device. `get_all()` returns a dict of sliced tensors (same keys). No `insert_returns()` method — callers insert GAE-computed returns/advantages directly.

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
- `TrainConfig`: Mutable dataclass with `__post_init__` for computed fields (`model_path`, `num_update`, `device`). Fields: `model_name`, `model_save_path`, `timestamp`, `rollout_steps`.

No `PPOConfig` (replaced by `AlgoConfig`). No `WandbConfig`.

### BaseTrain

```python
BaseTrain(agent, env, buffer, update_weights, train_config, algo_config=None, optimizer=None, require_buffer_size=10)
```

`update_weights` is a `Callable[[BaseAgent, Buffer, Optimizer, dict[str,Tensor], AlgoConfig | None], dict[str, Tensor]]`. Methods: `rollout_phase(state)`, `_log_metrics(metrics, step, use_wandb, use_tb)`, `train(use_wandb, use_tb)`, `save_model()`.

### Utility modules

- `function.py`: `linear_schedule(step, num_update)`, `get_buffer_params_model(model) -> (params, buffers)`
- `processing.py`: `NormMeanStd(shape, device, epsilon)`, `NormMinMax(low, high, device)` — both have `normalize(x)` and `NormMeanStd` has `update(x)`

## Conventions

- Abstract base classes use `@abstractmethod` with concrete bodies (template method pattern) — subclasses override and optionally call `super()`
- `Buffer` pre-allocates torch tensors and uses a slice pointer with a `size` property and bounds checking in `insert()`
- GPU detection is automatic in `TrainConfig.device`
- Tests use pytest, import zerorl as a package (`from zerorl.common import Buffer`)
- `train.py` includes wandb and tensorboard logging via `_log_metrics()`

## Testing

```
python -m pytest tests/ -v    # run all 143 tests
```
