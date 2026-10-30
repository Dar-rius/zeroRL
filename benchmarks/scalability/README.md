# Scalability benchmark (zeroRL vs SB3 vs TorchRL)

Measures how the three frameworks scale — environment count, worker count,
and two PPO training protocols — on CPU with exact transition budgets,
repeated seeds, and raw per-run results.

Full framework findings: [`INSPECTION.md`](INSPELECTION.md).
Generated results/report: `results/` (gitignored) → `results/REPORT.md`,
`results/figures/*.png`.

## Experiments

| Exp | What varies | Budget per run | Reps | Frameworks |
|---|---|---|---|---|
| A | env count N ∈ {1,2,4,8,16}, single process | warmup 16,000 (excluded) + measure 100,000 transitions | 5 seeds | zeroRL, SB3, TorchRL |
| B | workers W ∈ {1,2,4,8} at fixed N=8 | same as A | 5 seeds | zeroRL: W=1 only (sync-only); SB3: W=1 (Dummy), W=N (Subproc); TorchRL: all W (`MultiSyncCollector`) |
| C | N, fixed **2048 steps/env** per rollout (global batch 2048·N, updates 128/N) | exactly 262,144 PPO transitions | 3 seeds | all three |
| D | N, fixed **global rollout 2048** (steps/env 2048/N, updates 128) | exactly 262,144 PPO transitions | 3 seeds | all three |

C and D are never merged. Frameworks/configurations that cannot express a
cell record a `status="unsupported"` row (never silently dropped); errors
record `status="failed"` with the traceback. Budgets must be exact or the
row is marked non-exact and surfaced in the report.

## Setup

```bash
uv sync --all-extras --dev            # torchrl/tensordict are core deps
uv pip install -e ".[benchmarks]"     # SB3 + matplotlib (the 'benchmarks' extra)
```

**Mandatory on the dev box** (GTX 1050 Ti, sm_61 — CUDA kernels crash):
prefix every command with `CUDA_VISIBLE_DEVICES=""`.

## Running

```bash
# full suite (A, B, C, D — strictly sequential; parallel runs would
# distort the throughput measurements):
CUDA_VISIBLE_DEVICES="" bash /path/to/run_all.sh   # or individually:

CUDA_VISIBLE_DEVICES="" python benchmarks/scalability/run.py --experiment A
CUDA_VISIBLE_DEVICES="" python benchmarks/scalability/run.py --experiment B
CUDA_VISIBLE_DEVICES="" python benchmarks/scalability/run.py --experiment C
CUDA_VISIBLE_DEVICES="" python benchmarks/scalability/run.py --experiment D

# smoke (small budgets; C/D require the budget to be a multiple of the
# per-update batch: e.g. --total 2048 --envs 1 for C):
CUDA_VISIBLE_DEVICES="" python benchmarks/scalability/run.py \
    --experiment A --envs 2 --reps 1 --warmup 1000 --measure 2000 \
    --out /tmp/smokeA.jsonl

# figures + report from results/*.jsonl:
python -m benchmarks.scalability.plots
python -m benchmarks.scalability.report
```

Useful flags: `--frameworks zeroRL,SB3,TorchRL|all`, `--envs 1,2,4,8,16`,
`--workers 1,2,4,8`, `--envs-b 8`, `--reps N` (default 5 for A/B, 3 for
C/D), `--seed 1000` (rep `r` uses `seed+r`), `--warmup/--measure` (A/B),
`--total 262144` (C/D), `--out FILE`.

## Result rows (`results/exp*.jsonl`)

One JSON object per run (`results_schema.py`, `SCHEMA_VERSION = 1`):
`framework, experiment (A|B|C|D), mode (sync|mp), num_envs, num_workers,
protocol (collection|ppo), rep, seed, status (ok|failed|unsupported),
phases {init, warmup (A/B), measure}_seconds, transitions {target, actual,
exact}, metrics {throughput_mean, ...}, config {full workload + seeds +
hyperparameters}, environment {python/torch/sb3/torchrl versions, git_sha},
timestamp_utc`; `extra` carries fingerprints (A/B) or episode returns (C/D).

Aggregation helpers: `scaling_metrics.py` (pure formulas),
`aggregate.py` (row loading/grouping), used by `plots.py`/`report.py`.
Statistics always aggregate **all** ok repetitions (mean ± std, never
best-of).

## Tests

```bash
CUDA_VISIBLE_DEVICES="" uv run pytest tests/test_benchmark_scaling.py -q
CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q      # full suite
ruff check zerorl/ tests/ benchmarks/
```

SB3-dependent tests `importorskip` (the CI CPU job does not install the
`benchmarks` extra); TorchRL tests run in CI (core dep).
