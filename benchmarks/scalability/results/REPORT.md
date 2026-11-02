# Scalability benchmark report

Generated: 2026-11-01T05:03:06.155123+00:00 · rows: 225 · figures in `figures/`

## Environment

| key | value |
| --- | --- |
| python | 3.11.14 |
| platform | Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.39 |
| processor | x86_64 |
| device | cpu |
| hostname | darrius |
| torch | 2.13.0+cu130 |
| torch_num_threads | 6 |
| cuda_available | False |
| gymnasium | 1.3.0 |
| sb3 | 2.9.2 |
| torchrl | 0.14.1 |
| tensordict | 0.14.3 |
| git_sha | 6e59107451973271173ada17576bd74858297f8c |

## Methodology (summary)

- Frameworks: zeroRL (this repo, official `rollout_phase` / `BaseTrain.train()`), Stable-Baselines3, TorchRL — same shared-trunk policy/weights where policies apply (see `policy_reference.py`).
- Experiments A/B count individual env transitions with a warmup of 16,000 (excluded) + 100,000 measured transitions per run; Experiments C/D train PPO for exactly 262,144 transitions per run.
- Repetitions: 5 seeds per cell for A/B, 3 for C/D; every reported statistic aggregates **all** ok repetitions (mean ± std), never best-of.
- Metrics: `throughput = measured / measure_seconds`, `speedup(N) = t(baseline) / t(N)`, `environment_scaling_efficiency(N) = speedup(N) / N`, `worker_parallel_efficiency(W) = speedup(W) / (W / 1)`.
- Phase timing: `init` = env + policy construction + weight load (+ TorchRL collector/worker spawn); `warmup` (A/B only, excluded); `boundary` (A/B only) = warmup→measure phase reset / collector-worker restart, timed as `boundary_seconds`; `measure` = the official training/collection steady state. Full framework-specific details in `INSPECTION.md`.
- Device: CPU (`CUDA_VISIBLE_DEVICES=""`); protocol C/D hyperparameters in `ppo_protocols.py` (lr 3e-4 linear-decay, gamma .99, gae_lambda .95, clip .2, entropy .01, value .5, epochs 10, batch 64, grad-clip .5).

## Experiment A — environment scaling (collection)

| framework | N | throughput (trans/s) | mean time (s) | speedup (paired) | env-scaling efficiency (paired) | pairs (n) |
| --- | --- | --- | --- | --- | --- | --- |
| zeroRL | 1 | 1,926.8 ± 25.1 | 51.91 | 1.00 | 1.00 | 5 |
| zeroRL | 2 | 2,344.1 ± 64.4 | 42.69 | 1.22 ± 0.04 | 0.61 ± 0.02 | 5 |
| zeroRL | 4 | 3,760.2 ± 106.7 | 26.61 | 1.95 ± 0.06 | 0.49 ± 0.01 | 5 |
| zeroRL | 8 | 5,598.6 ± 87.7 | 17.87 | 2.91 ± 0.06 | 0.36 ± 0.01 | 5 |
| zeroRL | 16 | 7,283.1 ± 200.1 | 13.74 | 3.78 ± 0.14 | 0.24 ± 0.01 | 5 |
| SB3 | 1 | 1,887.7 ± 32.6 | 52.99 | 1.00 | 1.00 | 5 |
| SB3 | 2 | 2,321.5 ± 49.4 | 43.09 | 1.23 ± 0.03 | 0.62 ± 0.02 | 5 |
| SB3 | 4 | 3,770.5 ± 94.6 | 26.54 | 2.00 ± 0.06 | 0.50 ± 0.02 | 5 |
| SB3 | 8 | 5,547.9 ± 263.8 | 18.06 | 2.94 ± 0.10 | 0.37 ± 0.01 | 5 |
| SB3 | 16 | 7,280.2 ± 370.6 | 13.76 | 3.86 ± 0.22 | 0.24 ± 0.01 | 5 |
| TorchRL | 1 | 645.3 ± 5.6 | 154.97 | 1.00 | 1.00 | 5 |
| TorchRL | 2 | 848.1 ± 17.2 | 117.96 | 1.31 ± 0.03 | 0.66 ± 0.02 | 5 |
| TorchRL | 4 | 1,521.4 ± 10.1 | 65.73 | 2.36 ± 0.03 | 0.59 ± 0.01 | 5 |
| TorchRL | 8 | 2,660.1 ± 41.4 | 37.60 | 4.12 ± 0.04 | 0.52 ± 0.01 | 5 |
| TorchRL | 16 | 4,293.9 ± 174.2 | 23.32 | 6.65 ± 0.30 | 0.42 ± 0.02 | 5 |

![throughput](figures/expA_throughput.png)

![speedup](figures/expA_speedup.png)

![efficiency](figures/expA_efficiency.png)


## Experiment B — worker scaling (collection, N=8)

| framework | W | throughput (trans/s) | mean time (s) | speedup (paired) | worker parallel efficiency (paired) | pairs (n) |
| --- | --- | --- | --- | --- | --- | --- |
| zeroRL | 1 | 5,639.2 ± 181.1 | 17.75 | 1.00 | 1.00 | 5 |
| SB3 | 1 | 5,460.7 ± 248.0 | 18.34 | 1.00 | 1.00 | 5 |
| SB3 | 8 | 2,886.7 ± 56.7 | 34.65 | 0.53 ± 0.02 | 0.07 ± 0.00 | 5 |
| TorchRL | 1 | 2,619.0 ± 106.1 | 38.23 | 1.00 | 1.00 | 5 |
| TorchRL | 2 | 3,853.9 ± 161.9 | 25.99 | 1.47 ± 0.10 | 0.74 ± 0.05 | 5 |
| TorchRL | 4 | 3,105.6 ± 120.1 | 32.24 | 1.19 ± 0.04 | 0.30 ± 0.01 | 5 |
| TorchRL | 8 | 2,119.0 ± 76.3 | 47.24 | 0.81 ± 0.05 | 0.10 ± 0.01 | 5 |

**Unsupported (recorded, not run)**:

- zeroRL W=2: zeroRL has no multi-process environment path (vectorize_env -> SyncVectorEnv only); W>=2 unsupported
- zeroRL W=2: zeroRL has no multi-process environment path (vectorize_env -> SyncVectorEnv only); W>=2 unsupported
- zeroRL W=2: zeroRL has no multi-process environment path (vectorize_env -> SyncVectorEnv only); W>=2 unsupported
- zeroRL W=2: zeroRL has no multi-process environment path (vectorize_env -> SyncVectorEnv only); W>=2 unsupported
- zeroRL W=2: zeroRL has no multi-process environment path (vectorize_env -> SyncVectorEnv only); W>=2 unsupported
- zeroRL W=4: zeroRL has no multi-process environment path (vectorize_env -> SyncVectorEnv only); W>=2 unsupported
- zeroRL W=4: zeroRL has no multi-process environment path (vectorize_env -> SyncVectorEnv only); W>=2 unsupported
- zeroRL W=4: zeroRL has no multi-process environment path (vectorize_env -> SyncVectorEnv only); W>=2 unsupported
- zeroRL W=4: zeroRL has no multi-process environment path (vectorize_env -> SyncVectorEnv only); W>=2 unsupported
- zeroRL W=4: zeroRL has no multi-process environment path (vectorize_env -> SyncVectorEnv only); W>=2 unsupported
- zeroRL W=8: zeroRL has no multi-process environment path (vectorize_env -> SyncVectorEnv only); W>=2 unsupported
- zeroRL W=8: zeroRL has no multi-process environment path (vectorize_env -> SyncVectorEnv only); W>=2 unsupported
- zeroRL W=8: zeroRL has no multi-process environment path (vectorize_env -> SyncVectorEnv only); W>=2 unsupported
- zeroRL W=8: zeroRL has no multi-process environment path (vectorize_env -> SyncVectorEnv only); W>=2 unsupported
- zeroRL W=8: zeroRL has no multi-process environment path (vectorize_env -> SyncVectorEnv only); W>=2 unsupported
- SB3 W=2: SB3 SubprocVecEnv runs exactly one process per env, so only W=1 (DummyVecEnv) and W=8 (SubprocVecEnv) exist; W=2 unsupported
- SB3 W=2: SB3 SubprocVecEnv runs exactly one process per env, so only W=1 (DummyVecEnv) and W=8 (SubprocVecEnv) exist; W=2 unsupported
- SB3 W=2: SB3 SubprocVecEnv runs exactly one process per env, so only W=1 (DummyVecEnv) and W=8 (SubprocVecEnv) exist; W=2 unsupported
- SB3 W=2: SB3 SubprocVecEnv runs exactly one process per env, so only W=1 (DummyVecEnv) and W=8 (SubprocVecEnv) exist; W=2 unsupported
- SB3 W=2: SB3 SubprocVecEnv runs exactly one process per env, so only W=1 (DummyVecEnv) and W=8 (SubprocVecEnv) exist; W=2 unsupported
- SB3 W=4: SB3 SubprocVecEnv runs exactly one process per env, so only W=1 (DummyVecEnv) and W=8 (SubprocVecEnv) exist; W=4 unsupported
- SB3 W=4: SB3 SubprocVecEnv runs exactly one process per env, so only W=1 (DummyVecEnv) and W=8 (SubprocVecEnv) exist; W=4 unsupported
- SB3 W=4: SB3 SubprocVecEnv runs exactly one process per env, so only W=1 (DummyVecEnv) and W=8 (SubprocVecEnv) exist; W=4 unsupported
- SB3 W=4: SB3 SubprocVecEnv runs exactly one process per env, so only W=1 (DummyVecEnv) and W=8 (SubprocVecEnv) exist; W=4 unsupported
- SB3 W=4: SB3 SubprocVecEnv runs exactly one process per env, so only W=1 (DummyVecEnv) and W=8 (SubprocVecEnv) exist; W=4 unsupported

![throughput](figures/expB_throughput.png)

![speedup](figures/expB_speedup.png)

![efficiency](figures/expB_efficiency.png)


## Experiment C — PPO training protocol (fixed 2048 steps/env per rollout)

| framework | N | updates | throughput (trans/s) | mean time (s) | mean return (last 100) | reps |
| --- | --- | --- | --- | --- | --- | --- |
| zeroRL | 1 | 128 | 1,121.7 ± 131.9 | 236.0 | -93.9 ± 16.0 | 3 |
| zeroRL | 2 | 64 | 1,316.5 ± 27.9 | 199.2 | -89.2 ± 106.6 | 3 |
| zeroRL | 4 | 32 | 1,712.3 ± 12.7 | 153.1 | -85.6 ± 38.8 | 3 |
| zeroRL | 8 | 16 | 2,003.3 ± 40.8 | 130.9 | -36.8 ± 20.8 | 3 |
| zeroRL | 16 | 8 | 2,200.9 ± 21.6 | 119.1 | -43.6 ± 16.3 | 3 |
| SB3 | 1 | 128 | 1,037.1 ± 13.7 | 252.8 | -71.5 ± 33.5 | 3 |
| SB3 | 2 | 64 | 1,148.1 ± 6.4 | 228.3 | -105.7 ± 64.5 | 3 |
| SB3 | 4 | 32 | 1,389.1 ± 23.3 | 188.7 | -63.7 ± 44.0 | 3 |
| SB3 | 8 | 16 | 1,163.9 ± 248.5 | 231.5 | -77.8 ± 12.0 | 3 |
| SB3 | 16 | 8 | 1,014.6 ± 88.8 | 259.7 | -77.0 ± 18.0 | 3 |
| TorchRL | 1 | 128 | 347.0 ± 33.4 | 759.9 | -71.0 ± 29.0 | 3 |
| TorchRL | 2 | 64 | 477.9 ± 13.0 | 548.8 | -108.1 ± 58.3 | 3 |
| TorchRL | 4 | 32 | 634.0 ± 4.3 | 413.5 | -92.8 ± 10.4 | 3 |
| TorchRL | 8 | 16 | 782.3 ± 13.3 | 335.1 | -57.7 ± 39.9 | 3 |
| TorchRL | 16 | 8 | 884.6 ± 1.1 | 296.4 | -39.4 ± 11.4 | 3 |

![throughput](figures/expC_throughput.png)

![returns](figures/expC_returns.png)


## Experiment D — PPO training protocol (fixed global rollout of 2048)

| framework | N | updates | throughput (trans/s) | mean time (s) | mean return (last 100) | reps |
| --- | --- | --- | --- | --- | --- | --- |
| zeroRL | 1 | 128 | 1,092.9 ± 26.9 | 240.0 | -93.9 ± 16.0 | 3 |
| zeroRL | 2 | 128 | 1,221.0 ± 15.4 | 214.7 | -74.5 ± 140.2 | 3 |
| zeroRL | 4 | 128 | 1,585.6 ± 9.8 | 165.3 | -142.7 ± 52.2 | 3 |
| zeroRL | 8 | 128 | 1,823.5 ± 39.3 | 143.8 | -101.0 ± 98.2 | 3 |
| zeroRL | 16 | 128 | 1,881.7 ± 42.7 | 139.4 | -37.3 ± 67.3 | 3 |
| SB3 | 1 | 128 | 832.5 ± 23.6 | 315.0 | -71.5 ± 33.5 | 3 |
| SB3 | 2 | 128 | 941.6 ± 1.3 | 278.4 | -117.1 ± 104.9 | 3 |
| SB3 | 4 | 128 | 1,145.3 ± 5.0 | 228.9 | -118.7 ± 21.7 | 3 |
| SB3 | 8 | 128 | 1,287.2 ± 7.1 | 203.7 | -54.4 ± 36.8 | 3 |
| SB3 | 16 | 128 | 1,365.6 ± 10.2 | 192.0 | -145.7 ± 24.8 | 3 |
| TorchRL | 1 | 128 | 367.3 ± 0.5 | 713.8 | -71.0 ± 29.0 | 3 |
| TorchRL | 2 | 128 | 433.1 ± 3.3 | 605.3 | -138.5 ± 64.1 | 3 |
| TorchRL | 4 | 128 | 572.8 ± 12.2 | 457.8 | -82.0 ± 54.6 | 3 |
| TorchRL | 8 | 128 | 705.3 ± 19.0 | 371.8 | -118.4 ± 39.7 | 3 |
| TorchRL | 16 | 128 | 802.6 ± 10.7 | 326.6 | -118.9 ± 71.6 | 3 |

![throughput](figures/expD_throughput.png)

![returns](figures/expD_returns.png)


## Findings (computed from the data)

- Exp A collection, N=1: fastest = **zeroRL** (1,927 trans/s); also SB3 1,888, TorchRL 645.
- Exp A collection, N=2: fastest = **zeroRL** (2,344 trans/s); also SB3 2,322, TorchRL 848.
- Exp A collection, N=4: fastest = **SB3** (3,770 trans/s); also zeroRL 3,760, TorchRL 1,521.
- Exp A collection, N=8: fastest = **zeroRL** (5,599 trans/s); also SB3 5,548, TorchRL 2,660.
- Exp A collection, N=16: fastest = **zeroRL** (7,283 trans/s); also SB3 7,280, TorchRL 4,294.
- Exp B collection, W=1 (N=8): fastest = **zeroRL** (5,639 trans/s); also SB3 5,461, TorchRL 2,619.
- Exp B collection, W=2 (N=8): fastest = **TorchRL** (3,854 trans/s).
- Exp B collection, W=4 (N=8): fastest = **TorchRL** (3,106 trans/s).
- Exp B collection, W=8 (N=8): fastest = **SB3** (2,887 trans/s); also TorchRL 2,119.
- Exp C PPO, N=1: fastest = **zeroRL** (1,122 trans/s); also SB3 1,037, TorchRL 347.
- Exp C PPO, N=2: fastest = **zeroRL** (1,316 trans/s); also SB3 1,148, TorchRL 478.
- Exp C PPO, N=4: fastest = **zeroRL** (1,712 trans/s); also SB3 1,389, TorchRL 634.
- Exp C PPO, N=8: fastest = **zeroRL** (2,003 trans/s); also SB3 1,164, TorchRL 782.
- Exp C PPO, N=16: fastest = **zeroRL** (2,201 trans/s); also SB3 1,015, TorchRL 885.
- Exp D PPO, N=1: fastest = **zeroRL** (1,093 trans/s); also SB3 833, TorchRL 367.
- Exp D PPO, N=2: fastest = **zeroRL** (1,221 trans/s); also SB3 942, TorchRL 433.
- Exp D PPO, N=4: fastest = **zeroRL** (1,586 trans/s); also SB3 1,145, TorchRL 573.
- Exp D PPO, N=8: fastest = **zeroRL** (1,824 trans/s); also SB3 1,287, TorchRL 705.
- Exp D PPO, N=16: fastest = **zeroRL** (1,882 trans/s); also SB3 1,366, TorchRL 803.
- zeroRL env-scaling efficiency at N=16: 0.24 ± 0.01 (speedup 3.78 ± 0.14, n=5 paired reps, 16x envs).
- SB3 env-scaling efficiency at N=16: 0.24 ± 0.01 (speedup 3.86 ± 0.22, n=5 paired reps, 16x envs).
- TorchRL env-scaling efficiency at N=16: 0.42 ± 0.02 (speedup 6.65 ± 0.30, n=5 paired reps, 16x envs).
- zeroRL worker parallel efficiency at W=1: 1.00 (speedup 1.00, n=5 paired reps, 1x workers).
- SB3 worker parallel efficiency at W=8: 0.07 ± 0.00 (speedup 0.53 ± 0.02, n=5 paired reps, 8x workers).
- TorchRL worker parallel efficiency at W=8: 0.10 ± 0.01 (speedup 0.81 ± 0.05, n=5 paired reps, 8x workers).
- zeroRL N=1: training throughput C=1,122 vs D=1,093 trans/s (C faster).
- SB3 N=1: training throughput C=1,037 vs D=833 trans/s (C faster).
- TorchRL N=1: training throughput C=347 vs D=367 trans/s (D faster).

## Data integrity

| experiment | ok | failed | unsupported |
| --- | --- | --- | --- |
| A | 75 | 0 | 0 |
| B | 35 | 0 | 25 |
| C | 45 | 0 | 0 |
| D | 45 | 0 | 0 |

All ok rows recorded an exact transition budget.

Total rows: 225.

## Appendix — protocol C/D configuration (from a result row)

| key | value |
| --- | --- |
| env_id | LunarLander-v3 |
| rollout_steps_per_env | 2048 |
| rollout_global | 2048 |
| updates | 128 |
| total | 262144 |
| lr | 0.0003 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_eps | 0.2 |
| ent_coef | 0.01 |
| vf_coef | 0.5 |
| epochs | 10 |
| batch_size | 64 |
| grad_clip | 0.5 |
| lr_start | 0.0003 |
| lr_end | 0.0 |
| optimizer | {'type': 'Adam', 'eps': 1e-05, 'schedule': 'linear_decay_to_zero'} |
| net_arch | [64, 64] |
| reference_seed | 4242 |

---
See `README.md` (reproduction), `INSPECTION.md` (framework findings), `results/*.jsonl` (raw data).
