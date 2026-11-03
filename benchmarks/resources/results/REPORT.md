# Resource-consumption benchmark report

Generated: 2026-11-03T19:43:24.430693+00:00 · results: /home/darrius/project/zerorl/benchmarks/resources/results

## Resource consumption (experiment R)

Fixed workloads, device CPU. CPU% = sum over the live process tree (multiprocess sums may exceed 100%); RSS summed across tree processes (shared pages may repeat — PSS/USS recorded in the raw rows); GPU = device-level telemetry only (workload does not use the GPU); energy recorded only when reliable telemetry exists.

| workload | framework | wall (s) | cpu util avg (%) | cpu total (s) | cpu s / 1k | rss peak (MiB) | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| collect | zeroRL | 20.6 ± 2.1 | 603.9 ± 18.2 | 124.2 ± 11.2 | 1.24 ± 0.11 | 932.7 ± 22.8 | 5 |
| collect | SB3 | 17.2 ± 0.4 | 627.4 ± 8.4 | 108.2 ± 1.7 | 1.08 ± 0.02 | 964.8 ± 6.2 | 5 |
| collect | TorchRL | 36.9 ± 3.7 | 447.1 ± 12.5 | 164.7 ± 11.8 | 1.65 ± 0.12 | 983.2 ± 0.6 | 5 |
| ppo | zeroRL | 201.8 ± 7.0 | 619.7 ± 12.8 | 1,250.1 ± 34.6 | 4.77 ± 0.13 | 1,153.2 ± 217.5 | 3 |
| ppo | SB3 | 225.2 ± 4.7 | 639.6 ± 2.1 | 1,440.1 ± 26.2 | 5.49 ± 0.10 | 1,038.6 ± 3.3 | 3 |
| ppo | TorchRL | 479.5 ± 50.4 | 546.8 ± 17.4 | 2,609.5 ± 205.7 | 9.95 ± 0.78 | 1,057.4 ± 3.1 | 3 |

- GPU telemetry: available (device-level only).
- Energy: unavailable — GPU power telemetry unsupported on this device and no RAPL/powercap energy interface for CPU/whole-system energy.
- Raw per-run rows: `results/resources.jsonl` (validation runs kept separately in `resources_validation.*`).

_Memory note: raw values are bytes (v2; v1 rows/sample files used KiB); averages are time-weighted (trapezoidal); summed RSS may double-count shared pages — PSS/USS are the attributable views; v1 rows show — (never 0)._

### Memory (RSS / PSS / USS) — collect

**Peak (end-to-end)**

| framework | rss (MiB) | pss (MiB) | uss (MiB) | peak rss — measure phase (MiB) | n |
| --- | --- | --- | --- | --- | --- |
| zeroRL | 932.7 ± 22.8 | 930.1 ± 22.7 | 929.6 ± 22.7 | 928.9 ± 23.5 | 5 |
| SB3 | 964.8 ± 6.2 | 962.2 ± 6.1 | 961.6 ± 6.1 | 962.6 ± 6.5 | 5 |
| TorchRL | 983.2 ± 0.6 | 980.6 ± 0.6 | 980.0 ± 0.7 | 983.2 ± 0.6 | 5 |

**Average (end-to-end, time-weighted)**

| framework | rss (MiB) | pss (MiB) | uss (MiB) | n |
| --- | --- | --- | --- | --- |
| zeroRL | 927.3 ± 24.2 | 924.8 ± 24.0 | 924.2 ± 24.0 | 5 |
| SB3 | 961.4 ± 5.5 | 958.8 ± 5.5 | 958.2 ± 5.5 | 5 |
| TorchRL | 983.1 ± 0.6 | 980.5 ± 0.6 | 980.0 ± 0.7 | 5 |

### Memory (RSS / PSS / USS) — ppo

**Peak (end-to-end)**

| framework | rss (MiB) | pss (MiB) | uss (MiB) | peak rss — measure phase (MiB) | n |
| --- | --- | --- | --- | --- | --- |
| zeroRL | 1,153.2 ± 217.5 | 1,087.7 ± 123.6 | 1,028.1 ± 33.0 | 1,153.2 ± 217.6 | 3 |
| SB3 | 1,038.6 ± 3.3 | 1,027.5 ± 3.5 | 1,020.2 ± 3.5 | 1,038.6 ± 3.3 | 3 |
| TorchRL | 1,057.4 ± 3.1 | 1,046.2 ± 3.0 | 1,039.1 ± 3.2 | 1,057.4 ± 3.1 | 3 |

**Average (end-to-end, time-weighted)**

| framework | rss (MiB) | pss (MiB) | uss (MiB) | n |
| --- | --- | --- | --- | --- |
| zeroRL | 1,024.6 ± 3.4 | 1,013.4 ± 3.3 | 1,006.0 ± 3.5 | 3 |
| SB3 | 1,037.4 ± 3.7 | 1,026.3 ± 3.7 | 1,019.1 ± 3.7 | 3 |
| TorchRL | 1,056.5 ± 2.9 | 1,045.5 ± 2.7 | 1,038.3 ± 2.8 | 3 |
