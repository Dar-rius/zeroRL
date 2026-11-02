# Resource-consumption benchmark report

Generated: 2026-11-01T12:28:12.122124+00:00 · results: /home/darrius/project/zerorl/benchmarks/resources/results

## Resource consumption (experiment R)

Fixed workloads, device CPU. CPU% = sum over the live process tree (multiprocess sums may exceed 100%); RSS summed across tree processes (shared pages may repeat — PSS/USS recorded in the raw rows); GPU = device-level telemetry only (workload does not use the GPU); energy recorded only when reliable telemetry exists.

| workload | framework | wall (s) | cpu util avg (%) | cpu total (s) | cpu s / 1k | rss peak (MiB) | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| collect | zeroRL | 29.5 ± 13.6 | 556.0 ± 45.4 | 158.0 ± 60.1 | 1.58 ± 0.60 | 821.3 ± 28.2 | 10 |
| collect | SB3 | 26.3 ± 8.2 | 574.8 ± 32.2 | 149.2 ± 38.0 | 1.49 ± 0.38 | 853.1 ± 19.5 | 10 |
| collect | TorchRL | 48.5 ± 10.7 | 413.0 ± 27.9 | 197.3 ± 29.9 | 1.97 ± 0.30 | 872.6 ± 18.6 | 10 |
| ppo | zeroRL | 212.6 ± 43.8 | 589.5 ± 16.4 | 1,245.5 ± 228.3 | 4.75 ± 0.87 | 1,092.4 ± 205.2 | 6 |
| ppo | SB3 | 230.2 ± 30.9 | 607.9 ± 8.8 | 1,397.1 ± 167.9 | 5.33 ± 0.64 | 980.4 ± 77.0 | 6 |
| ppo | TorchRL | 438.9 ± 63.2 | 544.5 ± 17.3 | 2,375.1 ± 266.9 | 9.06 ± 1.02 | 998.8 ± 76.8 | 6 |

- GPU telemetry: available (device-level only).
- Energy: unavailable — GPU power telemetry unsupported on this device and no RAPL/powercap energy interface for CPU/whole-system energy.
- Raw per-run rows: `results/resources.jsonl` (validation runs kept separately in `resources_validation.*`).
