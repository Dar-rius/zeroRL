# Resource-consumption benchmark

Measures CPU, memory (process tree), GPU telemetry and energy for fixed RL
workloads — separate from the throughput benchmark in `benchmarks/scalability/`.
Reuses scalability's collection/PPO runners and row schema via imports.

Workloads (device CPU, seed = 1000 + rep):

- `collect` = Experiment A protocol, LunarLander-v3, N=8, 16,000 warmup +
  100,000 measured transitions, 5 reps × 3 frameworks (zeroRL, SB3, TorchRL).
- `ppo` = protocol D, N=4, exactly 262,144 transitions, 3 reps.

Entrypoints: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m benchmarks.resources.run | plots | report`

Outputs under `benchmarks/resources/results/`: `resources.jsonl` (campaign rows),
`REPORT.md`, `figures/` (`resources_*.png`), `resources_validation.*` (short
validation + monitor-overhead probe).
