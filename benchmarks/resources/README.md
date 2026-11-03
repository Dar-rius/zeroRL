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
validation + monitor-overhead probe), `samples/<workload>_<fw>_rep<n>.jsonl`
(raw per-interval monitor samples for every run; disable with `--no-samples`,
the kept path is recorded as `extra.samples_file`).

Requires `psutil` — installed by the `benchmarks` extra or the `dev`
dependency group; CPU/whole-system energy and GPU power are recorded as
unavailable when the host exposes no reliable telemetry (never fabricated).

GPU campaign files (`exp{X}_gpu.jsonl` / `resources_gpu.jsonl`) are auto-detected and rendered as separate device sections/figures.
