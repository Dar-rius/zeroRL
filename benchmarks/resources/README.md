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

## Memory measurement v2 (units rule)

- **Raw samples (`samples/*.jsonl`) are BYTES** (`rss`, `pss`, `uss`) since
  memory-measurement v2; v1 sample files used KiB. Every sample also carries
  `n_procs` and `full_ok` (procs where `psutil.memory_full_info()` succeeded).
  MiB/GiB appear only in report/plots — never in raw rows or samples.
- **Aggregates** (`extra.resources.memory`): canonical nested block
  `{"rss"|"pss"|"uss": {initial_bytes, average_bytes, peak_bytes, final_bytes},
  "availability": {backend: "psutil.memory_full_info", pss_available,
  uss_available, pss_missing_samples, uss_missing_samples}}` alongside the
  legacy flat `*_kib` fields (derived as `bytes // 1024`; v1 avg was
  arithmetic-of-kib, v2 avg is kib-of-time-weighted-bytes).
  - `average_*` = time-weighted (trapezoidal between successive valid samples);
    `peak_*` = max over the **aggregate** tree-sum time series (never summed
    per-proc maxima); partial samples (`full_ok < n_procs`) are excluded from
    that metric's series (never zero-filled) and counted as missing — a metric
    never fully available reports `available: false` with `None` avg/peak.
  - Summed RSS can double-count shared pages; PSS/USS are the attributable
    views.
- **Phase metrics** (`extra.resources.memory_by_phase`): per artifact phase
  (`init`, `warmup`, `boundary`, `measure` for collect; `init`, `measure` for
  ppo) the same 4 byte fields + `window_seconds`, sliced by cumulative phase
  bounds relative to the first sample (tail beyond the last bound stays
  end-to-end only). Failed runs (no artifact) get no `memory_by_phase`.
- **v1 vs v2**: existing `resources.jsonl` rows are v1 (flat `*_kib`, no
  nested block, no `samples_file`) and remain fully readable — report shows
  `—` for missing v2 fields, plots skip them; v1 files are versioned at
  campaign time, never rewritten here.

## New figures (additive; `resources_memory.png` legacy RSS figure untouched)

- `resources_memory_peak{suffix}.png` — per-workload subplots, framework
  groups × RSS/PSS/USS bars (MiB, yerr std over reps; unavailable metric
  skipped, never zero).
- `resources_memory_avg{suffix}.png` — same for time-weighted averages.
- `resources_memory_timeseries{suffix}.png` — (workload × framework) panels,
  RSS/PSS/USS MiB vs seconds from the per-run raw samples referenced by
  `extra.samples_file` (v1 rows/files skipped with a printed reason).

Requires `psutil` — installed by the `benchmarks` extra or the `dev`
dependency group; CPU/whole-system energy and GPU power are recorded as
unavailable when the host exposes no reliable telemetry (never fabricated).

GPU campaign files (`exp{X}_gpu.jsonl` / `resources_gpu.jsonl`) are auto-detected and rendered as separate device sections/figures.
