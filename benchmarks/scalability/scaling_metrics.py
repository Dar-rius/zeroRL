"""Scaling metric formulas for the scalability benchmark.

Pure functions (stdlib + numpy only) so they can be unit-tested without any
RL framework installed.

Metric definitions (see INSPECTION.md / README.md):

- throughput = actual_transitions / measurement_seconds
- speedup(N) = time_at_N1 / time_at_N                 (equal-work times only)
- environment_scaling_efficiency(N) = speedup(N) / N
  -> efficiency relative to the environment-count increase. NOT physical CPU
     parallel efficiency when everything runs inside one process.
- worker_parallel_efficiency(W) = speedup(W) / (W / W_baseline)
  -> only defined when the worker count actually varies.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def throughput(transitions: int, seconds: float) -> float:
    """Transitions per second (individual env transitions)."""
    if seconds <= 0:
        raise ValueError(f"seconds must be > 0, got {seconds}")
    if transitions < 0:
        raise ValueError(f"transitions must be >= 0, got {transitions}")
    return transitions / seconds


def speedup(time_baseline: float, time_n: float) -> float:
    """speedup = time_at_baseline / time_at_n (same work budget)."""
    if time_baseline <= 0 or time_n <= 0:
        raise ValueError(
            f"times must be > 0 (baseline={time_baseline}, n={time_n})"
        )
    return time_baseline / time_n


def environment_scaling_efficiency(speedup_n: float, num_envs: int,
                                   baseline_envs: int = 1) -> float:
    """speedup(N) / (N / N_baseline) for environment scaling.

    Label this as efficiency relative to the environment-count increase; when
    execution stays in a single process it is not a CPU parallel efficiency.
    """
    if num_envs <= 0 or baseline_envs <= 0:
        raise ValueError(f"env counts must be > 0 ({num_envs}, {baseline_envs})")
    return speedup_n / (num_envs / baseline_envs)


def worker_parallel_efficiency(speedup_w: float, num_workers: int,
                               baseline_workers: int) -> float:
    """speedup(W) / (W / W_baseline) for worker scaling."""
    if num_workers <= 0 or baseline_workers <= 0:
        raise ValueError(
            f"worker counts must be > 0 ({num_workers}, {baseline_workers})"
        )
    if num_workers < baseline_workers:
        raise ValueError(
            f"num_workers {num_workers} below baseline {baseline_workers} "
            f"cannot produce a defined efficiency"
        )
    return speedup_w / (num_workers / baseline_workers)


def summary_stats(values: Iterable[float]) -> dict:
    """Mean/std/min/max over ALL repetitions (never best-of)."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("summary_stats requires at least one value")
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def aggregate_runs(
    runs: Sequence[dict],
    *,
    time_key: str = "measure_seconds",
    transitions_key: str = "transitions_measured",
) -> dict:
    """Aggregate repeated runs of one (framework, experiment, mode, N, W) cell.

    All runs must share the same work budget; the aggregate reports
    throughput stats over every repetition and mean time for speedup
    computation. Raises if the transitions differ between runs (equal-work
    requirement for speedup).
    """
    ok = [r for r in runs if r.get("status") == "ok"]
    if not ok:
        return {"n_ok": 0, "n_total": len(runs)}
    counts = {r[transitions_key] for r in ok}
    if len(counts) != 1:
        raise ValueError(
            f"runs in one cell must share the same transition count for "
            f"speedup math, got {sorted(counts)}"
        )
    throughputs = [throughput(r[transitions_key], r[time_key]) for r in ok]
    times = [r[time_key] for r in ok]
    t_stats = summary_stats(times)
    tp_stats = summary_stats(throughputs)
    return {
        "n_ok": len(ok),
        "n_total": len(runs),
        "transitions": next(iter(counts)),
        "time_seconds": t_stats,
        "throughput": tp_stats,
        "seeds": [r.get("seed") for r in ok],
    }


def scaling_curve(
    by_n: dict[int, dict],
    *,
    baseline_n: int = 1,
    kind: str = "environment",
    baseline_workers: int | None = None,
) -> list[dict]:
    """Build a per-N scaling table from aggregated cells.

    Args:
        by_n: mapping N (or W) -> aggregate dict from :func:`aggregate_runs`.
        baseline_n: baseline point for speedup (time at baseline).
        kind: "environment" or "worker".
        baseline_workers: required for kind == "worker".

    Returns one row per point with speedup and the matching efficiency
    metric (labelled explicitly).
    """
    if kind not in ("environment", "worker"):
        raise ValueError(f"kind must be environment|worker, got {kind}")
    if kind == "worker" and baseline_workers is None:
        raise ValueError("baseline_workers is required for worker curves")
    if baseline_n not in by_n or by_n[baseline_n].get("n_ok", 0) == 0:
        raise ValueError(f"baseline point {baseline_n} missing or failed")
    base_time = by_n[baseline_n]["time_seconds"]["mean"]
    rows: list[dict] = []
    for point in sorted(by_n):
        cell = by_n[point]
        if cell.get("n_ok", 0) == 0:
            rows.append({"point": point, "status": "no_successful_runs"})
            continue
        sp = speedup(base_time, cell["time_seconds"]["mean"])
        row = {
            "point": point,
            "n_ok": cell["n_ok"],
            "mean_time_s": cell["time_seconds"]["mean"],
            "throughput_mean": cell["throughput"]["mean"],
            "throughput_std": cell["throughput"]["std"],
            "speedup": sp,
        }
        if kind == "environment":
            row["environment_scaling_efficiency"] = environment_scaling_efficiency(
                sp, point, baseline_n
            )
        else:
            row["worker_parallel_efficiency"] = worker_parallel_efficiency(
                sp, point, int(baseline_workers)
            )
        rows.append(row)
    return rows


def is_finite_positive(x: float) -> bool:
    return math.isfinite(x) and x > 0
