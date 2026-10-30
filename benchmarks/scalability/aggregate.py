"""Load and group benchmark JSONL rows for plots/report generation.

Builds on :mod:`scaling_metrics` (which stays pure): this module knows the
*row shape* produced by ``run.py`` (nested ``phases``/``transitions``/
``metrics``) and flattens it into the flat keys ``aggregate_runs`` expects.

All aggregation is over **every** ok repetition of a cell (never best-of);
failed/unsupported rows are preserved for reporting but never enter stats.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from benchmarks.scalability.results_schema import load_rows
from benchmarks.scalability.scaling_metrics import (
    aggregate_runs,
    environment_scaling_efficiency,
    scaling_curve,
    speedup,
    summary_stats,
    worker_parallel_efficiency,
)

FRAMEWORK_ORDER = ("zeroRL", "SB3", "TorchRL")


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    """Inject the flat keys ``aggregate_runs`` consumes (non-destructive)."""
    flat = dict(row)
    flat["transitions_measured"] = row["transitions"].get("measured", 0)
    flat["measure_seconds"] = row["phases"].get("measure_seconds", 0.0)
    flat["init_seconds"] = row["phases"].get("init_seconds", 0.0)
    flat["throughput_mean"] = row.get("metrics", {}).get("throughput_mean")
    return flat


def load_flat(path: Path) -> list[dict[str, Any]]:
    return [flatten_row(r) for r in load_rows(path)]


def group_cells(rows: Iterable[dict[str, Any]], experiment: str, *,
                point_key: str = "num_envs") -> dict[str, dict[int, list[dict]]]:
    """framework -> point (N or W) -> all rows of that cell.

    Rows are flattened on the way in, so both raw ``make_row`` dicts and
    ``load_flat`` output work.
    """
    grouped: dict[str, dict[int, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row["experiment"] != experiment:
            continue
        row = flatten_row(row)
        grouped[row["framework"]][int(row[point_key])].append(row)
    return {
        fw: dict(cells) for fw, cells in grouped.items()
        if fw in FRAMEWORK_ORDER
    }


def aggregate_cells(rows: Iterable[dict[str, Any]], experiment: str, *,
                    point_key: str = "num_envs"
                    ) -> dict[str, dict[int, dict]]:
    """framework -> point -> ``aggregate_runs`` stats (ok rows only)."""
    grouped = group_cells(rows, experiment, point_key=point_key)
    return {
        fw: {point: aggregate_runs(cell) for point, cell in cells.items()}
        for fw, cells in grouped.items()
    }


def curve(rows: Iterable[dict[str, Any]], experiment: str, framework: str, *,
          kind: str = "environment", point_key: str = "num_envs",
          baseline_workers: int | None = 1) -> list[dict]:
    """scaling_curve for one framework of one experiment."""
    cells = aggregate_cells(rows, experiment, point_key=point_key).get(
        framework, {}
    )
    if not cells:
        return []
    return scaling_curve(cells, kind=kind, baseline_workers=baseline_workers)


def paired_curve(rows: Iterable[dict[str, Any]], experiment: str,
                 framework: str, *, kind: str = "environment",
                 point_key: str = "num_envs",
                 baseline: int = 1) -> list[dict]:
    """P4: per-repetition speedup/efficiency WITH real dispersion.

    Each ok row at point ``p`` is paired with the SAME repetition
    (``rep`` + ``seed``) at ``baseline``:

        speedup_r   = t_baseline_r / t_p_r
        efficiency_r = speedup_r / (p / baseline)     (env or worker)

    Reps present at ``p`` but missing at the baseline are skipped and
    counted in ``n_skipped``. Aggregation via :func:`summary_stats`
    (mean, sample std) + normal-approx 95% CI (mean ± 1.96·std/√n).

    Raises ``ValueError`` when the baseline point has no ok rows (same
    contract as :func:`curve`, so plot callers can catch/warn).
    """
    if kind not in ("environment", "worker"):
        raise ValueError(f"kind must be environment|worker, got {kind}")
    grouped: dict[int, dict[tuple[int, Any], float]] = defaultdict(dict)
    for row in rows:
        if row["experiment"] != experiment or row["framework"] != framework:
            continue
        if row.get("status") != "ok":
            continue
        flat = flatten_row(row)
        grouped[int(flat[point_key])][(flat["rep"], flat.get("seed"))] = float(
            flat["measure_seconds"]
        )
    if baseline not in grouped:
        raise ValueError(f"baseline point {baseline} missing or failed")

    def _half_width(stats: dict) -> float:
        if stats["n"] < 2:
            return 0.0
        return 1.96 * stats["std"] / math.sqrt(stats["n"])

    base = grouped[baseline]
    out: list[dict] = []
    for point in sorted(grouped):
        cell = grouped[point]
        speedups: list[float] = []
        skipped = 0
        for key, t_p in cell.items():
            t_b = base.get(key)
            if t_b is None:
                skipped += 1
                continue
            speedups.append(speedup(t_b, t_p))
        if not speedups:
            out.append({"point": point, "n_pairs": 0, "n_skipped": skipped,
                        "status": "no_pairs"})
            continue
        if kind == "environment":
            effs = [environment_scaling_efficiency(s, point, baseline)
                    for s in speedups]
        else:
            effs = [worker_parallel_efficiency(s, point, baseline)
                    for s in speedups]
        s_sp = summary_stats(speedups)
        s_ef = summary_stats(effs)
        half = _half_width(s_sp)
        out.append({
            "point": point,
            "n_pairs": s_sp["n"],
            "n_skipped": skipped,
            "speedup_mean": s_sp["mean"],
            "speedup_std": s_sp["std"],
            "speedup_ci95_lo": s_sp["mean"] - half,
            "speedup_ci95_hi": s_sp["mean"] + half,
            "efficiency_mean": s_ef["mean"],
            "efficiency_std": s_ef["std"],
        })
    return out


def return_stats_by_point(rows: Iterable[dict[str, Any]], experiment: str) -> dict[str, dict[int, dict]]:
    """framework -> N -> summary stats of ``mean_return_last100`` over reps
    (PPO experiments C/D only)."""
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["experiment"] != experiment or row["status"] != "ok":
            continue
        value = row["metrics"].get("mean_return_last100")
        if value is not None:
            grouped[(row["framework"], int(row["num_envs"]))].append(float(value))
    for (fw, point), values in grouped.items():
        out[fw][point] = summary_stats(values)
    return dict(out)


def integrity_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Row counts by experiment/status + budget-exactness flags for REPORT."""
    rows = list(rows)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row["experiment"]][row["status"]] += 1
    inexact = [
        (row["experiment"], row["framework"], row["num_envs"], row["rep"],
         row["transitions"].get("measured"))
        for row in rows
        if row["status"] == "ok" and not row["transitions"].get("exact", False)
    ]
    failed_errors = [
        (row["experiment"], row["framework"], row["num_envs"], row["rep"],
         (row.get("error") or "").splitlines()[-1][:200])
        for row in rows if row["status"] == "failed"
    ]
    unsupported = [
        (row["experiment"], row["framework"], row["num_envs"], row["num_workers"],
         row.get("error"))
        for row in rows if row["status"] == "unsupported"
    ]
    return {
        "counts": {k: dict(v) for k, v in sorted(counts.items())},
        "inexact_budgets": inexact,
        "failed": failed_errors,
        "unsupported": unsupported,
        "n_rows": len(rows),
    }
