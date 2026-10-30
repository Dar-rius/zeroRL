"""Generate benchmark figures (PNG) from the result JSONL files.

Every figure is derived from aggregated *all-repetition* statistics (mean ±
std); nothing is hand-picked. Experiments with no result file (or a missing
baseline point, e.g. while runs are still in flight) are skipped with a
warning instead of crashing.

Usage::

    CUDA_VISIBLE_DEVICES="" python -m benchmarks.scalability.plots
    python -m benchmarks.scalability.plots --results-dir /tmp/smoke --out /tmp/figs
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scalability.aggregate import (  # noqa: E402
    FRAMEWORK_ORDER,
    aggregate_cells,
    load_flat,
    paired_curve,
    return_stats_by_point,
)

FW_STYLE = {
    "zeroRL": {"color": "#1f77b4", "marker": "o"},
    "SB3": {"color": "#ff7f0e", "marker": "s"},
    "TorchRL": {"color": "#2ca02c", "marker": "^"},
}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"[fig]  {path}")
    return path


def _line_plot(series: dict[str, list[tuple[float, float, float]]],
               *, title: str, xlabel: str, ylabel: str,
               ideal: tuple[str, str] | None = None,
               hline: float | None = None,
               logy: bool = False, xscale_log2: bool = False):
    """series: framework -> [(x, mean, std)]; ideal = (label, "identity")."""
    plt = _pyplot()
    from matplotlib.ticker import ScalarFormatter

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fw in FRAMEWORK_ORDER:
        if fw not in series or not series[fw]:
            continue
        xs = [p[0] for p in series[fw]]
        ys = [p[1] for p in series[fw]]
        es = [p[2] for p in series[fw]]
        style = FW_STYLE[fw]
        ax.errorbar(xs, ys, yerr=es, label=fw, color=style["color"],
                    marker=style["marker"], linewidth=1.8, capsize=3)
    if ideal is not None:
        label, kind = ideal
        if kind != "identity":
            raise ValueError(f"unknown ideal kind {kind!r}")
        xs_all = sorted({p[0] for pts in series.values() for p in pts})
        ax.plot(xs_all, xs_all, "--", color="gray", linewidth=1.2,
                label=label)
    if hline is not None:
        ax.axhline(hline, linestyle="--", color="gray", linewidth=1.2,
                    label="ideal 1.0")
    if xscale_log2:
        ax.set_xscale("log", base=2)
        ticks = sorted({p[0] for pts in series.values() for p in pts})
        ax.set_xticks(ticks)
        fmt = ScalarFormatter()
        fmt.set_scientific(False)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_formatter(fmt)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def _tp_series(rows, experiment, point_key="num_envs"):
    cells = aggregate_cells(rows, experiment, point_key=point_key)
    series: dict[str, list[tuple[float, float, float]]] = {}
    for fw, by_point in cells.items():
        pts = [
            (float(point), cell["throughput"]["mean"], cell["throughput"]["std"])
            for point, cell in sorted(by_point.items())
            if cell.get("n_ok", 0) > 0
        ]
        if pts:
            series[fw] = pts
    return series


def _curve_series(rows, experiment, *, kind, point_key="num_envs",
                  baseline: int = 1) -> dict[str, list]:
    """Efficiency series with PAIRED per-repetition dispersion (P4).

    y-errors are the std of per-rep efficiencies (not throughput_std).
    """
    series: dict[str, list] = {}
    for fw in FRAMEWORK_ORDER:
        try:
            c = paired_curve(rows, experiment, fw, kind=kind,
                             point_key=point_key, baseline=baseline)
        except ValueError as exc:
            warnings.warn(f"{experiment}/{fw}: curve skipped ({exc})")
            continue
        pts = [(r["point"], r["efficiency_mean"], r["efficiency_std"])
               for r in c if r.get("n_pairs", 0) > 0]
        if pts:
            series[fw] = pts
    return series


def _speedup_series(rows, experiment, *, kind, point_key="num_envs",
                    baseline: int = 1) -> dict[str, list]:
    """Speedup series with PAIRED per-repetition dispersion (P4).

    y-errors are the std of per-rep speedups (was a hardcoded 0.0).
    """
    series: dict[str, list] = {}
    for fw in FRAMEWORK_ORDER:
        try:
            c = paired_curve(rows, experiment, fw, kind=kind,
                             point_key=point_key, baseline=baseline)
        except ValueError as exc:
            warnings.warn(f"{experiment}/{fw}: curve skipped ({exc})")
            continue
        pts = [(r["point"], r["speedup_mean"], r["speedup_std"])
               for r in c if r.get("n_pairs", 0) > 0]
        if pts:
            series[fw] = pts
    return series


def plot_experiment_a(rows, out_dir: Path) -> list[Path]:
    paths = []
    paths.append(_save(_line_plot(
        _tp_series(rows, "A"),
        title="Exp A — collection throughput vs env count (N)",
        xlabel="number of envs N (single process)",
        ylabel="transitions / second",
        logy=True, xscale_log2=True,
    ), out_dir, "expA_throughput.png"))
    paths.append(_save(_line_plot(
        _speedup_series(rows, "A", kind="environment"),
        title="Exp A — speedup vs N (equal-work collection)",
        xlabel="N", ylabel="speedup = t(N=1) / t(N)",
        ideal=("ideal (=N)", "identity"), xscale_log2=True,
    ), out_dir, "expA_speedup.png"))
    paths.append(_save(_line_plot(
        _curve_series(rows, "A", kind="environment"),
        title="Exp A — environment-scaling efficiency vs N",
        xlabel="N", ylabel="speedup(N) / N",
        hline=1.0, xscale_log2=True,
    ), out_dir, "expA_efficiency.png"))
    return paths


def plot_experiment_b(rows, out_dir: Path) -> list[Path]:
    paths = []
    paths.append(_save(_line_plot(
        _tp_series(rows, "B", point_key="num_workers"),
        title="Exp B — collection throughput vs worker count (N=8)",
        xlabel="workers W", ylabel="transitions / second",
        logy=True, xscale_log2=True,
    ), out_dir, "expB_throughput.png"))
    paths.append(_save(_line_plot(
        _speedup_series(rows, "B", kind="worker", point_key="num_workers"),
        title="Exp B — speedup vs W (equal-work collection)",
        xlabel="W", ylabel="speedup = t(W=1) / t(W)",
        ideal=("ideal (=W)", "identity"), xscale_log2=True,
    ), out_dir, "expB_speedup.png"))
    paths.append(_save(_line_plot(
        _curve_series(rows, "B", kind="worker", point_key="num_workers"),
        title="Exp B — worker parallel efficiency vs W",
        xlabel="W", ylabel="speedup(W) / W",
        hline=1.0, xscale_log2=True,
    ), out_dir, "expB_efficiency.png"))
    return paths


def plot_ppo_experiment(rows, experiment: str, out_dir: Path) -> list[Path]:
    paths = []
    paths.append(_save(_line_plot(
        _tp_series(rows, experiment),
        title=f"Exp {experiment} — PPO training throughput vs env count",
        xlabel="N (steps/env per update: "
               f"{'2048' if experiment == 'C' else '2048/N'})",
        ylabel="training transitions / second",
        logy=True, xscale_log2=True,
    ), out_dir, f"exp{experiment}_throughput.png"))
    returns = return_stats_by_point(rows, experiment)
    series = {
        fw: [(float(point), stats["mean"], stats["std"])
             for point, stats in sorted(by_point.items())]
        for fw, by_point in returns.items() if by_point
    }
    if series:
        paths.append(_save(_line_plot(
            series,
            title=f"Exp {experiment} — mean return (last 100 episodes) vs N",
            xlabel="N", ylabel="episode return (mean ± std over reps)",
            xscale_log2=True,
        ), out_dir, f"exp{experiment}_returns.png"))
    return paths


def _resource_cells(path: Path) -> dict[str, dict[str, list[dict]]]:
    from benchmarks.scalability.results_schema import load_rows

    cells: dict[str, dict[str, list[dict]]] = {}
    for r in load_rows(path):
        if r["status"] != "ok":
            continue
        wl = r["config"]["workload"]
        cells.setdefault(wl, {}).setdefault(r["framework"], []).append(r)
    return cells


def plot_resources(results_dir: Path, out_dir: Path) -> list[Path]:
    """Resource-consumption figures (experiment R).

    All configured runs are device=CPU; GPU/energy panels are generated
    only when the corresponding telemetry actually exists.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from benchmarks.scalability.results_schema import load_rows

    res_file = results_dir / "resources.jsonl"
    if not res_file.exists():
        return []
    rows = [r for r in load_rows(res_file) if r["status"] == "ok"]
    if not rows:
        return []
    cells = _resource_cells(res_file)
    workloads = [w for w in ("collect", "ppo") if w in cells]
    fws = [fw for fw in FRAMEWORK_ORDER
           if any(fw in cells[w] for w in workloads)]
    paths: list[Path] = []

    def _vals(wl: str, fw: str, getter):
        return [v for r in cells[wl].get(fw, [])
                if (v := getter(r)) is not None]

    def _grouped(ax, getter, ylabel: str, title: str, *, scale: float = 1.0,
                 digits: int = 2) -> None:
        xs = np.arange(len(fws))
        width = 0.8 / max(len(workloads), 1)
        for wi, wl in enumerate(workloads):
            means, stds = [], []
            for fw in fws:
                vals = [scale * v for v in _vals(wl, fw, getter)]
                if vals:
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals, ddof=1))
                                if len(vals) > 1 else 0.0)
                else:
                    means.append(0.0)
                    stds.append(0.0)
            ax.bar(xs - 0.4 + width * (wi + 0.5), means, width, yerr=stds,
                   label=f"{wl}", capsize=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(fws)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    M = lambda key: (lambda r: r["metrics"].get(key))  # noqa: E731

    # 1) memory peak/avg
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _grouped(axes[0], M("rss_peak_kib"), "MiB", "peak RSS (process tree)",
             scale=1 / 1024)
    _grouped(axes[1], M("rss_avg_kib"), "MiB", "average RSS (process tree)",
             scale=1 / 1024)
    fig.suptitle("System memory — sum of tree RSS (shared pages may repeat)", fontsize=10)
    fig.tight_layout()
    paths.append(_save(fig, out_dir, "resources_memory.png"))

    # 2) CPU utilization + cumulative CPU time
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _grouped(axes[0], M("cpu_util_avg_pct"), "avg CPU (%)",
             "average CPU utilization (tree sum, >100% possible)")
    _grouped(axes[1], M("cpu_total_s"), "CPU seconds",
             "cumulative user+system CPU time")
    fig.tight_layout()
    paths.append(_save(fig, out_dir, "resources_cpu.png"))

    # 3) CPU seconds per 1k transitions
    fig, ax = plt.subplots(figsize=(6.5, 4))
    _grouped(ax, M("cpu_s_per_1k"), "CPU s / 1k transitions",
             "CPU cost per 1,000 individual transitions")
    fig.tight_layout()
    paths.append(_save(fig, out_dir, "resources_cpu_per_1k.png"))

    # 4) GPU telemetry (only when available; workload device is cpu)
    gpu_rows = [r for r in rows
                if r["extra"]["resources"].get("gpu", {}).get("available")]
    if gpu_rows:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        g = lambda r, k: r["extra"]["resources"]["gpu"].get(k)  # noqa: E731
        _grouped(axes[0], lambda r: g(r, "util_avg_pct"), "util %",
                 "GPU utilization (device-level; workload device=cpu)")
        _grouped(axes[1], lambda r: g(r, "mem_peak_bytes"), "MiB",
                 "peak GPU memory (device-level)", scale=1 / 2**20)
        fig.tight_layout()
        paths.append(_save(fig, out_dir, "resources_gpu.png"))
    else:
        print("[fig] resources_gpu.png skipped: no GPU telemetry "
              "(workload device=cpu)")

    # 5) energy (only when reliable telemetry exists)
    energy_rows = [r for r in rows
                   if r["extra"]["resources"].get("energy", {}).get("available")]
    if energy_rows:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        _grouped(ax, lambda r: r["extra"]["resources"]["energy"]
                 .get("gpu_joules"), "joules",
                 "GPU energy (NVML integration; GPU-only, not system)")
        fig.tight_layout()
        paths.append(_save(fig, out_dir, "resources_energy.png"))
    else:
        reason = rows[0]["extra"]["resources"].get("energy", {}).get(
            "reason", "no reliable telemetry")
        print(f"[fig] resources_energy.png skipped: {reason}")

    # 6) end-to-end wall-clock
    fig, ax = plt.subplots(figsize=(6.5, 4))
    _grouped(ax, M("wall_s"), "seconds",
             "end-to-end wall time for the fixed workload")
    fig.tight_layout()
    paths.append(_save(fig, out_dir, "resources_wall.png"))

    # 7) phase wall-times (from the reused runners' artifacts)
    fig, axes = plt.subplots(1, len(workloads),
                             figsize=(5.5 * len(workloads), 4), squeeze=False)
    phase_keys = ["init_seconds", "warmup_seconds", "boundary_seconds",
                  "measure_seconds"]
    for ax, wl in zip(axes[0], workloads):
        bottoms = np.zeros(len(fws))
        for key in phase_keys:
            vals = []
            for fw in fws:
                pv = [r["phases"].get(key, 0.0) for r in cells[wl].get(fw, [])]
                vals.append(float(np.mean(pv)) if pv else 0.0)
            vals_a = np.array(vals)
            if float(vals_a.sum()) <= 0:
                continue
            ax.bar(fws, vals_a, bottom=bottoms,
                   label=key.replace("_seconds", ""))
            bottoms += vals_a
        ax.set_ylabel("seconds")
        ax.set_title(f"phase wall time — {wl}", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    paths.append(_save(fig, out_dir, "resources_phases.png"))

    return paths


def generate_all(results_dir: Path, out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or results_dir / "figures"
    paths: list[Path] = []
    for exp, plotter in (
        ("A", plot_experiment_a),
        ("B", plot_experiment_b),
    ):
        file = results_dir / f"exp{exp}.jsonl"
        if not file.exists():
            warnings.warn(f"{file} missing — skipping exp {exp}")
            continue
        paths.extend(plotter(load_flat(file), out_dir))
    for exp in ("C", "D"):
        file = results_dir / f"exp{exp}.jsonl"
        if not file.exists():
            warnings.warn(f"{file} missing — skipping exp {exp}")
            continue
        paths.extend(plot_ppo_experiment(load_flat(file), exp, out_dir))
    paths.extend(plot_resources(results_dir, out_dir))
    return paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="render benchmark figures")
    p.add_argument("--results-dir",
                   default=str(REPO_ROOT / "benchmarks/scalability/results"))
    p.add_argument("--out", default=None,
                   help="figure output dir (default: <results-dir>/figures)")
    args = p.parse_args(argv)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out) if args.out else None
    paths = generate_all(results_dir, out_dir)
    if not paths:
        print("no figures generated (no result files found?)")
        return 1
    print(f"generated {len(paths)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
