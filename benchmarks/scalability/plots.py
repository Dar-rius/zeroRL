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


def plot_experiment_a(rows, out_dir: Path, suffix: str = "") -> list[Path]:
    """Exp A figures; ``suffix`` (e.g. ``"_gpu"``) only changes filenames."""
    paths = []
    paths.append(_save(_line_plot(
        _tp_series(rows, "A"),
        title="Exp A — collection throughput vs env count (N)",
        xlabel="number of envs N (single process)",
        ylabel="transitions / second",
        logy=True, xscale_log2=True,
    ), out_dir, f"expA{suffix}_throughput.png"))
    paths.append(_save(_line_plot(
        _speedup_series(rows, "A", kind="environment"),
        title="Exp A — speedup vs N (equal-work collection)",
        xlabel="N", ylabel="speedup = t(N=1) / t(N)",
        ideal=("ideal (=N)", "identity"), xscale_log2=True,
    ), out_dir, f"expA{suffix}_speedup.png"))
    paths.append(_save(_line_plot(
        _curve_series(rows, "A", kind="environment"),
        title="Exp A — environment-scaling efficiency vs N",
        xlabel="N", ylabel="speedup(N) / N",
        hline=1.0, xscale_log2=True,
    ), out_dir, f"expA{suffix}_efficiency.png"))
    return paths


def plot_experiment_b(rows, out_dir: Path, suffix: str = "") -> list[Path]:
    """Exp B figures; ``suffix`` (e.g. ``"_gpu"``) only changes filenames."""
    paths = []
    paths.append(_save(_line_plot(
        _tp_series(rows, "B", point_key="num_workers"),
        title="Exp B — collection throughput vs worker count (N=8)",
        xlabel="workers W", ylabel="transitions / second",
        logy=True, xscale_log2=True,
    ), out_dir, f"expB{suffix}_throughput.png"))
    paths.append(_save(_line_plot(
        _speedup_series(rows, "B", kind="worker", point_key="num_workers"),
        title="Exp B — speedup vs W (equal-work collection)",
        xlabel="W", ylabel="speedup = t(W=1) / t(W)",
        ideal=("ideal (=W)", "identity"), xscale_log2=True,
    ), out_dir, f"expB{suffix}_speedup.png"))
    paths.append(_save(_line_plot(
        _curve_series(rows, "B", kind="worker", point_key="num_workers"),
        title="Exp B — worker parallel efficiency vs W",
        xlabel="W", ylabel="speedup(W) / W",
        hline=1.0, xscale_log2=True,
    ), out_dir, f"expB{suffix}_efficiency.png"))
    return paths


def plot_ppo_experiment(rows, experiment: str, out_dir: Path,
                        suffix: str = "") -> list[Path]:
    """PPO protocol figures; ``suffix`` only changes output filenames."""
    paths = []
    paths.append(_save(_line_plot(
        _tp_series(rows, experiment),
        title=f"Exp {experiment} — PPO training throughput vs env count",
        xlabel="N (steps/env per update: "
               f"{'2048' if experiment == 'C' else '2048/N'})",
        ylabel="training transitions / second",
        logy=True, xscale_log2=True,
    ), out_dir, f"exp{experiment}{suffix}_throughput.png"))
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
        ), out_dir, f"exp{experiment}{suffix}_returns.png"))
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
        else:
            paths.extend(plotter(load_flat(file), out_dir))
        gpu_file = results_dir / f"exp{exp}_gpu.jsonl"   # silent when absent
        if gpu_file.exists():
            paths.extend(plotter(load_flat(gpu_file), out_dir, suffix="_gpu"))
    for exp in ("C", "D"):
        file = results_dir / f"exp{exp}.jsonl"
        if not file.exists():
            warnings.warn(f"{file} missing — skipping exp {exp}")
        else:
            paths.extend(plot_ppo_experiment(load_flat(file), exp, out_dir))
        gpu_file = results_dir / f"exp{exp}_gpu.jsonl"   # silent when absent
        if gpu_file.exists():
            paths.extend(plot_ppo_experiment(load_flat(gpu_file), exp, out_dir,
                                             suffix="_gpu"))
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
