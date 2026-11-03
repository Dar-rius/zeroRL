"""Generate resource-consumption benchmark figures (PNG) from the result JSONL.

Every figure is derived from aggregated *all-repetition* statistics (mean ±
std); nothing is hand-picked. GPU/energy panels are only generated when the
corresponding telemetry actually exists in the rows.

Usage::

    CUDA_VISIBLE_DEVICES="" python -m benchmarks.resources.plots
    python -m benchmarks.resources.plots --results-dir /tmp/smoke --out /tmp/figs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scalability.aggregate import FRAMEWORK_ORDER  # noqa: E402


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"[fig]  {path}")
    return path


def _resource_cells(path: Path) -> dict[str, dict[str, list[dict]]]:
    from benchmarks.scalability.results_schema import load_rows

    cells: dict[str, dict[str, list[dict]]] = {}
    for r in load_rows(path):
        if r["status"] != "ok":
            continue
        wl = r["config"]["workload"]
        cells.setdefault(wl, {}).setdefault(r["framework"], []).append(r)
    return cells


def _plot_resource_file(res_file: Path, out_dir: Path,
                        suffix: str = "") -> list[Path]:
    """Figures for ONE result file; ``suffix`` (e.g. ``"_gpu"``) only
    changes output filenames. Availability rules (GPU/energy telemetry)
    are evaluated on this file's own rows — one device side per call."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from benchmarks.scalability.results_schema import load_rows

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
    workload_device = "cuda" if suffix else "cpu"

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
    paths.append(_save(fig, out_dir, f"resources_memory{suffix}.png"))

    # 2) CPU utilization + cumulative CPU time
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _grouped(axes[0], M("cpu_util_avg_pct"), "avg CPU (%)",
             "average CPU utilization (tree sum, >100% possible)")
    _grouped(axes[1], M("cpu_total_s"), "CPU seconds",
             "cumulative user+system CPU time")
    fig.tight_layout()
    paths.append(_save(fig, out_dir, f"resources_cpu{suffix}.png"))

    # 3) CPU seconds per 1k transitions
    fig, ax = plt.subplots(figsize=(6.5, 4))
    _grouped(ax, M("cpu_s_per_1k"), "CPU s / 1k transitions",
             "CPU cost per 1,000 individual transitions")
    fig.tight_layout()
    paths.append(_save(fig, out_dir, f"resources_cpu_per_1k{suffix}.png"))

    # 4) GPU telemetry (only when available in THIS side's rows)
    gpu_rows = [r for r in rows
                if r["extra"]["resources"].get("gpu", {}).get("available")]
    telemetry_name = (f"resources_gpu{suffix}.png" if not suffix
                      else f"resources_telemetry{suffix}.png")
    if gpu_rows:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        g = lambda r, k: r["extra"]["resources"]["gpu"].get(k)  # noqa: E731
        _grouped(axes[0], lambda r: g(r, "util_avg_pct"), "util %",
                 f"GPU utilization (device-level; workload device={workload_device})")
        _grouped(axes[1], lambda r: g(r, "mem_peak_bytes"), "MiB",
                 "peak GPU memory (device-level)", scale=1 / 2**20)
        fig.tight_layout()
        paths.append(_save(fig, out_dir, telemetry_name))
    else:
        print(f"[fig] {telemetry_name} skipped: no GPU telemetry "
              f"(workload device={workload_device})")

    # 5) energy (only when reliable telemetry exists in THIS side)
    energy_rows = [r for r in rows
                   if r["extra"]["resources"].get("energy", {}).get("available")]
    if energy_rows:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        _grouped(ax, lambda r: r["extra"]["resources"]["energy"]
                 .get("gpu_joules"), "joules",
                 "GPU energy (NVML integration; GPU-only, not system)")
        fig.tight_layout()
        paths.append(_save(fig, out_dir, f"resources_energy{suffix}.png"))
    else:
        reason = rows[0]["extra"]["resources"].get("energy", {}).get(
            "reason", "no reliable telemetry")
        print(f"[fig] resources_energy{suffix}.png skipped: {reason}")

    # 6) end-to-end wall-clock
    fig, ax = plt.subplots(figsize=(6.5, 4))
    _grouped(ax, M("wall_s"), "seconds",
             "end-to-end wall time for the fixed workload")
    fig.tight_layout()
    paths.append(_save(fig, out_dir, f"resources_wall{suffix}.png"))

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
    paths.append(_save(fig, out_dir, f"resources_phases{suffix}.png"))

    return paths


def plot_resources(results_dir: Path, out_dir: Path,
                   suffix: str = "") -> list[Path]:
    """Resource-consumption figures (experiment R), both device sides.

    Renders ``resources{suffix}.jsonl`` (CPU side with the default
    ``suffix=""``; existing figure names unchanged), then — only when no
    explicit suffix is given — auto-detects ``resources_gpu.jsonl`` and
    renders the GPU side with ``suffix="_gpu"``. Statistics/telemetry
    availability are always evaluated within one side, never combined.
    """
    paths = _plot_resource_file(results_dir / f"resources{suffix}.jsonl",
                                out_dir, suffix)
    if not suffix:
        gpu_file = results_dir / "resources_gpu.jsonl"
        if gpu_file.exists():
            paths += _plot_resource_file(gpu_file, out_dir, "_gpu")
    return paths


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="render resource benchmark figures")
    p.add_argument("--results-dir",
                   default=str(REPO_ROOT / "benchmarks/resources/results"))
    p.add_argument("--out", default=None,
                   help="figure output dir (default: <results-dir>/figures)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out) if args.out else results_dir / "figures"
    paths = plot_resources(results_dir, out_dir)
    for p in paths:
        print(f"[fig] {p}")
    print(f"generated {len(paths)} figures")
    return 0 if paths else 1


if __name__ == "__main__":
    raise SystemExit(main())
