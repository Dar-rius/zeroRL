"""Generate REPORT.md for the resource-consumption benchmark (experiment R).

All numbers are computed from the raw rows (aggregated over every
repetition, never best-of). GPU/energy notes reflect what the telemetry
actually recorded — unavailable stays explicit, never fabricated.

Usage::

    CUDA_VISIBLE_DEVICES="" python -m benchmarks.resources.report
    python -m benchmarks.resources.report --results-dir /tmp/smoke --out /tmp/REPORT.md
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scalability.aggregate import FRAMEWORK_ORDER  # noqa: E402


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join([" --- "] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _pm2(vals, digits: int = 1) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    m = statistics.fmean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:,.{digits}f} ± {s:,.{digits}f}"


def _resources_table(ok: list[dict]) -> str:
    """Workload×framework table over one side's ok rows (device-homogeneous)."""
    header = ["workload", "framework", "wall (s)", "cpu util avg (%)",
              "cpu total (s)", "cpu s / 1k", "rss peak (MiB)", "n"]
    body: list[list[str]] = []
    for wl in ("collect", "ppo"):
        for fw in FRAMEWORK_ORDER:
            rs = [r for r in ok
                  if r["config"]["workload"] == wl
                  and r["framework"] == fw]
            if not rs:
                continue
            ms = [r["metrics"] for r in rs]
            body.append([
                wl, fw,
                _pm2([m.get("wall_s") for m in ms]),
                _pm2([m.get("cpu_util_avg_pct") for m in ms]),
                _pm2([m.get("cpu_total_s") for m in ms]),
                _pm2([m.get("cpu_s_per_1k") for m in ms], 2),
                _pm2([(m.get("rss_peak_kib") or 0) / 1024 for m in ms]),
                str(len(rs)),
            ])
    return _table(header, body)


def _section_resources(results_dir: Path) -> list[str]:
    """Table + telemetry notes for the resource-consumption campaign (R)."""
    from benchmarks.scalability.results_schema import load_rows

    res_file = results_dir / "resources.jsonl"
    if not res_file.exists():
        return []
    rows_all = load_rows(res_file)
    ok = [r for r in rows_all if r["status"] == "ok"]
    out = [
        "## Resource consumption (experiment R)", "",
        "Fixed workloads, device CPU. CPU% = sum over the live process "
        "tree (multiprocess sums may exceed 100%); RSS summed across "
        "tree processes (shared pages may repeat — PSS/USS recorded in "
        "the raw rows); GPU = device-level telemetry only (workload does "
        "not use the GPU); energy recorded only when reliable telemetry "
        "exists.", "",
    ]
    if not ok:
        out += [f"_no ok rows ({len(rows_all)} total) — campaign "
                "incomplete or still running._", ""]
        return out

    out += [_resources_table(ok), ""]
    res0 = ok[0]["extra"]["resources"]
    gpu = res0.get("gpu", {})
    energy = res0.get("energy", {})
    out += [
        f"- GPU telemetry: {'available (device-level only)' if gpu.get('available') else 'unavailable — ' + str(gpu.get('reason', ''))}.",
        f"- Energy: {'available — ' + str(energy.get('scope', '')) if energy.get('available') else 'unavailable — ' + str(energy.get('reason', ''))}.",
        "- Raw per-run rows: `results/resources.jsonl` (validation runs "
        "kept separately in `resources_validation.*`).", "",
    ]
    return out


def _section_resources_gpu(results_dir: Path) -> list[str]:
    """Second side: optional GPU campaign rows (never mixed with CPU stats)."""
    from benchmarks.scalability.results_schema import load_rows

    res_file = results_dir / "resources_gpu.jsonl"
    if not res_file.exists():
        return []
    rows_all = load_rows(res_file)
    ok = [r for r in rows_all if r["status"] == "ok"]
    out = [
        "## Resource consumption — GPU (device=cuda)", "",
        "Fixed workloads, device GPU (row `device=cuda`). Every statistic "
        "below is computed from GPU rows only and never combined with the "
        "CPU side above. CPU% = sum over the live process tree; RSS summed "
        "across tree processes; GPU = device-level telemetry of the "
        "workload's own device; energy recorded only when reliable "
        "telemetry exists.", "",
    ]
    if not ok:
        out += [f"_no ok rows ({len(rows_all)} total) — campaign "
                "incomplete or still running._", ""]
        return out

    out += [_resources_table(ok), ""]
    res0 = ok[0]["extra"]["resources"]
    gpu = res0.get("gpu", {})
    energy = res0.get("energy", {})
    out += [
        f"- GPU telemetry (device side cuda): {'available (device-level)' if gpu.get('available') else 'unavailable — ' + str(gpu.get('reason', ''))}.",
        f"- Energy (device side cuda): {'available — ' + str(energy.get('scope', '')) if energy.get('available') else 'unavailable — ' + str(energy.get('reason', ''))}.",
        "- Raw per-run rows: `results/resources_gpu.jsonl`.", "",
    ]
    return out


def generate_report(results_dir: Path, out_path: Path) -> Path:
    body = _section_resources(results_dir)
    gpu_body = _section_resources_gpu(results_dir)
    if not body and not gpu_body:
        body = ["## Resource consumption (experiment R)", "",
                "_no ok rows yet._", ""]
    lines = [
        "# Resource-consumption benchmark report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()} · "
        f"results: {results_dir}",
        "",
        *body,
        *gpu_body,
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {out_path}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="generate resource REPORT.md")
    p.add_argument("--results-dir",
                   default=str(REPO_ROOT / "benchmarks/resources/results"))
    p.add_argument("--out", default=None,
                   help="output markdown (default: <results-dir>/REPORT.md)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = Path(args.results_dir)
    out = Path(args.out) if args.out else results_dir / "REPORT.md"
    generate_report(results_dir, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
