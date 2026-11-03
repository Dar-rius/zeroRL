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


# -- memory-measurement v2 (nested bytes block) -----------------------------

def _nested_memory(r: dict) -> dict | None:
    """The v2 nested memory block, or None for v1 rows (never coerced)."""
    mem = (r.get("extra", {}).get("resources", {}) or {}).get("memory") or {}
    return mem if isinstance(mem.get("rss"), dict) else None


def _mem_mib(r: dict, metric: str, field: str) -> float | None:
    """One row's metric value in MiB; None when absent/unavailable (v1)."""
    mem = _nested_memory(r)
    if mem is None:
        return None
    if metric != "rss":
        if (mem.get("availability") or {}).get(f"{metric}_available") is False:
            return None
    v = (mem.get(metric) or {}).get(field)
    return None if v is None else v / 2**20


def _measure_peak_rss_mib(r: dict) -> float | None:
    mbp = (r.get("extra", {}).get("resources", {}) or {}).get(
        "memory_by_phase") or {}
    v = ((mbp.get("measure") or {}).get("rss") or {}).get("peak_bytes")
    return None if v is None else v / 2**20


def _memory_subsections(ok: list[dict]) -> list[str]:
    """`### Memory (RSS / PSS / USS) — {workload}` tables (v2 rows only).

    Rendered ONLY when at least one ok row carries the nested bytes block;
    v1 rows within a mixed file show `—` (never 0). Cells = mean ± std in
    MiB over that framework's non-missing rows; `n` = framework row count.
    """
    if not any(_nested_memory(r) for r in ok):
        return []
    out = [
        "_Memory note: raw values are bytes (v2; v1 rows/sample files "
        "used KiB); averages are time-weighted (trapezoidal); summed RSS "
        "may double-count shared pages — PSS/USS are the attributable "
        "views; v1 rows show — (never 0)._", "",
    ]

    def _fw_table(headers: list[str], ok_rows: list[dict],
                  getters) -> str:
        body: list[list[str]] = []
        for fw in FRAMEWORK_ORDER:
            rs = [r for r in ok_rows if r["framework"] == fw]
            if not rs:
                continue
            body.append([fw] + [_pm2([g(r) for r in rs]) for g in getters]
                        + [str(len(rs))])
        return _table(headers, body)

    for wl in ("collect", "ppo"):
        wl_rows = [r for r in ok if r["config"]["workload"] == wl]
        if not wl_rows:
            continue
        out += [f"### Memory (RSS / PSS / USS) — {wl}", "",
                "**Peak (end-to-end)**", ""]
        out.append(_fw_table(
            ["framework", "rss (MiB)", "pss (MiB)", "uss (MiB)",
             "peak rss — measure phase (MiB)", "n"],
            wl_rows,
            [lambda r: _mem_mib(r, "rss", "peak_bytes"),
             lambda r: _mem_mib(r, "pss", "peak_bytes"),
             lambda r: _mem_mib(r, "uss", "peak_bytes"),
             _measure_peak_rss_mib]))
        out += ["", "**Average (end-to-end, time-weighted)**", ""]
        out.append(_fw_table(
            ["framework", "rss (MiB)", "pss (MiB)", "uss (MiB)", "n"],
            wl_rows,
            [lambda r: _mem_mib(r, "rss", "average_bytes"),
             lambda r: _mem_mib(r, "pss", "average_bytes"),
             lambda r: _mem_mib(r, "uss", "average_bytes")]))
        out.append("")
    return out



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
    out += _memory_subsections(ok)
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
    out += _memory_subsections(ok)
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
