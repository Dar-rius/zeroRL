"""Generate REPORT.md from the benchmark result JSONL files.

All numbers are computed from the raw rows (aggregated over every
repetition, never best-of); findings sentences are derived programmatically
so the report cannot claim anything the data does not show. Missing or
partial experiment files degrade to skipped sections with a warning.

Usage::

    CUDA_VISIBLE_DEVICES="" python -m benchmarks.scalability.report
    python -m benchmarks.scalability.report --results-dir /tmp/smoke --out /tmp/REPORT.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scalability.aggregate import (  # noqa: E402
    FRAMEWORK_ORDER,
    aggregate_cells,
    curve,
    integrity_summary,
    load_flat,
    paired_curve,
    return_stats_by_point,
)

WARNINGS: list[str] = []


def _warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"[report] WARNING: {msg}", file=sys.stderr)


def _device(row: dict) -> str:
    """Device side of a result row (config wins, then environment)."""
    dev = ((row.get("config") or {}).get("device")
           or (row.get("environment") or {}).get("device"))
    return dev or "cpu"


def _load_sides(results_dir: Path, exp: str) -> dict[str, list[dict]]:
    """STRICT per-experiment file list (never glob): ``exp{X}.jsonl`` first,
    then the optional ``exp{X}_gpu.jsonl``; missing files are silently
    skipped. Rows are split by their own device value — sides other than
    ``cpu``/``cuda`` are keyed by the value itself."""
    sides: dict[str, list[dict]] = {}
    for name in (f"exp{exp}.jsonl", f"exp{exp}_gpu.jsonl"):
        file = results_dir / name
        if not file.exists():
            continue
        for row in load_flat(file):
            sides.setdefault(_device(row), []).append(row)
    return sides


def _fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}"


def _pm(mean: float | None, std: float | None, digits: int = 1) -> str:
    if mean is None:
        return "—"
    if std:
        return f"{mean:,.{digits}f} ± {std:,.{digits}f}"
    return f"{mean:,.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join([" --- "] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _section_env(rows: list[dict]) -> list[str]:
    sample = next((r for r in rows if r["status"] == "ok"), rows[0])
    env = sample["environment"]
    lines = [
        "## Environment",
        "",
        _table(["key", "value"], [[k, str(v)] for k, v in env.items()]),
        "",
    ]
    return lines


def _section_method() -> list[str]:
    return [
        "## Methodology (summary)",
        "",
        "- Frameworks: zeroRL (this repo, official `rollout_phase` / "
        "`BaseTrain.train()`), Stable-Baselines3, TorchRL — same shared-trunk "
        "policy/weights where policies apply (see `policy_reference.py`).",
        "- Experiments A/B count individual env transitions with a warmup of "
        "16,000 (excluded) + 100,000 measured transitions per run; "
        "Experiments C/D train PPO for exactly 262,144 transitions per run.",
        "- Repetitions: 5 seeds per cell for A/B, 3 for C/D; every reported "
        "statistic aggregates **all** ok repetitions (mean ± std), never "
        "best-of.",
        "- Metrics: `throughput = measured / measure_seconds`, "
        "`speedup(N) = t(baseline) / t(N)`, "
        "`environment_scaling_efficiency(N) = speedup(N) / N`, "
        "`worker_parallel_efficiency(W) = speedup(W) / (W / 1)`.",
        "- Phase timing: `init` = env + policy construction + weight load "
        "(+ TorchRL collector/worker spawn); `warmup` (A/B only, excluded); "
        "`boundary` (A/B only) = warmup→measure phase reset / collector-"
        "worker restart, timed as `boundary_seconds`; `measure` = the "
        "official training/collection steady state. Full framework-specific "
        "details in `INSPECTION.md`.",
        "- Device: CPU (`CUDA_VISIBLE_DEVICES=\"\"`); protocol C/D "
        "hyperparameters in `ppo_protocols.py` (lr 3e-4 linear-decay, "
        "gamma .99, gae_lambda .95, clip .2, entropy .01, value .5, epochs 10, "
        "batch 64, grad-clip .5).",
        "",
    ]


def _exp_a_table(rows: list[dict], suffix: str = "") -> tuple[str, dict[str, list]]:
    cells = aggregate_cells(rows, "A")
    curve_by_fw = {}
    lines = []
    header = ["framework", "N", "throughput (trans/s)", "mean time (s)",
              "speedup (paired)", "env-scaling efficiency (paired)",
              "pairs (n)"]
    body: list[list[str]] = []
    for fw in FRAMEWORK_ORDER:
        if fw not in cells:
            continue
        try:
            c = paired_curve(rows, "A", fw, kind="environment")
        except ValueError as exc:
            _warn(f"exp A/{fw}: curve unavailable ({exc})")
            continue
        curve_by_fw[fw] = c
        for row in c:
            cell = cells[fw].get(row["point"], {})
            tp = cell.get("throughput", {})
            if row.get("n_pairs", 0) == 0:
                continue
            body.append([
                fw, str(row["point"]),
                _pm(tp.get("mean"), tp.get("std")),
                _fmt(cell.get("time_seconds", {}).get("mean"), 2),
                _pm(row["speedup_mean"], row["speedup_std"], 2),
                _pm(row["efficiency_mean"], row["efficiency_std"], 2),
                str(row["n_pairs"]),
            ])
    lines.append(_table(header, body))
    lines += ["", f"![throughput](figures/expA{suffix}_throughput.png)", "",
              f"![speedup](figures/expA{suffix}_speedup.png)", "",
              f"![efficiency](figures/expA{suffix}_efficiency.png)", ""]
    return "\n".join(lines), curve_by_fw


def _exp_b_table(rows: list[dict], suffix: str = "") -> tuple[str, dict[str, list]]:
    cells = aggregate_cells(rows, "B", point_key="num_workers")
    curve_by_fw: dict[str, list] = {}
    header = ["framework", "W", "throughput (trans/s)", "mean time (s)",
              "speedup (paired)", "worker parallel efficiency (paired)",
              "pairs (n)"]
    body: list[list[str]] = []
    for fw in FRAMEWORK_ORDER:
        if fw not in cells:
            continue
        try:
            c = paired_curve(rows, "B", fw, kind="worker",
                             point_key="num_workers")
        except ValueError as exc:
            _warn(f"exp B/{fw}: curve unavailable ({exc})")
            continue
        curve_by_fw[fw] = c
        for row in c:
            cell = cells[fw].get(row["point"], {})
            tp = cell.get("throughput", {})
            if row.get("n_pairs", 0) == 0:
                continue
            body.append([
                fw, str(row["point"]),
                _pm(tp.get("mean"), tp.get("std")),
                _fmt(cell.get("time_seconds", {}).get("mean"), 2),
                _pm(row["speedup_mean"], row["speedup_std"], 2),
                _pm(row["efficiency_mean"], row["efficiency_std"], 2),
                str(row["n_pairs"]),
            ])
    lines = [_table(header, body)]
    unsupported = integrity_summary(rows)["unsupported"]
    b_unsup = [u for u in unsupported if u[0] == "B"]
    if b_unsup:
        lines += ["", "**Unsupported (recorded, not run)**:", ""]
        for _, fw, _n, w, err in b_unsup:
            lines.append(f"- {fw} W={w}: {err}")
        lines.append("")
    lines += [f"![throughput](figures/expB{suffix}_throughput.png)", "",
              f"![speedup](figures/expB{suffix}_speedup.png)", "",
              f"![efficiency](figures/expB{suffix}_efficiency.png)", ""]
    return "\n".join(lines), curve_by_fw


def _exp_ppo_table(rows: list[dict], experiment: str,
                   suffix: str = "") -> tuple[str, dict[str, list]]:
    cells = aggregate_cells(rows, experiment)
    returns = return_stats_by_point(rows, experiment)
    header = ["framework", "N", "updates", "throughput (trans/s)",
              "mean time (s)", "mean return (last 100)", "reps"]
    body: list[list[str]] = []
    curve_by_fw: dict[str, list] = {}
    for fw in FRAMEWORK_ORDER:
        if fw not in cells:
            continue
        try:
            c = curve(rows, experiment, fw, kind="environment")
        except ValueError as exc:
            _warn(f"exp {experiment}/{fw}: curve unavailable ({exc})")
            continue
        curve_by_fw[fw] = c
        for point, cell in sorted(cells[fw].items()):
            if cell.get("n_ok", 0) == 0:
                body.append([fw, str(point), "—", "—", "—", "—", "0"])
                continue
            tp = cell["throughput"]
            ret = returns.get(fw, {}).get(point)
            update_counts = sorted({
                r["config"].get("updates", "—")
                for r in rows
                if (r["experiment"] == experiment and r["framework"] == fw
                    and r["num_envs"] == point and r["status"] == "ok")
            })
            updates = str(update_counts[0]) if len(update_counts) == 1 else "/".join(
                str(u) for u in update_counts)
            body.append([
                fw, str(point), updates,
                _pm(tp["mean"], tp["std"]),
                _fmt(cell["time_seconds"]["mean"], 1),
                _pm(ret["mean"] if ret else None,
                    ret["std"] if ret else None),
                str(cell["n_ok"]),
            ])
    lines = [_table(header, body), "",
             f"![throughput](figures/exp{experiment}{suffix}_throughput.png)", ""]
    if returns:
        lines += [f"![returns](figures/exp{experiment}{suffix}_returns.png)", ""]
    return "\n".join(lines), curve_by_fw


def _findings(curves: dict[str, dict[str, dict[str, list]]],
              rows: list[dict]) -> list[str]:
    lines = ["## Findings (computed from the data)", ""]
    a, b = curves["A"], curves["B"]

    def best_at(cell_map: dict[str, dict[int, dict]], point: int,
                label: str) -> str:
        ranked = []
        for fw in FRAMEWORK_ORDER:
            cell = cell_map.get(fw, {}).get(point)
            if cell and cell.get("n_ok", 0) > 0:
                ranked.append((cell["throughput"]["mean"], fw))
        if not ranked:
            return f"- {label}: no successful runs."
        ranked.sort(reverse=True)
        top = ranked[0]
        others = ", ".join(f"{fw} {tp:,.0f}" for tp, fw in ranked[1:])
        return (f"- {label}: fastest = **{top[1]}** "
                f"({top[0]:,.0f} trans/s)" + (f"; also {others}." if others else "."))

    cells_a = aggregate_cells(rows, "A")
    cells_b = aggregate_cells(rows, "B", point_key="num_workers")
    cells_c = aggregate_cells(rows, "C")
    cells_d = aggregate_cells(rows, "D")
    for point in sorted({p for cells in cells_a.values() for p in cells}):
        lines.append(best_at(cells_a, point, f"Exp A collection, N={point}"))
    for point in sorted({p for cells in cells_b.values() for p in cells}):
        lines.append(best_at(cells_b, point,
                             f"Exp B collection, W={point} (N=8)"))
    for point in sorted({p for cells in cells_c.values() for p in cells}):
        lines.append(best_at(cells_c, point, f"Exp C PPO, N={point}"))
    for point in sorted({p for cells in cells_d.values() for p in cells}):
        lines.append(best_at(cells_d, point, f"Exp D PPO, N={point}"))

    for fw in FRAMEWORK_ORDER:
        c_curve = a.get(fw) or []
        last = [r for r in c_curve if r.get("n_pairs", 0) > 0]
        if last:
            r = last[-1]
            lines.append(
                f"- {fw} env-scaling efficiency at N={r['point']}: "
                f"{_pm(r['efficiency_mean'], r['efficiency_std'], 2)} "
                f"(speedup {_pm(r['speedup_mean'], r['speedup_std'], 2)}, "
                f"n={r['n_pairs']} paired reps, {r['point']}x envs)."
            )
    for fw in FRAMEWORK_ORDER:
        b_curve = b.get(fw) or []
        last = [r for r in b_curve if r.get("n_pairs", 0) > 0]
        if last:
            r = last[-1]
            lines.append(
                f"- {fw} worker parallel efficiency at W={r['point']}: "
                f"{_pm(r['efficiency_mean'], r['efficiency_std'], 2)} "
                f"(speedup {_pm(r['speedup_mean'], r['speedup_std'], 2)}, "
                f"n={r['n_pairs']} paired reps, {r['point']}x workers)."
            )

    # C vs D: same budget, different batch structure
    for fw in FRAMEWORK_ORDER:
        cc = cells_c.get(fw, {})
        dd = cells_d.get(fw, {})
        common = sorted(set(cc) & set(dd))
        for point in common:
            if cc[point].get("n_ok") and dd[point].get("n_ok"):
                tpc = cc[point]["throughput"]["mean"]
                tpd = dd[point]["throughput"]["mean"]
                mid = (tpc + tpd) / 2
                if abs(tpc - tpd) / mid < 0.01:
                    verdict = "≈ equal"
                else:
                    verdict = f"{'C' if tpc > tpd else 'D'} faster"
                lines.append(
                    f"- {fw} N={point}: training throughput "
                    f"C={tpc:,.0f} vs D={tpd:,.0f} trans/s ({verdict})."
                )
                break  # report the shared-N comparison once per framework
    lines.append("")
    return lines


def _integrity(rows: list[dict],
               side_counts: list[tuple[str, int]] | None = None) -> list[str]:
    summary = integrity_summary(rows)
    lines = ["## Data integrity", "",
             _table(["experiment", "ok", "failed", "unsupported"],
                    [[exp, str(s.get("ok", 0)), str(s.get("failed", 0)),
                      str(s.get("unsupported", 0))]
                     for exp, s in summary["counts"].items()]),
             ""]
    if summary["inexact_budgets"]:
        lines += ["**Budget deviations (must be empty for ok rows):**", ""]
        lines += [f"- {x}" for x in summary["inexact_budgets"]]
        lines.append("")
    else:
        lines += ["All ok rows recorded an exact transition budget.", ""]
    if summary["failed"]:
        lines += ["**Failed rows:**", ""]
        lines += [f"- exp {e} {fw} N={n} rep={r}: {err}"
                  for e, fw, n, r, err in summary["failed"]]
        lines.append("")
    n_other = 0
    if side_counts:
        for dev, n in side_counts:
            n_other += n
            label = "GPU" if dev == "cuda" else dev
            lines += [f"{label} rows: {n} (device={dev})", ""]
    lines += [f"Total rows: {summary['n_rows'] + n_other}.", ""]
    return lines


def _hyperparams(rows: list[dict]) -> list[str]:
    sample = next((r for r in rows
                   if r["experiment"] in ("C", "D") and r["status"] == "ok"),
                  None)
    if sample is None:
        return []
    cfg = {k: v for k, v in sample["config"].items() if k != "seeds"}
    return ["## Appendix — protocol C/D configuration (from a result row)", "",
            _table(["key", "value"], [[k, str(v)] for k, v in cfg.items()]), ""]


def generate_report(results_dir: Path, out_path: Path) -> Path:
    WARNINGS.clear()
    sides_by_exp: dict[str, dict[str, list[dict]]] = {}
    rows: list[dict] = []                      # CPU side (drives findings etc.)
    other_rows: dict[str, list[dict]] = {}     # non-cpu sides, keyed by device
    for exp in ("A", "B", "C", "D"):
        file = results_dir / f"exp{exp}.jsonl"
        if not file.exists():
            _warn(f"{file.name} missing — section skipped")
        sides = _load_sides(results_dir, exp)
        sides_by_exp[exp] = sides
        rows.extend(sides.get("cpu", []))
        for dev, dev_rows in sides.items():
            if dev != "cpu":
                other_rows.setdefault(dev, []).extend(dev_rows)
    n_other = sum(len(v) for v in other_rows.values())
    if not rows and not n_other:
        raise SystemExit(f"no result rows found under {results_dir}")
    total = len(rows) + n_other

    curves: dict[str, dict[str, dict[str, list]]] = {
        "A": {}, "B": {}, "C": {}, "D": {},
    }
    lines = [
        "# Scalability benchmark report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()} · "
        f"rows: {total} · figures in `figures/`",
        "",
        *(_section_env(rows or [r for rs in other_rows.values() for r in rs])),
        *(_section_method()),
    ]

    titles = {
        "A": "## Experiment A — environment scaling (collection)",
        "B": "## Experiment B — worker scaling (collection, N=8)",
        "C": ("## Experiment C — PPO training protocol "
              "(fixed 2048 steps/env per rollout)"),
        "D": ("## Experiment D — PPO training protocol "
              "(fixed global rollout of 2048)"),
    }
    for exp in ("A", "B", "C", "D"):
        sides = sides_by_exp[exp]
        gpu_sides = {d: rs for d, rs in sides.items() if d != "cpu"}
        # Existing CPU section first (byte-identical for CPU-only dirs).
        if (results_dir / f"exp{exp}.jsonl").exists() and (
                sides.get("cpu") or not gpu_sides):
            if exp == "A":
                table, curves["A"] = _exp_a_table(sides.get("cpu", []))
            elif exp == "B":
                table, curves["B"] = _exp_b_table(sides.get("cpu", []))
            else:
                table, curves[exp] = _exp_ppo_table(sides.get("cpu", []), exp)
            lines += [titles[exp], "", table, ""]
        # Sibling section(s) for every non-cpu side — same renderer, own
        # rows, `_gpu` figure infix; never mixed with the CPU statistics.
        for dev, dev_rows in gpu_sides.items():
            label = "GPU" if dev == "cuda" else dev
            lines += [
                f"## Experiment {exp} — {label} (device={dev})", "",
                f"Rows recorded with device={dev}; every statistic in this "
                "section is computed on this side only and never combined "
                "with CPU rows.", "",
            ]
            if exp == "A":
                table, _ = _exp_a_table(dev_rows, suffix="_gpu")
            elif exp == "B":
                table, _ = _exp_b_table(dev_rows, suffix="_gpu")
            else:
                table, _ = _exp_ppo_table(dev_rows, exp, suffix="_gpu")
            lines += [table, ""]

    lines += _findings(curves, rows)
    lines += _integrity(rows, [(d, len(rs)) for d, rs in other_rows.items()])
    lines += _hyperparams(rows)
    lines += ["---",
              "See `README.md` (reproduction), `INSPECTION.md` "
              "(framework findings), `results/*.jsonl` (raw data).", ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="generate REPORT.md")
    p.add_argument("--results-dir",
                   default=str(REPO_ROOT / "benchmarks/scalability/results"))
    p.add_argument("--out", default=None,
                   help="output markdown (default: <results-dir>/REPORT.md)")
    args = p.parse_args(argv)
    results_dir = Path(args.results_dir)
    out = Path(args.out) if args.out else results_dir / "REPORT.md"
    generate_report(results_dir, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
