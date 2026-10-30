"""Scalability benchmark CLI.

Examples::

    # full Experiment A (all frameworks, all env counts, 5 reps)
    CUDA_VISIBLE_DEVICES="" python benchmarks/scalability/run.py --experiment A

    # full Experiment B (worker scaling, N=8)
    CUDA_VISIBLE_DEVICES="" python benchmarks/scalability/run.py --experiment B

    # full PPO protocol runs (C: fixed 2048 steps/env, D: fixed global 2048;
    # 262144 transitions each, 3 reps default)
    CUDA_VISIBLE_DEVICES="" python benchmarks/scalability/run.py --experiment C
    CUDA_VISIBLE_DEVICES="" python benchmarks/scalability/run.py --experiment D

    # quick smoke
    CUDA_VISIBLE_DEVICES="" python benchmarks/scalability/run.py \
        --experiment A --frameworks all --envs 2 --reps 1 \
        --warmup 1000 --measure 2000 --out /tmp/smokeA.jsonl

Every run appends one JSONL row (see results_schema.py); failures are
recorded as ``status="failed"`` rows with the traceback and configurations a
framework cannot express are ``status="unsupported"`` — never silently
dropped, never faked. ``CUDA_VISIBLE_DEVICES=""`` is mandatory on the dev
box (sm_61).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scalability.collection import (  # noqa: E402
    FRAMEWORKS_A,
    CollectionConfig,
    run_collection,
)
from benchmarks.scalability.ppo_protocols import (  # noqa: E402
    DEFAULT_PPO_TOTAL,
    EXPERIMENTS_CD,
    PPOProtocolConfig,
    run_ppo_training,
)
from benchmarks.scalability.results_schema import append_row, make_row  # noqa: E402
from benchmarks.scalability.workers import (  # noqa: E402
    NUM_ENVS_B,
    UnsupportedConfiguration,
    run_b_collection,
)


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_frameworks(text: str) -> list[str]:
    if text.strip().lower() == "all":
        return list(FRAMEWORKS_A)
    names = [x.strip() for x in text.split(",") if x.strip()]
    bad = [n for n in names if n not in FRAMEWORKS_A]
    if bad:
        raise SystemExit(f"unknown frameworks {bad}; valid: {list(FRAMEWORKS_A)}")
    return names


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="zeroRL scalability benchmark")
    p.add_argument("--experiment", choices=["A", "B", "C", "D"], default="A",
                   help="A=env scaling, B=worker scaling, "
                        "C/D=PPO training protocols")
    p.add_argument("--frameworks", default="all",
                   help="comma list: zeroRL,SB3,TorchRL (or 'all')")
    p.add_argument("--envs", default="1,2,4,8,16",
                   help="env counts (experiments A/C/D)")
    p.add_argument("--envs-b", type=int, default=NUM_ENVS_B,
                   help="fixed env count for experiment B (spec: 8)")
    p.add_argument("--workers", default="1,2,4,8",
                   help="worker counts (experiment B)")
    p.add_argument("--reps", type=int, default=None,
                   help="repetitions (default: 5 for A/B, 3 for C/D)")
    p.add_argument("--seed", type=int, default=1000,
                   help="base seed; rep r runs with seed+r (recorded per row)")
    p.add_argument("--warmup", type=int, default=16_000)
    p.add_argument("--measure", type=int, default=100_000)
    p.add_argument("--total", type=int, default=DEFAULT_PPO_TOTAL,
                   help="total PPO transitions for experiments C/D")
    p.add_argument("--out", default=None,
                   help="JSONL output (default: benchmarks/scalability/results/"
                        "exp<EXPERIMENT>.jsonl)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frameworks = _parse_frameworks(args.frameworks)
    is_ppo = args.experiment in EXPERIMENTS_CD
    reps = args.reps if args.reps is not None else (3 if is_ppo else 5)
    out = Path(args.out) if args.out else (
        REPO_ROOT / "benchmarks/scalability" / "results" / f"exp{args.experiment}.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    ppo_cfg: PPOProtocolConfig | None = None
    cfg = CollectionConfig(warmup=args.warmup, measure=args.measure)

    # (framework, num_envs, num_workers, mode)
    if args.experiment == "A":
        specs = [
            (fw, n, 1, "sync")
            for fw in frameworks
            for n in _parse_ints(args.envs)
        ]
    elif args.experiment == "B":
        workers = _parse_ints(args.workers)
        specs = [
            (fw, args.envs_b, w, "sync" if w == 1 else "mp")
            for fw in frameworks
            for w in workers
        ]
    else:
        ppo_cfg = PPOProtocolConfig(total=args.total)
        env_counts = _parse_ints(args.envs)
        for n in env_counts:
            try:
                ppo_cfg.updates(args.experiment, n)
            except ValueError as exc:
                raise SystemExit(f"invalid C/D configuration for N={n}: {exc}")
        specs = [(fw, n, 1, "sync") for fw in frameworks for n in env_counts]

    n_rows = n_failed = n_unsupported = 0
    for framework, num_envs, num_workers, mode in specs:
        for rep in range(reps):
            seed = args.seed + rep
            label = (f"{framework:>7} N={num_envs:>2} W={num_workers} "
                     f"rep={rep} seed={seed}")
            status, art, err = "ok", None, None
            try:
                if args.experiment == "A":
                    art = run_collection(framework, cfg, num_envs, seed)
                elif args.experiment == "B":
                    art = run_b_collection(framework, cfg, num_workers,
                                           seed, num_envs)
                else:
                    art = run_ppo_training(framework, args.experiment,
                                           num_envs, seed, ppo_cfg)
            except UnsupportedConfiguration as exc:
                status, err = "unsupported", str(exc)
            except Exception:
                status, err = "failed", traceback.format_exc()

            if is_ppo:
                row, throughput = _ppo_row(
                    args, framework, mode, num_envs, num_workers, rep, seed,
                    status, art, err, ppo_cfg,
                )
            elif status == "ok":
                assert art is not None
                phases = art["phases"]
                throughput = art["transitions"]["measured"] / phases["measure_seconds"]
                row = make_row(
                    framework=framework, experiment=args.experiment, mode=mode,
                    num_envs=num_envs, num_workers=num_workers,
                    protocol="collection", rep=rep, seed=seed, status="ok",
                    phases=phases, transitions=art["transitions"],
                    metrics={
                        "throughput_mean": throughput,
                        "init_seconds": phases["init_seconds"],
                        "warmup_seconds": phases["warmup_seconds"],
                        "measure_seconds": phases["measure_seconds"],
                    },
                    config={
                        "env_id": cfg.env_id, "warmup": cfg.warmup,
                        "measure": cfg.measure,
                        "reference_seed": cfg.reference_seed,
                        **art["config_detail"],
                    },
                    extra={"fingerprint": art["fingerprint"]},
                )
            else:
                phases = {"init_seconds": 0.0, "warmup_seconds": 0.0,
                          "measure_seconds": 0.0}
                row = make_row(
                    framework=framework, experiment=args.experiment, mode=mode,
                    num_envs=num_envs, num_workers=num_workers,
                    protocol="collection", rep=rep, seed=seed,
                    status=status, phases=phases,
                    transitions={"warmup_target": cfg.warmup, "warmup_actual": 0,
                                 "measure_target": cfg.measure,
                                 "measured": 0, "exact": False},
                    metrics={},
                    config={"env_id": cfg.env_id, "warmup": cfg.warmup,
                            "measure": cfg.measure},
                    error=err,
                )

            append_row(out, row)
            n_rows += 1
            if status == "ok":
                if is_ppo:
                    assert art is not None
                    print(f"[ok]   {label}  {throughput:9.1f} trans/s  "
                          f"(init {art['phases']['init_seconds']:.2f}s / "
                          f"meas {art['phases']['measure_seconds']:.2f}s, "
                          f"episodes {art['metrics']['episodes']})", flush=True)
                else:
                    phases = art["phases"]
                    print(f"[ok]   {label}  {throughput:9.1f} trans/s  "
                          f"(init {phases['init_seconds']:.2f}s / "
                          f"warm {phases['warmup_seconds']:.2f}s / "
                          f"meas {phases['measure_seconds']:.2f}s)", flush=True)
            elif status == "unsupported":
                n_unsupported += 1
                print(f"[UNS]  {label}  {err}", flush=True)
            else:
                n_failed += 1
                print(f"[FAIL] {label}\n{err}", flush=True)

    print(f"done: {n_rows} rows ({n_failed} failed, "
          f"{n_unsupported} unsupported) -> {out}")
    return 1 if n_failed else 0


def _ppo_row(args: argparse.Namespace, framework: str, mode: str,
             num_envs: int, num_workers: int, rep: int, seed: int,
             status: str, art: dict | None, err: str | None,
             ppo_cfg: PPOProtocolConfig):
    """Build (row, throughput) for one C/D run (ok or failed)."""
    assert ppo_cfg is not None
    if status == "ok":
        assert art is not None
        phases = art["phases"]
        throughput = art["transitions"]["measured"] / phases["measure_seconds"]
        row = make_row(
            framework=framework, experiment=args.experiment, mode=mode,
            num_envs=num_envs, num_workers=num_workers,
            protocol="ppo", rep=rep, seed=seed, status="ok",
            phases=phases, transitions=art["transitions"],
            metrics={
                "throughput_mean": throughput,
                "init_seconds": phases["init_seconds"],
                "measure_seconds": phases["measure_seconds"],
                **art["metrics"],
            },
            config={
                "env_id": ppo_cfg.env_id,
                **art["config_detail"],
            },
            extra={"episode_returns": art["episode_returns"]},
        )
        return row, throughput
    phases = {"init_seconds": 0.0, "measure_seconds": 0.0}
    row = make_row(
        framework=framework, experiment=args.experiment, mode=mode,
        num_envs=num_envs, num_workers=num_workers,
        protocol="ppo", rep=rep, seed=seed, status=status, phases=phases,
        transitions={"warmup_target": 0, "warmup_actual": 0,
                     "measure_target": ppo_cfg.total,
                     "measured": 0, "exact": False},
        metrics={},
        config={"env_id": ppo_cfg.env_id, "total": ppo_cfg.total},
        error=err,
    )
    return row, 0.0


if __name__ == "__main__":
    raise SystemExit(main())
