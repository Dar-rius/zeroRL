"""Resource-consumption benchmark (experiment R) — zeroRL vs SB3 vs TorchRL.

Answers how much CPU, memory, (GPU/energy where telemetry exists) and
wall-clock a framework needs to complete the SAME fixed RL workload as the
scalability benchmark. It REUSES the existing runners (``run_collection`` /
``run_ppo_training``) — no duplicated training loops — and only wraps them
in a process-tree resource monitor.

Workloads (existing configurations, nothing invented):

- ``collect``: Experiment A collection protocol — LunarLander-v3, N=8,
  warmup 16,000 + measure 100,000 individual transitions, 5 reps.
- ``ppo``:     protocol D PPO — fixed global rollout 2048, N=4,
  exactly 262,144 training transitions, 3 reps (seed = 1000 + rep).

Conventions (documented because they affect interpretation):

- CPU utilization is the SUM over the live process tree (root + all
  descendants discovered every sample) of per-interval CPU-time deltas /
  wall dt — multiprocessing workloads can exceed 100% and are NOT capped.
  PIDs born during the window contribute their full CPU history from first
  sight (they were created inside the window); PIDs present at ``start()``
  are baselined there. A PID that exits loses at most the final
  ``interval`` seconds of CPU time.
- Memory: RSS/PSS/USS are summed across live tree processes each sample
  (summed RSS can double-count shared pages — PSS/USS are also recorded).
- The monitor window is the FULL runner call (init + training); the
  artifact's phase wall-times are recorded alongside for separation.
- GPU: workload device is CPU (dev-box sm_61 cannot run these CUDA
  builds); NVML provides device telemetry only. Energy is unavailable on
  this host (NVML power unsupported, no RAPL) and is recorded as such —
  never fabricated.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import statistics
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Literal

import psutil

from benchmarks.scalability.results_schema import append_row, make_row

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMEWORKS_R = ("zeroRL", "SB3", "TorchRL")
WORKLOADS = ("collect", "ppo")
DEFAULT_INTERVAL_S = 0.2
COLLECT_NUM_ENVS = 8
PPO_EXPERIMENT = "D"
PPO_NUM_ENVS = 4
DEFAULT_OUT = REPO_ROOT / "benchmarks/scalability/results/resources.jsonl"


# ---------------------------------------------------------------------------
# hardware / GPU metadata (NVML via ctypes — no extra dependency)
# ---------------------------------------------------------------------------

def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def _gpu_name_oneshot() -> str:
    """Static metadata only (never used for high-frequency sampling)."""
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return "none"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return (out.stdout.strip().splitlines() or ["unknown"])[0]
    except Exception:
        return "unknown"


class NvmlBackend:
    """Minimal ctypes NVML sampler (utilization + memory; power probed)."""

    name = "nvml-ctypes"

    def __init__(self) -> None:
        lib_path = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
        self._nv = ctypes.CDLL(lib_path)
        if self._nv.nvmlInit_v2() != 0:
            raise RuntimeError("nvmlInit_v2 failed")
        self._handle = ctypes.c_void_p()
        if self._nv.nvmlDeviceGetHandleByIndex_v2(
                0, ctypes.byref(self._handle)) != 0:
            raise RuntimeError("nvmlDeviceGetHandleByIndex_v2 failed")

        class _U(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("mem", ctypes.c_uint)]

        class _M(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong),
                        ("free", ctypes.c_ulonglong),
                        ("used", ctypes.c_ulonglong)]
        self._u_type, self._m_type = _U, _M
        self._nv.nvmlDeviceGetUtilizationRates.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_U)]
        self._nv.nvmlDeviceGetMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_M)]
        self._nv.nvmlDeviceGetPowerUsage.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        mem = _M()
        if self._nv.nvmlDeviceGetMemoryInfo(
                self._handle, ctypes.byref(mem)) != 0:
            raise RuntimeError("nvmlDeviceGetMemoryInfo failed")
        self.total_bytes = int(mem.total)
        # power support probe (WSL often returns NOT_SUPPORTED)
        pwr = ctypes.c_uint()
        rc = self._nv.nvmlDeviceGetPowerUsage(self._handle, ctypes.byref(pwr))
        self.power_supported = rc == 0
        self.power_reason = ("" if rc == 0
                             else f"nvmlDeviceGetPowerUsage rc={rc} "
                                  "(NOT_SUPPORTED on this device)")

    def sample(self) -> dict[str, Any]:
        u, m = self._u_type(), self._m_type()
        out: dict[str, Any] = {}
        if self._nv.nvmlDeviceGetUtilizationRates(
                self._handle, ctypes.byref(u)) == 0:
            out["util_pct"] = int(u.gpu)
            out["mem_util_pct"] = int(u.mem)
        if self._nv.nvmlDeviceGetMemoryInfo(
                self._handle, ctypes.byref(m)) == 0:
            out["mem_used_bytes"] = int(m.used)
            out["mem_total_bytes"] = int(m.total)
        if self.power_supported:
            pwr = ctypes.c_uint()
            if self._nv.nvmlDeviceGetPowerUsage(
                    self._handle, ctypes.byref(pwr)) == 0:
                out["power_mW"] = int(pwr.value)
        return out


def make_gpu_backend() -> NvmlBackend | None:
    try:
        return NvmlBackend()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# energy integration (pure; used only when power telemetry exists)
# ---------------------------------------------------------------------------

def integrate_power_mw(times_s: list[float], power_mw: list[float]) -> float:
    """Trapezoidal integration of sampled power into joules."""
    if len(times_s) != len(power_mw) or len(times_s) < 2:
        return 0.0
    joules = 0.0
    for i in range(1, len(times_s)):
        dt = times_s[i] - times_s[i - 1]
        joules += (power_mw[i] + power_mw[i - 1]) / 2.0 / 1000.0 * dt
    return joules


# ---------------------------------------------------------------------------
# process-tree monitor
# ---------------------------------------------------------------------------

def aggregate_samples(
    samples: list[dict[str, Any]],
    *,
    duration_s: float,
    interval_s: float,
    gpu_backend_name: str | None,
    energy_note: str,
) -> dict[str, Any]:
    """Pure aggregation of recorded samples (unit-testable)."""
    cpu_pcts = [s["cpu_pct"] for s in samples if s.get("cpu_pct") is not None]
    user_s = sum(s.get("d_user", 0.0) for s in samples)
    sys_s = sum(s.get("d_sys", 0.0) for s in samples)

    def _mm(vals: list[float]) -> dict[str, float | None]:
        if not vals:
            return {"avg": None, "peak": None}
        return {"avg": round(statistics.fmean(vals), 3),
                "peak": round(max(vals), 3)}

    mem: dict[str, Any] = {
        "accounting": "sum over live process-tree processes per sample "
                      "(summed RSS may double-count shared pages; PSS/USS "
                      "recorded alongside)",
        "n_procs_avg": round(statistics.fmean(
            [s.get("n_procs", 1) for s in samples]), 2) if samples else 0,
    }
    for key in ("rss", "pss", "uss"):
        vals = [s[key] for s in samples if s.get(key) is not None]
        if vals:
            mem[f"{key}_init_kib"] = vals[0]
            mem[f"{key}_final_kib"] = vals[-1]
            mem[f"{key}_avg_kib"] = round(statistics.fmean(vals), 1)
            mem[f"{key}_peak_kib"] = max(vals)

    gpu_samples = [s["gpu"] for s in samples if s.get("gpu")]
    if gpu_backend_name and gpu_samples:
        gpu: dict[str, Any] = {
            "available": True,
            "backend": gpu_backend_name,
            "scope": "device-level telemetry; workload device is cpu",
        }
        utils = [g["util_pct"] for g in gpu_samples if "util_pct" in g]
        mems = [g["mem_used_bytes"] for g in gpu_samples if "mem_used_bytes" in g]
        if utils:
            gpu["util_avg_pct"] = round(statistics.fmean(utils), 2)
            gpu["util_peak_pct"] = max(utils)
        if mems:
            gpu["mem_avg_bytes"] = round(statistics.fmean(mems))
            gpu["mem_peak_bytes"] = max(mems)
        powers = [(s["t"], s["gpu"]["power_mW"])
                  for s in samples
                  if s.get("gpu") and "power_mW" in s["gpu"]]
        if len(powers) >= 2:
            gpu["power_avg_mW"] = round(statistics.fmean(
                [p for _, p in powers]), 1)
            gpu["power_peak_mW"] = max(p for _, p in powers)
    else:
        gpu = {"available": False,
               "reason": "NVML unavailable or produced no samples"}

    if any(s.get("gpu", {}).get("power_mW") is not None
           for s in samples if s.get("gpu")):
        times = [s["t"] for s in samples
                 if s.get("gpu") and "power_mW" in s["gpu"]]
        mws = [s["gpu"]["power_mW"] for s in samples
               if s.get("gpu") and "power_mW" in s["gpu"]]
        energy = {"available": True, "scope": "GPU only (not whole-system)",
                  "source": "NVML power, trapezoidal integration",
                  "gpu_joules": round(integrate_power_mw(times, mws), 3)}
    else:
        energy = {"available": False, "gpu_joules": None,
                  "cpu_joules": None, "system_joules": None,
                  "reason": energy_note}

    return {
        "window_seconds": round(duration_s, 3),
        "cpu": {
            "user_s": round(user_s, 3),
            "system_s": round(sys_s, 3),
            "total_cpu_s": round(user_s + sys_s, 3),
            "util_avg_pct": (round(statistics.fmean(cpu_pcts), 2)
                             if cpu_pcts else None),
            "util_peak_pct": (round(max(cpu_pcts), 2)
                              if cpu_pcts else None),
            "cores_logical": os.cpu_count(),
            "accounting": "sum of per-interval tree CPU-time deltas / dt; "
                          "can exceed 100% for multiprocess workloads",
        },
        "memory": mem,
        "gpu": gpu,
        "energy": energy,
        "sampling": {
            "interval_s": interval_s,
            "n_samples": len(samples),
            "backend": "psutil",
            "gpu_backend": gpu_backend_name,
        },
    }


class ResourceMonitor:
    """Samples the process tree (+ GPU) on a fixed interval in a thread.

    Discovers children created AFTER start() and tolerates children that
    exit before stop() (per-PID interval deltas; a lost exit tail is at
    most one interval).
    """

    def __init__(self, *, root_pid: int | None = None,
                 interval: float = DEFAULT_INTERVAL_S,
                 gpu: NvmlBackend | None | Callable[[], NvmlBackend | None] = None) -> None:
        if interval <= 0:
            raise ValueError(f"interval must be > 0, got {interval}")
        self.root_pid = root_pid if root_pid is not None else os.getpid()
        self.interval = interval
        if callable(gpu):
            gpu = gpu()
        self.gpu: NvmlBackend | None = gpu
        self._samples: list[dict[str, Any]] = []
        self._prev: dict[int, tuple[float, float]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t_start: float | None = None
        self.energy_note = (
            "GPU power telemetry unsupported on this device and no "
            "RAPL/powercap energy interface for CPU/whole-system energy"
        )

    # -- sampling ---------------------------------------------------------
    def _procs(self) -> list[psutil.Process]:
        root = psutil.Process(self.root_pid)
        out = [root]
        try:
            out += root.children(recursive=True)
        except psutil.Error:
            pass
        return out

    def _baseline(self) -> None:
        self._prev = {}
        for p in self._procs():
            try:
                ct = p.cpu_times()
                self._prev[p.pid] = (ct.user, ct.system)
            except psutil.Error:
                continue

    def _tick(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        d_user = d_sys = 0.0
        rss = pss = uss = 0
        n = 0
        for p in self._procs():
            try:
                ct = p.cpu_times()
            except psutil.Error:
                continue
            n += 1
            prev = self._prev.get(p.pid)
            if prev is None:
                # born after start(): its whole history is inside the window
                base_user, base_sys = 0.0, 0.0
            else:
                base_user, base_sys = prev
            self._prev[p.pid] = (ct.user, ct.system)
            d_user += max(0.0, ct.user - base_user)
            d_sys += max(0.0, ct.system - base_sys)
            try:
                mi = p.memory_full_info()
                rss += getattr(mi, "rss", 0)
                pss += getattr(mi, "pss", 0)
                uss += getattr(mi, "uss", 0)
            except psutil.Error:
                try:
                    rss += p.memory_info().rss
                except psutil.Error:
                    continue
        sample: dict[str, Any] = {
            "t": now, "n_procs": n,
            "d_user": d_user, "d_sys": d_sys,
            # psutil reports bytes; record KiB to match the field names
            "rss": rss // 1024, "pss": pss // 1024, "uss": uss // 1024,
        }
        prev_samples = self._samples
        if prev_samples:
            dt = now - prev_samples[-1]["t"]
            if dt > 0:
                sample["cpu_pct"] = (d_user + d_sys) / dt * 100.0
        if self.gpu is not None:
            try:
                sample["gpu"] = self.gpu.sample()
            except Exception:
                sample["gpu"] = {}
        self._samples.append(sample)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._tick()

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        self._baseline()
        self._t_start = time.monotonic()
        self._tick(self._t_start)          # first sample: init footprint
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 4 + 1.0)
        self._tick()
        duration = time.monotonic() - (self._t_start or time.monotonic())
        return aggregate_samples(
            self._samples,
            duration_s=duration,
            interval_s=self.interval,
            gpu_backend_name=self.gpu.name if self.gpu else None,
            energy_note=self.energy_note,
        )


# ---------------------------------------------------------------------------
# workload execution (reuses the existing runners — no duplicate loops)
# ---------------------------------------------------------------------------

def hardware_info(gpu: NvmlBackend | None) -> dict[str, Any]:
    return {
        "cpu_model": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "ram_total_gib": round(psutil.virtual_memory().total / 2**30, 2),
        "gpu_name": _gpu_name_oneshot(),
        "gpu_total_bytes": gpu.total_bytes if gpu else None,
        "monitor_backend": "psutil",
        "gpu_backend": gpu.name if gpu else None,
        "rapl_available": Path("/sys/class/powercap").is_dir(),
    }


def run_resource_workload(
    framework: str,
    workload: str,
    rep: int,
    *,
    interval: float = DEFAULT_INTERVAL_S,
    out_path: Path | None = None,
    collection_cfg: Any = None,
    ppo_cfg: Any = None,
    num_envs: int | None = None,
    gpu: Literal["auto"] | NvmlBackend | None
    | Callable[[], NvmlBackend | None] = "auto",
    monitor: ResourceMonitor | None = None,
) -> dict[str, Any]:
    """Run one fixed workload under monitoring and append a schema row.

    ``gpu="auto"`` (default) resolves to a fresh ``make_gpu_backend()``;
    explicit ``None`` disables GPU sampling and a callable is resolved by
    ``ResourceMonitor``. This is the single place that interprets ``"auto"``.
    """
    from benchmarks.scalability.collection import run_collection
    from benchmarks.scalability.ppo_protocols import run_ppo_training

    if gpu == "auto":  # sentinel: one place for auto-resolution
        gpu = make_gpu_backend()

    if workload == "collect":
        cfg = collection_cfg if collection_cfg is not None else _collect_cfg()
        n = num_envs if num_envs is not None else COLLECT_NUM_ENVS
    elif workload == "ppo":
        cfg = ppo_cfg if ppo_cfg is not None else _ppo_cfg()
        n = num_envs if num_envs is not None else PPO_NUM_ENVS
    else:
        raise ValueError(f"workload must be one of {WORKLOADS}, got {workload!r}")

    seed = 1000 + rep
    mon = monitor if monitor is not None else ResourceMonitor(
        interval=interval, gpu=gpu)
    status, err, art = "ok", None, None
    mon.start()
    t0 = time.perf_counter()
    try:
        if workload == "ppo":
            art = run_ppo_training(framework, PPO_EXPERIMENT, n, seed, cfg)
        else:
            art = run_collection(framework, cfg, n, seed)
    except Exception as exc:
        status = ("unsupported"
                  if type(exc).__name__ == "UnsupportedConfiguration"
                  else "failed")
        err = traceback.format_exc()
    wall_s = time.perf_counter() - t0
    stats = mon.stop()

    if art is not None:
        phases = art["phases"]
        transitions = art["transitions"]
    else:
        phases = {"init_seconds": 0.0, "warmup_seconds": 0.0,
                  "measure_seconds": 0.0}
        transitions = {"warmup_target": 0, "warmup_actual": 0,
                       "measure_target": 0, "measured": 0, "exact": False}

    measured = int(transitions.get("measured", 0) or 0)
    cpu = stats["cpu"]
    mem = stats["memory"]
    if status == "ok":
        metrics: dict[str, Any] = {
            "wall_s": round(wall_s, 3),
            "init_seconds": phases.get("init_seconds"),
            "measure_seconds": phases.get("measure_seconds"),
            "throughput": (measured / phases["measure_seconds"]
                           if phases.get("measure_seconds") else None),
            "cpu_util_avg_pct": cpu["util_avg_pct"],
            "cpu_util_peak_pct": cpu["util_peak_pct"],
            "cpu_total_s": cpu["total_cpu_s"],
            "cpu_s_per_1k": (round(cpu["total_cpu_s"] / measured * 1000, 4)
                             if measured else None),
            "rss_peak_kib": mem.get("rss_peak_kib"),
            "rss_avg_kib": mem.get("rss_avg_kib"),
            "uss_peak_kib": mem.get("uss_peak_kib"),
        }
    else:
        metrics = {"wall_s": round(wall_s, 3)}

    if workload == "collect":
        budget = {"warmup": cfg.warmup, "measure": cfg.measure}
    else:
        budget = {"total": cfg.total, "experiment": PPO_EXPERIMENT,
                  "updates": cfg.updates(PPO_EXPERIMENT, n)}
    config = {
        "workload": workload,
        "env_id": cfg.env_id,
        "device": "cpu",
        "interval_s": interval,
        "budget": budget,
        "num_envs": n,
        "seed_schedule": "1000 + rep",
    }

    row = make_row(
        framework=framework,
        experiment="R",
        mode="sync",
        num_envs=n,
        num_workers=1,
        protocol=workload,
        rep=rep,
        seed=seed,
        status=status,
        phases=phases,
        transitions=transitions,
        metrics=metrics,
        config=config,
        error=err,
        extra={"resources": stats, "hardware": hardware_info(mon.gpu)},
    )
    out = out_path if out_path is not None else DEFAULT_OUT
    append_row(out, row)
    # verify the row was written correctly before proceeding
    last_line = out.read_text(encoding="utf-8").strip().splitlines()[-1]
    json.loads(last_line)
    return row


def _collect_cfg() -> Any:
    from benchmarks.scalability.collection import CollectionConfig

    return CollectionConfig()


def _ppo_cfg() -> Any:
    from benchmarks.scalability.ppo_protocols import PPOProtocolConfig

    return PPOProtocolConfig()


# ---------------------------------------------------------------------------
# validation helper (short runs + monitoring-overhead probe; separate file)
# ---------------------------------------------------------------------------

def run_validation(out_dir: Path | None = None,
                   reps: int = 1) -> dict[str, Any]:
    """Short per-framework runs + overhead probe -> resources_validation.*.

    Validation output never lands in the campaign file, so preliminary
    runs can never be mistaken for repetitions.
    """
    from benchmarks.scalability.collection import CollectionConfig, run_collection

    out_dir = out_dir if out_dir is not None else REPO_ROOT / "benchmarks/scalability/results"
    val_rows = out_dir / "resources_validation.jsonl"
    if val_rows.exists():
        val_rows.unlink()
    tiny_collect = CollectionConfig(warmup=500, measure=1000)
    summary: dict[str, Any] = {"runs": [], "overhead_probe": []}
    for fw in FRAMEWORKS_R:
        row = run_resource_workload(
            fw, "collect", 99, interval=DEFAULT_INTERVAL_S,
            out_path=val_rows, collection_cfg=tiny_collect, num_envs=2)
        summary["runs"].append({
            "framework": fw, "status": row["status"],
            "cpu_total_s": row["metrics"].get("cpu_total_s"),
            "rss_peak_kib": row["extra"]["resources"]["memory"]
                            .get("rss_peak_kib"),
        })

    # overhead probe: identical tiny workload, monitor on vs off
    for i in range(2):
        t0 = time.perf_counter()
        run_collection("zeroRL", tiny_collect, 2, 950 + i)
        off = time.perf_counter() - t0
        mon = ResourceMonitor(interval=DEFAULT_INTERVAL_S, gpu=None)
        t0 = time.perf_counter()
        mon.start()
        run_collection("zeroRL", tiny_collect, 2, 950 + i)
        on = time.perf_counter() - t0
        mon.stop()
        summary["overhead_probe"].append({
            "pair": i, "monitor_off_s": round(off, 3),
            "monitor_on_s": round(on, 3),
            "overhead_pct": round((on - off) / off * 100.0, 2)
            if off else None,
        })
    path = out_dir / "resources_validation.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="resource-consumption benchmark (experiment R)")
    p.add_argument("--workload", choices=[*WORKLOADS, "all"], default="all")
    p.add_argument("--frameworks", default="all",
                   help="comma list: zeroRL,SB3,TorchRL (or 'all')")
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S,
                   help="resource sampling interval in seconds")
    p.add_argument("--reps-collect", type=int, default=5,
                   help="repetitions for the collect workload (A schedule)")
    p.add_argument("--reps-ppo", type=int, default=3,
                   help="repetitions for the ppo workload (C/D schedule)")
    p.add_argument("--out", default=None,
                   help="JSONL output (default: results/resources.jsonl)")
    p.add_argument("--validate", action="store_true",
                   help="short validation runs + overhead probe only")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out) if args.out else DEFAULT_OUT
    if args.validate:
        summary = run_validation()
        print(json.dumps(summary, indent=2))
        return 0

    if args.frameworks.strip().lower() == "all":
        frameworks = list(FRAMEWORKS_R)
    else:
        frameworks = [x.strip() for x in args.frameworks.split(",")
                      if x.strip()]
        bad = [f for f in frameworks if f not in FRAMEWORKS_R]
        if bad:
            raise SystemExit(f"unknown frameworks {bad}; valid: {list(FRAMEWORKS_R)}")

    workloads = list(WORKLOADS) if args.workload == "all" else [args.workload]
    n_rows = n_failed = 0
    for workload in workloads:
        reps = args.reps_collect if workload == "collect" else args.reps_ppo
        for fw in frameworks:
            for rep in range(reps):
                label = f"{fw:>7} {workload:>7} rep={rep} seed={1000 + rep}"
                try:
                    row = run_resource_workload(
                        fw, workload, rep, interval=args.interval,
                        out_path=out)
                except Exception:
                    n_failed += 1
                    print(f"[FAIL] {label}\n{traceback.format_exc()}")
                    continue
                if row["status"] == "failed":
                    n_failed += 1
                    print(f"[FAIL] {label}  "
                          f"{(row.get('error') or '').strip().splitlines()[-1]}")
                elif row["status"] == "unsupported":
                    print(f"[unsu] {label}  {(row.get('error') or '').strip()}")
                else:
                    m = row["metrics"]
                    print(
                        f"[ok]   {label}  wall {m['wall_s']:7.1f}s  "
                        f"cpu {m['cpu_total_s']:7.1f}s "
                        f"({m['cpu_util_avg_pct']}% avg)  "
                        f"rss_peak {m['rss_peak_kib'] / 1024:6.0f} MiB  "
                        f"cpu/1k {m['cpu_s_per_1k']}"
                    )
                n_rows += 1
    print(f"done: {n_rows} rows ({n_failed} failed) -> {out}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
