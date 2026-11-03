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
  UNITS: raw samples (``samples/*.jsonl``) are BYTES since memory-
  measurement v2 — v1 sample files used KiB. Aggregates expose canonical
  ``memory`` blocks in bytes (time-weighted averages) plus legacy flat
  ``*_kib`` fields derived as ``bytes // 1024``; MiB/GiB appear only in
  report/plots. A sample whose ``full_ok < n_procs`` is partial: its
  PSS/USS values are excluded from those metrics' series (never
  zero-filled) and counted in ``availability.*_missing_samples``.
- The monitor window is the FULL runner call (init + training); the
  artifact's phase wall-times are recorded alongside for separation and,
  when present, aggregated into ``memory_by_phase`` (per-phase time-
  weighted memory over cumulative phase bounds relative to the first
  sample).
- GPU: workload device is CPU (dev-box sm_61 cannot run these CUDA
  builds); NVML provides device telemetry only. Energy is unavailable on
  this host (NVML power unsupported, no RAPL) and is recorded as such —
  never fabricated.

Entrypoint::

    CUDA_VISIBLE_DEVICES="" python -m benchmarks.resources.run
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
DEFAULT_OUT = REPO_ROOT / "benchmarks/resources/results/resources.jsonl"


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

def _time_weighted_mean(times: list[float],
                        vals: list[float]) -> float | None:
    """Trapezoidal time-weighted mean of a (already filtered) series.

    ``sum((v[i]+v[i+1])/2 * (t[i+1]-t[i])) / (t[-1]-t[0])`` — irregular
    gaps are handled naturally. A single sample yields that value; an
    empty series yields ``None``. Missing samples are excluded from the
    series upstream, NEVER zero-filled.
    """
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    span = times[-1] - times[0]
    if span <= 0:
        return float(vals[0])
    acc = 0.0
    for i in range(1, len(vals)):
        acc += (vals[i] + vals[i - 1]) / 2.0 * (times[i] - times[i - 1])
    return acc / span


def _metric_valid(sample: dict[str, Any], key: str) -> bool:
    """A sample contributes to ``key``'s series iff present AND complete.

    ``rss`` is always present (memory_info fallback still yields it).
    ``pss``/``uss`` require a FULL sample (``full_ok >= n_procs``): partial
    samples are excluded, not zero-filled. Samples without the v2
    ``full_ok`` field (hand-made/legacy) count as present.
    """
    if sample.get(key) is None:
        return False
    if key == "rss":
        return True
    full_ok = sample.get("full_ok")
    if full_ok is None:
        return True
    return full_ok >= sample.get("n_procs", full_ok)


def _metric_block(samples: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """{initial, average(time-weighted), peak(max of aggregate series),
    final} in BYTES for one metric over a sample slice."""
    series = [(s["t"], s[key]) for s in samples if _metric_valid(s, key)]
    if not series:
        return {"initial_bytes": None, "average_bytes": None,
                "peak_bytes": None, "final_bytes": None}
    times = [t for t, _ in series]
    vals = [v for _, v in series]
    twa = _time_weighted_mean(times, vals)
    return {
        "initial_bytes": vals[0],
        "average_bytes": round(twa) if twa is not None else None,
        # peak = max over the AGGREGATE time series (never per-proc sums)
        "peak_bytes": max(vals),
        "final_bytes": vals[-1],
    }


def _availability(samples: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"backend": "psutil.memory_full_info"}
    for key in ("pss", "uss"):
        missing = sum(1 for s in samples if not _metric_valid(s, key))
        out[f"{key}_available"] = bool(samples) and missing == 0
        out[f"{key}_missing_samples"] = missing
    return out


_PHASE_ORDER = ("init", "warmup", "boundary", "measure")


def phase_bounds_from(phases: dict[str, float]) -> list[tuple[str, float]]:
    """Cumulative (name, end_seconds) bounds from artifact phase durations.

    Order: init → warmup → boundary → measure; zero/absent phases are
    skipped (ppo artifacts only carry init + measure).
    """
    bounds: list[tuple[str, float]] = []
    end = 0.0
    for name in _PHASE_ORDER:
        dur = phases.get(f"{name}_seconds")
        if dur is None or dur <= 0:
            continue
        end += float(dur)
        bounds.append((name, round(end, 6)))
    return bounds


def aggregate_samples(
    samples: list[dict[str, Any]],
    *,
    duration_s: float,
    interval_s: float,
    gpu_backend_name: str | None,
    energy_note: str,
    phase_bounds: list[tuple[str, float]] | None = None,
) -> dict[str, Any]:
    """Pure aggregation of recorded samples (unit-testable).

    Raw sample ``rss/pss/uss`` are BYTES (v2; v1 sample files were KiB).
    Legacy flat ``*_kib`` fields are derived as ``bytes // 1024`` from the
    same byte values (averages = time-weighted bytes // 1024). The
    canonical nested block lives under the returned ``"memory"`` key
    alongside the flat fields; ``phase_bounds`` (relative to the first
    sample) additionally yields ``"memory_by_phase"`` — omitted entirely
    when no bounds are given (failed runs).
    """
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
        "units": "bytes (raw samples; *_kib fields = bytes // 1024)",
        "n_procs_avg": round(statistics.fmean(
            [s.get("n_procs", 1) for s in samples]), 2) if samples else 0,
    }
    # canonical nested block: bytes + time-weighted averages + availability
    for key in ("rss", "pss", "uss"):
        mem[key] = _metric_block(samples, key)
    mem["availability"] = _availability(samples)
    # legacy flat fields, byte-compatible: bytes // 1024
    for key in ("rss", "pss", "uss"):
        blk = mem[key]
        if blk["initial_bytes"] is not None:
            mem[f"{key}_init_kib"] = blk["initial_bytes"] // 1024
            mem[f"{key}_final_kib"] = blk["final_bytes"] // 1024
            mem[f"{key}_peak_kib"] = blk["peak_bytes"] // 1024
            # NOTE: avg = kib-of-time-weighted-bytes (v1 was arithmetic-of-kib)
            mem[f"{key}_avg_kib"] = blk["average_bytes"] // 1024

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

    result = {
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

    # phase-specific memory: slices of the SAME aggregate series between
    # cumulative bounds (relative to the first sample); the tail beyond the
    # last bound stays out of phases but inside the end-to-end block above.
    if phase_bounds:
        t0 = samples[0]["t"] if samples else 0.0
        by_phase: dict[str, Any] = {}
        start = 0.0
        for i, (name, end) in enumerate(phase_bounds):
            first = i == 0
            sl = [s for s in samples
                  if (s["t"] - t0) <= end
                  and ((s["t"] - t0) >= start if first
                       else (s["t"] - t0) > start)]
            entry: dict[str, Any] = {"window_seconds": round(end - start, 3)}
            for key in ("rss", "pss", "uss"):
                entry[key] = _metric_block(sl, key)
            by_phase[name] = entry
            start = end
        result["memory_by_phase"] = by_phase

    return result


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
        full_ok = 0
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
                full_ok += 1   # only full-info procs make PSS/USS complete
            except psutil.Error:
                # fallback: RSS only — this proc does NOT count in full_ok
                try:
                    rss += p.memory_info().rss
                except psutil.Error:
                    continue
        sample: dict[str, Any] = {
            "t": now, "n_procs": n,
            "d_user": d_user, "d_sys": d_sys,
            # memory-measurement v2: raw samples are BYTES (v1 files used KiB)
            "rss": rss, "pss": pss, "uss": uss,
            "full_ok": full_ok,
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

    @property
    def samples(self) -> list[dict[str, Any]]:
        """Raw per-interval samples recorded between start() and stop()."""
        return list(self._samples)

    def stop(self, *, phase_bounds: list[tuple[str, float]] | None = None
             ) -> dict[str, Any]:
        """Final sample + aggregation; ``phase_bounds`` (see
        :func:`phase_bounds_from`) adds ``memory_by_phase`` when given."""
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
            phase_bounds=phase_bounds,
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
    write_samples: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run one fixed workload under monitoring and append a schema row.

    ``gpu="auto"`` (default) resolves to a fresh ``make_gpu_backend()``;
    explicit ``None`` disables GPU sampling and a callable is resolved by
    ``ResourceMonitor``. This is the single place that interprets ``"auto"``.

    With ``write_samples=True`` (default) the RAW per-interval samples are
    persisted under ``<out-dir>/samples/`` so aggregated numbers stay
    auditable; the repo-relative path is stored as ``extra.samples_file``.
    The CLI flag ``--no-samples`` disables the dump. Raw sample
    ``rss/pss/uss`` are BYTES (v2 — v1 sample files used KiB); on success
    ``extra.resources`` carries the nested ``memory`` block (+legacy
    ``*_kib`` fields) and ``memory_by_phase`` from the artifact phases —
    both omitted from the ``memory``'s phase key on failed runs.
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
    out = out_path if out_path is not None else DEFAULT_OUT
    mon = monitor if monitor is not None else ResourceMonitor(
        interval=interval, gpu=gpu)
    status, err, art = "ok", None, None
    mon.start()
    t0 = time.perf_counter()
    try:
        if workload == "ppo":
            art = run_ppo_training(framework, PPO_EXPERIMENT, n, seed, cfg,
                                   device=device)
        else:
            art = run_collection(framework, cfg, n, seed, device=device)
    except Exception as exc:
        status = ("unsupported"
                  if type(exc).__name__ == "UnsupportedConfiguration"
                  else "failed")
        err = traceback.format_exc()
    wall_s = time.perf_counter() - t0
    # phase memory: cumulative artifact-phase bounds relative to the first
    # sample; failed runs (art is None) get NO memory_by_phase.
    bounds = phase_bounds_from(art["phases"]) if art is not None else None
    stats = mon.stop(phase_bounds=bounds)

    # raw per-sample dump: auditable raw measurements next to the rows
    samples_rel: str | None = None
    if write_samples and mon.samples:
        sdir = out.parent / "samples"
        sdir.mkdir(parents=True, exist_ok=True)
        # `_cuda` suffix keeps GPU samples from overwriting CPU sample files
        sfile = sdir / (
            f"{workload}_{framework}_rep{rep}"
            f"{'_cuda' if device == 'cuda' else ''}.jsonl")
        sfile.write_text(
            "\n".join(json.dumps(s) for s in mon.samples) + "\n",
            encoding="utf-8")
        samples_rel = str(sfile.relative_to(out.parent))

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
        "device": device,   # actual device (fallback legs add device_used)
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
        extra={"resources": stats, "hardware": hardware_info(mon.gpu),
               "samples_file": samples_rel},
        device=device,
    )
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

    out_dir = out_dir if out_dir is not None else REPO_ROOT / "benchmarks/resources/results"
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
                   help="JSONL output (default: "
                        "benchmarks/resources/results/resources.jsonl)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                   help="compute device for the workload (default cpu; "
                        "env stepping is always CPU)")
    p.add_argument("--no-samples", action="store_true",
                   help="skip raw per-sample JSONL dumps under samples/")
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
                        out_path=out,
                        write_samples=not args.no_samples,
                        device=args.device)
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
    print(f"done: {n_rows} rows ({n_failed} failed, device={args.device}) "
          f"-> {out}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
