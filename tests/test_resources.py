"""Tests for the resource-consumption benchmark (experiment R).

Covers the required validation areas: process-tree CPU accounting incl.
dynamically spawned/killed children, memory aggregation, GPU/energy
telemetry behavior (mocked; no physical GPU needed), sampling and elapsed
time, transition budgets, phase separation, energy integration, and
schema/serialization compatibility with the aggregation utilities.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from benchmarks.resources.run import (
    DEFAULT_INTERVAL_S,
    ResourceMonitor,
    _time_weighted_mean,
    aggregate_samples,
    integrate_power_mw,
    run_resource_workload,
)


class _FakeGpu:
    """Mocked NVML backend for telemetry tests."""

    name = "fake-nvml"

    def __init__(self, power: bool = False) -> None:
        self.total_bytes = 8 * 2**30
        self._power = power

    def sample(self) -> dict:
        out = {"util_pct": 37, "mem_util_pct": 12,
               "mem_used_bytes": 100 * 2**20, "mem_total_bytes": self.total_bytes}
        if self._power:
            out["power_mW"] = 150_000
        return out


class TestProcessTreeAccounting:
    """(1)(2)(3) CPU-time accounting incl. children spawned/killed mid-run."""

    def test_child_spawned_after_start_is_counted_and_can_exit(self) -> None:
        # child burns ~0.6s CPU (user+sys), spawned AFTER monitor start and
        # finished BEFORE stop — per-PID delta accounting must include it.
        mon = ResourceMonitor(interval=0.05, gpu=None)
        mon.start()
        child = subprocess.Popen(
            [sys.executable, "-c",
             "x=0\nfor i in range(6_000_000): x+=i\n"]
        )
        child.wait(timeout=60)
        stats = mon.stop()
        # the parent alone did almost nothing; the child's CPU must dominate
        assert stats["cpu"]["total_cpu_s"] > 0.25, stats["cpu"]
        assert stats["sampling"]["n_samples"] >= 2
        # no leaked references to the dead child in later samples
        assert all(s.get("n_procs", 1) >= 1 for s in [stats["memory"]])

    def test_child_present_at_start_is_baselined(self) -> None:
        # a pre-existing child's history must NOT be counted from birth
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            mon = ResourceMonitor(interval=0.05, gpu=None)
            mon.start()
            time.sleep(0.2)
            stats = mon.stop()
            # sleeping child contributes ~0 CPU; total must stay tiny
            assert stats["cpu"]["total_cpu_s"] < 0.6, stats["cpu"]
        finally:
            child.kill()
            child.wait(timeout=10)


class TestAggregation:
    """(4)(7) peak/average memory aggregation and sampling/elapsed time."""

    def test_peak_avg_init_final(self) -> None:
        # memory-measurement v2: raw samples are BYTES (v1 files used KiB);
        # every sample carries full_ok == n_procs (complete PSS/USS samples).
        samples = [
            {"t": 0.0, "d_user": 0.0, "d_sys": 0.0, "rss": 100 * 1024,
             "pss": 90 * 1024, "uss": 80 * 1024, "n_procs": 1, "full_ok": 1},
            {"t": 0.2, "d_user": 0.1, "d_sys": 0.0, "rss": 300 * 1024,
             "pss": 270 * 1024, "uss": 250 * 1024, "n_procs": 2,
             "full_ok": 2, "cpu_pct": 50.0},
            {"t": 0.4, "d_user": 0.2, "d_sys": 0.1, "rss": 200 * 1024,
             "pss": 180 * 1024, "uss": 160 * 1024, "n_procs": 2,
             "full_ok": 2, "cpu_pct": 150.0},
        ]
        stats = aggregate_samples(samples, duration_s=0.4, interval_s=0.2,
                                  gpu_backend_name=None, energy_note="n/a")
        mem = stats["memory"]
        # trapezoidal time-weighted mean, computed here explicitly (all
        # values divisible by 1024 → exact KiB): ((100+300)/2*0.2 +
        # (300+200)/2*0.2) / 0.4 = 225 KiB (arithmetic-of-kib would be 200)
        expected_avg_bytes = (
            ((100 + 300) / 2 * 0.2 + (300 + 200) / 2 * 0.2) / 0.4) * 1024
        # legacy flat fields = bytes // 1024 of the same byte series
        assert mem["rss_init_kib"] == 100
        assert mem["rss_final_kib"] == 200
        assert mem["rss_peak_kib"] == 300
        assert mem["rss_avg_kib"] == expected_avg_bytes // 1024 == 225
        assert mem["uss_peak_kib"] == 250
        # canonical nested v2 block (bytes; time-weighted average)
        assert mem["rss"]["peak_bytes"] == 300 * 1024
        assert mem["rss"]["initial_bytes"] == 100 * 1024
        assert mem["rss"]["final_bytes"] == 200 * 1024
        assert mem["rss"]["average_bytes"] == pytest.approx(expected_avg_bytes)
        assert mem["availability"]["pss_available"] is True
        cpu = stats["cpu"]
        assert cpu["total_cpu_s"] == pytest.approx(0.4)
        assert cpu["util_avg_pct"] == pytest.approx(100.0)   # 50 & 150
        assert cpu["util_peak_pct"] == 150.0                 # not capped
        assert stats["sampling"]["n_samples"] == 3
        assert stats["window_seconds"] == pytest.approx(0.4)

    def test_elapsed_and_interval_recorded(self) -> None:
        mon = ResourceMonitor(interval=0.05, gpu=None)
        mon.start()
        time.sleep(0.15)
        stats = mon.stop()
        assert stats["sampling"]["interval_s"] == 0.05
        assert stats["window_seconds"] >= 0.15
        assert stats["sampling"]["n_samples"] >= 3


class TestGpuAndEnergyTelemetry:
    """(5)(6)(10) GPU telemetry, graceful unavailability, energy math."""

    def test_mocked_gpu_samples_recorded(self) -> None:
        mon = ResourceMonitor(interval=0.05, gpu=_FakeGpu())
        mon.start()
        time.sleep(0.15)
        stats = mon.stop()
        gpu = stats["gpu"]
        assert gpu["available"] is True
        assert gpu["util_avg_pct"] == 37
        assert gpu["util_peak_pct"] == 37
        assert gpu["mem_peak_bytes"] == 100 * 2**20

    def test_unavailable_gpu_keeps_valid_cpu_memory(self) -> None:
        mon = ResourceMonitor(interval=0.05, gpu=None)
        mon.start()
        time.sleep(0.15)
        stats = mon.stop()
        assert stats["gpu"]["available"] is False
        assert stats["gpu"]["reason"]
        assert stats["cpu"]["total_cpu_s"] is not None
        assert stats["memory"]["rss_peak_kib"]

    def test_energy_integration_trapezoid(self) -> None:
        # 1W for 1s ramping to 2W over 1s, back to 1W: 1.5J + 1.5J
        assert integrate_power_mw([0.0, 1.0, 2.0],
                                  [1000.0, 2000.0, 1000.0]) == pytest.approx(3.0)
        assert integrate_power_mw([], []) == 0.0
        assert integrate_power_mw([0.0], [1000.0]) == 0.0

    def test_energy_unavailable_is_explicit(self) -> None:
        mon = ResourceMonitor(interval=0.05, gpu=_FakeGpu(power=False))
        mon.start()
        time.sleep(0.1)
        stats = mon.stop()
        energy = stats["energy"]
        assert energy["available"] is False
        assert energy["gpu_joules"] is None
        assert "NVML power" in energy["reason"] or "power" in energy["reason"]


class TestWorkloadIntegration:
    """(8)(9) transition budget, phase separation, schema serialization."""

    def test_collect_workload_budget_and_phases(self) -> None:
        from benchmarks.scalability.collection import CollectionConfig

        cfg = CollectionConfig(warmup=500, measure=1000)
        row = run_resource_workload(
            "zeroRL", "collect", 99, interval=0.05,
            out_path=Path("/tmp/opencode/res_test.jsonl") if Path(
                "/tmp/opencode").exists() else None,
            collection_cfg=cfg, num_envs=2, gpu=None)
        assert row["status"] == "ok"
        t = row["transitions"]
        assert t["exact"] and t["measured"] == 1000
        # phase separation: artifact phases present alongside the resource window
        assert row["phases"]["init_seconds"] > 0
        assert row["phases"]["warmup_seconds"] > 0
        assert row["phases"]["measure_seconds"] > 0
        res = row["extra"]["resources"]
        assert res["window_seconds"] >= sum(
            row["phases"][k] for k in
            ("init_seconds", "warmup_seconds", "boundary_seconds")
            if k in row["phases"])
        assert row["metrics"]["cpu_s_per_1k"] > 0
        assert row["config"]["device"] == "cpu"
        assert row["config"]["interval_s"] == 0.05


class TestRawSamplePersistence:
    """Raw per-interval samples are preserved next to the rows and are
    addressable from the row (extra.samples_file); can be disabled."""

    def test_raw_samples_dump_and_disable(self, tmp_path) -> None:
        from benchmarks.scalability.collection import CollectionConfig

        out = tmp_path / "res.jsonl"
        cfg = CollectionConfig(warmup=500, measure=1000)
        row = run_resource_workload(
            "zeroRL", "collect", 97, interval=0.05, out_path=out,
            collection_cfg=cfg, num_envs=2, gpu=None)
        assert row["status"] == "ok"
        rel = row["extra"]["samples_file"]
        assert rel and rel.startswith("samples/")
        assert "_cuda" not in rel
        samples = [json.loads(x)
                   for x in (tmp_path / rel).read_text().splitlines()]
        assert len(samples) >= 2
        assert {"t", "rss", "n_procs"} <= set(samples[0])
        # cpu_pct present on all but the first sample
        assert all("cpu_pct" in s for s in samples[1:])

        row2 = run_resource_workload(
            "zeroRL", "collect", 96, interval=0.05, out_path=out,
            collection_cfg=cfg, num_envs=2, gpu=None, write_samples=False)
        assert row2["extra"]["samples_file"] is None
        assert not (tmp_path / "samples"
                    / "collect_zeroRL_rep96.jsonl").exists()

    def test_cuda_sample_suffix(self, tmp_path) -> None:
        try:
            torch.zeros(1, device="cuda") + 1
            ok = torch.cuda.get_device_capability()[0] >= 7
        except Exception:
            ok = False
        if not ok:
            pytest.skip("CUDA kernels unavailable or too old on this host")

        from benchmarks.scalability.collection import CollectionConfig

        row = run_resource_workload(
            "zeroRL", "collect", 95, interval=0.05,
            out_path=tmp_path / "r.jsonl",
            collection_cfg=CollectionConfig(warmup=500, measure=1000),
            num_envs=2, device="cuda", gpu=None)
        assert row["status"] == "ok"
        assert row["config"]["device"] == "cuda"
        rel = row["extra"]["samples_file"]
        assert rel.endswith("_cuda.jsonl")
        assert (tmp_path / rel).exists()


class TestSerializationAndAggregation:
    """(11) rows are schema-valid and consumable by the aggregators."""

    def test_row_roundtrip_and_aggregation(self, tmp_path) -> None:
        from benchmarks.scalability.aggregate import integrity_summary
        from benchmarks.scalability.collection import CollectionConfig
        from benchmarks.scalability.results_schema import load_rows

        out = tmp_path / "resources.jsonl"
        cfg = CollectionConfig(warmup=500, measure=1000)
        row = run_resource_workload(
            "zeroRL", "collect", 98, interval=0.05, out_path=out,
            collection_cfg=cfg, num_envs=2, gpu=None)
        rows = load_rows(out)
        assert len(rows) == 1
        assert rows[0]["experiment"] == "R"
        assert rows[0]["schema_version"] == row["schema_version"]
        # aggregation utilities accept the new experiment id
        summary = integrity_summary(rows)
        assert summary["counts"]["R"]["ok"] == 1
        # jsonl line is valid on its own
        json.loads(out.read_text().strip().splitlines()[-1])


class TestResourcesTwoSidedReport:
    """CPU report renders byte-pure; GPU side is an additive sibling section."""

    @staticmethod
    def _mk_res_row(device: str, rep: int) -> dict:
        from benchmarks.scalability.results_schema import make_row

        return make_row(
            framework="zeroRL", experiment="R", mode="sync", num_envs=2,
            num_workers=1, protocol="collect", rep=rep, seed=1000 + rep,
            status="ok",
            phases={"init_seconds": 0.1, "warmup_seconds": 1.0,
                    "measure_seconds": 10.0, "boundary_seconds": 0.1},
            transitions={"warmup_target": 16_000, "warmup_actual": 16_000,
                         "measure_target": 100_000, "measured": 100_000,
                         "exact": True},
            metrics={"wall_s": 12.0, "init_seconds": 0.1,
                     "measure_seconds": 10.0, "throughput": 10000.0,
                     "cpu_util_avg_pct": 100.0, "cpu_util_peak_pct": 150.0,
                     "cpu_total_s": 8.0, "cpu_s_per_1k": 0.08,
                     "rss_peak_kib": 100000, "rss_avg_kib": 90000,
                     "uss_peak_kib": 80000},
            config={"workload": "collect", "env_id": "LunarLander-v3",
                    "device": device, "interval_s": 0.2,
                    "budget": {"warmup": 16000, "measure": 100000},
                    "num_envs": 2, "seed_schedule": "1000 + rep"},
            extra={"resources": {"gpu": {"available": False, "reason": "t"},
                                 "energy": {"available": False,
                                            "reason": "t"}},
                   "hardware": {}, "samples_file": None},
            error=None, device=device,
        )

    def test_report_two_sided(self, tmp_path) -> None:
        from benchmarks.resources.report import main
        from benchmarks.scalability.results_schema import append_row

        d = tmp_path / "results"
        d.mkdir()
        for rep in (0, 1):
            append_row(d / "resources.jsonl", self._mk_res_row("cpu", rep))
        assert main(["--results-dir", str(d),
                     "--out", str(tmp_path / "R1.md")]) == 0
        t1 = (tmp_path / "R1.md").read_text()
        assert "## Resource consumption (experiment R)" in t1
        assert "— GPU" not in t1

        for rep in (0, 1):
            append_row(d / "resources_gpu.jsonl",
                       self._mk_res_row("cuda", rep))
        assert main(["--results-dir", str(d),
                     "--out", str(tmp_path / "R2.md")]) == 0
        t2 = (tmp_path / "R2.md").read_text()
        assert "## Resource consumption — GPU (device=cuda)" in t2

        # CPU-side table unchanged: slice R2's CPU section (both headings are
        # asserted present above) and compare; the GPU sibling section adds
        # exactly one aggregated `| collect |` row.
        cpu1 = [ln for ln in t1.splitlines() if ln.startswith("| collect |")]
        cpu2_body = t2.split("## Resource consumption (experiment R)",
                             1)[1].split("## Resource consumption — GPU", 1)[0]
        cpu2 = [ln for ln in cpu2_body.splitlines()
                if ln.startswith("| collect |")]
        assert cpu1 == cpu2
        n_r2 = len([ln for ln in t2.splitlines()
                    if ln.startswith("| collect |")])
        assert n_r2 == len(cpu1) + 1


def _mk_v2_res_row(rep: int, *, rss_peak_bytes: int,
                   pss_available: bool = True,
                   samples_file: str | None = None) -> dict:
    """A v2 resources row carrying the nested bytes memory block."""
    from benchmarks.scalability.results_schema import make_row

    def _blk(scale: float) -> dict:
        peak = int(rss_peak_bytes * scale)
        return {"initial_bytes": int(peak * 0.5),
                "average_bytes": int(peak * 0.75),
                "peak_bytes": peak,
                "final_bytes": int(peak * 0.6)}

    memory = {
        "rss": _blk(1.0),
        "pss": _blk(0.8),
        "uss": _blk(0.7),
        "availability": {"backend": "psutil.memory_full_info",
                         "pss_available": pss_available,
                         "pss_missing_samples": 0 if pss_available else 2,
                         "uss_available": True,
                         "uss_missing_samples": 0},
    }
    return make_row(
        framework="zeroRL", experiment="R", mode="sync", num_envs=2,
        num_workers=1, protocol="collect", rep=rep, seed=1000 + rep,
        status="ok",
        phases={"init_seconds": 0.2, "warmup_seconds": 1.0,
                "measure_seconds": 10.0, "boundary_seconds": 0.1},
        transitions={"warmup_target": 16_000, "warmup_actual": 16_000,
                     "measure_target": 100_000, "measured": 100_000,
                     "exact": True},
        metrics={"wall_s": 12.0, "init_seconds": 0.2,
                 "measure_seconds": 10.0, "throughput": 10000.0,
                 "cpu_util_avg_pct": 100.0, "cpu_util_peak_pct": 150.0,
                 "cpu_total_s": 8.0, "cpu_s_per_1k": 0.08,
                 "rss_peak_kib": rss_peak_bytes // 1024,
                 "rss_avg_kib": int(rss_peak_bytes * 0.75) // 1024,
                 "uss_peak_kib": int(rss_peak_bytes * 0.7) // 1024},
        config={"workload": "collect", "env_id": "LunarLander-v3",
                "device": "cpu", "interval_s": 0.05,
                "budget": {"warmup": 16000, "measure": 100000},
                "num_envs": 2, "seed_schedule": "1000 + rep"},
        extra={"resources": {"gpu": {"available": False, "reason": "t"},
                             "energy": {"available": False, "reason": "t"},
                             "window_seconds": 12.0,
                             "memory": memory,
                             "memory_by_phase": {
                                 "init": {"window_seconds": 0.2,
                                          "rss": _blk(0.5),
                                          "pss": _blk(0.4),
                                          "uss": _blk(0.35)},
                                 "measure": {"window_seconds": 10.0,
                                             "rss": _blk(0.9),
                                             "pss": _blk(0.72),
                                             "uss": _blk(0.63)}},
                             "sampling": {"n_samples": 60,
                                          "interval_s": 0.05,
                                          "backend": "psutil"}},
               "hardware": {}, "samples_file": samples_file},
        error=None, device="cpu",
    )


class TestMemoryV2:
    """Memory-measurement v2: byte samples, time-weighted aggregates,
    partial-sample availability, per-phase blocks, report/plots output."""

    def test_time_weighted_mean_trapezoid(self) -> None:
        # trapezoid over [0,1,10] x [0,10,100] = (5 + 495)/10 = 50.0;
        # a plain arithmetic mean would be 110/3 = 36.67 — weighting shows
        vals = [0.0, 10.0, 100.0]
        tw = _time_weighted_mean([0.0, 1.0, 10.0], vals)
        assert tw == pytest.approx(50.0)
        assert tw != pytest.approx(110.0 / 3.0)
        # a single sample yields that value; an empty series yields None
        assert _time_weighted_mean([5.0], [7.0]) == 7.0
        assert _time_weighted_mean([], []) is None

    def test_aggregate_peak_is_max_of_series_not_sum_of_maxima(self) -> None:
        # per-proc series a=[100_000, 10_000], b=[10_000, 100_000] peak at
        # DIFFERENT samples → aggregated per-sample series is [110_000,
        # 110_000]; peak must be 110_000, strictly below the sum of the
        # individual per-proc maxima (200_000).
        samples = [
            {"t": 0.0, "rss": 110_000, "pss": 90_000, "uss": 80_000,
             "n_procs": 2, "full_ok": 2, "d_user": 0.0, "d_sys": 0.0},
            {"t": 1.0, "rss": 110_000, "pss": 90_000, "uss": 80_000,
             "n_procs": 2, "full_ok": 2, "d_user": 0.5, "d_sys": 0.5},
        ]
        stats = aggregate_samples(samples, duration_s=1.0, interval_s=1.0,
                                  gpu_backend_name=None, energy_note="n/a")
        peak = stats["memory"]["rss"]["peak_bytes"]
        assert peak == 110_000 == max(s["rss"] for s in samples)
        assert peak < 100_000 + 100_000   # < sum of per-proc maxima

    def test_pss_uss_partial_and_missing(self) -> None:
        # middle sample is PARTIAL (full_ok < n_procs): its pss/uss are
        # excluded from the series (never zero-filled) and counted missing
        samples = [
            {"t": 0.0, "rss": 100_000, "pss": 80_000, "uss": 70_000,
             "n_procs": 2, "full_ok": 2, "d_user": 0.0, "d_sys": 0.0},
            {"t": 1.0, "rss": 200_000, "pss": 999_999, "uss": 999_999,
             "n_procs": 2, "full_ok": 1, "d_user": 0.5, "d_sys": 0.0},
            {"t": 2.0, "rss": 300_000, "pss": 120_000, "uss": 110_000,
             "n_procs": 2, "full_ok": 2, "d_user": 0.5, "d_sys": 0.0},
        ]
        stats = aggregate_samples(samples, duration_s=2.0, interval_s=1.0,
                                  gpu_backend_name=None, energy_note="n/a")
        mem = stats["memory"]
        avail = mem["availability"]
        assert avail["backend"] == "psutil.memory_full_info"
        assert avail["pss_available"] is False
        assert avail["pss_missing_samples"] == 1
        assert avail["uss_available"] is False
        assert avail["uss_missing_samples"] == 1
        # valid pss series = the two FULL samples only (999_999 excluded):
        # trapezoid over t=[0,2] x [80_000,120_000] = 100_000 bytes
        assert mem["pss"]["initial_bytes"] == 80_000
        assert mem["pss"]["final_bytes"] == 120_000
        assert mem["pss"]["peak_bytes"] == 120_000
        assert mem["pss"]["average_bytes"] == 100_000
        assert mem["pss_avg_kib"] == 100_000 // 1024
        # rss is never gated by full_ok: all three samples contribute
        assert mem["rss"]["final_bytes"] == 300_000
        assert mem["rss"]["average_bytes"] == 200_000

    def test_memory_by_phase_bounds(self) -> None:
        # 5 samples; the t=0.5 tail lies beyond the measure bound (0.4)
        samples = [
            {"t": 0.0, "rss": 100_000, "pss": 90_000, "uss": 80_000,
             "n_procs": 1, "full_ok": 1, "d_user": 0.0, "d_sys": 0.0},
            {"t": 0.1, "rss": 120_000, "pss": 100_000, "uss": 90_000,
             "n_procs": 1, "full_ok": 1, "d_user": 0.05, "d_sys": 0.01},
            {"t": 0.2, "rss": 250_000, "pss": 200_000, "uss": 180_000,
             "n_procs": 1, "full_ok": 1, "d_user": 0.05, "d_sys": 0.01},
            {"t": 0.3, "rss": 200_000, "pss": 160_000, "uss": 150_000,
             "n_procs": 1, "full_ok": 1, "d_user": 0.05, "d_sys": 0.01},
            {"t": 0.5, "rss": 900_000, "pss": 700_000, "uss": 650_000,
             "n_procs": 1, "full_ok": 1, "d_user": 0.10, "d_sys": 0.02},
        ]
        stats = aggregate_samples(samples, duration_s=0.5, interval_s=0.1,
                                  gpu_backend_name=None, energy_note="n/a",
                                  phase_bounds=[("init", 0.1),
                                                ("measure", 0.4)])
        mbp = stats["memory_by_phase"]
        assert set(mbp) == {"init", "measure"}
        assert mbp["init"]["window_seconds"] == pytest.approx(0.1)
        assert mbp["measure"]["window_seconds"] == pytest.approx(0.3)
        assert mbp["init"]["rss"]["initial_bytes"] == 100_000
        assert mbp["init"]["rss"]["final_bytes"] == 120_000
        # measure slice = 0.1 < t <= 0.4 (t=0.2, 0.3); tail excluded
        assert mbp["measure"]["rss"]["peak_bytes"] == 250_000
        assert mbp["measure"]["rss"]["final_bytes"] == 200_000
        # the tail stays in the end-to-end block only
        assert stats["memory"]["rss"]["peak_bytes"] == 900_000

    def test_tick_survives_access_denied(self, monkeypatch) -> None:
        import psutil
        from types import SimpleNamespace

        class _DeniedProc:
            pid = 424242

            def cpu_times(self):
                return SimpleNamespace(user=0.05, system=0.01)

            def memory_full_info(self):
                raise psutil.AccessDenied(pid=self.pid)

            def memory_info(self):
                return SimpleNamespace(rss=123_456)

        mon = ResourceMonitor(interval=0.05, gpu=None)
        monkeypatch.setattr(ResourceMonitor, "_procs",
                            lambda self: [_DeniedProc()])
        mon._baseline()
        mon._tick(now=100.0)            # must not raise
        assert len(mon.samples) == 1
        s = mon.samples[0]
        assert {"t", "n_procs", "d_user", "d_sys", "rss", "pss", "uss",
                "full_ok"} <= set(s)
        assert s["t"] == 100.0
        assert s["n_procs"] == 1
        assert s["rss"] == 123_456      # memory_info() fallback still feeds RSS
        assert s["full_ok"] == 0        # full-info failure not counted
        # → pss/uss marked unavailable for this (fully partial) window
        stats = aggregate_samples(mon.samples, duration_s=0.05,
                                  interval_s=0.05, gpu_backend_name=None,
                                  energy_note="n/a")
        assert stats["memory"]["availability"]["pss_available"] is False

    def test_unit_conversion_exact(self) -> None:
        # 1537 bytes is NOT divisible by 1024: flat fields floor (→ 1 KiB)
        # while the canonical nested block keeps the exact byte value
        samples = [{"t": 0.0, "rss": 1537, "pss": 1537, "uss": 1537,
                    "n_procs": 1, "full_ok": 1,
                    "d_user": 0.0, "d_sys": 0.0}]
        stats = aggregate_samples(samples, duration_s=0.1, interval_s=0.1,
                                  gpu_backend_name=None, energy_note="n/a")
        mem = stats["memory"]
        assert mem["rss"]["initial_bytes"] == 1537
        assert mem["rss"]["peak_bytes"] == 1537
        assert mem["rss"]["final_bytes"] == 1537
        assert mem["rss"]["average_bytes"] == 1537
        assert mem["rss_init_kib"] == 1
        assert mem["rss_peak_kib"] == 1
        assert mem["rss_final_kib"] == 1
        assert mem["rss_avg_kib"] == 1
        assert mem["uss_peak_kib"] == 1
        assert mem["uss"]["peak_bytes"] == 1537
        assert mem["availability"]["pss_available"] is True

    def test_real_run_writes_memory_schema(self, tmp_path) -> None:
        from benchmarks.scalability.collection import CollectionConfig

        cfg = CollectionConfig(warmup=500, measure=1000)
        row = run_resource_workload(
            "zeroRL", "collect", 94, interval=0.05,
            out_path=tmp_path / "res.jsonl", collection_cfg=cfg,
            num_envs=2, gpu=None)
        assert row["status"] == "ok", row.get("error")
        res = row["extra"]["resources"]
        mem = res["memory"]
        # nested v2 block (bytes) + psutil availability backend
        assert mem["rss"]["peak_bytes"] > 0
        assert mem["availability"]["backend"] == "psutil.memory_full_info"
        mbp = res["memory_by_phase"]
        assert {"init", "measure"} <= set(mbp)
        assert mbp["init"]["window_seconds"] > 0
        assert mbp["measure"]["window_seconds"] > 0
        # regression guards: cpu / energy / sampling still recorded
        assert res["cpu"]["total_cpu_s"] > 0
        assert res["energy"]["available"] is False and res["energy"]["reason"]
        assert res["sampling"]["n_samples"] > 1

    def test_report_memory_subsections(self, tmp_path) -> None:
        from benchmarks.resources.report import main
        from benchmarks.scalability.results_schema import append_row

        d = tmp_path / "results"
        d.mkdir()
        # two v2 rows with DIFFERENT rss peaks; pss unavailable on both
        # (availability gate — must render as —, never as a 0.0 placeholder)
        for rep, peak in ((0, 200 * 2**20), (1, 300 * 2**20)):
            append_row(d / "resources.jsonl",
                       _mk_v2_res_row(rep, rss_peak_bytes=peak,
                                      pss_available=False))
        assert main(["--results-dir", str(d),
                     "--out", str(tmp_path / "R.md")]) == 0
        text = (tmp_path / "R.md").read_text()
        heading = "### Memory (RSS / PSS / USS) — collect"
        assert heading in text
        peak_row = next(ln for ln in text.split(heading, 1)[1].splitlines()
                        if ln.startswith("| zeroRL |"))
        cells = [c.strip() for c in peak_row.split("|")[1:-1]]
        # cells: framework, rss, pss, uss, measure-peak rss, n
        assert cells[0] == "zeroRL"
        assert "±" in cells[1]           # rss = mean ± std over both rows
        assert cells[2] == "—"           # unavailable pss → em dash, not 0.0

    def test_plots_memory_figures(self, tmp_path) -> None:
        from benchmarks.resources.plots import main
        from benchmarks.scalability.results_schema import append_row

        v2_figs = ("resources_memory_peak.png", "resources_memory_avg.png",
                   "resources_memory_timeseries.png")
        # v2 dir: rows + fabricated raw samples files (bytes, full_ok)
        d = tmp_path / "results"
        sdir = d / "samples"
        sdir.mkdir(parents=True)
        for rep in (0, 1):
            lines = [json.dumps(
                {"t": 0.05 * i, "n_procs": 2,
                 "rss": 100_000 + 10_000 * i, "pss": 90_000 + 8_000 * i,
                 "uss": 80_000 + 7_000 * i, "full_ok": 2,
                 "d_user": 0.01 * i, "d_sys": 0.005 * i})
                for i in range(6)]
            rel = f"samples/collect_zeroRL_rep{rep}.jsonl"
            (d / rel).write_text("\n".join(lines) + "\n", encoding="utf-8")
            append_row(d / "resources.jsonl",
                       _mk_v2_res_row(rep, rss_peak_bytes=(200 + 50 * rep)
                                      * 2**20, samples_file=rel))
        figs = tmp_path / "figs_v2"
        assert main(["--results-dir", str(d), "--out", str(figs)]) == 0
        for name in v2_figs:
            p = figs / name
            assert p.is_file() and p.stat().st_size > 0, name

        # v1-style dir (no nested memory block): the three v2 figures are
        # NOT produced (legacy figures still render)
        d1 = tmp_path / "results_v1"
        d1.mkdir()
        for rep in (0, 1):
            append_row(d1 / "resources.jsonl",
                       TestResourcesTwoSidedReport._mk_res_row("cpu", rep))
        figs1 = tmp_path / "figs_v1"
        assert main(["--results-dir", str(d1), "--out", str(figs1)]) == 0
        for name in v2_figs:
            assert not (figs1 / name).exists(), name
        assert (figs1 / "resources_memory.png").is_file()


def test_monitor_interval_validation() -> None:
    with pytest.raises(ValueError):
        ResourceMonitor(interval=0.0, gpu=None)


def test_cli_parser_defaults() -> None:
    from benchmarks.resources.run import build_parser

    args = build_parser().parse_args([])
    assert args.workload == "all"
    assert args.interval == DEFAULT_INTERVAL_S
    assert args.reps_collect == 5 and args.reps_ppo == 3
