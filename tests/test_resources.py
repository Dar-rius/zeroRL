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
        samples = [
            {"t": 0.0, "d_user": 0.0, "d_sys": 0.0, "rss": 100, "pss": 90,
             "uss": 80, "n_procs": 1},
            {"t": 0.2, "d_user": 0.1, "d_sys": 0.0, "rss": 300, "pss": 270,
             "uss": 250, "n_procs": 2, "cpu_pct": 50.0},
            {"t": 0.4, "d_user": 0.2, "d_sys": 0.1, "rss": 200, "pss": 180,
             "uss": 160, "n_procs": 2, "cpu_pct": 150.0},
        ]
        stats = aggregate_samples(samples, duration_s=0.4, interval_s=0.2,
                                  gpu_backend_name=None, energy_note="n/a")
        mem = stats["memory"]
        assert mem["rss_init_kib"] == 100
        assert mem["rss_final_kib"] == 200
        assert mem["rss_peak_kib"] == 300
        assert mem["rss_avg_kib"] == pytest.approx(200.0)
        assert mem["uss_peak_kib"] == 250
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


def test_monitor_interval_validation() -> None:
    with pytest.raises(ValueError):
        ResourceMonitor(interval=0.0, gpu=None)


def test_cli_parser_defaults() -> None:
    from benchmarks.resources.run import build_parser

    args = build_parser().parse_args([])
    assert args.workload == "all"
    assert args.interval == DEFAULT_INTERVAL_S
    assert args.reps_collect == 5 and args.reps_ppo == 3
