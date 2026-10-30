"""Unit tests for benchmarks/scalability — budget accounting, metric formulas,
result schema and the shared reference policy adapters.

Framework-independent core only: no torchrl import; SB3-dependent tests are
guarded with importorskip (the CI CPU job does not install the
``benchmarks`` extra).
"""

import json

import pytest
import torch

from benchmarks.scalability import policy_reference as ref
from benchmarks.scalability import results_schema as schema
from benchmarks.scalability import scaling_metrics as metrics
from benchmarks.scalability.budget import (
    BudgetExceeded, BudgetTracker, remaining_batches,
)


class TestBudgetTracker:
    def test_exact_budget_reached(self) -> None:
        t = BudgetTracker(100)
        for _ in range(4):
            t.record(25)
        assert t.done and t.exact
        assert t.actual == 100 and t.remaining == 0 and t.deviation == 0
        assert t.batches == [25, 25, 25, 25]

    def test_partial_batches_rejected_without_tolerance(self) -> None:
        t = BudgetTracker(10)
        t.record(5)
        with pytest.raises(BudgetExceeded):
            t.record(6)

    def test_overshoot_within_tolerance(self) -> None:
        t = BudgetTracker(10, tolerate_overshoot=True, overshoot_tolerance=5)
        t.record(15)
        assert t.actual == 15 and t.exact is False and t.deviation == 5

    def test_overshoot_beyond_tolerance_rejected(self) -> None:
        t = BudgetTracker(10, tolerate_overshoot=True, overshoot_tolerance=2)
        with pytest.raises(BudgetExceeded):
            t.record(13)

    def test_remaining_and_not_done(self) -> None:
        t = BudgetTracker(100)
        t.record(30)
        assert not t.done and not t.exact
        assert t.remaining == 70

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            BudgetTracker(0)
        t = BudgetTracker(10)
        with pytest.raises(ValueError):
            t.record(0)
        with pytest.raises(ValueError):
            t.record(-3)

    def test_summary_is_json_safe(self) -> None:
        t = BudgetTracker(100)
        t.record(100)
        s = t.summary()
        assert s == {"target": 100, "actual": 100, "exact": True,
                     "deviation": 0, "num_batches": 1}
        json.dumps(s)


class TestRemainingBatches:
    def test_divisible(self) -> None:
        assert remaining_batches(100, 25) == 4

    def test_indivisible_rejected(self) -> None:
        with pytest.raises(ValueError, match="not divisible"):
            remaining_batches(100, 30)

    def test_bad_batch_size(self) -> None:
        with pytest.raises(ValueError):
            remaining_batches(100, 0)


class TestThroughput:
    def test_value(self) -> None:
        assert metrics.throughput(1000, 2.0) == 500.0

    def test_invalid_seconds(self) -> None:
        with pytest.raises(ValueError):
            metrics.throughput(10, 0.0)

    def test_invalid_transitions(self) -> None:
        with pytest.raises(ValueError):
            metrics.throughput(-1, 1.0)


class TestSpeedupAndEfficiency:
    def test_speedup(self) -> None:
        assert metrics.speedup(10.0, 2.5) == 4.0

    def test_speedup_invalid(self) -> None:
        with pytest.raises(ValueError):
            metrics.speedup(0.0, 2.5)
        with pytest.raises(ValueError):
            metrics.speedup(10.0, -1.0)

    def test_environment_scaling_efficiency(self) -> None:
        assert metrics.environment_scaling_efficiency(4.0, 8) == 0.5
        assert metrics.environment_scaling_efficiency(8.0, 8) == 1.0

    def test_environment_efficiency_invalid(self) -> None:
        with pytest.raises(ValueError):
            metrics.environment_scaling_efficiency(4.0, 0)

    def test_worker_parallel_efficiency(self) -> None:
        assert metrics.worker_parallel_efficiency(2.0, 4, baseline_workers=2) == 1.0

    def test_worker_efficiency_invalid(self) -> None:
        with pytest.raises(ValueError):
            metrics.worker_parallel_efficiency(2.0, 1, baseline_workers=2)
        with pytest.raises(ValueError):
            metrics.worker_parallel_efficiency(2.0, 0, baseline_workers=1)


class TestSummaryStats:
    def test_mean_std_min_max(self) -> None:
        s = metrics.summary_stats([1.0, 2.0, 3.0])
        assert s["n"] == 3 and s["mean"] == 2.0
        assert s["std"] == pytest.approx(1.0)  # ddof=1
        assert s["min"] == 1.0 and s["max"] == 3.0

    def test_single_value_std_zero(self) -> None:
        s = metrics.summary_stats([5.0])
        assert s["n"] == 1 and s["std"] == 0.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            metrics.summary_stats([])


def _run(seed: int, *, status: str = "ok", time: float = 2.0,
         transitions: int = 100_000) -> dict:
    return {
        "seed": seed, "status": status,
        "measure_seconds": time, "transitions_measured": transitions,
    }


def _paired_a_row(n: int, rep: int, secs: float, fw: str = "zeroRL") -> dict:
    """Minimal A-experiment row for paired_curve tests."""
    return {
        "framework": fw, "experiment": "A", "mode": "sync",
        "num_envs": n, "num_workers": 1, "rep": rep, "seed": 1000 + rep,
        "status": "ok",
        "phases": {"init_seconds": 1.0, "warmup_seconds": 2.0,
                   "measure_seconds": secs, "boundary_seconds": 0.1},
        "transitions": {"measured": 100_000, "exact": True},
        "metrics": {},
    }


class TestPairedCurve:
    """P4: per-repetition speedup/efficiency with real dispersion."""

    def test_paired_curve_known_values(self) -> None:
        import math
        import statistics

        from benchmarks.scalability.aggregate import paired_curve

        rows = (
            [_paired_a_row(1, 0, 10.0), _paired_a_row(1, 1, 20.0),
             _paired_a_row(1, 2, 30.0)]
            + [_paired_a_row(2, 0, 5.0), _paired_a_row(2, 1, 8.0),
               _paired_a_row(2, 2, 15.0)]
        )
        c = paired_curve(rows, "A", "zeroRL", kind="environment",
                         point_key="num_envs", baseline=1)
        by_p = {r["point"]: r for r in c}
        assert by_p[1]["n_pairs"] == 3
        assert by_p[1]["speedup_mean"] == pytest.approx(1.0)
        assert by_p[1]["speedup_std"] == 0.0
        # per-rep speedups: 10/5=2, 20/8=2.5, 30/15=2
        r2 = by_p[2]
        sp = [2.0, 2.5, 2.0]
        assert r2["n_pairs"] == 3
        assert r2["speedup_mean"] == pytest.approx(statistics.mean(sp))
        assert r2["speedup_std"] == pytest.approx(statistics.stdev(sp))
        assert r2["efficiency_mean"] == pytest.approx(
            statistics.mean([s / 2 for s in sp]))
        assert r2["efficiency_std"] == pytest.approx(
            statistics.stdev([s / 2 for s in sp]))
        se = r2["speedup_std"] / math.sqrt(3)
        assert r2["speedup_ci95_lo"] == pytest.approx(
            r2["speedup_mean"] - 1.96 * se)
        assert r2["speedup_ci95_hi"] == pytest.approx(
            r2["speedup_mean"] + 1.96 * se)

    def test_paired_curve_missing_baseline_rep(self) -> None:
        from benchmarks.scalability.aggregate import paired_curve

        rows = (
            # baseline N=1 lost rep 1 (failed); N=2 has all three reps
            [_paired_a_row(1, 0, 10.0), _paired_a_row(1, 2, 30.0)]
            + [_paired_a_row(2, 0, 5.0), _paired_a_row(2, 1, 8.0),
               _paired_a_row(2, 2, 15.0)]
        )
        c = paired_curve(rows, "A", "zeroRL", kind="environment",
                         point_key="num_envs", baseline=1)
        by_p = {r["point"]: r for r in c}
        assert by_p[2]["n_pairs"] == 2       # rep 1 skipped: no baseline
        assert by_p[2]["n_skipped"] == 1
        assert by_p[2]["speedup_mean"] == pytest.approx(
            (10.0 / 5.0 + 30.0 / 15.0) / 2)
        assert by_p[2]["speedup_std"] == 0.0
        assert by_p[1]["n_pairs"] == 2

    def test_plot_series_use_paired_dispersion(self) -> None:
        """Series y--errors must be the paired speedup/efficiency stds —
        not throughput_std (old _curve_series) and not a hardcoded 0.0
        (old _speedup_series)."""
        from benchmarks.scalability.plots import _curve_series, _speedup_series

        rows = (
            [_paired_a_row(1, 0, 10.0), _paired_a_row(1, 1, 20.0),
             _paired_a_row(1, 2, 30.0)]
            + [_paired_a_row(2, 0, 5.0), _paired_a_row(2, 1, 8.0),
               _paired_a_row(2, 2, 15.0)]
        )
        ser = _speedup_series(rows, "A", kind="environment")
        pts = {p: (y, e) for p, y, e in ser["zeroRL"]}
        # paired speedup std at N=2 is 0.2887 (throughput_std would be
        # ~6686; the old hardcoded error bar was 0.0)
        assert pts[2][1] == pytest.approx(0.2886751, rel=1e-4)
        assert pts[1][1] == pytest.approx(0.0)

        eff = _curve_series(rows, "A", kind="environment")
        epts = {p: (y, e) for p, y, e in eff["zeroRL"]}
        assert epts[2][1] == pytest.approx(0.1443376, rel=1e-4)
        assert epts[1][1] == pytest.approx(0.0)


class TestAggregateRuns:
    def test_aggregates_all_reps(self) -> None:
        runs = [_run(1, time=2.0), _run(2, time=4.0), _run(3, status="failed")]
        agg = metrics.aggregate_runs(runs)
        assert agg["n_ok"] == 2 and agg["n_total"] == 3
        assert agg["transitions"] == 100_000
        assert agg["time_seconds"]["mean"] == 3.0
        assert agg["seeds"] == [1, 2]
        assert agg["throughput"]["mean"] == pytest.approx(
            (100_000 / 2.0 + 100_000 / 4.0) / 2
        )

    def test_mismatched_transitions_rejected(self) -> None:
        runs = [_run(1), _run(2, transitions=50_000)]
        with pytest.raises(ValueError, match="same transition count"):
            metrics.aggregate_runs(runs)

    def test_all_failed(self) -> None:
        agg = metrics.aggregate_runs([_run(1, status="failed")])
        assert agg["n_ok"] == 0 and agg["n_total"] == 1


class TestScalingCurve:
    @staticmethod
    def _cells() -> dict[int, dict]:
        return {
            1: metrics.aggregate_runs([_run(1, time=10.0)]),
            2: metrics.aggregate_runs([_run(2, time=5.0)]),
            4: metrics.aggregate_runs([_run(3, time=4.0)]),
        }

    def test_environment_curve(self) -> None:
        rows = metrics.scaling_curve(self._cells(), baseline_n=1,
                                     kind="environment")
        by_point = {r["point"]: r for r in rows}
        assert by_point[2]["speedup"] == 2.0
        assert by_point[2]["environment_scaling_efficiency"] == 1.0
        assert by_point[4]["speedup"] == 2.5
        assert by_point[4]["environment_scaling_efficiency"] == 0.625

    def test_worker_curve_requires_baseline_workers(self) -> None:
        with pytest.raises(ValueError, match="baseline_workers"):
            metrics.scaling_curve(self._cells(), baseline_n=1, kind="worker")

    def test_worker_curve(self) -> None:
        rows = metrics.scaling_curve(self._cells(), baseline_n=1,
                                     kind="worker", baseline_workers=1)
        by_point = {r["point"]: r for r in rows}
        assert by_point[4]["worker_parallel_efficiency"] == 2.5 / 4

    def test_missing_baseline_rejected(self) -> None:
        cells = {2: metrics.aggregate_runs([_run(2, time=5.0)])}
        with pytest.raises(ValueError, match="baseline"):
            metrics.scaling_curve(cells, baseline_n=1, kind="environment")

    def test_failed_cell_reported_not_crashing(self) -> None:
        cells = self._cells()
        cells[8] = metrics.aggregate_runs([_run(4, status="failed")])
        rows = metrics.scaling_curve(cells, baseline_n=1, kind="environment")
        row8 = [r for r in rows if r["point"] == 8][0]
        assert row8["status"] == "no_successful_runs"

    def test_bad_kind(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            metrics.scaling_curve(self._cells(), baseline_n=1, kind="mixed")


class TestSchema:
    @staticmethod
    def _ok_row(**over: object) -> dict:
        base: dict = dict(
            framework="zeroRL", experiment="A", mode="sync",
            num_envs=4, num_workers=1, protocol="collection",
            rep=0, seed=42, status="ok",
            phases={"init": 0.1, "warmup": 1.0, "measure_seconds": 2.0},
            transitions={"target": 100_000, "measured": 100_000, "exact": True},
            metrics={"throughput_mean": 50_000.0},
            config={"env_id": "LunarLander-v3"},
        )
        base.update(over)
        return schema.make_row(**base)  # type: ignore[arg-type]

    def test_ok_row_valid(self) -> None:
        row = self._ok_row()
        assert row["framework"] == "zeroRL"
        assert row["schema_version"] == schema.SCHEMA_VERSION
        assert row["environment"]["device"] == "cpu"

    def test_bad_framework_rejected(self) -> None:
        with pytest.raises(ValueError, match="framework"):
            self._ok_row(framework="CleanRL")

    def test_bad_experiment_rejected(self) -> None:
        with pytest.raises(ValueError, match="experiment"):
            self._ok_row(experiment="E")

    def test_bad_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            self._ok_row(mode="cuda")

    def test_failed_row_requires_error(self) -> None:
        with pytest.raises(ValueError, match="error"):
            self._ok_row(status="failed", error=None)

    def test_ok_row_requires_measure_time(self) -> None:
        with pytest.raises(ValueError, match="measure_seconds"):
            self._ok_row(
                phases={"init": 0.1, "warmup": 1.0, "measure_seconds": 0.0}
            )

    def test_unsupported_row_no_transitions_needed(self) -> None:
        row = self._ok_row(
            status="unsupported", error="zeroRL has no multi-process envs",
            transitions={"target": 100_000, "measured": 0, "exact": False},
        )
        assert row["status"] == "unsupported"

    def test_append_and_load_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "results" / "expA.jsonl"
        schema.append_row(path, self._ok_row())
        schema.append_row(path, self._ok_row(rep=1, seed=43))
        rows = schema.load_rows(path)
        assert len(rows) == 2
        assert {r["framework"] for r in rows} == {"zeroRL"}
        assert [r["seed"] for r in rows] == [42, 43]

    def test_validate_missing_field(self) -> None:
        row = self._ok_row()
        del row["metrics"]
        with pytest.raises(ValueError, match="missing fields"):
            schema.validate_row(row)

    def test_environment_info_versions(self) -> None:
        info = schema.environment_info()
        for key in ("python", "torch", "gymnasium", "git_sha", "device"):
            assert key in info


class TestReferencePolicy:
    def test_build_is_deterministic(self) -> None:
        sd1 = ref.build_reference_state_dict()
        sd2 = ref.build_reference_state_dict()
        assert set(sd1) == set(ref.REFERENCE_KEYS)
        for k in sd1:
            assert torch.equal(sd1[k], sd2[k]), f"non-deterministic tensor {k}"

    def test_different_seed_differs(self) -> None:
        sd1 = ref.build_reference_state_dict(seed=1)
        sd2 = ref.build_reference_state_dict(seed=2)
        assert any(not torch.equal(sd1[k], sd2[k]) for k in sd1)

    def test_load_into_zerorl_roundtrip(self) -> None:
        sd = ref.build_reference_state_dict()
        agent = ref.build_deterministic_zerorl_agent(sd)
        with torch.no_grad():  # perturb, then reload
            agent.actor.weight.add_(1.0)
        ref.load_into_zerorl(agent, sd)  # asserts tensor-equal internally

    def test_deterministic_get_action_is_argmax(self) -> None:
        agent = ref.build_deterministic_zerorl_agent()
        torch.manual_seed(0)
        obs = torch.randn(6, ref.OBS_DIM)
        out = agent.get_action(obs)
        with torch.no_grad():
            logits, value = agent.forward(obs)
        assert torch.equal(out["action"], logits.argmax(dim=-1))
        assert out["action"].shape == (6,)
        assert out["value"].shape == (6,)
        assert torch.isfinite(out["log_prob"]).all()
        assert torch.isfinite(out["entropy"]).all()

    def test_trunk_sharing_and_param_count(self) -> None:
        trunk = ref.TorchRLSharedTrunk()
        assert trunk.critic_path[0] is trunk.trunk
        assert trunk.actor_path[0] is trunk.trunk
        n_params = sum(p.numel() for p in trunk.parameters())
        assert n_params == 5061  # 8-64-64-4 shared trunk + 64-1 critic head

    def test_torchrl_module_matches_zerorl_actions(self) -> None:
        torch.manual_seed(7)
        obs = torch.randn(32, ref.OBS_DIM)
        agent = ref.build_deterministic_zerorl_agent()
        trunk = ref.TorchRLSharedTrunk()
        assert torch.equal(agent.get_action(obs)["action"], trunk.act(obs))

    def test_cross_adapter_parity(self) -> None:
        result = ref.assert_deterministic_parity()
        assert isinstance(result["fingerprint"], float)


class TestSB3Adapter:
    def test_shared_policy_builds_shared_trunk(self) -> None:
        pytest.importorskip("stable_baselines3")
        import gymnasium as gym

        policy = ref.build_sb3_shared_policy(
            gym.spaces.Box(-1.0, 1.0, (ref.OBS_DIM,)),
            gym.spaces.Discrete(ref.ACT_DIM),
        )
        assert policy.mlp_extractor.value_net is policy.mlp_extractor.policy_net
        n_params = sum(p.numel() for p in policy.parameters())
        assert n_params == 5061  # shared topology == zeroRL

    def test_load_and_deterministic_parity(self) -> None:
        pytest.importorskip("stable_baselines3")
        import gymnasium as gym

        policy = ref.build_sb3_shared_policy(
            gym.spaces.Box(-1.0, 1.0, (ref.OBS_DIM,)),
            gym.spaces.Discrete(ref.ACT_DIM),
        )
        ref.load_into_sb3(policy, ref.build_reference_state_dict())
        ref.assert_deterministic_parity_with_sb3(policy)

    def test_optimizer_has_no_duplicate_params(self) -> None:
        pytest.importorskip("stable_baselines3")
        import gymnasium as gym

        policy = ref.build_sb3_shared_policy(
            gym.spaces.Box(-1.0, 1.0, (ref.OBS_DIM,)),
            gym.spaces.Discrete(ref.ACT_DIM),
        )
        opt_params = {id(p) for g in policy.optimizer.param_groups
                      for p in g["params"]}
        model_params = {id(p) for p in policy.parameters()}
        assert opt_params == model_params
        assert len(policy.optimizer.param_groups[0]["params"]) == 8


_SMOKE_CFG_KWARGS = {"warmup": 500, "measure": 1000}


class TestTransitionCountVerification:
    """P2: every framework's transition counts are verified against the
    requested budget by a single helper that reports BOTH numbers."""

    def test_verify_transition_count_equal_passes(self) -> None:
        from benchmarks.scalability.collection import _verify_transition_count

        _verify_transition_count(100, 100, "warmup")   # no raise

    def test_verify_transition_count_raises_with_both_numbers(self) -> None:
        from benchmarks.scalability.collection import _verify_transition_count

        with pytest.raises(
            RuntimeError,
            match="warmup budget mismatch: actual=99 target=100",
        ):
            _verify_transition_count(99, 100, "warmup")
        with pytest.raises(
            RuntimeError,
            match="measure budget mismatch: actual=101 target=100",
        ):
            _verify_transition_count(101, 100, "measure")


class TestCollectionSmoke:
    """End-to-end Experiment A runs at tiny budgets (integration smoke)."""

    @staticmethod
    def _cfg():
        from benchmarks.scalability.collection import CollectionConfig

        return CollectionConfig(**_SMOKE_CFG_KWARGS)

    @staticmethod
    def _assert_exact(art: dict) -> None:
        t = art["transitions"]
        assert t["exact"], t
        assert t["warmup_actual"] == _SMOKE_CFG_KWARGS["warmup"]
        assert t["measured"] == _SMOKE_CFG_KWARGS["measure"]
        assert min(art["phases"].values()) > 0

    def test_zerorl_exact_budget_and_determinism(self) -> None:
        from benchmarks.scalability.collection import run_zerorl_collection

        cfg = self._cfg()
        a = run_zerorl_collection(cfg, num_envs=2, seed=777)
        b = run_zerorl_collection(cfg, num_envs=2, seed=777)
        self._assert_exact(a)
        assert a["fingerprint"] == b["fingerprint"]

    def test_zerorl_counts_come_from_buffer(self, monkeypatch) -> None:
        """P2: zeroRL counts must be READ from the buffer (slices × N),
        never fabricated from the target. A no-op rollout_phase writes no
        slices, so the runner must raise with the real count (0) instead of
        reporting the requested target."""
        from zerorl.helpers.train import BaseTrain

        from benchmarks.scalability.collection import run_zerorl_collection

        monkeypatch.setattr(BaseTrain, "rollout_phase", lambda self: None)
        cfg = self._cfg()
        with pytest.raises(RuntimeError) as exc:
            run_zerorl_collection(cfg, num_envs=2, seed=777)
        msg = str(exc.value)
        assert "warmup budget mismatch" in msg
        assert "actual=0" in msg
        assert f"target={cfg.warmup}" in msg

    def test_failure_mid_collection_preserves_error_and_closes(
        self, monkeypatch
    ) -> None:
        """P6: a mid-collection failure must surface the ORIGINAL error
        (with a cleanup note when a closer fails), attempt env close, and
        leave no new active children."""
        import multiprocessing

        import zerorl.functions as zf
        from zerorl.helpers.train import BaseTrain

        from benchmarks.scalability.collection import run_zerorl_collection

        def boom(self):
            raise RuntimeError("boom-marker")

        monkeypatch.setattr(BaseTrain, "rollout_phase", boom)

        real_ve = zf.vectorize_env
        close_calls: list[int] = []

        def spy_ve(*args, **kwargs):
            env = real_ve(*args, **kwargs)

            def close():
                close_calls.append(1)
                raise RuntimeError("close-failed-marker")

            env.close = close
            return env

        monkeypatch.setattr(zf, "vectorize_env", spy_ve)

        before = {id(p) for p in multiprocessing.active_children()}
        with pytest.raises(RuntimeError, match="boom-marker") as exc:
            run_zerorl_collection(self._cfg(), num_envs=2, seed=777)
        # original error preserved AND the failing closer is attached
        assert close_calls == [1], "env close was never attempted"
        notes = getattr(exc.value, "__notes__", [])
        assert any("cleanup [env] failed" in n for n in notes), notes
        assert any("close-failed-marker" in n for n in notes), notes
        after = {id(p) for p in multiprocessing.active_children()}
        assert after == before, "failure path leaked processes"

    def test_sb3_exact_budget_and_determinism(self) -> None:
        pytest.importorskip("stable_baselines3")
        from benchmarks.scalability.collection import run_sb3_collection

        cfg = self._cfg()
        a = run_sb3_collection(cfg, num_envs=2, seed=777)
        b = run_sb3_collection(cfg, num_envs=2, seed=777)
        self._assert_exact(a)
        assert a["fingerprint"] == b["fingerprint"]

    def test_torchrl_exact_budget_and_determinism(self) -> None:
        pytest.importorskip("torchrl")
        import multiprocessing

        from benchmarks.scalability.collection import run_torchrl_collection

        before = set(id(p) for p in multiprocessing.active_children())
        cfg = self._cfg()
        a = run_torchrl_collection(cfg, num_envs=2, seed=777)
        b = run_torchrl_collection(cfg, num_envs=2, seed=777)
        self._assert_exact(a)
        assert a["fingerprint"] == b["fingerprint"]
        after = set(id(p) for p in multiprocessing.active_children())
        assert after == before, "torchrl sync collection leaked processes"

    def test_three_framework_initial_state_and_action_parity(self) -> None:
        """Same canonical seeds + same weights => bit-equal trajectories.

        All three frameworks must START from the identical canonical initial
        state (TorchRL's GymWrapper info-reader check consumes the first
        reset, so a re-arm/restore procedure must run after setup) and then
        produce identical deterministic actions and per-phase action sums.
        """
        pytest.importorskip("stable_baselines3")
        pytest.importorskip("torchrl")
        from benchmarks.scalability.collection import (
            run_sb3_collection, run_torchrl_collection, run_zerorl_collection,
        )

        cfg = self._cfg()
        z = run_zerorl_collection(cfg, num_envs=2, seed=777)
        s = run_sb3_collection(cfg, num_envs=2, seed=777)
        t = run_torchrl_collection(cfg, num_envs=2, seed=777)
        for key in ("first_obs_head", "actions_head",
                    "action_sum_warmup", "action_sum_measure"):
            zv = z["fingerprint"][key]
            assert zv == s["fingerprint"][key], f"zeroRL vs SB3 differ on {key}"
            assert zv == t["fingerprint"][key], f"zeroRL vs TorchRL differ on {key}"

    def test_different_seeds_differ(self) -> None:
        from benchmarks.scalability.collection import run_zerorl_collection

        a = run_zerorl_collection(self._cfg(), num_envs=2, seed=777)
        b = run_zerorl_collection(self._cfg(), num_envs=2, seed=778)
        assert a["fingerprint"]["actions_head"] != b["fingerprint"]["actions_head"]

    def test_collection_fields_recorded_for_all_frameworks(self) -> None:
        """Exp A workload contract: every path RECORDS action/log_prob/
        entropy/value for exactly the measured number of transitions."""
        pytest.importorskip("stable_baselines3")
        pytest.importorskip("torchrl")
        from benchmarks.scalability.collection import (
            run_sb3_collection, run_torchrl_collection, run_zerorl_collection,
        )

        cfg = self._cfg()
        runners = (run_zerorl_collection, run_sb3_collection,
                   run_torchrl_collection)
        measured = _SMOKE_CFG_KWARGS["measure"]
        for runner in runners:
            art = runner(cfg, num_envs=2, seed=777)
            fields = art["config_detail"]["fields"]
            for key in ("action", "log_prob", "entropy", "value"):
                assert fields[key] == measured, \
                    f"{runner.__name__}: fields[{key}]={fields[key]} != {measured}"

    def test_sb3_single_pass_collection_forward(self, monkeypatch) -> None:
        """SB3 must compute all four fields in ONE trunk pass per step
        (forward_collection), not two (get_distribution + predict_values)."""
        pytest.importorskip("stable_baselines3")
        from benchmarks.scalability import policy_reference as ref
        from benchmarks.scalability.collection import run_sb3_collection

        cls = ref.shared_trunk_policy_cls()
        orig = cls.forward_collection
        calls = {"n": 0}

        def spy(self, obs):
            calls["n"] += 1
            return orig(self, obs)

        monkeypatch.setattr(cls, "forward_collection", spy)
        monkeypatch.setattr(ref, "shared_trunk_policy_cls", lambda *a, **k: cls)
        art = run_sb3_collection(self._cfg(), num_envs=2, seed=777)
        warm_steps = _SMOKE_CFG_KWARGS["warmup"] // 2
        meas_steps = _SMOKE_CFG_KWARGS["measure"] // 2
        assert calls["n"] == warm_steps + meas_steps, calls
        assert art["transitions"]["measured"] == _SMOKE_CFG_KWARGS["measure"]

    def test_torchrl_policy_outputs_four_fields(self) -> None:
        """One TorchRL collection-policy forward returns all four per-step
        fields: (action, log_prob, entropy, value), each shape (N,)."""
        pytest.importorskip("torchrl")
        import torch

        from benchmarks.scalability.collection import _TorchRLCollectPolicy
        from benchmarks.scalability.policy_reference import (
            OBS_DIM, TorchRLSharedTrunk, build_reference_state_dict,
        )

        trunk = TorchRLSharedTrunk(build_reference_state_dict())
        policy = _TorchRLCollectPolicy(trunk)
        obs = torch.zeros(3, OBS_DIM)
        out = policy(obs)
        assert isinstance(out, tuple) and len(out) == 4, \
            f"expected 4-field tuple, got {type(out)}"
        for tensor in out:
            assert tensor.shape == (3,), tensor.shape


class TestWorkerHelpers:
    def test_supported_workers_mapping(self) -> None:
        from benchmarks.scalability.workers import supported_workers

        assert supported_workers("zeroRL", 8) == {1}
        assert supported_workers("SB3", 8) == {1, 8}
        assert supported_workers("TorchRL", 8) == {1, 2, 4, 8}
        assert supported_workers("TorchRL", 16) == {1, 2, 4, 8}

    def test_split_seeds(self) -> None:
        from benchmarks.scalability.workers import split_seeds

        seeds = list(range(8))
        assert split_seeds(seeds, 2) == [list(range(4)), list(range(4, 8))]
        assert split_seeds(seeds, 8) == [[i] for i in range(8)]
        with pytest.raises(ValueError):
            split_seeds(seeds, 3)   # 8 % 3 != 0

    def test_unsupported_configs_raise(self) -> None:
        from benchmarks.scalability.collection import CollectionConfig
        from benchmarks.scalability.workers import (
            UnsupportedConfiguration, run_b_collection,
        )

        cfg = CollectionConfig(warmup=800, measure=1600)
        with pytest.raises(UnsupportedConfiguration, match="multi-process"):
            run_b_collection("zeroRL", cfg, 2, seed=1)
        with pytest.raises(UnsupportedConfiguration, match="one process per env"):
            run_b_collection("SB3", cfg, 4, seed=1)

    def test_choose_collector_fpb(self) -> None:
        from benchmarks.scalability.collection import choose_collector_fpb

        fpb = choose_collector_fpb(16_000, 16, 8_000)
        assert fpb % 16 == 0 and fpb <= 8_000 and 16_000 % fpb == 0
        fpb = choose_collector_fpb(100_000, 8, 8_000)
        assert fpb % 8 == 0 and fpb <= 8_000 and 100_000 % fpb == 0
        with pytest.raises(ValueError):
            choose_collector_fpb(500, 8, 8_000)   # 500 % 8 != 0


_WORKER_SMOKE_KWARGS = {"warmup": 400, "measure": 800}   # divisible by N=8


class TestWorkerSmoke:
    """Experiment B integration smoke (N=8, tiny budgets)."""

    @staticmethod
    def _cfg():
        from benchmarks.scalability.collection import CollectionConfig

        return CollectionConfig(**_WORKER_SMOKE_KWARGS)

    @staticmethod
    def _assert_exact(art: dict) -> None:
        t = art["transitions"]
        assert t["exact"], t
        assert t["measured"] == _WORKER_SMOKE_KWARGS["measure"]
        assert min(art["phases"].values()) > 0

    def test_torchrl_mp_exact_budget_and_determinism(self) -> None:
        pytest.importorskip("torchrl")
        import multiprocessing

        from benchmarks.scalability.workers import run_b_collection

        cfg = self._cfg()
        before = {id(p) for p in multiprocessing.active_children()}
        a = run_b_collection("TorchRL", cfg, 2, seed=777)
        b = run_b_collection("TorchRL", cfg, 2, seed=777)
        self._assert_exact(a)
        assert a["fingerprint"] == b["fingerprint"]
        after = {id(p) for p in multiprocessing.active_children()}
        assert after == before, "torchrl mp collection leaked worker processes"

    def test_failure_mp_collection_reaps_workers(self, monkeypatch) -> None:
        """P6 failure path for the worker-based runner: original error
        preserved, workers reaped (no new active children)."""
        pytest.importorskip("torchrl")
        import multiprocessing

        from benchmarks.scalability import collection as coll_mod
        from benchmarks.scalability.workers import run_torchrl_mp_collection

        def boom(*_args, **_kwargs):
            raise RuntimeError("mp-boom-marker")

        # _absorb_actions runs in the PARENT while processing warmup tds —
        # after collectors (and their workers) are already spawned.
        monkeypatch.setattr(coll_mod, "_absorb_actions", boom)

        cfg = self._cfg()
        before = {p.pid for p in multiprocessing.active_children()}
        try:
            with pytest.raises(RuntimeError, match="mp-boom-marker"):
                run_torchrl_mp_collection(cfg, num_envs=4, num_workers=2,
                                          seed=777)
            after = {p.pid for p in multiprocessing.active_children()}
            assert after <= before, "mp failure path leaked worker processes"
        finally:
            # never let a RED run's leaked workers hang pytest at exit
            for p in multiprocessing.active_children():
                if p.pid not in before:
                    p.terminate()
                    p.join(timeout=5)

    def test_zerorl_w1_runs(self) -> None:
        from benchmarks.scalability.workers import run_b_collection

        art = run_b_collection("zeroRL", self._cfg(), 1, seed=777)
        self._assert_exact(art)
        assert art["config_detail"]["seeds"] is not None

    def test_sb3_subproc_exact_budget(self) -> None:
        pytest.importorskip("stable_baselines3")
        import multiprocessing

        from benchmarks.scalability.workers import run_b_collection

        before = {id(p) for p in multiprocessing.active_children()}
        art = run_b_collection("SB3", self._cfg(), 8, seed=777)
        self._assert_exact(art)
        assert art["config_detail"]["vec_env"] == "SubprocVecEnv"
        after = {id(p) for p in multiprocessing.active_children()}
        assert after == before, "SB3 SubprocVecEnv leaked worker processes"

    def test_torchrl_w1_equals_sync_runner(self) -> None:
        pytest.importorskip("torchrl")
        from benchmarks.scalability.workers import run_b_collection

        art = run_b_collection("TorchRL", self._cfg(), 1, seed=777)
        self._assert_exact(art)
        assert art["config_detail"]["collector"] == "Collector(backend='direct')"


class TestPPOProtocolConfig:
    """Protocol C/D budget math (spec: C updates = 128/N, D updates = 128)."""

    def test_protocol_c_rollout_and_updates(self) -> None:
        from benchmarks.scalability.ppo_protocols import PPOProtocolConfig

        cfg = PPOProtocolConfig()
        for n in (1, 2, 4, 8, 16):
            assert cfg.rollout_steps("C", n) == 2048
            assert cfg.updates("C", n) == 128 // n

    def test_protocol_d_rollout_and_updates(self) -> None:
        from benchmarks.scalability.ppo_protocols import PPOProtocolConfig

        cfg = PPOProtocolConfig()
        for n in (1, 2, 4, 8, 16):
            rollout = cfg.rollout_steps("D", n)
            assert rollout == 2048 // n
            assert rollout * n == 2048        # global batch fixed
            assert cfg.updates("D", n) == 128  # constant across N

    def test_invalid_inputs(self) -> None:
        from benchmarks.scalability.ppo_protocols import PPOProtocolConfig

        cfg = PPOProtocolConfig()
        with pytest.raises(ValueError, match="divisible"):
            cfg.rollout_steps("D", 3)          # 2048 % 3 != 0
        with pytest.raises(ValueError, match="experiment"):
            cfg.rollout_steps("A", 1)
        with pytest.raises(ValueError, match="not divisible"):
            PPOProtocolConfig(total=1000).updates("C", 1)
        with pytest.raises(ValueError, match="not divisible"):
            PPOProtocolConfig(total=1024).updates("C", 2)   # 1024 < 2048*2
        with pytest.raises(ValueError):
            PPOProtocolConfig(total=1024).updates("C", 4)

    def test_dispatcher_rejects_bad_args(self) -> None:
        from benchmarks.scalability.ppo_protocols import run_ppo_training

        with pytest.raises(ValueError, match="experiment"):
            run_ppo_training("zeroRL", "A", 1, 0)
        with pytest.raises(ValueError, match="framework"):
            run_ppo_training("CleanRL", "C", 1, 0)


class TestPPOParamAlignment:
    """P5: explicit protocol -> AlgoConfig mapping, recorded lr/optimizer
    evidence (empirically: all three decay 3e-4 -> 0 linearly, Adam eps 1e-5),
    and boundary phase timing."""

    def test_zerorl_algo_config_mapped_from_protocol(self) -> None:
        from benchmarks.scalability.ppo_protocols import (
            PPOProtocolConfig,
            _algo_config_from,
        )

        proto = PPOProtocolConfig(lr=1e-3, gamma=0.9, gae_lambda=0.8,
                                  clip_eps=0.1, ent_coef=0.02, vf_coef=0.3,
                                  epochs=3, batch_size=32)
        ac = _algo_config_from(proto)
        assert (ac.lr, ac.gamma, ac.gae_lambda, ac.clip_eps, ac.ent_coef,
                ac.value_coef, ac.epochs, ac.batch_size) == (
            1e-3, 0.9, 0.8, 0.1, 0.02, 0.3, 3, 32,
        )

    @pytest.mark.parametrize("framework", ["zeroRL", "SB3", "TorchRL"])
    def test_lr_and_optimizer_recorded_aligned(self, framework: str) -> None:
        """tiny protocol-C run: lr_start/lr_end/optimizer recorded and
        identical across frameworks (3e-4 -> 0, Adam eps=1e-5)."""
        if framework == "SB3":
            pytest.importorskip("stable_baselines3")
        if framework == "TorchRL":
            pytest.importorskip("torchrl")
        from benchmarks.scalability.ppo_protocols import (
            PPOProtocolConfig,
            run_ppo_training,
        )

        cfg = PPOProtocolConfig(total=8192)   # 4 updates of 2048 @ N=1
        art = run_ppo_training(framework, "C", 1, 777, cfg)
        detail = art["config_detail"]
        assert detail["lr_start"] == pytest.approx(3e-4, rel=1e-6)
        assert detail["lr_end"] == pytest.approx(0.0, abs=1e-9)
        opt = detail["optimizer"]
        assert opt["type"] == "Adam"
        assert opt["eps"] == pytest.approx(1e-5)
        assert opt["schedule"] == "linear_decay_to_zero"

    @pytest.mark.parametrize("framework", ["zeroRL", "SB3", "TorchRL"])
    def test_boundary_seconds_present_and_positive(self, framework: str) -> None:
        """A/B artifacts must time the warmup|measure boundary as its own
        phase (collector/worker restart, phase reset)."""
        if framework == "SB3":
            pytest.importorskip("stable_baselines3")
        if framework == "TorchRL":
            pytest.importorskip("torchrl")
        from benchmarks.scalability.collection import CollectionConfig
        from benchmarks.scalability.collection import run_collection

        cfg = CollectionConfig(**_SMOKE_CFG_KWARGS)
        art = run_collection(framework, cfg, num_envs=2, seed=777)
        assert art["phases"]["boundary_seconds"] > 0, art["phases"]


class TestTorchRLPPOGradientFlow:
    """ClipPPOLoss must share the trunk parameters with the collector's
    policy: backward() has to put grads on the exact params the optimizer
    steps, otherwise TorchRL 'training' silently does nothing."""

    def test_loss_updates_trunk_params(self) -> None:
        """Production wiring (build_torchrl_ppo_wiring +
        _torchrl_train_minibatch): backward() puts grads on trunk + both
        heads and the optimizer step changes every trunk parameter."""
        pytest.importorskip("torchrl")
        from tensordict import TensorDict

        from benchmarks.scalability.ppo_protocols import (
            PPOProtocolConfig,
            _torchrl_train_minibatch,
            build_torchrl_ppo_wiring,
        )

        cfg = PPOProtocolConfig()
        trunk = ref.TorchRLSharedTrunk(ref.build_reference_state_dict())
        wiring = build_torchrl_ppo_wiring(trunk, cfg, cfg.updates("C", 1))
        n_trunk = len(list(trunk.parameters()))
        assert len(wiring["params"]) == n_trunk == 8

        torch.manual_seed(0)
        td = TensorDict({
            "observation": torch.randn(64, ref.OBS_DIM),
            "action": torch.randint(0, ref.ACT_DIM, (64,)),
            wiring["lp_key"]: torch.zeros(64, 1),
            "advantage": torch.randn(64, 1),
            "value_target": torch.randn(64, 1),
            "state_value": torch.randn(64, 1),
        }, batch_size=[64])
        before = [p.clone() for p in trunk.parameters()]
        _torchrl_train_minibatch(trunk, wiring["loss_module"], td,
                                 wiring["optimizer"], wiring["params"],
                                 cfg.grad_clip)
        with_grad = sum(
            1 for p in trunk.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert with_grad == n_trunk
        changed = sum(
            (b - a).abs().sum().item() > 0
            for b, a in zip(before, trunk.parameters())
        )
        assert changed == n_trunk


class TestSharedTrunkSinglePass:
    """Shared-trunk fix: the actor/critic share ONE trunk execution per
    actor-critic pass, for BOTH collection and PPO training, in every
    framework. zeroRL (`extract_layer` once in forward) is the reference."""

    # ------------------------------------------------------------------ #
    # unit: trunk executions per pass
    # ------------------------------------------------------------------ #

    def test_zerorl_get_action_single_trunk(self) -> None:
        from zerorl.helpers.factory import ActorCriticAgent

        agent = ActorCriticAgent(ref.OBS_DIM, ref.ACT_DIM, is_discrete=True,
                                 hidden_dim=ref.HIDDEN)
        count = [0]
        agent.extract_layer.register_forward_hook(
            lambda *_: count.__setitem__(0, count[0] + 1))
        agent.get_action(torch.randn(7, ref.OBS_DIM))
        assert count[0] == 1, f"zeroRL get_action ran trunk {count[0]}x"

    def test_sb3_paths_single_trunk(self) -> None:
        """forward_collection (Exp A), stock forward (C/D collect) and
        evaluate_actions (C/D training) each run the shared trunk ONCE."""
        pytest.importorskip("stable_baselines3")
        import gymnasium as gym

        policy = ref.build_sb3_shared_policy(
            gym.spaces.Box(-1.0, 1.0, shape=(ref.OBS_DIM,),
                           dtype="float32"),
            gym.spaces.Discrete(ref.ACT_DIM),
        )
        count = [0]
        policy.mlp_extractor.policy_net.register_forward_hook(
            lambda *_: count.__setitem__(0, count[0] + 1))
        obs = torch.randn(7, ref.OBS_DIM)

        policy.forward_collection(obs)
        assert count[0] == 1, f"forward_collection ran trunk {count[0]}x"

        policy.forward(obs)
        assert count[0] == 2, f"stock forward ran trunk {count[0] - 1}x"

        actions = torch.randint(0, ref.ACT_DIM, (7,))
        policy.evaluate_actions(obs, actions)
        assert count[0] == 3, f"evaluate_actions ran trunk {count[0] - 2}x"

    def test_torchrl_collect_policy_single_trunk(self) -> None:
        pytest.importorskip("torchrl")
        from benchmarks.scalability.collection import _TorchRLCollectPolicy

        trunk = ref.TorchRLSharedTrunk(ref.build_reference_state_dict())
        count = [0]
        trunk.trunk.register_forward_hook(
            lambda *_: count.__setitem__(0, count[0] + 1))
        policy = _TorchRLCollectPolicy(trunk)
        out = policy(torch.randn(7, ref.OBS_DIM))
        assert len(out) == 4
        assert count[0] == 1, f"TorchRL collect policy ran trunk {count[0]}x"

    def test_torchrl_minibatch_single_trunk_and_grads(self) -> None:
        """One PPO minibatch = exactly one trunk pass (features recomputed
        with grad), grads land on trunk + both heads, optimizer params are
        de-duplicated."""
        pytest.importorskip("torchrl")
        from tensordict import TensorDict

        from benchmarks.scalability.ppo_protocols import (
            PPOProtocolConfig,
            _torchrl_train_minibatch,
            build_torchrl_ppo_wiring,
        )

        cfg = PPOProtocolConfig()
        trunk = ref.TorchRLSharedTrunk(ref.build_reference_state_dict())
        wiring = build_torchrl_ppo_wiring(trunk, cfg,
                                          cfg.updates("C", 1))
        loss_module, params = wiring["loss_module"], wiring["params"]
        n_trunk = len(list(trunk.parameters()))
        assert len(params) == n_trunk == 8          # 4 trunk + 2 actor + 2 critic
        assert sum(p.numel() for p in params) == 5061   # shared elements
        assert len({id(p) for p in params}) == len(params)

        count = [0]
        trunk.trunk.register_forward_hook(
            lambda *_: count.__setitem__(0, count[0] + 1))
        torch.manual_seed(0)
        td = TensorDict({
            "observation": torch.randn(64, ref.OBS_DIM),
            "action": torch.randint(0, ref.ACT_DIM, (64,)),
            wiring["lp_key"]: torch.zeros(64, 1),
            "advantage": torch.randn(64, 1),
            "value_target": torch.randn(64, 1),
            "state_value": torch.randn(64, 1),
        }, batch_size=[64])
        opt = torch.optim.Adam(params, lr=3e-4, eps=1e-5)
        _torchrl_train_minibatch(trunk, loss_module, td, opt, params, 0.5)
        assert count[0] == 1, f"minibatch ran trunk {count[0]}x"
        with_grad = sum(1 for p in trunk.parameters()
                        if p.grad is not None and p.grad.abs().sum() > 0)
        assert with_grad == n_trunk, f"only {with_grad}/{n_trunk} got grads"

    # ------------------------------------------------------------------ #
    # e2e: actual trunk executions of a complete tiny protocol-C run
    # ------------------------------------------------------------------ #

    @staticmethod
    def _c_cfg():
        from benchmarks.scalability.ppo_protocols import PPOProtocolConfig

        return PPOProtocolConfig(total=_PPO_SMOKE_TOTAL)   # N=1: 1 update

    def test_zerorl_ppo_e2e_trunk_count(self, monkeypatch) -> None:
        """2048 env-step get_action + 1 rollout bootstrap + 320 minibatch
        forwards (10 epochs x 32) = 2369 trunk executions."""
        import zerorl.helpers.factory as zf

        from benchmarks.scalability import ppo_protocols as pp

        counter = [0]

        class CountingAgent(zf.ActorCriticAgent):
            # run_zerorl_ppo imports ActorCriticAgent from zf at call time
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.extract_layer.register_forward_hook(
                    lambda *_: counter.__setitem__(0, counter[0] + 1))

        monkeypatch.setattr(zf, "ActorCriticAgent", CountingAgent)
        art = pp.run_zerorl_ppo(self._c_cfg(), "C", 1, 777)
        assert art["transitions"]["exact"]
        assert counter[0] == 2048 + 1 + 320, (
            f"zeroRL trunk executions: {counter[0]} (expected 2369)")

    def test_sb3_ppo_e2e_trunk_count(self, monkeypatch) -> None:
        """Trunk executions decompose exactly into: 2048 collect forwards
        (one per env step, single-pass after the fix) + 320 minibatch
        evaluate_actions + official TimeLimit bootstrap calls (SB3's
        timeout block calls policy.predict_values once per truncated
        episode — observed 1 for seed 777; the decomposition equality is
        the anti-duplication check: a double-trunk regression cannot
        satisfy it)."""
        pytest.importorskip("stable_baselines3")
        import stable_baselines3.common.policies as s3pol

        from benchmarks.scalability import ppo_protocols as pp

        counter = [0]
        pv_calls = [0]
        orig_pv = s3pol.ActorCriticPolicy.predict_values

        def pv_spy(self, obs):
            pv_calls[0] += 1
            return orig_pv(self, obs)

        monkeypatch.setattr(s3pol.ActorCriticPolicy, "predict_values", pv_spy)

        def counting_cls():
            base = ref.shared_trunk_policy_cls()

            class Counting(base):  # type: ignore[misc,valid-type]
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.mlp_extractor.policy_net.register_forward_hook(
                        lambda *_: counter.__setitem__(0, counter[0] + 1))

            return Counting

        monkeypatch.setattr(pp, "shared_trunk_policy_cls", counting_cls)
        art = pp.run_sb3_ppo(self._c_cfg(), "C", 1, 777)
        assert art["transitions"]["exact"]
        assert counter[0] == 2048 + 320 + pv_calls[0], (
            f"SB3 trunk executions: {counter[0]} != 2048+320+{pv_calls[0]}")

    def test_torchrl_ppo_e2e_trunk_count(self, monkeypatch) -> None:
        """Trunk executions decompose exactly into: collector-policy
        invocations (observed 2049 for 2048 counted frames — one extra
        boundary forward inside torchrl's Collector) + 1 GAE
        next-features injection per td yield (total == fpb: one yield)
        + 320 minibatch feature passes. The equality is the
        anti-duplication check: a double-trunk regression cannot satisfy
        it."""
        pytest.importorskip("torchrl")
        from benchmarks.scalability import ppo_protocols as pp

        counter = [0]
        policy_calls = [0]
        orig_wiring = pp.build_torchrl_ppo_wiring

        def wiring_spy(trunk, cfg, updates):
            wiring = orig_wiring(trunk, cfg, updates)
            wiring["collector_policy"].register_forward_hook(
                lambda *_: policy_calls.__setitem__(0, policy_calls[0] + 1))
            return wiring

        monkeypatch.setattr(pp, "build_torchrl_ppo_wiring", wiring_spy)

        class CountingTrunk(ref.TorchRLSharedTrunk):
            def __init__(self, sd=None):
                super().__init__(sd)
                self.trunk.register_forward_hook(
                    lambda *_: counter.__setitem__(0, counter[0] + 1))

        monkeypatch.setattr(pp, "TorchRLSharedTrunk", CountingTrunk)
        art = pp.run_torchrl_ppo(self._c_cfg(), "C", 1, 777)
        assert art["transitions"]["exact"]
        assert policy_calls[0] >= 2048, policy_calls[0]
        assert counter[0] == policy_calls[0] + 1 + 320, (
            f"TorchRL trunk executions: {counter[0]} != "
            f"{policy_calls[0]}+1+320")

    # ------------------------------------------------------------------ #
    # numerical equivalence + gradients + optimizer de-duplication
    # ------------------------------------------------------------------ #

    def test_adapters_match_zeroRL_reference_batched(self) -> None:
        """Same reference weights => single-pass SB3/TorchRL outputs equal
        zeroRL's, for a vectorized batch of observations."""
        pytest.importorskip("stable_baselines3")
        pytest.importorskip("torchrl")
        import gymnasium as gym

        from benchmarks.scalability.collection import _TorchRLCollectPolicy

        sd = ref.build_reference_state_dict()
        agent = ref.build_deterministic_zerorl_agent(sd)
        sb3 = ref.build_sb3_shared_policy(
            gym.spaces.Box(-1.0, 1.0, shape=(ref.OBS_DIM,),
                           dtype="float32"),
            gym.spaces.Discrete(ref.ACT_DIM))
        ref.load_into_sb3(sb3, sd)
        trunk = ref.TorchRLSharedTrunk(sd)
        torchrl_policy = _TorchRLCollectPolicy(trunk)

        obs = torch.randn(64, ref.OBS_DIM)
        z_logits, z_value = agent.forward(obs)
        z_dist = agent.build_distribution(z_logits)
        action = z_dist.probs.argmax(dim=-1)          # deterministic
        z_lp = z_dist.log_prob(action)
        z_ent = z_dist.entropy()
        z_value = z_value.squeeze(-1)

        s_actions, s_values, s_lp, s_ent = sb3.forward_collection(obs)
        assert torch.equal(s_actions, action)
        assert torch.allclose(s_values.squeeze(-1), z_value, atol=1e-6)
        assert torch.allclose(s_lp, z_lp, atol=1e-6)
        assert torch.allclose(s_ent, z_ent, atol=1e-6)
        # training-path single pass must agree too
        e_values, e_lp, e_ent = sb3.evaluate_actions(obs, action)
        assert torch.allclose(e_values.squeeze(-1), z_value, atol=1e-6)
        assert torch.allclose(e_lp, z_lp, atol=1e-6)
        assert torch.allclose(e_ent, z_ent, atol=1e-6)

        t_actions, t_lp, t_ent, t_value = torchrl_policy(obs)
        assert torch.equal(t_actions, action)
        assert torch.allclose(t_value, z_value, atol=1e-6)
        assert torch.allclose(t_lp, z_lp, atol=1e-6)
        assert torch.allclose(t_ent, z_ent, atol=1e-6)

    def test_sb3_evaluate_actions_grads_reach_trunk_and_heads(self) -> None:
        pytest.importorskip("stable_baselines3")
        import gymnasium as gym

        policy = ref.build_sb3_shared_policy(
            gym.spaces.Box(-1.0, 1.0, shape=(ref.OBS_DIM,),
                           dtype="float32"),
            gym.spaces.Discrete(ref.ACT_DIM))
        obs = torch.randn(16, ref.OBS_DIM)
        actions = torch.randint(0, ref.ACT_DIM, (16,))
        values, lp, ent = policy.evaluate_actions(obs, actions)
        (values.sum() + lp.sum() + ent.sum()).backward()
        for name, module in (("trunk", policy.mlp_extractor.policy_net),
                             ("actor_head", policy.action_net),
                             ("critic_head", policy.value_net)):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            assert grads, f"{name}: no gradients"
            assert any(g.abs().sum() > 0 for g in grads), f"{name}: zero grads"

    def test_optimizer_parameters_unique_and_complete(self) -> None:
        """No duplicated parameters in any framework's optimizer: 8 unique
        parameter tensors carrying exactly 5061 elements (4736 trunk + 260
        actor + 65 critic — any dropped or duplicated tensor changes this)."""
        import gymnasium as gym

        sd = ref.build_reference_state_dict()

        def _check(params, label):
            ids = [id(p) for p in params]
            assert len(ids) == len(set(ids)) == 8, f"{label}: {len(ids)} tensors"
            assert sum(p.numel() for p in params) == 5061, (
                f"{label}: {sum(p.numel() for p in params)} elements != 5061")

        # zeroRL
        agent = ref.build_deterministic_zerorl_agent(sd)
        _check(list(agent.parameters()), "zeroRL")
        # SB3
        pytest.importorskip("stable_baselines3")
        sb3 = ref.build_sb3_shared_policy(
            gym.spaces.Box(-1.0, 1.0, shape=(ref.OBS_DIM,),
                           dtype="float32"),
            gym.spaces.Discrete(ref.ACT_DIM))
        _check([p for g in sb3.optimizer.param_groups for p in g["params"]],
               "SB3 optimizer")
        # TorchRL
        pytest.importorskip("torchrl")
        from benchmarks.scalability.ppo_protocols import (
            PPOProtocolConfig, build_torchrl_ppo_wiring,
        )
        cfg = PPOProtocolConfig()
        trunk = ref.TorchRLSharedTrunk(sd)
        wiring = build_torchrl_ppo_wiring(trunk, cfg, cfg.updates("C", 1))
        _check(wiring["params"], "TorchRL wiring")


_PPO_SMOKE_TOTAL = 2048   # smallest exact budget: one update under C and D


class TestPPOProtocolSmoke:
    """End-to-end C/D runs at the smallest exact budget (integration)."""

    @staticmethod
    def _assert_art(art: dict, experiment: str, num_envs: int) -> None:
        from benchmarks.scalability.ppo_protocols import PPOProtocolConfig

        cfg = PPOProtocolConfig(total=_PPO_SMOKE_TOTAL)
        t = art["transitions"]
        assert t["exact"], t
        assert t["measured"] == _PPO_SMOKE_TOTAL
        assert min(art["phases"].values()) > 0
        m = art["metrics"]
        assert m["updates"] == cfg.updates(experiment, num_envs)
        assert m["throughput"] > 0
        assert m["episodes"] >= 1
        assert isinstance(m["mean_return"], float)
        assert len(art["episode_returns"]) == m["episodes"]
        d = art["config_detail"]
        assert d["lr"] == 3e-4 and d["grad_clip"] == 0.5
        assert d["epochs"] == 10 and d["batch_size"] == 64
        assert d["optimizer"]["schedule"] == "linear_decay_to_zero"
        assert d["lr_start"] == pytest.approx(3e-4, rel=1e-6)
        assert d["lr_end"] == pytest.approx(0.0, abs=1e-9)
        if experiment == "C":
            assert d["rollout_steps_per_env"] == 2048
        else:
            assert d["rollout_global"] == 2048

    @pytest.mark.parametrize("experiment,num_envs", [("C", 1), ("D", 2)])
    def test_zerorl(self, experiment: str, num_envs: int) -> None:
        from benchmarks.scalability.ppo_protocols import (
            PPOProtocolConfig, run_ppo_training,
        )

        art = run_ppo_training("zeroRL", experiment, num_envs, seed=777,
                               cfg=PPOProtocolConfig(total=_PPO_SMOKE_TOTAL))
        self._assert_art(art, experiment, num_envs)

    @pytest.mark.parametrize("experiment,num_envs", [("C", 1), ("D", 2)])
    def test_sb3(self, experiment: str, num_envs: int) -> None:
        pytest.importorskip("stable_baselines3")
        from benchmarks.scalability.ppo_protocols import (
            PPOProtocolConfig, run_ppo_training,
        )

        art = run_ppo_training("SB3", experiment, num_envs, seed=777,
                               cfg=PPOProtocolConfig(total=_PPO_SMOKE_TOTAL))
        self._assert_art(art, experiment, num_envs)

    @pytest.mark.parametrize("experiment,num_envs", [("C", 1), ("D", 2)])
    def test_torchrl(self, experiment: str, num_envs: int) -> None:
        pytest.importorskip("torchrl")
        import multiprocessing

        from benchmarks.scalability.ppo_protocols import (
            PPOProtocolConfig, run_ppo_training,
        )

        before = {id(p) for p in multiprocessing.active_children()}
        art = run_ppo_training("TorchRL", experiment, num_envs, seed=777,
                               cfg=PPOProtocolConfig(total=_PPO_SMOKE_TOTAL))
        self._assert_art(art, experiment, num_envs)
        after = {id(p) for p in multiprocessing.active_children()}
        assert after == before, "torchrl PPO run leaked processes"

    def test_zerorl_same_seed_repeats(self) -> None:
        from benchmarks.scalability.ppo_protocols import (
            PPOProtocolConfig, run_ppo_training,
        )

        cfg = PPOProtocolConfig(total=_PPO_SMOKE_TOTAL)
        a = run_ppo_training("zeroRL", "D", 2, seed=777, cfg=cfg)
        b = run_ppo_training("zeroRL", "D", 2, seed=777, cfg=cfg)
        self._assert_art(a, "D", 2)
        assert a["episode_returns"] == b["episode_returns"]
        assert a["metrics"]["updates"] == b["metrics"]["updates"]


def _mk_row(framework: str, experiment: str, *, num_envs: int = 1,
            num_workers: int = 1, rep: int = 0, status: str = "ok",
            measured: int = 100_000, measure_s: float = 10.0,
            exact: bool = True, mode: str = "sync",
            protocol: str = "collection", error: str | None = None,
            return_last100: float | None = None) -> dict:
    phases = {"init_seconds": 0.1, "warmup_seconds": 1.0,
              "measure_seconds": measure_s}
    if experiment in ("C", "D"):
        phases = {"init_seconds": 0.1, "measure_seconds": measure_s}
    metrics: dict = {"throughput_mean": measured / measure_s}
    if return_last100 is not None:
        metrics.update({"updates": 128, "episodes": 10,
                        "mean_return": return_last100,
                        "mean_return_last100": return_last100})
    return schema.make_row(
        framework=framework, experiment=experiment, mode=mode,
        num_envs=num_envs, num_workers=num_workers, protocol=protocol,
        rep=rep, seed=1000 + rep, status=status, phases=phases,
        transitions={"warmup_target": 16_000, "warmup_actual": 16_000,
                     "measure_target": measured, "measured": measured,
                     "exact": exact},
        metrics=metrics, config={"env_id": "LunarLander-v3",
                                 "updates": 128},
        error=error,
    )


class TestAggregate:
    def test_flatten_row(self) -> None:
        from benchmarks.scalability.aggregate import flatten_row

        flat = flatten_row(_mk_row("zeroRL", "A"))
        assert flat["transitions_measured"] == 100_000
        assert flat["measure_seconds"] == 10.0
        assert flat["init_seconds"] == 0.1

    def test_group_and_aggregate_cells(self) -> None:
        from benchmarks.scalability.aggregate import (
            aggregate_cells, group_cells,
        )

        rows = [
            _mk_row("zeroRL", "A", num_envs=1, rep=0, measure_s=10.0),
            _mk_row("zeroRL", "A", num_envs=1, rep=1, measure_s=12.0),
            _mk_row("SB3", "A", num_envs=2, rep=0, measure_s=5.0),
            _mk_row("zeroRL", "B", num_envs=8, num_workers=2, rep=0,
                    status="unsupported", measured=0, measure_s=0.01,
                    exact=False, mode="mp",
                    error="unsupported: sync-only"),
        ]
        grouped = group_cells(rows, "A")
        assert set(grouped["zeroRL"]) == {1}
        assert set(grouped["SB3"]) == {2}
        cells = aggregate_cells(rows, "A")
        assert cells["zeroRL"][1]["n_ok"] == 2
        assert cells["zeroRL"][1]["throughput"]["n"] == 2
        # unsupported-only cell: n_ok == 0 but rows are counted
        b = aggregate_cells(rows, "B", point_key="num_workers")
        assert b["zeroRL"][2]["n_ok"] == 0
        assert b["zeroRL"][2]["n_total"] == 1

    def test_curve_environment(self) -> None:
        from benchmarks.scalability.aggregate import curve

        rows = [
            _mk_row("TorchRL", "A", num_envs=1, rep=0, measure_s=10.0),
            _mk_row("TorchRL", "A", num_envs=4, rep=0, measure_s=15.0),
        ]
        c = curve(rows, "A", "TorchRL", kind="environment")
        assert len(c) == 2
        assert c[0]["speedup"] == 1.0
        # speedup = t(1)/t(4) = 10/15 (mean times); efficiency = speedup / 4
        assert c[1]["speedup"] == pytest.approx(10.0 / 15.0)
        assert c[1]["environment_scaling_efficiency"] == pytest.approx(
            (10.0 / 15.0) / 4
        )

    def test_curve_worker_requires_baseline(self) -> None:
        from benchmarks.scalability.aggregate import curve

        rows = [
            _mk_row("SB3", "B", num_envs=8, num_workers=1, rep=0,
                    measure_s=10.0),
            _mk_row("SB3", "B", num_envs=8, num_workers=8, rep=0,
                    measure_s=20.0, mode="mp"),
        ]
        c = curve(rows, "B", "SB3", kind="worker", point_key="num_workers")
        assert [r["point"] for r in c] == [1, 8]
        assert c[1]["worker_parallel_efficiency"] == pytest.approx(
            (10.0 / 20.0) / 8
        )

    def test_return_stats_by_point(self) -> None:
        from benchmarks.scalability.aggregate import return_stats_by_point

        rows = [
            _mk_row("SB3", "C", num_envs=2, rep=0, return_last100=-100.0),
            _mk_row("SB3", "C", num_envs=2, rep=1, return_last100=-200.0),
            _mk_row("SB3", "C", num_envs=4, rep=0, status="failed",
                    measured=0, measure_s=0.01, exact=False,
                    error="RuntimeError: boom"),
        ]
        stats = return_stats_by_point(rows, "C")
        assert stats["SB3"][2]["n"] == 2
        assert stats["SB3"][2]["mean"] == pytest.approx(-150.0)
        assert 4 not in stats["SB3"]

    def test_integrity_summary(self) -> None:
        from benchmarks.scalability.aggregate import integrity_summary

        rows = [
            _mk_row("zeroRL", "A", rep=0),
            _mk_row("zeroRL", "A", rep=1, measured=99_999, exact=False),
            _mk_row("SB3", "B", num_workers=4, num_envs=8, status="unsupported",
                    measured=0, measure_s=0.01, exact=False, mode="mp",
                    error="unsupported: one process per env"),
            _mk_row("TorchRL", "C", rep=0, status="failed", measured=0,
                    measure_s=0.01, exact=False, protocol="ppo",
                    error="RuntimeError: boom"),
        ]
        summary = integrity_summary(rows)
        assert summary["n_rows"] == 4
        assert summary["counts"]["A"] == {"ok": 2}
        assert summary["counts"]["B"] == {"unsupported": 1}
        assert summary["counts"]["C"] == {"failed": 1}
        assert len(summary["inexact_budgets"]) == 1
        assert len(summary["failed"]) == 1
        assert len(summary["unsupported"]) == 1


class TestPlotsAndReport:
    """Render figures + REPORT.md from a synthetic results directory."""

    @staticmethod
    def _write_results(out_dir) -> None:
        from benchmarks.scalability.results_schema import append_row

        for exp in ("A", "B", "C", "D"):
            path = out_dir / f"exp{exp}.jsonl"
            if path.exists():
                path.unlink()
        for fw in ("zeroRL", "SB3", "TorchRL"):
            for n in (1, 2):
                for rep in range(2):
                    append_row(out_dir / "expA.jsonl",
                               _mk_row(fw, "A", num_envs=n, rep=rep,
                                       measure_s=100_000 / (1000 * n)))
            for w in ({ "zeroRL": [1], "SB3": [1, 8],
                        "TorchRL": [1, 2] }[fw]):
                append_row(out_dir / "expB.jsonl",
                           _mk_row(fw, "B", num_envs=8, num_workers=w, rep=0,
                                   mode="sync" if w == 1 else "mp",
                                   measure_s=100_000 / (1000 * w)))
            for exp in ("C", "D"):
                for n in (1, 2):
                    append_row(out_dir / f"exp{exp}.jsonl",
                               _mk_row(fw, exp, num_envs=n, rep=0,
                                       protocol="ppo", measure_s=200.0,
                                       return_last100=-100.0 - 10 * n))
        append_row(out_dir / "expB.jsonl",
                   _mk_row("zeroRL", "B", num_envs=8, num_workers=4, rep=0,
                           mode="mp", status="unsupported", measured=0,
                           measure_s=0.01, exact=False,
                           error="unsupported: multi-process not supported"))

    def test_generate_all_figures(self, tmp_path) -> None:
        pytest.importorskip("matplotlib")
        from benchmarks.scalability.plots import generate_all

        results = tmp_path / "results"
        results.mkdir()
        self._write_results(results)
        paths = generate_all(results, tmp_path / "figs")
        assert len(paths) == 10
        for path in paths:
            assert path.exists() and path.stat().st_size > 0

    def test_generate_report(self, tmp_path) -> None:
        from benchmarks.scalability.report import generate_report

        results = tmp_path / "results"
        results.mkdir()
        self._write_results(results)
        out = generate_report(results, tmp_path / "REPORT.md")
        text = out.read_text()
        for heading in ("# Scalability benchmark report",
                        "## Experiment A", "## Experiment B",
                        "## Experiment C", "## Experiment D",
                        "## Findings", "## Data integrity"):
            assert heading in text, heading
        assert "All ok rows recorded an exact transition budget." in text
        assert "unsupported: multi-process not supported" in text


class TestCLIErrorRecording:
    """P6: a failed CLI run must persist a full traceback in the row."""

    def test_failed_cli_row_stores_traceback(self, tmp_path, monkeypatch) -> None:
        import benchmarks.scalability.run as run_mod

        def boom(*_args, **_kwargs):
            raise RuntimeError("cli-boom-marker")

        monkeypatch.setattr(run_mod, "run_collection", boom)
        out = tmp_path / "failed.jsonl"
        rc = run_mod.main([
            "--experiment", "A", "--frameworks", "zeroRL",
            "--envs", "2", "--reps", "1",
            "--warmup", "8", "--measure", "8", "--out", str(out),
        ])
        assert rc == 1
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(rows) == 1
        row = rows[0]
        assert row["status"] == "failed"
        assert "cli-boom-marker" in row["error"]
        assert "Traceback" in row["error"]
