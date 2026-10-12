"""Unit tests for zerorl.logger (create_logger, profile, PhaseProfiler)."""

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from zerorl.config import AlgoConfig, TrainConfig
from zerorl.logger import (
    PhaseMetrics,
    PhaseProfiler,
    ProfileMetrics,
    create_logger,
    profile,
)


def _cfg(**overrides) -> TrainConfig:
    cfg = TrainConfig(model_name="logtest", model_save_path="/tmp", project_name="test")
    cfg.rollout_steps = overrides.get("rollout_steps", 8)
    cfg.num_envs = overrides.get("num_envs", 2)
    return cfg


# ---------------------------------------------------------------------------
# create_logger — real, unpatched closure
# ---------------------------------------------------------------------------

class TestCreateLogger:
    def test_returns_callable_with_close(self) -> None:
        log = create_logger(_cfg(), AlgoConfig())
        assert callable(log)
        assert hasattr(log, "close")
        log.close()

    def test_logs_scalar_tensor_and_float_without_error(self) -> None:
        log = create_logger(_cfg(), AlgoConfig())
        log({"loss": torch.tensor(1.5), "lr": 0.001}, step=0)
        log.close()

    def test_non_scalar_tensor_metric_raises_value_error(self) -> None:
        log = create_logger(_cfg(), AlgoConfig())
        with pytest.raises(ValueError, match="must be scalar"):
            log({"loss": torch.zeros(4)}, step=0)
        log.close()

    def test_missing_wandb_raises_import_error(self) -> None:
        with patch("zerorl.logger.wandb", None):
            with pytest.raises(ImportError, match="Wandb"):
                create_logger(_cfg(), AlgoConfig(), use_wandb=True)

    def test_missing_tensorboard_raises_import_error(self) -> None:
        with patch("zerorl.logger.SummaryWriter", None):
            with pytest.raises(ImportError, match="TensorBoard"):
                create_logger(_cfg(), AlgoConfig(), use_tb=True)

    def test_wandb_backend_receives_cleaned_metrics(self) -> None:
        mock_wandb = MagicMock()
        with patch("zerorl.logger.wandb", mock_wandb):
            log = create_logger(_cfg(), AlgoConfig(), use_wandb=True)
            mock_wandb.init.assert_called_once()
            assert mock_wandb.init.call_args.kwargs["project"] == "test"
            log({"loss": torch.tensor(2.0), "reward": 3}, step=5)
            mock_wandb.log.assert_called_once_with(
                {"loss": 2.0, "reward": 3.0}, step=5
            )
            log.close()
            mock_wandb.finish.assert_called_once()

    def test_tb_backend_receives_cleaned_metrics(self) -> None:
        mock_writer = MagicMock()
        with patch("zerorl.logger.SummaryWriter", mock_writer):
            log = create_logger(_cfg(), AlgoConfig(), use_tb=True)
            mock_writer.assert_called_once()
            log({"loss": torch.tensor(2.0)}, step=7)
            mock_writer.return_value.add_scalar.assert_called_once_with(
                "loss", 2.0, 7
            )
            log.close()
            mock_writer.return_value.close.assert_called_once()


# ---------------------------------------------------------------------------
# profile decorator + ProfileMetrics
# ---------------------------------------------------------------------------

class TestProfileDecorator:
    def test_returns_result_and_profile_metrics(self) -> None:
        @profile("work")
        def work(x: int) -> int:
            return x + 1

        result, pm = work(1)
        assert result == 2
        assert isinstance(pm, ProfileMetrics)
        assert pm.name == "work"
        assert pm.duration_ms >= 0.0
        assert pm.vram_peak_gb == 0.0

    def test_duration_reflects_work(self) -> None:
        @profile("sleepy")
        def sleepy() -> None:
            time.sleep(0.01)

        _, pm = sleepy()
        assert pm.duration_ms >= 9.0

    def test_cuda_branch_reads_peak_memory(self, monkeypatch) -> None:
        sync = MagicMock()
        reset = MagicMock()
        monkeypatch.setattr(torch.cuda, "synchronize", sync)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated",
                            lambda: 2 * 1024**3)

        @profile("gpu_work", is_cuda=True)
        def gpu_work() -> int:
            return 42

        result, pm = gpu_work()
        assert result == 42
        assert sync.call_count == 2
        reset.assert_called_once()
        assert pm.vram_peak_gb == pytest.approx(2.0)

    def test_preserves_function_metadata(self) -> None:
        @profile("named")
        def documented() -> None:
            """docstring here"""

        assert documented.__name__ == "documented"
        assert documented.__doc__ == "docstring here"


# ---------------------------------------------------------------------------
# PhaseProfiler
# ---------------------------------------------------------------------------

class TestPhaseProfiler:
    def test_init_fields(self) -> None:
        prof = PhaseProfiler(_cfg(rollout_steps=8, num_envs=2))
        assert prof.is_cuda is False
        assert prof.total_rollout == 16
        assert isinstance(prof.metrics, PhaseMetrics)
        assert prof.metrics.total_ms == 0.0

    def test_start_phase_resets_metrics(self) -> None:
        prof = PhaseProfiler(_cfg())
        prof.metrics.rollout_ms = 99.0
        prof.start_phase()
        assert prof.metrics.rollout_ms == 0.0

    def test_track_sets_phase_duration(self) -> None:
        prof = PhaseProfiler(_cfg())
        prof.start_phase()
        with prof.track("rollout"):
            time.sleep(0.01)
        with prof.track("update"):
            pass
        assert prof.metrics.rollout_ms >= 9.0
        assert prof.metrics.update_ms >= 0.0

    def test_track_sets_duration_even_on_exception(self) -> None:
        prof = PhaseProfiler(_cfg())
        prof.start_phase()
        with pytest.raises(RuntimeError, match="boom"):
            with prof.track("rollout"):
                raise RuntimeError("boom")
        assert prof.metrics.rollout_ms >= 0.0

    def test_end_phase_returns_filled_metrics(self) -> None:
        prof = PhaseProfiler(_cfg(rollout_steps=8, num_envs=2))
        prof.start_phase()
        with prof.track("rollout"):
            time.sleep(0.01)
        with prof.track("update"):
            pass
        pm = prof.end_phase()
        assert pm is prof.metrics
        assert pm.total_ms >= pm.rollout_ms
        assert pm.ram_mb > 0.0
        assert pm.fps > 0.0

    def test_fps_zero_without_rollout(self) -> None:
        prof = PhaseProfiler(_cfg())
        prof.start_phase()
        pm = prof.end_phase()
        assert pm.rollout_ms == 0.0
        assert pm.fps == 0.0

    def test_fps_uses_total_rollout(self) -> None:
        prof = PhaseProfiler(_cfg(rollout_steps=4, num_envs=2))
        prof.start_phase()
        with prof.track("rollout"):
            time.sleep(0.01)
        pm = prof.end_phase()
        expected = prof.total_rollout / (pm.total_ms / 1000)
        assert pm.fps == pytest.approx(expected)

    def test_cuda_branch_reads_gpu_stats(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "synchronize", MagicMock())
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", MagicMock())
        monkeypatch.setattr(torch.cuda, "memory_allocated",
                            lambda: 1024**3)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated",
                            lambda: 2 * 1024**3)
        prof = PhaseProfiler(_cfg(), is_cuda=True)
        prof.start_phase()
        with prof.track("rollout"):
            pass
        pm = prof.end_phase()
        assert pm.vram_allocated_gb == pytest.approx(1.0)
        assert pm.vram_peak_gb == pytest.approx(2.0)
