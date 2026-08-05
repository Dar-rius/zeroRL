"""Unit tests for processing utilities (zerorl.processing).

Tests NormMeanStd (running mean/std normalizer) and NormMinMax (min-max normalizer).
"""

import torch
from zerorl.processing import NormMeanStd, NormMinMax


# =============================================================================
# Test NormMeanStd
# =============================================================================


class TestNormMeanStd:
    """Verify NormMeanStd running statistics and normalization."""

    def test_initial_state(self) -> None:
        norm = NormMeanStd(shape=(4,), device=torch.device("cpu"))
        torch.testing.assert_close(norm.mean, torch.zeros(4))
        torch.testing.assert_close(norm.var, torch.ones(4))
        assert norm.count == 1e-4

    def test_update_single_sample(self) -> None:
        norm = NormMeanStd(shape=(4,), device=torch.device("cpu"))
        x = torch.ones(4)
        norm.update(x)
        assert norm.count > 1e-4

    def test_update_batch(self) -> None:
        norm = NormMeanStd(shape=(3,), device=torch.device("cpu"))
        batch = torch.tensor([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])
        norm.update(batch)
        assert norm.count > 1e-4

    def test_normalize_after_update(self) -> None:
        norm = NormMeanStd(shape=(4,), device=torch.device("cpu"))
        # Update with known data
        for _ in range(100):
            norm.update(torch.randn(4))
        x = torch.randn(4)
        normalized = norm.normalize(x)
        assert normalized.shape == (4,)
        assert torch.isfinite(normalized).all()

    def test_running_mean_converges(self) -> None:
        norm = NormMeanStd(shape=(2,), device=torch.device("cpu"))
        true_mean = torch.tensor([5.0, 10.0])
        for _ in range(1000):
            norm.update(true_mean + torch.randn(2) * 0.1)
        torch.testing.assert_close(norm.mean, true_mean, atol=0.1, rtol=0.1)

    def test_running_var_converges(self) -> None:
        norm = NormMeanStd(shape=(2,), device=torch.device("cpu"))
        true_std = torch.tensor([1.0, 2.0])
        for _ in range(1000):
            norm.update(true_std * torch.randn(2))
        expected_var = true_std ** 2
        torch.testing.assert_close(norm.var, expected_var, atol=0.3, rtol=0.3)

    def test_unbiased_after_many_updates(self) -> None:
        norm = NormMeanStd(shape=(3,), device=torch.device("cpu"))
        for _ in range(500):
            norm.update(torch.randn(3))
        # After many updates, mean should be close to 0
        torch.testing.assert_close(norm.mean, torch.zeros(3), atol=0.2, rtol=0.2)


# =============================================================================
# Test NormMinMax
# =============================================================================


class TestNormMinMax:
    """Verify NormMinMax normalization and scale computation."""

    def test_basic_normalization(self) -> None:
        low = torch.tensor([0.0, 0.0])
        high = torch.tensor([1.0, 1.0])
        norm = NormMinMax(low, high, device="cpu")
        x = torch.tensor([0.5, 0.5])
        result = norm.normalize(x)
        torch.testing.assert_close(result, torch.tensor([0.5, 0.5]))

    def test_maps_to_unit_range(self) -> None:
        low = torch.tensor([-1.0, -2.0])
        high = torch.tensor([1.0, 2.0])
        norm = NormMinMax(low, high, device="cpu")
        # Low should map to 0
        result_low = norm.normalize(low)
        torch.testing.assert_close(result_low, torch.zeros(2), atol=1e-5, rtol=1e-5)
        # High should map to ~1
        result_high = norm.normalize(high)
        torch.testing.assert_close(result_high, torch.ones(2), atol=1e-5, rtol=1e-5)

    def test_midpoint(self) -> None:
        low = torch.tensor([0.0])
        high = torch.tensor([10.0])
        norm = NormMinMax(low, high, device="cpu")
        x = torch.tensor([5.0])
        result = norm.normalize(x)
        torch.testing.assert_close(result, torch.tensor([0.5]))

    def test_scale_computation(self) -> None:
        low = torch.tensor([0.0, 0.0])
        high = torch.tensor([2.0, 4.0])
        norm = NormMinMax(low, high, device="cpu")
        expected_scale = 1.0 / (high - low + 1e-8)
        torch.testing.assert_close(norm.scale, expected_scale, atol=1e-6, rtol=1e-6)

    def test_negative_range(self) -> None:
        low = torch.tensor([-10.0])
        high = torch.tensor([-5.0])
        norm = NormMinMax(low, high, device="cpu")
        x = torch.tensor([-7.5])
        result = norm.normalize(x)
        torch.testing.assert_close(result, torch.tensor([0.5]))

    def test_batch_normalization(self) -> None:
        low = torch.tensor([0.0, 0.0])
        high = torch.tensor([1.0, 1.0])
        norm = NormMinMax(low, high, device="cpu")
        batch = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        result = norm.normalize(batch)
        expected = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
