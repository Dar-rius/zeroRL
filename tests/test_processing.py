import pytest
import torch
from zerorl.processing import NormMeanStd, NormMinMax

@pytest.fixture(params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def device(request):
    return torch.device(request.param)

class TestNormMeanStd:
    @pytest.mark.gpu
    def test_initial_state(self, device) -> None:
        norm = NormMeanStd(shape=(4,), device=device)
        torch.testing.assert_close(norm.mean, torch.zeros(4, device=device))

    @pytest.mark.gpu
    def test_running_mean_converges(self, device) -> None:
        norm = NormMeanStd(shape=(2,), device=device)
        true_mean = torch.tensor([5.0, 10.0], device=device)
        for _ in range(1000):
            norm.update(true_mean + torch.randn(2, device=device) * 0.1)
        torch.testing.assert_close(norm.mean, true_mean, atol=0.1, rtol=0.1)

class TestNormMinMax:
    @pytest.mark.gpu
    def test_basic_normalization(self, device) -> None:
        low = torch.tensor([0.0, 0.0], device=device)
        high = torch.tensor([1.0, 1.0], device=device)
        norm = NormMinMax(low, high, device=device)
        x = torch.tensor([0.5, 0.5], device=device)
        result = norm.normalize(x)
        torch.testing.assert_close(result, torch.tensor([0.5, 0.5], device=device))
