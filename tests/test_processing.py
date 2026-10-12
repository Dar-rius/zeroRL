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


class TestNormMeanStdStateDict:
    @pytest.mark.gpu
    def test_state_dict_keys_and_clones(self, device) -> None:
        norm = NormMeanStd(shape=(3,), device=device)
        for _ in range(10):
            norm.update(torch.ones(3, device=device) * 2.0)
        sd = norm.state_dict()
        assert set(sd.keys()) == {"mean", "var", "count"}
        assert sd["count"].dtype == torch.float32
        # clones: later updates must not mutate the saved dict
        saved_mean = sd["mean"].clone()
        norm.update(torch.ones(3, device=device) * 100.0)
        torch.testing.assert_close(sd["mean"], saved_mean)

    @pytest.mark.gpu
    def test_load_state_dict_roundtrip(self, device) -> None:
        src = NormMeanStd(shape=(3,), device=device)
        for _ in range(30):
            src.update(torch.randn(3, device=device))
        dst = NormMeanStd(shape=(3,), device=device)
        dst.load_state_dict(src.state_dict())
        torch.testing.assert_close(dst.mean, src.mean)
        torch.testing.assert_close(dst.var, src.var)
        assert dst.count == pytest.approx(src.count)

    @pytest.mark.gpu
    def test_loaded_normalizer_normalizes_same_as_source(self, device) -> None:
        src = NormMeanStd(shape=(2,), device=device)
        for _ in range(50):
            src.update(torch.randn(2, device=device) + 5.0)
        dst = NormMeanStd(shape=(2,), device=device)
        dst.load_state_dict(src.state_dict())
        x = torch.randn(2, device=device)
        torch.testing.assert_close(dst.normalize(x), src.normalize(x))

    @pytest.mark.gpu
    def test_load_moves_tensors_to_destination_device(self) -> None:
        src = NormMeanStd(shape=(2,), device=torch.device("cpu"))
        for _ in range(5):
            src.update(torch.randn(2))
        dst = NormMeanStd(shape=(2,), device=torch.device("cpu"))
        dst.load_state_dict(src.state_dict())
        assert dst.mean.device.type == "cpu"
        assert dst.var.device.type == "cpu"
