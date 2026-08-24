"""Integration tests for Buffer (zerorl.buffer)."""

import pytest
import torch
from zerorl.buffer import Buffer
from zerorl.config import TrainConfig

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _make_config(step: int, num_envs: int = 1, device: torch.device = torch.device("cpu")) -> TrainConfig:
    cfg = TrainConfig(model_name="test", model_save_path="/tmp", project_name="test")
    cfg.rollout_steps = step
    cfg.num_envs = num_envs
    cfg.device = device
    return cfg

class TestBufferIntegration:
    @pytest.mark.gpu
    def test_full_insert_cycle_with_get_all(self, device) -> None:
        buf = Buffer(data={"state": (8,), "action": (4,)}, config=_make_config(20, device=device))
        for _ in range(20):
            buf.insert(state=torch.randn(8, device=device), action=torch.randn(4, device=device))
        result = buf.get_all()
        assert result["state"].shape == (20, 1, 8)

    @pytest.mark.gpu
    def test_buffer_scalar_actions(self, device) -> None:
        buf = Buffer(data={"state": (4,), "action": ()}, config=_make_config(10, device=device))
        for i in range(10):
            buf.insert(state=torch.ones(4, device=device), action=torch.tensor(float(i), device=device))
        result = buf.get_all()
        assert result["action"].shape == (10, 1)

    @pytest.mark.gpu
    def test_clear_and_refill(self, device) -> None:
        buf = Buffer(data={"state": (2,)}, config=_make_config(5, device=device))
        for i in range(5):
            buf.insert(state=torch.tensor([float(i), float(i)], device=device))
        buf.clear()
        assert buf.size == 0
        for i in range(3):
            buf.insert(state=torch.tensor([float(i + 10), float(i + 10)], device=device))
        assert buf.size == 3

    @pytest.mark.gpu
    def test_large_buffer(self, device) -> None:
        n = 2048
        buf = Buffer(data={"state": (4,), "action": ()}, config=_make_config(n, device=device))
        for _ in range(n):
            buf.insert(state=torch.randn(4, device=device), action=torch.tensor(0, device=device))
        assert buf.size == n
        assert buf.get_all()["state"].shape == (n, 1, 4)
