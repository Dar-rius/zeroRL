"""Integration tests for Buffer (zerorl.common)."""

import pytest
import torch
from zerorl.buffer import Buffer

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestBufferIntegration:
    @pytest.mark.gpu
    def test_full_insert_cycle_with_get_all(self, device) -> None:
        buf = Buffer(step=20, data={"state": (8,), "action": (4,)}, device=device)
        for _ in range(20):
            buf.insert(state=torch.randn(8, device=device), action=torch.randn(4, device=device))
        result = buf.get_all()
        assert result["state"].shape == (20, 1, 8)

    @pytest.mark.gpu
    def test_buffer_scalar_actions(self, device) -> None:
        buf = Buffer(step=10, data={"state": (4,), "action": ()}, device=device)
        for i in range(10):
            buf.insert(state=torch.ones(4, device=device), action=torch.tensor(float(i), device=device))
        result = buf.get_all()
        assert result["action"].shape == (10, 1)

    @pytest.mark.gpu
    def test_clear_and_refill(self, device) -> None:
        buf = Buffer(step=5, data={"state": (2,)}, device=device)
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
        buf = Buffer(step=n, data={"state": (4,), "action": ()}, device=device)
        for _ in range(n):
            buf.insert(state=torch.randn(4, device=device), action=torch.tensor(0, device=device))
        assert buf.size == n
        assert buf.get_all()["state"].shape == (n, 1, 4)
