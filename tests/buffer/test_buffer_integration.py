"""Integration tests for Buffer (zerorl.buffer)."""

import pytest
import torch
from zerorl.buffer import Buffer

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _req(device, **extra):
    kw = {
        "state": torch.zeros(4, device=device),
        "action": torch.tensor(0.0, device=device),
        "reward": torch.tensor(0.0, device=device),
        "terminated": torch.tensor(0.0, device=device),
        "truncated": torch.tensor(0.0, device=device),
    }
    kw.update(extra)
    return kw

class TestBufferIntegration:
    @pytest.mark.gpu
    def test_full_insert_cycle_with_get_all(self, device) -> None:
        schema = {"state": (8,), "action": (4,), "reward": (), "terminated": (), "truncated": ()}
        buf = Buffer(capacity=20, num_envs=1, schema=schema, device=device)
        for _ in range(20):
            buf.insert(
                state=torch.randn(8, device=device),
                action=torch.randn(4, device=device),
                reward=torch.tensor(0.0, device=device),
                terminated=torch.tensor(0.0, device=device),
                truncated=torch.tensor(0.0, device=device),
            )
        result = buf.get_all()
        assert result["state"].shape == (20, 1, 8)

    @pytest.mark.gpu
    def test_buffer_scalar_actions(self, device) -> None:
        schema = {"state": (4,), "action": (), "reward": (), "terminated": (), "truncated": ()}
        buf = Buffer(capacity=10, num_envs=1, schema=schema, device=device)
        for i in range(10):
            buf.insert(
                state=torch.ones(4, device=device),
                action=torch.tensor(float(i), device=device),
                reward=torch.tensor(0.0, device=device),
                terminated=torch.tensor(0.0, device=device),
                truncated=torch.tensor(0.0, device=device),
            )
        result = buf.get_all()
        assert result["action"].shape == (10, 1)

    @pytest.mark.gpu
    def test_clear_and_refill(self, device) -> None:
        schema = {"state": (2,), "action": (), "reward": (), "terminated": (), "truncated": ()}
        buf = Buffer(capacity=5, num_envs=1, schema=schema, device=device)
        for i in range(5):
            buf.insert(**_req(device, state=torch.tensor([float(i), float(i)], device=device)))
        buf.clear()
        assert buf.size == 0
        for i in range(3):
            buf.insert(**_req(device, state=torch.tensor([float(i + 10), float(i + 10)], device=device)))
        assert buf.size == 3

    @pytest.mark.gpu
    def test_large_buffer(self, device) -> None:
        n = 2048
        schema = {"state": (4,), "action": (), "reward": (), "terminated": (), "truncated": ()}
        buf = Buffer(capacity=n, num_envs=1, schema=schema, device=device)
        for _ in range(n):
            buf.insert(**_req(device))
        assert buf.size == n
        assert buf.get_all()["state"].shape == (n, 1, 4)
