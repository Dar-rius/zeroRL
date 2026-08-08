"""Unit tests for the Buffer class (zerorl.common)."""

import pytest
import torch
from zerorl.common import Buffer

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestBufferInit:
    @pytest.mark.gpu
    def test_creates_arrays_with_correct_shapes(self, device) -> None:
        buf = Buffer(step=10, data={"states": (4,)}, device=device)
        assert buf.data["states"].shape == (10, 1, 4)

    @pytest.mark.gpu
    def test_default_device(self, device) -> None:
        buf = Buffer(step=5, data={"x": (3,)}, device=device)
        assert buf.data["x"].device.type == device.type

class TestBufferInsert:
    @pytest.mark.gpu
    def test_single_insert_state(self, device) -> None:
        buf = Buffer(step=10, data={"states": (4,)}, device=device)
        states = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device).unsqueeze(0)
        buf.insert(states=states)
        torch.testing.assert_close(buf.data["states"][0], states)

    @pytest.mark.gpu
    def test_raises_valueerror_when_full(self, device) -> None:
        buf = Buffer(step=3, data={"states": (2,)}, device=device)
        for _ in range(3):
            buf.insert(states=torch.zeros(2, device=device))
        with pytest.raises(ValueError):
            buf.insert(states=torch.zeros(2, device=device))

class TestBufferGetAll:
    @pytest.mark.gpu
    def test_tensor_shapes_after_inserts(self, device) -> None:
        buf = Buffer(step=5, data={"states": (4,)}, device=device)
        for _ in range(5):
            buf.insert(states=torch.randn(4, device=device))
        assert buf.get_all()["states"].shape == (5, 1, 4)
