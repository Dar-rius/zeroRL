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
        buf = Buffer(step=10, data={"state": (4,)}, device=device)
        assert buf.data["state"].shape == (10, 1, 4)

    @pytest.mark.gpu
    def test_default_device(self, device) -> None:
        buf = Buffer(step=5, data={"x": (3,)}, device=device)
        assert buf.data["x"].device.type == device.type

class TestBufferInsert:
    @pytest.mark.gpu
    def test_single_insert_state(self, device) -> None:
        buf = Buffer(step=10, data={"state": (4,)}, device=device)
        state = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device).unsqueeze(0)
        buf.insert(state=state)
        torch.testing.assert_close(buf.data["state"][0], state)

    @pytest.mark.gpu
    def test_raises_valueerror_when_full(self, device) -> None:
        buf = Buffer(step=3, data={"state": (2,)}, device=device)
        for _ in range(3):
            buf.insert(state=torch.zeros(2, device=device))
        with pytest.raises(ValueError):
            buf.insert(state=torch.zeros(2, device=device))

class TestBufferGetAll:
    @pytest.mark.gpu
    def test_tensor_shapes_after_inserts(self, device) -> None:
        buf = Buffer(step=5, data={"state": (4,)}, device=device)
        for _ in range(5):
            buf.insert(state=torch.randn(4, device=device))
        assert buf.get_all()["state"].shape == (5, 1, 4)
