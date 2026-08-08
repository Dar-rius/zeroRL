"""Unit tests for the Buffer class (zerorl.common)."""

import pytest
import torch
from zerorl.common import Buffer
from zerorl.errors import KeyBufferError

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


class TestBufferInsertEdgeCases:
    @pytest.mark.gpu
    def test_insert_missing_key_zero_fills(self, device) -> None:
        buf = Buffer(step=3, data={"a": (), "b": ()}, device=device)
        buf.insert(a=torch.tensor(1.0, device=device))
        assert buf.size == 1
        torch.testing.assert_close(buf.data["a"][0], torch.tensor([1.0], device=device))
        torch.testing.assert_close(buf.data["b"][0], torch.tensor([0.0], device=device))


class TestBufferKeyError:
    @pytest.mark.gpu
    def test_insert_unknown_key_raises_key_buffer_error(self, device) -> None:
        buf = Buffer(step=3, data={"state": (4,)}, device=device)
        with pytest.raises(KeyBufferError):
            buf.insert(foo=torch.zeros(4, device=device))

    @pytest.mark.gpu
    def test_key_buffer_error_has_correct_arg_name(self, device) -> None:
        buf = Buffer(step=3, data={"state": (4,)}, device=device)
        with pytest.raises(KeyBufferError) as exc_info:
            buf.insert(missing=torch.zeros(4, device=device))
        assert exc_info.value.arg_name == "missing"

    @pytest.mark.gpu
    def test_slice_not_incremented_on_error(self, device) -> None:
        buf = Buffer(step=3, data={"state": (4,)}, device=device)
        with pytest.raises(KeyBufferError):
            buf.insert(foo=torch.zeros(4, device=device))
        assert buf.size == 0

    @pytest.mark.gpu
    def test_mix_known_unknown_key_raises(self, device) -> None:
        buf = Buffer(step=3, data={"state": (4,), "action": ()}, device=device)
        with pytest.raises(KeyBufferError):
            buf.insert(state=torch.zeros(4, device=device),
                       bogus=torch.zeros(4, device=device))
        assert buf.size == 0
