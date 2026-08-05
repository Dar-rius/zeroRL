"""Unit tests for the Buffer class (zerorl.common).

Covers initialization, insertion, bounds checking, tensor conversion,
and buffer clearing/reuse.
"""

import pytest
import torch
from zerorl.common import Buffer

# =============================================================================
# Test Initialization
# =============================================================================


class TestBufferInit:
    """Verify Buffer.__init__() creates arrays with correct shapes and types."""

    def test_creates_arrays_with_correct_shapes(self) -> None:
        buf = Buffer(step=10, data={"state": (4,), "action": (2,)}, device=torch.device("cpu"))
        assert buf.data["state"].shape == (10, 4)
        assert buf.data["action"].shape == (10, 2)

    def test_creates_arrays_with_correct_dtypes(self) -> None:
        buf = Buffer(step=5, data={"state": (3,)}, device=torch.device("cpu"))
        assert buf.data["state"].dtype == torch.float32

    def test_slice_starts_at_zero(self) -> None:
        buf = Buffer(step=10, data={"state": (4,)}, device=torch.device("cpu"))
        assert buf.slice == 0

    def test_step_stored(self) -> None:
        buf = Buffer(step=42, data={"state": (2,)}, device=torch.device("cpu"))
        assert buf.step == 42

    def test_scalar_shape(self) -> None:
        buf = Buffer(step=5, data={"state": (8,), "action": ()}, device=torch.device("cpu"))
        assert buf.data["action"].shape == (5,)

    def test_multiple_keys(self) -> None:
        buf = Buffer(
            step=10,
            data={"state": (4,), "action": (), "reward": (), "done": ()},
            device=torch.device("cpu"),
        )
        assert "state" in buf.data
        assert "action" in buf.data
        assert "reward" in buf.data
        assert "done" in buf.data

    def test_default_device_is_cpu(self) -> None:
        buf = Buffer(step=5, data={"x": (3,)}, device=torch.device("cpu"))
        assert buf.data["x"].device == torch.device("cpu")


# =============================================================================
# Test Size Property
# =============================================================================


class TestBufferSize:
    """Verify the size property tracks the current element count."""

    def test_initially_zero(self) -> None:
        buf = Buffer(step=10, data={"state": (4,)}, device=torch.device("cpu"))
        assert buf.size == 0

    def test_increments_after_insert(self) -> None:
        buf = Buffer(step=10, data={"state": (4,)}, device=torch.device("cpu"))
        buf.insert(state=torch.zeros(4))
        assert buf.size == 1

    def test_increments_correctly_multi_insert(self) -> None:
        buf = Buffer(step=10, data={"state": (4,)}, device=torch.device("cpu"))
        for i in range(5):
            buf.insert(state=torch.ones(4) * i)
        assert buf.size == 5


# =============================================================================
# Test Insert Method
# =============================================================================


class TestBufferInsert:
    """Verify Buffer.insert() stores data at the correct index."""

    def test_single_insert_state(self) -> None:
        buf = Buffer(step=10, data={"state": (4,)}, device=torch.device("cpu"))
        state = torch.tensor([1.0, 2.0, 3.0, 4.0])
        buf.insert(state=state)
        torch.testing.assert_close(buf.data["state"][0], state)

    def test_single_insert_scalar(self) -> None:
        buf = Buffer(step=10, data={"action": ()}, device=torch.device("cpu"))
        buf.insert(action=torch.tensor(3.0))
        assert buf.data["action"][0].item() == 3.0

    def test_multiple_inserts_correct_indices(self) -> None:
        buf = Buffer(step=10, data={"state": (3,)}, device=torch.device("cpu"))
        for i in range(7):
            buf.insert(state=torch.full((3,), float(i)))
        torch.testing.assert_close(buf.data["state"][6], torch.full((3,), 6.0))
        assert buf.slice == 7

    def test_insert_partial_keys(self) -> None:
        """Inserting only a subset of keys should only update those keys."""
        buf = Buffer(step=10, data={"state": (4,), "reward": ()}, device=torch.device("cpu"))
        buf.insert(state=torch.ones(4))
        assert buf.size == 1
        torch.testing.assert_close(buf.data["state"][0], torch.ones(4))
        # reward at index 0 should remain zeros (default)
        assert buf.data["reward"][0].item() == 0.0

    def test_insert_unknown_key_raises(self) -> None:
        buf = Buffer(step=10, data={"state": (4,)}, device=torch.device("cpu"))
        with pytest.raises(ValueError):
            buf.insert(state=torch.zeros(4), unknown_key=torch.tensor(1.0))


# =============================================================================
# Test Buffer Full (Bounds Checking)
# =============================================================================


class TestBufferInsertFull:
    """Verify that inserting into a full buffer raises ValueError."""

    def test_raises_valueerror_when_full(self) -> None:
        buf = Buffer(step=3, data={"state": (2,)}, device=torch.device("cpu"))
        for _ in range(3):
            buf.insert(state=torch.zeros(2))
        with pytest.raises(ValueError, match="Buffer is full"):
            buf.insert(state=torch.zeros(2))


# =============================================================================
# Test Get All (Tensor Conversion)
# =============================================================================


class TestBufferGetAll:
    """Verify Buffer.get_all() returns a dict of sliced tensors."""

    def test_returns_dict(self) -> None:
        buf = Buffer(step=5, data={"state": (4,)}, device=torch.device("cpu"))
        result = buf.get_all()
        assert isinstance(result, dict)

    def test_dict_keys_match(self) -> None:
        buf = Buffer(step=5, data={"state": (4,), "action": ()}, device=torch.device("cpu"))
        result = buf.get_all()
        assert set(result.keys()) == {"state", "action"}

    def test_tensor_shapes_after_inserts(self) -> None:
        buf = Buffer(step=5, data={"state": (4,), "action": ()}, device=torch.device("cpu"))
        for i in range(5):
            buf.insert(state=torch.randn(4), action=torch.tensor(float(i)))
        result = buf.get_all()
        assert result["state"].shape == (5, 4)
        assert result["action"].shape == (5,)

    def test_tensor_shapes_empty_buffer(self) -> None:
        buf = Buffer(step=5, data={"state": (4,), "action": ()}, device=torch.device("cpu"))
        result = buf.get_all()
        assert result["state"].shape == (0, 4)
        assert result["action"].shape == (0,)

    def test_tensor_dtypes(self) -> None:
        buf = Buffer(step=5, data={"state": (4,)}, device=torch.device("cpu"))
        result = buf.get_all()
        assert result["state"].dtype == torch.float32

    def test_get_all_reflects_inserted_data(self) -> None:
        buf = Buffer(step=3, data={"state": (2,), "reward": ()}, device=torch.device("cpu"))
        buf.insert(state=torch.tensor([1.0, 2.0]), reward=torch.tensor(0.5))
        result = buf.get_all()
        torch.testing.assert_close(result["state"][0], torch.tensor([1.0, 2.0]))
        assert result["reward"][0].item() == pytest.approx(0.5)

    def test_get_all_empty_buffer(self) -> None:
        buf = Buffer(step=5, data={"state": (4,)}, device=torch.device("cpu"))
        result = buf.get_all()
        assert result["state"].shape == (0, 4)

    def test_get_all_sliced_to_size(self) -> None:
        buf = Buffer(step=10, data={"state": (2,)}, device=torch.device("cpu"))
        for i in range(3):
            buf.insert(state=torch.ones(2) * i)
        result = buf.get_all()
        assert result["state"].shape == (3, 2)


# =============================================================================
# Test Clear (Buffer Reuse)
# =============================================================================


class TestBufferClear:
    """Verify clear() resets the buffer for reuse."""

    def test_resets_slice_to_zero(self) -> None:
        buf = Buffer(step=10, data={"state": (4,)}, device=torch.device("cpu"))
        for _ in range(5):
            buf.insert(state=torch.ones(4))
        buf.clear()
        assert buf.slice == 0
        assert buf.size == 0

    def test_allows_reuse_after_clear(self) -> None:
        buf = Buffer(step=3, data={"state": (2,)}, device=torch.device("cpu"))
        buf.insert(state=torch.tensor([1.0, 1.0]))
        buf.insert(state=torch.tensor([2.0, 2.0]))
        buf.clear()
        buf.insert(state=torch.tensor([9.0, 9.0]))
        assert buf.size == 1
        torch.testing.assert_close(buf.data["state"][0], torch.tensor([9.0, 9.0]))
