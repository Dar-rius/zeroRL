"""Unit tests for the Buffer class (zerorl.buffer)."""

import pytest
import torch
from zerorl.buffer import Buffer
from zerorl.errors import KeyBufferError

_REQUIRED = {"state", "action", "reward", "terminated", "truncated"}

def _schema(extra: dict[str, tuple] | None = None) -> dict[str, tuple]:
    """Schema that always includes the Buffer.required_keys."""
    base = {"state": (4,), "action": (), "reward": (), "terminated": (), "truncated": ()}
    if extra:
        base.update(extra)
    return base

def _required_kwargs(device: torch.device, **extra) -> dict[str, torch.Tensor]:
    """kwargs covering every required_key with scalar-compatible tensors."""
    kw = {
        "state": torch.zeros(4, device=device),
        "action": torch.tensor(0.0, device=device),
        "reward": torch.tensor(0.0, device=device),
        "terminated": torch.tensor(0.0, device=device),
        "truncated": torch.tensor(0.0, device=device),
    }
    kw.update(extra)
    return kw

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestBufferInit:
    @pytest.mark.gpu
    def test_creates_arrays_with_correct_shapes(self, device) -> None:
        buf = Buffer(capacity=10, num_envs=1, schema=_schema(), device=device)
        assert buf.data["state"].shape == (10, 1, 4)

    @pytest.mark.gpu
    def test_default_device(self, device) -> None:
        buf = Buffer(capacity=5, num_envs=1, schema=_schema({"x": (3,)}), device=device)
        assert buf.data["x"].device.type == device.type

class TestBufferInsert:
    @pytest.mark.gpu
    def test_single_insert_state(self, device) -> None:
        buf = Buffer(capacity=10, num_envs=1, schema=_schema(), device=device)
        state = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device).unsqueeze(0)
        buf.insert(**_required_kwargs(device, state=state))
        torch.testing.assert_close(buf.data["state"][0], state)

    @pytest.mark.gpu
    def test_raises_valueerror_when_full(self, device) -> None:
        buf = Buffer(capacity=3, num_envs=1, schema=_schema(), device=device)
        for _ in range(3):
            buf.insert(**_required_kwargs(device))
        with pytest.raises(ValueError):
            buf.insert(**_required_kwargs(device))

class TestBufferGetAll:
    @pytest.mark.gpu
    def test_tensor_shapes_after_inserts(self, device) -> None:
        buf = Buffer(capacity=5, num_envs=1, schema=_schema(), device=device)
        for _ in range(5):
            buf.insert(**_required_kwargs(device))
        assert buf.get_all()["state"].shape == (5, 1, 4)


class TestBufferInsertEdgeCases:
    @pytest.mark.gpu
    def test_insert_missing_key_zero_fills(self, device) -> None:
        buf = Buffer(capacity=3, num_envs=1, schema=_schema({"a": (), "b": ()}), device=device)
        # Pass only required keys; a and b should zero-fill.
        buf.insert(**_required_kwargs(device))
        assert buf.size == 1
        torch.testing.assert_close(buf.data["a"][0], torch.tensor([0.0], device=device))
        torch.testing.assert_close(buf.data["b"][0], torch.tensor([0.0], device=device))

    @pytest.mark.gpu
    def test_insert_missing_required_key_raises(self, device) -> None:
        buf = Buffer(capacity=3, num_envs=1, schema=_schema(), device=device)
        with pytest.raises(ValueError, match="Missing required keys"):
            buf.insert(state=torch.zeros(4, device=device))


class TestBufferKeyError:
    @pytest.mark.gpu
    def test_insert_unknown_key_raises_key_buffer_error(self, device) -> None:
        buf = Buffer(capacity=3, num_envs=1, schema=_schema(), device=device)
        with pytest.raises(KeyBufferError):
            buf.insert(**_required_kwargs(device, foo=torch.zeros(4, device=device)))

    @pytest.mark.gpu
    def test_key_buffer_error_has_correct_arg_name(self, device) -> None:
        buf = Buffer(capacity=3, num_envs=1, schema=_schema(), device=device)
        with pytest.raises(KeyBufferError) as exc_info:
            buf.insert(**_required_kwargs(device, missing=torch.zeros(4, device=device)))
        assert exc_info.value.arg_name == "missing"

    @pytest.mark.gpu
    def test_slice_not_incremented_on_error(self, device) -> None:
        buf = Buffer(capacity=3, num_envs=1, schema=_schema(), device=device)
        with pytest.raises(KeyBufferError):
            buf.insert(**_required_kwargs(device, foo=torch.zeros(4, device=device)))
        assert buf.size == 0

    @pytest.mark.gpu
    def test_mix_known_unknown_key_raises(self, device) -> None:
        buf = Buffer(capacity=3, num_envs=1, schema=_schema(), device=device)
        with pytest.raises(KeyBufferError):
            buf.insert(**_required_kwargs(device, bogus=torch.zeros(4, device=device)))
        assert buf.size == 0


class TestBufferGetAllReshape:
    @pytest.mark.gpu
    def test_reshape_flattens_leading_dims(self, device) -> None:
        buf = Buffer(capacity=5, num_envs=1, schema=_schema(), device=device)
        for _ in range(5):
            buf.insert(**_required_kwargs(device))
        result = buf.get_all(reshape=True)
        assert result["state"].shape == (5, 4)

    @pytest.mark.gpu
    def test_reshape_slices_before_reshape(self, device) -> None:
        buf = Buffer(capacity=5, num_envs=1, schema=_schema(), device=device)
        for _ in range(3):
            buf.insert(**_required_kwargs(device))
        result = buf.get_all(reshape=True)
        assert result["state"].shape == (3, 4)

    @pytest.mark.gpu
    def test_reshape_multi_env(self, device) -> None:
        buf = Buffer(capacity=3, num_envs=4, schema=_schema(), device=device)
        for _ in range(3):
            buf.insert(**_required_kwargs(device))
        result = buf.get_all(reshape=True)
        assert result["state"].shape == (12, 4)

    @pytest.mark.gpu
    def test_reshape_scalar_fields(self, device) -> None:
        buf = Buffer(capacity=3, num_envs=2, schema=_schema(), device=device)
        for _ in range(3):
            buf.insert(
                state=torch.zeros(4, device=device),
                action=torch.zeros(2, device=device),
                reward=torch.tensor([1.0, 2.0], device=device),
                terminated=torch.tensor([0.0, 0.0], device=device),
                truncated=torch.tensor([0.0, 0.0], device=device),
            )
        result = buf.get_all(reshape=True)
        assert result["reward"].shape == (6,)

    @pytest.mark.gpu
    def test_no_reshape_preserves_3d(self, device) -> None:
        buf = Buffer(capacity=3, num_envs=2, schema=_schema(), device=device)
        for _ in range(3):
            buf.insert(**_required_kwargs(device))
        result = buf.get_all(reshape=False)
        assert result["state"].shape == (3, 2, 4)


class TestBufferMultiEnv:
    @pytest.mark.gpu
    def test_multi_env_shapes(self, device) -> None:
        buf = Buffer(capacity=5, num_envs=4, schema=_schema(), device=device)
        assert buf.data["state"].shape == (5, 4, 4)

    @pytest.mark.gpu
    def test_multi_env_insert_and_retrieve(self, device) -> None:
        buf = Buffer(capacity=3, num_envs=2, schema=_schema(), device=device)
        for _ in range(3):
            buf.insert(**_required_kwargs(device))
        result = buf.get_all()
        assert result["state"].shape == (3, 2, 4)
        assert buf.size == 3


class TestBufferClear:
    @pytest.mark.gpu
    def test_clear_resets_size_to_zero(self, device) -> None:
        buf = Buffer(capacity=5, num_envs=1, schema=_schema(), device=device)
        for _ in range(5):
            buf.insert(**_required_kwargs(device))
        buf.clear()
        assert buf.size == 0

    @pytest.mark.gpu
    def test_get_all_after_clear_returns_empty(self, device) -> None:
        buf = Buffer(capacity=5, num_envs=1, schema=_schema(), device=device)
        for _ in range(5):
            buf.insert(**_required_kwargs(device))
        buf.clear()
        result = buf.get_all()
        assert result["state"].shape[0] == 0

    @pytest.mark.gpu
    def test_clear_does_not_zero_memory(self, device) -> None:
        buf = Buffer(capacity=3, num_envs=1, schema=_schema(), device=device)
        buf.insert(**_required_kwargs(device, state=torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)))
        buf.clear()
        buf.insert(**_required_kwargs(device, state=torch.tensor([10.0, 20.0, 30.0, 40.0], device=device)))
        result = buf.get_all()
        torch.testing.assert_close(
            result["state"][0],
            torch.tensor([[10.0, 20.0, 30.0, 40.0]], device=device),
        )
        assert result["state"].shape[0] == 1


class TestBufferDevice:
    @pytest.mark.gpu
    def test_device_property_returns_config_device(self, device) -> None:
        buf = Buffer(capacity=4, num_envs=1, schema=_schema(), device=device)
        assert buf.device.type == device.type
