"""Unit tests for custom exceptions (zerorl.errors).

Verifies EmptyBufferError (raised on update with insufficient buffer)
and KeyBufferError (raised by Buffer.insert() on unknown keys).
"""

import pytest
from zerorl.errors import EmptyBufferError, KeyBufferError


class TestEmptyBufferError:
    """Tests for the EmptyBufferError custom exception."""

    def test_is_exception_subclass(self) -> None:
        assert issubclass(EmptyBufferError, Exception)

    def test_stores_attributes(self) -> None:
        err = EmptyBufferError(current_size=3, require_buffer_size=10)
        assert err.current_size == 3
        assert err.require_buffer_size == 10

    def test_message_attribute(self) -> None:
        err = EmptyBufferError(current_size=0, require_buffer_size=5)
        assert "buffer is empty" in err.message.lower()

    def test_str_contains_details(self) -> None:
        err = EmptyBufferError(current_size=2, require_buffer_size=8)
        s = str(err)
        assert "2" in s
        assert "8" in s
        assert "buffer" in s.lower()

    def test_str_contains_suggestion(self) -> None:
        err = EmptyBufferError(current_size=0, require_buffer_size=1)
        s = str(err)
        assert "rollout" in s.lower() or "update" in s.lower()

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(EmptyBufferError) as exc_info:
            raise EmptyBufferError(current_size=0, require_buffer_size=10)
        assert exc_info.value.current_size == 0
        assert exc_info.value.require_buffer_size == 10

    def test_different_sizes(self) -> None:
        err = EmptyBufferError(current_size=99, require_buffer_size=100)
        assert err.current_size == 99
        assert err.require_buffer_size == 100


class TestKeyBufferError:
    """Tests for the KeyBufferError custom exception."""

    def test_is_exception_subclass(self) -> None:
        assert issubclass(KeyBufferError, Exception)

    def test_stores_attributes(self) -> None:
        data = {"state": object(), "action": object()}
        err = KeyBufferError("missing", data)
        assert err.arg_name == "missing"
        assert err.data_buffer is data

    def test_str_contains_arg_name(self) -> None:
        err = KeyBufferError("entropy", {"state": object()})
        s = str(err)
        assert "entropy" in s

    def test_str_lists_valid_keys(self) -> None:
        data = {"state": object(), "action": object(), "value": object()}
        err = KeyBufferError("foo", data)
        s = str(err)
        assert "state" in s
        assert "action" in s
        assert "value" in s

    def test_can_be_raised_and_caught(self) -> None:
        data = {"state": object()}
        with pytest.raises(KeyBufferError) as exc_info:
            raise KeyBufferError("bar", data)
        assert exc_info.value.arg_name == "bar"
        assert exc_info.value.data_buffer is data
