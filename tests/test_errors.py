"""Unit tests for EmptyBufferError (zerorl.errors).

Verifies the custom exception raised when update is attempted on an
insufficiently filled buffer.
"""

import pytest
from zerorl.errors import EmptyBufferError


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
