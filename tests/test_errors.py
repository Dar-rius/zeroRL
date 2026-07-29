"""Unit tests for EmptyBufferError (rl_template.errors).

Verifies the custom exception raised when update is attempted on an
insufficiently filled buffer.
"""

import pytest
from rl_template.errors import EmptyBufferError


class TestEmptyBufferError:
    """Tests for the EmptyBufferError custom exception."""

    def test_is_exception_subclass(self):
        """EmptyBufferError should be a subclass of Exception."""
        assert issubclass(EmptyBufferError, Exception)

    def test_stores_attributes(self):
        """The error should store both current_size and require_buffer_size."""
        err = EmptyBufferError(current_size=3, require_buffer_size=10)
        assert err.current_size == 3
        assert err.require_buffer_size == 10

    def test_message_attribute(self):
        """The error message should mention that the buffer is empty."""
        err = EmptyBufferError(current_size=0, require_buffer_size=5)
        assert "buffer is empty" in err.message.lower()

    def test_str_contains_details(self):
        """str(error) should include both the current and required buffer sizes."""
        err = EmptyBufferError(current_size=2, require_buffer_size=8)
        s = str(err)
        assert "2" in s
        assert "8" in s
        assert "buffer" in s.lower()

    def test_str_contains_suggestion(self):
        """str(error) should suggest running rollout before update."""
        err = EmptyBufferError(current_size=0, require_buffer_size=1)
        s = str(err)
        assert "rollout" in s.lower() or "update" in s.lower()

    def test_can_be_raised_and_caught(self):
        """EmptyBufferError should be raisable and catchable via pytest.raises."""
        with pytest.raises(EmptyBufferError) as exc_info:
            raise EmptyBufferError(current_size=0, require_buffer_size=10)
        assert exc_info.value.current_size == 0
        assert exc_info.value.require_buffer_size == 10

    def test_different_sizes(self):
        """Error should work correctly with any size values."""
        err = EmptyBufferError(current_size=99, require_buffer_size=100)
        assert err.current_size == 99
        assert err.require_buffer_size == 100
