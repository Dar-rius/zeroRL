"""Custom exceptions for the zerorl project."""


class EmptyBufferError(Exception):
    """Raised when an update is attempted on an insufficiently filled buffer.

    Attributes:
        current_size: Number of entries currently in the buffer.
        require_buffer_size: Minimum entries required for an update.
    """

    def __init__(self, current_size: int, require_buffer_size: int):
        self.current_size = current_size
        self.require_buffer_size = require_buffer_size

        self.message = "Training agent flow is incorrect: the buffer is empty"
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return a detailed error message with context and suggestion."""
        suggestion = "Call Rollout before update weights"
        details = f"[Crash Workflow] {self.message}\n"
        details += f"the current buffer size: {self.current_size}\n"
        details += f" the minimal size required is: {self.require_buffer_size}\n"
        details += f"{suggestion}"
        return details


class KeyBufferError(Exception):
    """Raised when the argument name don't exist in data buffer.

    Attributes:
        arg_name: the argument name.
        data_buffer: the data buffer
    """

    def __init__(self, arg_name: str, data_buffer: dict[str, object]):
        self.arg_name = arg_name
        self.data_buffer = data_buffer 

    def __str__(self) -> str:
        """Return a detailed error message with context and suggestion."""
        details = f"Key '{self.arg_name}' (returned by agent.get_action) does not exist in the Buffer. \n"
        details += f"Please ensure your Buffer is initialized with the key '{self.arg_name}\n"
        details += f"Current valid keys are: {list(self.data_buffer.keys())}"
        return details


