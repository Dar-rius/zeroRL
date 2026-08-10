"""Unit tests for custom exceptions (zerorl.errors).

Verifies EmptyBufferError (raised on update with insufficient buffer),
KeyBufferError (raised by Buffer.insert() on unknown keys), and
assert_agent_contract (runtime check for required agent methods).
"""

import pytest
import torch
import torch.nn as nn
from zerorl.agent import BaseAgent
from zerorl.errors import EmptyBufferError, KeyBufferError, assert_agent_contract


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


class _GoodAgent(BaseAgent):
    """Minimal agent satisfying the full contract for positive-case tests."""
    def __init__(self) -> None:
        super().__init__(); self.actor = nn.Linear(4, 2)

    def forward(self, state: torch.Tensor, **kwargs):
        return self.actor(state), torch.zeros(1)

    @staticmethod
    def build_distribution(logits: torch.Tensor):
        return torch.distributions.Categorical(logits=logits)

    def get_action(self, state: torch.Tensor, action=None, **kwargs):
        logits, v = self.forward(state)
        d = self.build_distribution(logits)
        a = d.sample() if action is None else action
        return {"action": a, "log_prob": torch.zeros(1),
                "entropy": torch.zeros(1), "value": v}


class TestAssertAgentContract:
    """Tests for assert_agent_contract (runtime agent method checks)."""

    def test_passes_when_all_attrs_present(self) -> None:
        agent = _GoodAgent()
        assert_agent_contract(agent, {
            "forward": "f", "get_action": "g", "build_distribution": "b"})

    def test_raises_not_implemented_for_missing_attr(self) -> None:
        # NOTE: can't test `forward` here — nn.Module defines a default
        # `forward`, so hasattr(agent, "forward") is always True. The
        # {"forward": ...} entry in ppo.py:90 is effectively a no-op.
        # Use `get_action` instead (not inherited from nn.Module).
        class NoGetAction(BaseAgent):
            def __init__(self): super().__init__(); self.actor = nn.Linear(4, 2)
            def forward(self, state, **kwargs):
                return self.actor(state), torch.zeros(1)
            @staticmethod
            def build_distribution(logits: torch.Tensor):
                return torch.distributions.Categorical(logits=logits)
            # NOTE: no get_action -> assert_agent_contract raises
        with pytest.raises(NotImplementedError, match="^g$"):
            assert_agent_contract(NoGetAction(), {"get_action": "g"})

    def test_message_matches_dict_value(self) -> None:
        with pytest.raises(NotImplementedError, match="expected message"):
            assert_agent_contract(_GoodAgent(), {"missing_attr": "expected message"})

    def test_empty_dict_no_raise(self) -> None:
        assert_agent_contract(_GoodAgent(), {})

    def test_first_missing_attr_raises_in_dict_order(self) -> None:
        # Documents dict-iteration semantics: the first missing attr raises,
        # and the raised message is the dict VALUE (not the key).
        agent = _GoodAgent()  # has forward, get_action, build_distribution
        with pytest.raises(NotImplementedError, match="^msg1$"):
            assert_agent_contract(agent, {"missing_one": "msg1", "missing_two": "msg2"})
