"""Unit tests for utility functions (zerorl.functions)."""

import pytest
import torch
import torch.nn as nn
from zerorl.helpers.agent import BaseAgent, eval_action
from zerorl.functions import get_buffer_params_model, fast_compile, _cxx_compiler_available

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestGetBufferParamsModel:
    @pytest.mark.gpu
    def test_returns_tuple_of_dicts(self, device) -> None:
        class SimpleAgent(BaseAgent):
            def __init__(self): super().__init__(); self.actor = nn.Linear(4, 2); self.register_buffer("running_mean", torch.zeros(4))
            def forward(self, state, **kwargs): return self.actor(state), torch.zeros(1)
            @staticmethod
            def build_distribution(logits): return torch.distributions.Categorical(logits=logits)
            def get_action(self, state: torch.Tensor, action: torch.Tensor | None = None, **kwargs):
                logits, value = self.forward(state, **kwargs)
                dist = self.build_distribution(logits)
                if action is None: action = dist.sample()
                log_prob, dist_entropy = eval_action(dist, action)
                return {"action": action, "log_prob": log_prob, "entropy":dist_entropy, "value":value}

        agent = SimpleAgent().to(device)
        params, buffers = get_buffer_params_model(agent)
        assert "actor.weight" in params
        assert "running_mean" in buffers


class TestCompilerCheck:
    def test_cxx_compiler_available_returns_bool(self) -> None:
        result = _cxx_compiler_available()
        assert isinstance(result, bool)

    def test_fast_compile_noop_when_no_compiler(self, monkeypatch) -> None:
        monkeypatch.setattr("zerorl.functions._cxx_compiler_available", lambda: False)
        @fast_compile
        def identity(x):
            return x
        inp = torch.randn(4)
        result = identity(inp)  # type: ignore[arg-type]
        assert result is inp

    def test_fast_compile_decorated_fn_can_be_called(self) -> None:
        @fast_compile
        def add(a, b):
            return a + b
        assert add(1, 2) == 3  # type: ignore[arg-type]
