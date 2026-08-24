"""Unit tests for utility functions (zerorl.functions)."""

import pytest
import torch
import torch.nn as nn
from zerorl.helpers.agent import BaseAgent, eval_action
from zerorl.functions import get_buffer_params_model, fast_compile, _cxx_compiler_available, get_obs_act, vectorize_env

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


class TestGetObsAct:
    def test_discrete_env_returns_five_tuple(self) -> None:
        env = vectorize_env("CartPole-v1", 1)
        result = get_obs_act(env)
        assert len(result) == 5
        env.close()

    def test_discrete_env_is_discrete_true(self) -> None:
        env = vectorize_env("CartPole-v1", 1)
        obs_shape, act_shape, obs_n, act_n, is_discrete = get_obs_act(env)
        assert is_discrete is True
        env.close()

    def test_discrete_env_action_count(self) -> None:
        env = vectorize_env("CartPole-v1", 1)
        _, _, _, act_n, _ = get_obs_act(env)
        assert act_n == 2
        env.close()

    def test_discrete_env_action_shape_is_empty(self) -> None:
        env = vectorize_env("CartPole-v1", 1)
        _, act_shape, _, _, _ = get_obs_act(env)
        assert act_shape == ()
        env.close()

    def test_discrete_env_obs_dim(self) -> None:
        env = vectorize_env("CartPole-v1", 1)
        obs_shape, _, obs_n, _, _ = get_obs_act(env)
        assert obs_shape == (4,)
        assert obs_n == 4
        env.close()

    def test_continuous_env_is_discrete_false(self) -> None:
        env = vectorize_env("Pendulum-v1", 1)
        _, _, _, _, is_discrete = get_obs_act(env)
        assert is_discrete is False
        env.close()

    def test_continuous_env_action_count(self) -> None:
        env = vectorize_env("Pendulum-v1", 1)
        _, _, _, act_n, _ = get_obs_act(env)
        assert act_n == 1
        env.close()

    def test_continuous_env_action_shape(self) -> None:
        env = vectorize_env("Pendulum-v1", 1)
        _, act_shape, _, _, _ = get_obs_act(env)
        assert act_shape == (1,)
        env.close()
