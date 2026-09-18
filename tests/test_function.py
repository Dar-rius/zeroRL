"""Unit tests for utility functions (zerorl.functions)."""

import os
import numpy as np
import pytest
import torch
import torch.nn as nn
from gymnasium.vector import SyncVectorEnv
from zerorl.helpers.agent import BaseAgent, eval_action
from zerorl.functions import (
    env_step,
    get_buffer_params_model,
    get_obs_act,
    parse_dict_to_tensor,
    processing_state,
    save_checkpoints,
    to_env_action,
    vectorize_env,
)
from zerorl.compiler import fast_compile, _cxx_compiler_available
from zerorl.processing import NormMeanStd


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleAgent(BaseAgent):
    """Minimal agent满足 BaseAgent contract for testing."""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2):
        super().__init__()
        self.actor = nn.Linear(obs_dim, act_dim)
        self.critic = nn.Linear(obs_dim, 1)
        self.register_buffer("running_mean", torch.zeros(obs_dim))

    def forward(self, state, **kwargs):
        return self.actor(state), self.critic(state)

    @staticmethod
    def build_distribution(logits):
        return torch.distributions.Categorical(logits=logits)

    def get_action(self, state: torch.Tensor, action: torch.Tensor | None = None, **kwargs):
        logits, value = self.forward(state, **kwargs)
        dist = self.build_distribution(logits)
        if action is None:
            action = dist.sample()
        log_prob, dist_entropy = eval_action(dist, action)
        return {
            "action": action,
            "log_prob": log_prob,
            "entropy": dist_entropy,
            "value": value.squeeze(-1),
        }


@pytest.fixture
def simple_agent() -> SimpleAgent:
    return SimpleAgent()


@pytest.fixture
def cartpole_env():
    env = vectorize_env("CartPole-v1", num_envs=1)
    yield env
    env.close()


@pytest.fixture
def pendulum_env():
    env = vectorize_env("Pendulum-v1", num_envs=1)
    yield env
    env.close()


# ===========================================================================
# Existing tests (preserved)
# ===========================================================================

class TestGetBufferParamsModel:
    @pytest.mark.gpu
    def test_returns_tuple_of_dicts(self, device) -> None:
        agent = SimpleAgent().to(device)
        params, buffers = get_buffer_params_model(agent)
        assert "actor.weight" in params
        assert "running_mean" in buffers


class TestCompilerCheck:
    def test_cxx_compiler_available_returns_bool(self) -> None:
        result = _cxx_compiler_available()
        assert isinstance(result, bool)

    def test_fast_compile_noop_when_no_compiler(self, monkeypatch) -> None:
        monkeypatch.setattr("zerorl.compiler._cxx_compiler_available", lambda: False)
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
    def test_discrete_env_returns_five_tuple(self, cartpole_env) -> None:
        result = get_obs_act(cartpole_env)
        assert len(result) == 5

    def test_discrete_env_is_discrete_true(self, cartpole_env) -> None:
        _, _, _, _, is_discrete = get_obs_act(cartpole_env)
        assert is_discrete is True

    def test_discrete_env_action_count(self, cartpole_env) -> None:
        _, _, _, act_n, _ = get_obs_act(cartpole_env)
        assert act_n == 2

    def test_discrete_env_action_shape_is_empty(self, cartpole_env) -> None:
        _, act_shape, _, _, _ = get_obs_act(cartpole_env)
        assert act_shape == ()

    def test_discrete_env_obs_dim(self, cartpole_env) -> None:
        obs_shape, _, obs_n, _, _ = get_obs_act(cartpole_env)
        assert obs_shape == (4,)
        assert obs_n == 4

    def test_continuous_env_is_discrete_false(self, pendulum_env) -> None:
        _, _, _, _, is_discrete = get_obs_act(pendulum_env)
        assert is_discrete is False

    def test_continuous_env_action_count(self, pendulum_env) -> None:
        _, _, _, act_n, _ = get_obs_act(pendulum_env)
        assert act_n == 1

    def test_continuous_env_action_shape(self, pendulum_env) -> None:
        _, act_shape, _, _, _ = get_obs_act(pendulum_env)
        assert act_shape == (1,)


# ===========================================================================
# vectorize_env
# ===========================================================================

class TestVectorizeEnv:
    def test_returns_sync_vector_env(self) -> None:
        env = vectorize_env("CartPole-v1", num_envs=1)
        assert isinstance(env, SyncVectorEnv)
        env.close()

    def test_num_envs_propagates(self) -> None:
        env = vectorize_env("CartPole-v1", num_envs=3)
        obs: np.ndarray
        obs, _ = env.reset(seed=0)
        assert obs.shape == (3, 4)
        env.close()

    def test_callable_spec(self) -> None:
        env = vectorize_env(lambda: __import__("gymnasium").make("CartPole-v1"), num_envs=1)
        obs: np.ndarray
        obs, _ = env.reset(seed=0)
        assert obs.shape == (1, 4)
        env.close()

    def test_class_spec(self) -> None:
        import gymnasium.envs.classic_control.cartpole as cp
        env = vectorize_env(cp.CartPoleEnv, num_envs=1)
        obs: np.ndarray
        obs, _ = env.reset(seed=0)
        assert obs.shape == (1, 4)
        env.close()

    def test_instance_spec_deepcopies(self) -> None:
        import gymnasium
        base = gymnasium.make("CartPole-v1")
        env = vectorize_env(base, num_envs=2)  # type: ignore[arg-type]
        obs: np.ndarray
        obs, _ = env.reset(seed=0)
        assert obs.shape == (2, 4)
        env.close()
        base.close()

    def test_render_mode_propagates(self) -> None:
        env = vectorize_env("CartPole-v1", num_envs=1, render_mode="rgb_array")
        obs: np.ndarray
        obs, _ = env.reset(seed=0)
        assert obs.shape == (1, 4)
        env.close()


# ===========================================================================
# env_step
# ===========================================================================

class TestEnvStep:
    def test_returns_expected_keys(self, simple_agent, cartpole_env) -> None:
        obs, _ = cartpole_env.reset(seed=0)
        state_tensor = processing_state(obs)
        result = env_step(cartpole_env, simple_agent, state_tensor)
        expected = {"state", "next_state", "reward", "terminated", "truncated", "info",
                    "action", "log_prob", "entropy", "value"}
        assert set(result.keys()) == expected

    def test_next_state_shape(self, simple_agent, cartpole_env) -> None:
        obs, _ = cartpole_env.reset(seed=0)
        state_tensor = processing_state(obs)
        result = env_step(cartpole_env, simple_agent, state_tensor)
        assert result["next_state"].shape == (1, 4)

    def test_reward_is_numeric(self, simple_agent, cartpole_env) -> None:
        obs, _ = cartpole_env.reset(seed=0)
        state_tensor = processing_state(obs)
        result = env_step(cartpole_env, simple_agent, state_tensor)
        reward_arr = np.asarray(result["reward"])
        assert np.all(np.isfinite(reward_arr))

    def test_action_matches_agent_output(self, simple_agent, cartpole_env) -> None:
        obs, _ = cartpole_env.reset(seed=0)
        state_tensor = processing_state(obs)
        with torch.inference_mode():
            agent_out = simple_agent.get_action(state_tensor)
        result = env_step(cartpole_env, simple_agent, state_tensor)
        assert result["action"].shape == agent_out["action"].shape

    def test_with_normalizer(self, simple_agent, cartpole_env) -> None:
        norm = NormMeanStd(shape=(4,))
        obs, _ = cartpole_env.reset(seed=0)
        state_tensor = processing_state(obs, normalizer=norm)
        result = env_step(cartpole_env, simple_agent, state_tensor)
        assert "next_state" in result
        assert norm.count > 1.0

    def test_device_cpu(self, simple_agent, cartpole_env) -> None:
        obs, _ = cartpole_env.reset(seed=0)
        state_tensor = processing_state(obs, device=torch.device("cpu"))
        result = env_step(cartpole_env, simple_agent, state_tensor)
        assert isinstance(result["next_state"], np.ndarray)


# ===========================================================================
# save_checkpoints
# ===========================================================================

class TestSaveCheckpoints:
    def test_creates_file(self, simple_agent, tmp_path) -> None:
        path = str(tmp_path / "model.pt")
        save_checkpoints(simple_agent, path)
        assert os.path.exists(path)

    def test_saved_keys(self, simple_agent, tmp_path) -> None:
        path = str(tmp_path / "model.pt")
        save_checkpoints(simple_agent, path)
        data = torch.load(path, weights_only=False)
        assert "agent_state_dict" in data
        assert "normalizer_state_dict" in data

    def test_agent_state_dict_roundtrip(self, simple_agent, tmp_path) -> None:
        path = str(tmp_path / "model.pt")
        save_checkpoints(simple_agent, path)
        data = torch.load(path, weights_only=False)
        for k in simple_agent.state_dict():
            assert k in data["agent_state_dict"]
            torch.testing.assert_close(
                simple_agent.state_dict()[k], data["agent_state_dict"][k]
            )

    def test_with_normalizer(self, simple_agent, tmp_path) -> None:
        norm = NormMeanStd(shape=(4,))
        norm.update(torch.randn(10, 4))
        path = str(tmp_path / "model.pt")
        save_checkpoints(simple_agent, path, normalizer=norm)
        data = torch.load(path, weights_only=False)
        assert data["normalizer_state_dict"] is not None
        assert "mean" in data["normalizer_state_dict"]

    def test_without_normalizer(self, simple_agent, tmp_path) -> None:
        path = str(tmp_path / "model.pt")
        save_checkpoints(simple_agent, path)
        data = torch.load(path, weights_only=False)
        assert data["normalizer_state_dict"] is None

    def test_creates_nested_dirs(self, simple_agent, tmp_path) -> None:
        path = str(tmp_path / "sub" / "deep" / "model.pt")
        save_checkpoints(simple_agent, path)
        assert os.path.exists(path)


# ===========================================================================
# processing_state
# ===========================================================================

class TestProcessingState:
    def test_numpy_input_returns_tensor(self) -> None:
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = processing_state(arr)
        assert isinstance(result, torch.Tensor)

    def test_tensor_input_returns_tensor(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0])
        result = processing_state(t)
        assert isinstance(result, torch.Tensor)

    def test_1d_input_unsqueezed(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = processing_state(arr)
        assert result.shape == (1, 4)

    def test_2d_input_unchanged(self) -> None:
        arr = np.random.randn(5, 4).astype(np.float32)
        result = processing_state(arr)
        assert result.shape == (5, 4)

    def test_dtype_is_float32(self) -> None:
        arr = np.array([1.0, 2.0], dtype=np.float64)
        result = processing_state(arr)
        assert result.dtype == torch.float32

    def test_device_cpu(self) -> None:
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = processing_state(arr, device=torch.device("cpu"))
        assert result.device == torch.device("cpu")

    def test_with_normalizer_update_true(self) -> None:
        norm = NormMeanStd(shape=(4,))
        arr = np.random.randn(4).astype(np.float32)
        processing_state(arr, normalizer=norm, update=True)
        assert norm.count > 1.0

    def test_with_normalizer_update_false(self) -> None:
        norm = NormMeanStd(shape=(4,))
        initial_count = norm.count
        arr = np.random.randn(4).astype(np.float32)
        processing_state(arr, normalizer=norm, update=False)
        assert norm.count == initial_count

    def test_with_normalizer_normalizes(self) -> None:
        norm = NormMeanStd(shape=(4,))
        for _ in range(100):
            norm.update(torch.randn(10, 4))
        arr = np.random.randn(4).astype(np.float32)
        result = processing_state(arr, normalizer=norm, update=False)
        assert result.shape == (1, 4)


# ===========================================================================
# parse_dict_to_tensor
# ===========================================================================

class TestParseDictToTensor:
    def _make_array_output(self):
        return {
            "next_state": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            "reward": np.array([1.0]),
            "terminated": np.array([False]),
            "truncated": np.array([False]),
        }

    def _make_scalar_output(self):
        return {
            "next_state": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            "reward": 1.0,
            "terminated": False,
            "truncated": False,
        }

    def test_returns_dict(self) -> None:
        out = parse_dict_to_tensor(self._make_array_output())
        assert isinstance(out, dict)

    def test_keys_preserved(self) -> None:
        out = parse_dict_to_tensor(self._make_array_output())
        for k in ("next_state", "reward", "terminated", "truncated"):
            assert k in out

    def test_values_are_tensors(self) -> None:
        out = parse_dict_to_tensor(self._make_array_output())
        for k in ("next_state", "reward", "terminated", "truncated"):
            assert isinstance(out[k], torch.Tensor)

    def test_dtype_is_float32(self) -> None:
        out = parse_dict_to_tensor(self._make_array_output())
        for k in ("next_state", "reward", "terminated", "truncated"):
            assert out[k].dtype == torch.float32

    def test_array_reward_becomes_tensor(self) -> None:
        out = parse_dict_to_tensor(self._make_array_output())
        assert out["reward"].ndim == 1

    def test_already_tensor_unchanged(self) -> None:
        output = {
            "next_state": torch.tensor([1.0, 2.0]),
            "reward": torch.tensor([0.5]),
            "terminated": torch.tensor([False]),
            "truncated": torch.tensor([False]),
        }
        out = parse_dict_to_tensor(output)
        for k in ("next_state", "reward", "terminated", "truncated"):
            assert out[k].dtype == torch.float32

    def test_scalar_unsqueeze(self) -> None:
        parse_dict_to_tensor(self._make_scalar_output())


# ===========================================================================
# to_env_action
# ===========================================================================

class TestToEnvAction:
    def test_cpu_env_returns_numpy(self) -> None:
        class MockEnv:
            device = "cpu"
        action = torch.tensor([0, 1])
        result = to_env_action(action, MockEnv())
        assert isinstance(result, np.ndarray)

    def test_no_device_attr_returns_numpy(self) -> None:
        class MockEnv:
            pass
        action = torch.tensor([0, 1])
        result = to_env_action(action, MockEnv())
        assert isinstance(result, np.ndarray)

    def test_tensor_input_cpu(self) -> None:
        class MockEnv:
            device = "cpu"
        action = torch.tensor([1.0, 0.0])
        result = to_env_action(action, MockEnv())
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [1.0, 0.0])
