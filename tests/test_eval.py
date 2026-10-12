"""Unit tests for eval/policy primitives (zerorl.functions)."""

import warnings

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from zerorl.config import TrainConfig
from zerorl.functions import (
    _resolve_env_spec,
    actions_limits,
    deterministic_action,
    evaluate_policy,
    get_last_obs,
    parse_to_tensor,
    try_agent,
    vectorize_env,
)
from zerorl.helpers.agent import BaseAgent
from zerorl.helpers.factory import ActorCriticAgent


# ---------------------------------------------------------------------------
# Test agents
# ---------------------------------------------------------------------------

class NoDistAgent(BaseAgent):
    """Agent with get_action but no build_distribution -> fallback path."""
    def __init__(self, obs_dim: int = 4, act_dim: int = 2) -> None:
        super().__init__()
        self.actor = nn.Linear(obs_dim, act_dim)
        self.fixed_action = 1

    def forward(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.actor(state)

    def get_action(self, state: torch.Tensor, action: torch.Tensor | None = None, **kwargs):
        batch = state.shape[0]
        act = torch.full((batch,), self.fixed_action, dtype=torch.long, device=state.device)
        return {"action": act, "log_prob": torch.zeros(batch, device=state.device),
                "entropy": torch.zeros(batch, device=state.device),
                "value": torch.zeros(batch, device=state.device)}


# ---------------------------------------------------------------------------
# _resolve_env_spec (C5 fix)
# ---------------------------------------------------------------------------

class TestResolveEnvSpec:
    def test_str_input_passthrough(self) -> None:
        assert _resolve_env_spec("CartPole-v1") == "CartPole-v1"

    def test_instance_input_passthrough(self) -> None:
        obj = object()
        assert _resolve_env_spec(obj) is obj

    def test_gym_vector_env_returns_factory_with_kwargs(self) -> None:
        env = vectorize_env(lambda: gym.make("Pendulum-v1", g=9.81), num_envs=1)
        resolved = _resolve_env_spec(env)
        assert callable(resolved) and not isinstance(resolved, type)
        fresh = resolved()
        try:
            assert fresh.spec.kwargs == {"g": 9.81}
        finally:
            fresh.close()
        env.close()

    def test_render_mode_forced(self) -> None:
        env = vectorize_env("CartPole-v1", num_envs=1)
        resolved = _resolve_env_spec(env, render_mode="rgb_array")
        fresh = resolved()
        try:
            assert fresh.render_mode == "rgb_array"
        finally:
            fresh.close()
        env.close()

    def test_none_id_falls_back_to_unwrapped_class(self) -> None:
        env = vectorize_env("CartPole-v1", num_envs=1)
        root_spec = env.envs[0].unwrapped.spec
        old_id = root_spec.id
        root_spec.id = None
        try:
            resolved = _resolve_env_spec(env)
            assert isinstance(resolved, type)
            fresh = resolved()  # must construct (wrapper class would TypeError)
            assert isinstance(fresh.unwrapped, type(env.envs[0].unwrapped))
            fresh.close()
        finally:
            root_spec.id = old_id
            env.close()

    def test_custom_base_env_returns_its_class(self) -> None:
        from zerorl.helpers.env import BaseEnv

        class PlainEnv(BaseEnv):
            def __init__(self) -> None:
                super().__init__()
                self.observation_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
                self.action_space = gym.spaces.Discrete(2)

            def reset(self, *, seed=None, options=None):
                return np.zeros(4, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(4, dtype=np.float32), 1.0, False, False, {}

            def close(self):
                pass

        env = vectorize_env(PlainEnv, num_envs=1)
        resolved = _resolve_env_spec(env)
        assert resolved is PlainEnv
        env.close()


# ---------------------------------------------------------------------------
# deterministic_action
# ---------------------------------------------------------------------------

class TestDeterministicAction:
    def test_fallback_without_build_distribution(self) -> None:
        agent = NoDistAgent()
        out = deterministic_action(agent, torch.randn(3, 4))
        assert torch.equal(out, torch.full((3,), 1, dtype=torch.long))

    def test_discrete_returns_argmax(self) -> None:
        torch.manual_seed(0)
        agent = ActorCriticAgent(4, 2, is_discrete=True)
        state = torch.randn(5, 4)
        logits, _ = agent(state)
        out = deterministic_action(agent, state)
        assert torch.equal(out, logits.argmax(dim=-1))

    def test_continuous_returns_distribution_mean(self) -> None:
        torch.manual_seed(0)
        agent = ActorCriticAgent(3, 1, is_discrete=False)
        state = torch.randn(4, 3)
        logits, _ = agent(state)
        expected = agent.build_distribution(logits).mean
        out = deterministic_action(agent, state)
        assert torch.allclose(out, expected)

    def test_flagless_categorical_agent_returns_argmax(self) -> None:
        """Agents without `is_discrete` must still get argmax, not Categorical.mean
        (which torch deliberately returns as nan)."""
        class FlaglessCategoricalAgent(BaseAgent):
            def __init__(self, obs_dim: int = 4, act_dim: int = 2) -> None:
                super().__init__()
                self.actor = nn.Linear(obs_dim, act_dim)

            def forward(self, state, **kwargs):
                return self.actor(state), torch.zeros(state.shape[0], device=state.device)

            def get_action(self, state, action=None, **kwargs):
                batch = state.shape[0]
                zeros = torch.zeros(batch, device=state.device)
                return {"action": zeros.long(), "log_prob": zeros,
                        "entropy": zeros, "value": zeros}

            def build_distribution(self, logits):
                return torch.distributions.Categorical(logits=logits)

        agent = FlaglessCategoricalAgent()
        assert not hasattr(agent, "is_discrete")
        state = torch.randn(5, 4)
        logits, _ = agent(state)
        out = deterministic_action(agent, state)
        assert torch.equal(out, logits.argmax(dim=-1))
        assert torch.isfinite(out.float()).all()


# ---------------------------------------------------------------------------
# evaluate_policy
# ---------------------------------------------------------------------------

class TestEvaluatePolicy:
    def test_returns_stats_dict(self) -> None:
        agent = ActorCriticAgent(4, 2, is_discrete=True)
        result = evaluate_policy("CartPole-v1", agent, seeds=[0, 1],
                                 normalizer=None, device=torch.device("cpu"))
        assert set(result) == {"mean", "std", "returns"}
        assert len(result["returns"]) == 2
        assert isinstance(result["mean"], float)
        assert isinstance(result["std"], float)

    def test_std_is_ddof_one(self) -> None:
        agent = ActorCriticAgent(4, 2, is_discrete=True)
        result = evaluate_policy("CartPole-v1", agent, seeds=[7, 8],
                                 normalizer=None, device=torch.device("cpu"))
        expected = float(np.std(result["returns"], ddof=1))
        assert result["std"] == pytest.approx(expected)

    def test_single_seed_std_zero(self) -> None:
        agent = ActorCriticAgent(4, 2, is_discrete=True)
        result = evaluate_policy("CartPole-v1", agent, seeds=[3],
                                 normalizer=None, device=torch.device("cpu"))
        assert result["std"] == 0.0

    def test_restores_training_mode(self) -> None:
        agent = ActorCriticAgent(4, 2, is_discrete=True)
        assert agent.training
        evaluate_policy("CartPole-v1", agent, seeds=[0],
                        normalizer=None, device=torch.device("cpu"))
        assert agent.training

        agent.eval()
        evaluate_policy("CartPole-v1", agent, seeds=[0],
                        normalizer=None, device=torch.device("cpu"))
        assert not agent.training

    def test_accepts_vector_env_input(self) -> None:
        env = vectorize_env("CartPole-v1", num_envs=2)
        agent = ActorCriticAgent(4, 2, is_discrete=True)
        result = evaluate_policy(env, agent, seeds=[0],
                                 normalizer=None, device=torch.device("cpu"))
        assert len(result["returns"]) == 1
        env.close()

    def test_none_id_env_does_not_crash(self) -> None:
        """Regression for C5: spec.id=None used to reach vectorize_env(None)."""
        env = vectorize_env("CartPole-v1", num_envs=1)
        root_spec = env.envs[0].unwrapped.spec
        old_id = root_spec.id
        root_spec.id = None
        try:
            agent = ActorCriticAgent(4, 2, is_discrete=True)
            result = evaluate_policy(env, agent, seeds=[0],
                                     normalizer=None, device=torch.device("cpu"))
            assert len(result["returns"]) == 1
        finally:
            root_spec.id = old_id
            env.close()


# ---------------------------------------------------------------------------
# try_agent (GIF)
# ---------------------------------------------------------------------------

class TestTryAgent:
    @pytest.fixture
    def cfg(self, tmp_path) -> TrainConfig:
        return TrainConfig(model_name="evaltest", model_save_path=str(tmp_path),
                           project_name="proj")

    def test_writes_gif_to_given_path(self, tmp_path, cfg) -> None:
        agent = ActorCriticAgent(4, 2, is_discrete=True)
        try_agent("CartPole-v1", agent, cfg, iterations=1,
                  gif_path=str(tmp_path / "e"))
        assert (tmp_path / "e_0.gif").is_file()

    def test_iterations_write_one_gif_each(self, tmp_path, cfg) -> None:
        agent = ActorCriticAgent(4, 2, is_discrete=True)
        try_agent("CartPole-v1", agent, cfg, iterations=2,
                  gif_path=str(tmp_path / "ep"))
        assert (tmp_path / "ep_0.gif").is_file()
        assert (tmp_path / "ep_1.gif").is_file()

    def test_default_path_uses_project_name(self, tmp_path, cfg, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        agent = ActorCriticAgent(4, 2, is_discrete=True)
        try_agent("CartPole-v1", agent, cfg, iterations=1)
        assert (tmp_path / "proj_0.gif").is_file()

    def test_restores_training_mode(self, tmp_path, cfg) -> None:
        agent = ActorCriticAgent(4, 2, is_discrete=True)
        assert agent.training
        try_agent("CartPole-v1", agent, cfg, gif_path=str(tmp_path / "m"))
        assert agent.training
        agent.eval()
        try_agent("CartPole-v1", agent, cfg, gif_path=str(tmp_path / "n"))
        assert not agent.training

    def test_try_agent_no_fps_deprecation_warning(self, tmp_path, cfg) -> None:
        agent = ActorCriticAgent(4, 2, is_discrete=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try_agent("CartPole-v1", agent, cfg, iterations=1,
                      gif_path=str(tmp_path / "w"))
        fps_warnings = [w for w in caught
                        if issubclass(w.category, DeprecationWarning)
                        and "fps" in str(w.message)]
        assert fps_warnings == []

    def test_gif_frame_durations_ms(self, tmp_path, cfg) -> None:
        from PIL import Image
        agent = ActorCriticAgent(4, 2, is_discrete=True)
        try_agent("CartPole-v1", agent, cfg, iterations=1,
                  gif_path=str(tmp_path / "d"))
        gif = tmp_path / "d_0.gif"
        assert gif.is_file()
        im = Image.open(gif)
        durations = []
        for i in range(im.n_frames):
            im.seek(i)
            durations.append(im.info.get("duration"))
        # Pillow merges pixel-identical consecutive frames and sums their
        # durations, so assert merge-invariant properties instead of an
        # exact per-frame list: every delay is a positive multiple of the
        # 200 ms base (not the old fps=25 40 ms, not a seconds misread),
        # and the 1200 ms terminal hold survives (possibly merged upward).
        assert durations, "gif has no frames"
        assert all(d > 0 and d % 200 == 0 for d in durations), durations
        assert durations[-1] >= 1200


# ---------------------------------------------------------------------------
# parse_to_tensor
# ---------------------------------------------------------------------------

class TestParseToTensor:
    def test_int_becomes_1d_float32(self) -> None:
        out = parse_to_tensor(3)
        assert out.shape == (1,)
        assert out.dtype == torch.float32
        assert out.item() == 3.0

    def test_float_becomes_1d(self) -> None:
        out = parse_to_tensor(2.5)
        assert out.shape == (1,)
        assert out.item() == pytest.approx(2.5)

    def test_ndarray_preserves_length(self) -> None:
        out = parse_to_tensor(np.array([1.0, 2.0, 3.0]))
        assert out.shape == (3,)
        assert out.dtype == torch.float32

    def test_numpy_scalar_unsqueezed(self) -> None:
        out = parse_to_tensor(np.float32(7.0))
        assert out.shape == (1,)

    def test_device_honored(self) -> None:
        out = parse_to_tensor(1.0, device=torch.device("cpu"))
        assert out.device.type == "cpu"


# ---------------------------------------------------------------------------
# get_last_obs
# ---------------------------------------------------------------------------

class TestGetLastObs:
    def test_missing_final_observation_returns_none(self) -> None:
        assert get_last_obs({}, 1, (4,), torch.device("cpu")) is None

    def test_full_final_observation_stacked(self) -> None:
        obs0 = np.array([1.0, 2.0], dtype=np.float32)
        obs1 = np.array([3.0, 4.0], dtype=np.float32)
        info = {"final_observation": [obs0, obs1]}
        out = get_last_obs(info, 2, (2,), torch.device("cpu"))
        assert out is not None
        assert out.shape == (2, 2)
        torch.testing.assert_close(out, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_none_slot_zero_filled(self) -> None:
        obs0 = np.array([1.0, 2.0], dtype=np.float32)
        info = {"final_observation": [obs0, None]}
        out = get_last_obs(info, 2, (2,), torch.device("cpu"))
        assert out is not None
        assert out.shape == (2, 2)
        assert out[1].tolist() == [0.0, 0.0]

    def test_all_none_slots_still_tensor(self) -> None:
        info = {"final_observation": [None, None]}
        out = get_last_obs(info, 2, (3,), torch.device("cpu"))
        assert out is not None
        assert out.shape == (2, 3)
        assert torch.count_nonzero(out) == 0


# ---------------------------------------------------------------------------
# actions_limits
# ---------------------------------------------------------------------------

class TestActionsLimits:
    def test_box_action_space_returns_bounds(self) -> None:
        env = vectorize_env("Pendulum-v1", num_envs=1)
        agent = ActorCriticAgent(3, 1, is_discrete=False)
        result = actions_limits(env, agent)
        assert result is not None
        low, high = result
        assert np.asarray(low).shape == (1,)
        assert np.asarray(high).shape == (1,)
        assert (np.asarray(low) < np.asarray(high)).all()
        env.close()

    def test_discrete_action_space_returns_none(self) -> None:
        env = vectorize_env("CartPole-v1", num_envs=1)
        agent = ActorCriticAgent(4, 2, is_discrete=True)
        assert actions_limits(env, agent) is None
        env.close()
