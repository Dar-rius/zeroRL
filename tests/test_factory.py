"""Unit tests for factory functions (zerorl.helpers.factory)."""

import pytest
import torch
from zerorl.helpers.factory import (
    get_env, get_actor_critic_buffer, ActorCriticAgent,
    get_policy_buffer, get_replay_buffer, PolicyAgent,
)
from zerorl.buffer import Buffer
from gymnasium.vector import SyncVectorEnv
from zerorl.config import TrainConfig


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tmp_config(tmp_path, device) -> TrainConfig:
    cfg = TrainConfig(model_name="factory_test", model_save_path=str(tmp_path), project_name="test")
    cfg.device = device
    cfg.rollout_steps = 8
    return cfg


class TestGetEnv:
    def test_get_env_returns_vector_env(self) -> None:
        env = get_env("CartPole-v1", 2, None)
        assert isinstance(env, SyncVectorEnv)
        env.close()

    def test_get_env_num_envs_propagates(self) -> None:
        env = get_env("CartPole-v1", 4, None)
        assert env.observation_space.shape == (4, 4)
        env.close()

    def test_get_env_render_mode_rgb_array(self) -> None:
        env = get_env("CartPole-v1", 1, "rgb_array")
        obs, _ = env.reset(seed=42)
        assert obs.shape[0] == 1
        env.close()

    def test_get_env_render_mode_none(self) -> None:
        env = get_env("CartPole-v1", 1, None)
        obs, _ = env.reset(seed=42)
        assert obs.shape[0] == 1
        env.close()


class TestGetActorCriticBuffer:
    def test_buffer_has_all_required_keys(self, tmp_config) -> None:
        buf = get_actor_critic_buffer((4,), (2,), tmp_config)
        expected_keys = {"state", "action", "reward", "done", "entropy",
                         "value", "return", "log_prob", "advantage",
                         "truncated"}
        assert set(buf.data.keys()) == expected_keys

    def test_buffer_shapes(self, tmp_config) -> None:
        buf = get_actor_critic_buffer((4,), (2,), tmp_config)
        assert buf.data["state"].shape == (8, 1, 4)
        assert buf.data["action"].shape == (8, 1, 2)
        assert buf.data["reward"].shape == (8, 1)
        assert buf.data["value"].shape == (8, 1)


class TestActorCriticAgent:
    def test_forward_returns_tuple(self, device) -> None:
        agent = ActorCriticAgent(input_dim=4, output_dim=2, is_discrete=True).to(device)
        state = torch.randn(1, 4, device=device)
        logits, value = agent.forward(state)
        assert logits.shape == (1, 2)
        assert value.shape == (1, 1)

    def test_get_action_returns_correct_keys(self, device) -> None:
        agent = ActorCriticAgent(input_dim=4, output_dim=2, is_discrete=True).to(device)
        state = torch.randn(1, 4, device=device)
        result = agent.get_action(state)
        assert set(result.keys()) == {"action", "log_prob", "entropy", "value"}

    def test_hidden_layer_default(self, device) -> None:
        agent = ActorCriticAgent(input_dim=4, output_dim=2, is_discrete=True, hidden_dim=128).to(device)
        state = torch.randn(1, 4, device=device)
        logits, _ = agent.forward(state)
        assert logits.shape == (1, 2)

    def test_get_action_value_is_squeezed_1d(self, device) -> None:
        agent = ActorCriticAgent(input_dim=4, output_dim=2, is_discrete=True).to(device)
        state = torch.randn(4, 4, device=device)
        result = agent.get_action(state)
        assert result["value"].dim() == 1
        assert result["value"].shape == (4,)


class TestGetPolicyBuffer:
    def test_has_correct_keys(self, tmp_config) -> None:
        buf = get_policy_buffer((4,), (), tmp_config)
        expected_keys = {"state", "action", "reward", "done", "truncated", "log_prob"}
        assert set(buf.data.keys()) == expected_keys

    def test_shapes(self, tmp_config) -> None:
        buf = get_policy_buffer((4,), (2,), tmp_config)
        assert buf.data["state"].shape == (8, 1, 4)
        assert buf.data["action"].shape == (8, 1, 2)
        assert buf.data["reward"].shape == (8, 1)


class TestGetReplayBuffer:
    def test_has_correct_keys(self, tmp_config) -> None:
        buf = get_replay_buffer((4,), (), tmp_config)
        expected_keys = {"state", "action", "reward", "done", "next_state", "truncated"}
        assert set(buf.data.keys()) == expected_keys

    def test_has_next_state(self, tmp_config) -> None:
        buf = get_replay_buffer((4,), (), tmp_config)
        assert "next_state" in buf.data
        assert buf.data["next_state"].shape == (8, 1)


class TestPolicyAgent:
    def test_discrete_forward_returns_logits(self, device) -> None:
        agent = PolicyAgent(input_dim=4, output_dim=2, is_discrete=True).to(device)
        state = torch.randn(1, 4, device=device)
        logits = agent.forward(state)
        assert logits.shape == (1, 2)

    def test_continuous_forward_returns_logits(self, device) -> None:
        agent = PolicyAgent(input_dim=3, output_dim=1, is_discrete=False).to(device)
        state = torch.randn(1, 3, device=device)
        logits = agent.forward(state)
        assert logits.shape == (1, 1)

    def test_discrete_get_action_keys(self, device) -> None:
        agent = PolicyAgent(input_dim=4, output_dim=2, is_discrete=True).to(device)
        state = torch.randn(1, 4, device=device)
        result = agent.get_action(state)
        assert set(result.keys()) == {"action", "log_prob"}

    def test_continuous_get_action_keys(self, device) -> None:
        agent = PolicyAgent(input_dim=3, output_dim=1, is_discrete=False).to(device)
        state = torch.randn(1, 3, device=device)
        result = agent.get_action(state)
        assert set(result.keys()) == {"action", "log_prob"}

    def test_discrete_builds_categorical(self, device) -> None:
        agent = PolicyAgent(input_dim=4, output_dim=2, is_discrete=True).to(device)
        state = torch.randn(1, 4, device=device)
        logits = agent.forward(state)
        dist = agent.build_distribution(logits)
        assert isinstance(dist, torch.distributions.Categorical)

    def test_continuous_builds_normal(self, device) -> None:
        agent = PolicyAgent(input_dim=3, output_dim=1, is_discrete=False).to(device)
        state = torch.randn(1, 3, device=device)
        logits = agent.forward(state)
        dist = agent.build_distribution(logits)
        assert isinstance(dist, torch.distributions.Normal)


class TestBufferDeviceProperty:
    def test_device_property(self, device) -> None:
        cfg = TrainConfig(model_name="test", model_save_path="/tmp", project_name="test")
        cfg.rollout_steps = 4
        cfg.num_envs = 1
        cfg.device = device
        buf = Buffer(data={"state": (4,)}, config=cfg)
        assert buf.device.type == device.type