"""Unit tests for factory functions (zerorl.factory)."""

import pytest
import torch
import torch.nn as nn
from zerorl.factory import get_env, get_actor_critic_buffer, ActorCriticAgent
from zerorl.vector_env import VectorEnv
from zerorl.buffer import Buffer
from zerorl.config import TrainConfig
from zerorl.agent import BaseAgent


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
        assert isinstance(env, VectorEnv)
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
                         "value", "adv", "return", "log_prob", "advantage"}
        assert set(buf.data.keys()) == expected_keys

    def test_buffer_shapes(self, tmp_config) -> None:
        buf = get_actor_critic_buffer((4,), (2,), tmp_config)
        assert buf.data["state"].shape == (8, 1, 4)
        assert buf.data["action"].shape == (8, 1, 2)
        assert buf.data["reward"].shape == (8, 1)
        assert buf.data["value"].shape == (8, 1)


class TestActorCriticAgent:
    def test_forward_returns_tuple(self, device) -> None:
        agent = ActorCriticAgent(input_layer=4, output_layer=2).to(device)
        state = torch.randn(1, 4, device=device)
        logits, value = agent.forward(state)
        assert logits.shape == (1, 2)
        assert value.shape == (1, 1)

    def test_get_action_returns_correct_keys(self, device) -> None:
        agent = ActorCriticAgent(input_layer=4, output_layer=2).to(device)
        state = torch.randn(1, 4, device=device)
        result = agent.get_action(state)
        assert set(result.keys()) == {"action", "log_prob", "entropy", "value"}

    def test_hidden_layer_default(self, device) -> None:
        agent = ActorCriticAgent(input_layer=4, output_layer=2, hidden_layer=128).to(device)
        state = torch.randn(1, 4, device=device)
        logits, _ = agent.forward(state)
        assert logits.shape == (1, 2)