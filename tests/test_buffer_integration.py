"""Integration tests for Buffer (rl_template.common).

Tests end-to-end buffer workflows: full insert cycles, various
shape combinations, clear/refill reuse, and GAE data flow.
"""

import numpy as np
import pytest
from rl_template.common import Buffer


class TestBufferIntegration:
    """End-to-end integration tests simulating real RL training workflows."""

    def test_full_insert_cycle_with_get_all(self):
        """Fill buffer, insert GAE returns, and verify tensor shapes.

        Simulates a full rollout -> GAE -> PPO data preparation cycle.
        """
        buf = Buffer(
            step=20,
            state_shape=(8,),
            action_shape=(4,),
            extra_shapes={"action_mask": (2,)},
        )
        for i in range(20):
            state = np.random.randn(8).astype(np.float32)
            action = np.random.randint(0, 4)
            buf.insert(
                state=state,
                action=action,
                old_log_prob=np.random.randn(),
                reward=np.random.randn(),
                value=np.random.randn(),
                dones=int(i == 19),
                action_mask=np.random.randn(2).astype(np.float32),
            )
        returns = np.random.randn(20).astype(np.float32)
        adv = np.random.randn(20).astype(np.float32)
        buf.insert_returns(returns, adv)

        result = buf.get_all()
        assert len(result) == 9
        states, actions, old_log_probs, ret, a, rewards, values, dones, extra = result
        assert states.shape == (20, 8)
        assert actions.shape == (20, 4)
        assert ret.shape == (20,)
        assert a.shape == (20,)
        assert extra["action_mask"].shape == (20, 2)

    def test_buffer_scalar_actions(self):
        """Buffer with action_shape=() should produce a 1D actions tensor."""
        buf = Buffer(step=10, state_shape=(4,), action_shape=())
        for i in range(10):
            buf.insert(
                state=np.ones(4),
                action=i,
                old_log_prob=0.0,
                reward=float(i),
                value=0.0,
                dones=0,
            )
        states, actions, _, _, _, rewards, _, _, _ = buf.get_all()
        assert actions.shape == (10,)
        assert rewards[5].item() == pytest.approx(5.0)

    def test_buffer_multidim_states(self):
        """Buffer should handle multi-dimensional states (e.g., images)."""
        buf = Buffer(step=5, state_shape=(3, 3), action_shape=())
        state = np.ones((3, 3), dtype=np.float32)
        for i in range(5):
            buf.insert(
                state=state * i,
                action=0,
                old_log_prob=0.0,
                reward=0.0,
                value=0.0,
                dones=0,
            )
        states, _, _, _, _, _, _, _, _ = buf.get_all()
        assert states.shape == (5, 3, 3)
        np.testing.assert_array_equal(states[2].numpy(), state * 2)

    def test_clear_and_refill(self):
        """Buffer should be fully reusable after clear().

        New data should start at index 0 after clearing.
        """
        buf = Buffer(step=5, state_shape=(2,))
        for i in range(5):
            buf.insert(
                state=np.array([float(i), float(i)]),
                action=i,
                old_log_prob=0.0,
                reward=0.0,
                value=0.0,
                dones=0,
            )
        buf.clear()
        assert buf.size == 0
        for i in range(3):
            buf.insert(
                state=np.array([float(i + 10), float(i + 10)]),
                action=i,
                old_log_prob=0.0,
                reward=0.0,
                value=0.0,
                dones=0,
            )
        assert buf.size == 3
        states, _, _, _, _, _, _, _, _ = buf.get_all()
        np.testing.assert_array_equal(states[0].numpy(), [10.0, 10.0])
        np.testing.assert_array_equal(states[2].numpy(), [12.0, 12.0])

    def test_insert_returns_then_get_all(self):
        """insert_returns() data should be faithfully returned by get_all()."""
        buf = Buffer(step=3, state_shape=(2,))
        buf.insert(
            state=np.array([1.0, 2.0]),
            action=0,
            old_log_prob=-0.1,
            reward=1.0,
            value=0.5,
            dones=0,
        )
        buf.insert(
            state=np.array([3.0, 4.0]),
            action=1,
            old_log_prob=-0.2,
            reward=2.0,
            value=0.6,
            dones=0,
        )
        buf.insert(
            state=np.array([5.0, 6.0]),
            action=0,
            old_log_prob=-0.3,
            reward=3.0,
            value=0.7,
            dones=1,
        )
        returns = np.array([1.5, 2.5, 3.5])
        adv = np.array([1.0, 2.0, 3.0])
        buf.insert_returns(returns, adv)
        _, _, _, ret, a, _, _, _, _ = buf.get_all()
        np.testing.assert_allclose(ret.numpy(), returns)
        np.testing.assert_allclose(a.numpy(), adv)

    def test_buffer_with_different_state_action_shapes(self):
        """Buffer should handle various state and action dimension combinations."""
        # Small state, small action
        buf1 = Buffer(step=3, state_shape=(2,), action_shape=(1,))
        buf1.insert(
            state=np.array([1.0, 2.0]),
            action=np.array([0.0]),
            old_log_prob=0.0,
            reward=0.0,
            value=0.0,
            dones=0,
        )
        s1, a1, _, _, _, _, _, _, _ = buf1.get_all()
        assert s1.shape == (3, 2)
        assert a1.shape == (3, 1)

        # Large state (image-like), scalar action
        buf2 = Buffer(step=3, state_shape=(64, 64, 3), action_shape=())
        buf2.insert(
            state=np.zeros((64, 64, 3)),
            action=0,
            old_log_prob=0.0,
            reward=0.0,
            value=0.0,
            dones=0,
        )
        s2, a2, _, _, _, _, _, _, _ = buf2.get_all()
        assert s2.shape == (3, 64, 64, 3)
        assert a2.shape == (3,)
