"""Integration tests for Buffer (zerorl.common).

Tests end-to-end buffer workflows: full insert cycles, various
shape combinations, clear/refill reuse, and multi-key data flow.
"""

import pytest
import torch
from zerorl.common import Buffer


class TestBufferIntegration:
    """End-to-end integration tests simulating real RL training workflows."""

    def test_full_insert_cycle_with_get_all(self) -> None:
        """Fill buffer with multiple keys and verify get_all() returns all of them."""
        buf = Buffer(
            step=20,
            data={
                "state": (8,),
                "action": (4,),
                "old_log_probs": (),
                "returns": (),
                "adv": (),
                "reward": (),
                "value": (),
                "done": (),
            },
            device=torch.device("cpu"),
        )
        for i in range(20):
            buf.insert(
                state=torch.randn(8),
                action=torch.randn(4),
                old_log_probs=torch.tensor(float(i) * -0.1),
                returns=torch.tensor(float(i)),
                adv=torch.tensor(float(i) * 0.5),
                reward=torch.tensor(float(i) * 0.1),
                value=torch.tensor(float(i) * 0.2),
                done=torch.tensor(1.0 if i == 19 else 0.0),
            )
        result = buf.get_all()
        assert len(result) == 8
        assert result["state"].shape == (20, 8)
        assert result["action"].shape == (20, 4)
        assert result["old_log_probs"].shape == (20,)
        assert result["returns"].shape == (20,)
        assert result["adv"].shape == (20,)

    def test_buffer_scalar_actions(self) -> None:
        """Buffer with scalar action shape should produce a 1D actions tensor."""
        buf = Buffer(step=10, data={"state": (4,), "action": ()}, device=torch.device("cpu"))
        for i in range(10):
            buf.insert(state=torch.ones(4), action=torch.tensor(float(i)))
        result = buf.get_all()
        assert result["action"].shape == (10,)
        assert result["action"][5].item() == pytest.approx(5.0)

    def test_buffer_multidim_states(self) -> None:
        """Buffer should handle multi-dimensional states (e.g., images)."""
        buf = Buffer(step=5, data={"state": (3, 3), "action": ()}, device=torch.device("cpu"))
        state = torch.ones((3, 3))
        for i in range(5):
            buf.insert(state=state * i, action=torch.tensor(0))
        result = buf.get_all()
        assert result["state"].shape == (5, 3, 3)
        torch.testing.assert_close(result["state"][2], state * 2)

    def test_clear_and_refill(self) -> None:
        """Buffer should be fully reusable after clear()."""
        buf = Buffer(step=5, data={"state": (2,)}, device=torch.device("cpu"))
        for i in range(5):
            buf.insert(state=torch.tensor([float(i), float(i)]))
        buf.clear()
        assert buf.size == 0
        for i in range(3):
            buf.insert(state=torch.tensor([float(i + 10), float(i + 10)]))
        assert buf.size == 3
        result = buf.get_all()
        torch.testing.assert_close(result["state"][0], torch.tensor([10.0, 10.0]))
        torch.testing.assert_close(result["state"][2], torch.tensor([12.0, 12.0]))

    def test_buffer_with_different_state_action_shapes(self) -> None:
        """Buffer should handle various state and action dimension combinations."""
        # Small state, small action
        buf1 = Buffer(step=3, data={"state": (2,), "action": (1,)}, device=torch.device("cpu"))
        for i in range(3):
            buf1.insert(state=torch.tensor([float(i), float(i)]), action=torch.tensor([float(i)]))
        result1 = buf1.get_all()
        assert result1["state"].shape == (3, 2)
        assert result1["action"].shape == (3, 1)

        # Large state (image-like), scalar action
        buf2 = Buffer(step=3, data={"state": (64, 64, 3), "action": ()}, device=torch.device("cpu"))
        for i in range(3):
            buf2.insert(state=torch.zeros(64, 64, 3), action=torch.tensor(float(i)))
        result2 = buf2.get_all()
        assert result2["state"].shape == (3, 64, 64, 3)
        assert result2["action"].shape == (3,)

    def test_ppo_workflow_keys(self) -> None:
        """Buffer should support the exact key names used by ppo()."""
        buf = Buffer(
            step=32,
            data={
                "state": (4,),
                "actions": (),
                "old_log_probs": (),
                "adv": (),
                "returns": (),
                "value": (),
            },
            device=torch.device("cpu"),
        )
        for _ in range(32):
            buf.insert(
                state=torch.randn(4),
                actions=torch.tensor(0),
                old_log_probs=torch.tensor(-0.5),
                adv=torch.tensor(1.0),
                returns=torch.tensor(2.0),
                value=torch.tensor(0.5),
            )
        result = buf.get_all()
        assert "actions" in result
        assert "old_log_probs" in result
        assert "adv" in result
        assert "returns" in result
        assert "state" in result

    def test_large_buffer(self) -> None:
        """Buffer should handle typical PPO rollout sizes."""
        n = 2048
        buf = Buffer(
            step=n,
            data={"state": (4,), "action": (), "reward": (), "done": ()},
            device=torch.device("cpu"),
        )
        for i in range(n):
            buf.insert(
                state=torch.randn(4),
                action=torch.tensor(0),
                reward=torch.tensor(1.0),
                done=torch.tensor(0.0),
            )
        assert buf.size == n
        result = buf.get_all()
        assert result["state"].shape == (n, 4)
        assert result["reward"].shape == (n,)
