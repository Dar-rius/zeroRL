"""Unit tests for the Buffer class (rl_template.common).

Covers initialization, insertion, bounds checking, tensor conversion,
GAE return insertion, and buffer clearing/reuse.
"""

import numpy as np
import pytest
import torch
from rl_template.common import Buffer

# =============================================================================
# Test Initialization
# =============================================================================


class TestBufferInit:
    """Verify Buffer.__init__() creates arrays with correct shapes and types."""

    def test_creates_arrays_with_correct_shapes(self):
        """All 8 internal arrays should match the (step, shape) dimensions."""
        buf = Buffer(step=10, state_shape=(4,), action_shape=(2,))
        assert buf.states.shape == (10, 4)
        assert buf.actions.shape == (10, 2)
        assert buf.old_log_probs.shape == (10,)
        assert buf.returns.shape == (10,)
        assert buf.adv.shape == (10,)
        assert buf.rewards.shape == (10,)
        assert buf.values.shape == (10,)
        assert buf.dones.shape == (10,)

    def test_creates_arrays_with_correct_dtypes(self):
        """All arrays should be float32 for GPU compatibility."""
        buf = Buffer(step=5, state_shape=(3,))
        assert buf.states.dtype == np.float32
        assert buf.actions.dtype == np.float32
        assert buf.old_log_probs.dtype == np.float32
        assert buf.returns.dtype == np.float32
        assert buf.adv.dtype == np.float32
        assert buf.rewards.dtype == np.float32
        assert buf.values.dtype == np.float32
        assert buf.dones.dtype == np.float32

    def test_slice_starts_at_zero(self):
        """The insertion pointer should start at 0 (empty buffer)."""
        buf = Buffer(step=10, state_shape=(4,))
        assert buf.slice == 0

    def test_step_stored(self):
        """The capacity (step) should be stored for later reference."""
        buf = Buffer(step=42, state_shape=(2,))
        assert buf.step == 42

    def test_scalar_action_shape(self):
        """Discrete actions with action_shape=() should produce a 1D actions array."""
        buf = Buffer(step=5, state_shape=(8,), action_shape=())
        assert buf.actions.shape == (5,)


# =============================================================================
# Test Size Property
# =============================================================================


class TestBufferSize:
    """Verify the size property tracks the current element count."""

    def test_initially_zero(self):
        """Empty buffer should report size=0."""
        buf = Buffer(step=10, state_shape=(4,))
        assert buf.size == 0

    def test_increments_after_insert(self):
        """Size should increase by 1 after each insert."""
        buf = Buffer(step=10, state_shape=(4,))
        buf.insert(
            state=np.zeros(4),
            action=0,
            old_log_prob=0.0,
            reward=1.0,
            value=0.5,
            dones=0,
        )
        assert buf.size == 1

    def test_increments_correctly_multi_insert(self):
        """Size should match the number of inserts performed."""
        buf = Buffer(step=10, state_shape=(4,))
        for i in range(5):
            buf.insert(
                state=np.ones(4) * i,
                action=i,
                old_log_prob=float(i),
                reward=float(i),
                value=float(i),
                dones=0,
            )
        assert buf.size == 5


# =============================================================================
# Test Insert Method
# =============================================================================


class TestBufferInsert:
    """Verify Buffer.insert() stores data at the correct index."""

    def test_single_insert_state(self):
        """Inserted state should be stored at index 0."""
        buf = Buffer(step=10, state_shape=(4,))
        state = np.array([1.0, 2.0, 3.0, 4.0])
        buf.insert(
            state=state, action=0, old_log_prob=0.0, reward=0.0, value=0.0, dones=0
        )
        np.testing.assert_array_equal(buf.states[0], state)

    def test_single_insert_action(self):
        """Inserted action should be stored as a float in the actions array."""
        buf = Buffer(step=10, state_shape=(4,))
        buf.insert(
            state=np.zeros(4),
            action=3,
            old_log_prob=0.0,
            reward=0.0,
            value=0.0,
            dones=0,
        )
        assert buf.actions[0] == 3.0

    def test_single_insert_metadata(self):
        """Log prob, reward, value, and done flag should all be stored correctly."""
        buf = Buffer(step=10, state_shape=(4,))
        buf.insert(
            state=np.zeros(4),
            action=0,
            old_log_prob=-0.5,
            reward=2.5,
            value=1.0,
            dones=1,
        )
        assert buf.old_log_probs[0] == pytest.approx(-0.5)
        assert buf.rewards[0] == pytest.approx(2.5)
        assert buf.values[0] == pytest.approx(1.0)
        assert buf.dones[0] == 1.0

    def test_multiple_inserts_correct_indices(self):
        """Each insert should go to the next index; slice should track the count."""
        buf = Buffer(step=10, state_shape=(3,))
        for i in range(7):
            buf.insert(
                state=np.full(3, i),
                action=i,
                old_log_prob=float(i),
                reward=float(i * 10),
                value=float(i),
                dones=0,
            )
        np.testing.assert_array_equal(buf.states[6], np.full(3, 6))
        assert buf.rewards[6] == pytest.approx(60.0)
        assert buf.slice == 7


# =============================================================================
# Test Buffer Full (Bounds Checking)
# =============================================================================


class TestBufferInsertFull:
    """Verify that inserting into a full buffer raises ValueError."""

    def test_raises_valueerror_when_full(self):
        """After filling the buffer, one more insert should raise ValueError."""
        buf = Buffer(step=3, state_shape=(2,))
        for _ in range(3):
            buf.insert(
                state=np.zeros(2),
                action=0,
                old_log_prob=0.0,
                reward=0.0,
                value=0.0,
                dones=0,
            )
        with pytest.raises(ValueError, match="Buffer is full"):
            buf.insert(
                state=np.zeros(2),
                action=0,
                old_log_prob=0.0,
                reward=0.0,
                value=0.0,
                dones=0,
            )


# =============================================================================
# Test Get All (Tensor Conversion)
# =============================================================================


class TestBufferGetAll:
    """Verify Buffer.get_all() returns correct tensors for PPO training."""

    def test_returns_eight_tensors(self):
        """get_all() should return exactly 8 tensors."""
        buf = Buffer(step=5, state_shape=(4,))
        result = buf.get_all()
        assert len(result) == 8

    def test_tensor_dtypes(self):
        """Actions and dones should be long (integer); everything else float32."""
        buf = Buffer(step=5, state_shape=(4,))
        states, actions, old_log_probs, returns, adv, rewards, values, dones = (
            buf.get_all()
        )
        assert states.dtype == torch.float32
        assert actions.dtype == torch.float32
        assert old_log_probs.dtype == torch.float32
        assert returns.dtype == torch.float32
        assert adv.dtype == torch.float32
        assert rewards.dtype == torch.float32
        assert values.dtype == torch.float32
        assert dones.dtype == torch.long

    def test_tensor_shapes(self):
        """Tensor shapes should match the buffer dimensions."""
        buf = Buffer(step=5, state_shape=(4,))
        states, actions, old_log_probs, returns, adv, rewards, values, dones = (
            buf.get_all()
        )
        assert states.shape == (5, 4)
        assert actions.shape == (5,)
        assert old_log_probs.shape == (5,)
        assert returns.shape == (5,)
        assert adv.shape == (5,)
        assert rewards.shape == (5,)
        assert values.shape == (5,)
        assert dones.shape == (5,)

    def test_get_all_reflects_inserted_data(self):
        """Tensor values should match the data inserted into the buffer."""
        buf = Buffer(step=3, state_shape=(2,))
        buf.insert(
            state=np.array([1.0, 2.0]),
            action=1,
            old_log_prob=-0.1,
            reward=0.5,
            value=0.3,
            dones=0,
        )
        states, actions, old_log_probs, _, _, rewards, values, dones = buf.get_all()
        assert states[0].tolist() == [1.0, 2.0]
        assert actions[0].item() == 1
        assert old_log_probs[0].item() == pytest.approx(-0.1)
        assert rewards[0].item() == pytest.approx(0.5)
        assert values[0].item() == pytest.approx(0.3)
        assert dones[0].item() == 0


# =============================================================================
# Test Insert Returns (GAE Output)
# =============================================================================


class TestBufferInsertReturns:
    """Verify insert_returns() stores GAE-computed returns and advantages."""

    def test_updates_returns_and_adv(self):
        """insert_returns() should write to the returns and adv arrays."""
        buf = Buffer(step=5, state_shape=(4,))
        returns = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        adv = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        buf.insert_returns(returns, adv)
        np.testing.assert_array_equal(buf.returns, returns)
        np.testing.assert_allclose(buf.adv, adv, atol=1e-6)

    def test_overwrites_previous_returns(self):
        """Calling insert_returns() twice should overwrite the first values."""
        buf = Buffer(step=3, state_shape=(2,))
        buf.insert_returns(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        buf.insert_returns(np.array([9.0, 9.0, 9.0]), np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(buf.returns, np.array([9.0, 9.0, 9.0]))
        np.testing.assert_array_equal(buf.adv, np.array([1.0, 1.0, 1.0]))


# =============================================================================
# Test Clear (Buffer Reuse)
# =============================================================================


class TestBufferClear:
    """Verify clear() resets the buffer for reuse."""

    def test_resets_slice_to_zero(self):
        """After clear(), the slice pointer and size should both be 0."""
        buf = Buffer(step=10, state_shape=(4,))
        for i in range(5):
            buf.insert(
                state=np.ones(4),
                action=0,
                old_log_prob=0.0,
                reward=0.0,
                value=0.0,
                dones=0,
            )
        buf.clear()
        assert buf.slice == 0
        assert buf.size == 0

    def test_allows_reuse_after_clear(self):
        """After clear(), new inserts should start from index 0."""
        buf = Buffer(step=3, state_shape=(2,))
        buf.insert(
            state=np.array([1.0, 1.0]),
            action=0,
            old_log_prob=0.0,
            reward=0.0,
            value=0.0,
            dones=0,
        )
        buf.insert(
            state=np.array([2.0, 2.0]),
            action=1,
            old_log_prob=0.0,
            reward=0.0,
            value=0.0,
            dones=0,
        )
        buf.clear()
        buf.insert(
            state=np.array([9.0, 9.0]),
            action=2,
            old_log_prob=0.0,
            reward=0.0,
            value=0.0,
            dones=0,
        )
        assert buf.size == 1
        np.testing.assert_array_equal(buf.states[0], np.array([9.0, 9.0]))
        assert buf.actions[0] == 2.0
