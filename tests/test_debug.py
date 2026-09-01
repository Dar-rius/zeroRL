"""Unit tests for the debug module (zerorl.debug)."""

import pytest
import torch

from zerorl.debug import RLSanityError, check_tensor, check_shape, check_reward_scale


class TestRLSanityError:
    def test_is_exception_subclass(self) -> None:
        assert issubclass(RLSanityError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(RLSanityError):
            raise RLSanityError("test")


class TestCheckTensor:
    def test_finite_tensor_passes(self) -> None:
        check_tensor(torch.tensor([1.0, 2.0, 3.0]), "x", step=0)

    def test_nan_tensor_raises(self) -> None:
        with pytest.raises(RLSanityError):
            check_tensor(torch.tensor([1.0, float("nan"), 3.0]), "x", step=0)

    def test_inf_tensor_raises(self) -> None:
        with pytest.raises(RLSanityError):
            check_tensor(torch.tensor([1.0, float("inf"), 3.0]), "x", step=0)

    def test_neg_inf_tensor_raises(self) -> None:
        with pytest.raises(RLSanityError):
            check_tensor(torch.tensor([1.0, float("-inf"), 3.0]), "x", step=0)

    def test_error_contains_tensor_name(self) -> None:
        with pytest.raises(RLSanityError, match="'my_tensor'"):
            check_tensor(torch.tensor([float("nan")]), "my_tensor", step=0)

    def test_error_contains_step(self) -> None:
        with pytest.raises(RLSanityError, match="Step 42"):
            check_tensor(torch.tensor([float("inf")]), "x", step=42)

    def test_loss_name_gives_gradient_message(self) -> None:
        with pytest.raises(RLSanityError, match="[Gg]radient"):
            check_tensor(torch.tensor([float("nan")]), "loss", step=0)

    def test_log_prob_name_gives_distribution_message(self) -> None:
        with pytest.raises(RLSanityError, match="out of bounds|distribution"):
            check_tensor(torch.tensor([float("nan")]), "log_prob", step=0)

    def test_reward_name_gives_env_message(self) -> None:
        with pytest.raises(RLSanityError, match="[Ee]nvironment"):
            check_tensor(torch.tensor([float("nan")]), "reward", step=0)

    def test_generic_name_gives_generic_message(self) -> None:
        with pytest.raises(RLSanityError, match="mathematical operations"):
            check_tensor(torch.tensor([float("nan")]), "foo", step=0)

    def test_error_reports_first_bad_value(self) -> None:
        with pytest.raises(RLSanityError, match="nan"):
            check_tensor(torch.tensor([1.0, 2.0, 3.0, float("nan")]), "x", step=0)


class TestCheckShape:
    def test_matching_shape_passes(self) -> None:
        check_shape(torch.zeros(3, 4), (3, 4), "x", step=0)

    def test_mismatched_shape_raises(self) -> None:
        with pytest.raises(RLSanityError):
            check_shape(torch.zeros(3, 4), (3, 5), "x", step=0)

    def test_error_contains_expected_shape(self) -> None:
        with pytest.raises(RLSanityError, match=r"\(3, 5\)"):
            check_shape(torch.zeros(3, 4), (3, 5), "x", step=0)

    def test_error_contains_actual_shape(self) -> None:
        with pytest.raises(RLSanityError, match=r"torch\.Size\(\[3, 4\]\)"):
            check_shape(torch.zeros(3, 4), (3, 5), "x", step=0)


class TestCheckRewardScale:
    def test_normal_rewards_pass(self) -> None:
        check_reward_scale(torch.tensor([1.0, -5.0, 100.0]), step=0)

    def test_large_rewards_raises(self) -> None:
        with pytest.raises(RLSanityError):
            check_reward_scale(torch.tensor([1.0, 2e5, 3.0]), step=0)

    def test_boundary_exactly_1e5_passes(self) -> None:
        check_reward_scale(torch.tensor([1e5]), step=0)

    def test_error_contains_max_reward(self) -> None:
        with pytest.raises(RLSanityError, match="200000"):
            check_reward_scale(torch.tensor([2e5]), step=0)

    def test_negative_large_rewards_raises(self) -> None:
        with pytest.raises(RLSanityError):
            check_reward_scale(torch.tensor([-1e6]), step=0)
