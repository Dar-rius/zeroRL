"""Debug utilities for detecting NaN/Inf and shape mismatches in RL tensors."""

import torch
from torch import Tensor


class RLSanityError(Exception):
    """Custom exception for RL-specific silent failures."""
    pass


def check_tensor(tensor: Tensor, name: str, step: int):
    """Raise RLSanityError if tensor contains NaN or Inf."""
    if not torch.isfinite(tensor).all():
        bad_idx = torch.where(~torch.isfinite(tensor))
        first_bad_idx = bad_idx[0][0].item() if len(bad_idx) > 0 else 0
        val = tensor.flatten()[int(first_bad_idx)].item()
        msg = (f"\n[DEBUG FATAL] Step {step}: NaN or Inf detected in '{name}'!\n"
               f"First bad value at index {first_bad_idx}: {val}\n")
        if name == "loss": msg += "Cause: Gradients likely exploded. check learning rate or reward scale."
        elif name == "log_prob": msg += "Action is out of bounds or distribution std collapsed (check network weights)."
        elif name == "reward": msg += "Cause: Environment returned an invalid reward."
        else:
            msg += "check the mathematical operations that produced this tensor."
        raise RLSanityError(msg)

def check_shape(tensor: Tensor, expected_shape: tuple, name: str, step: int):
    """Raise RLSanityError if tensor shape doesn't match expected shape."""
    if tensor.shape != expected_shape:
        msg = (
                f"\n[DEBUG FATAL] Step {step}: Shape mismatch for '{name}'\n"
                f"Expected shape: {expected_shape}, but got: {tensor.shape}\n"
                f"Cause : Agent output or Env output has wrong dimensions. Check num_envs or action_dim."
                )
        raise RLSanityError(msg)

def check_reward_scale(tensor: torch.Tensor, step:int):
    """Raise RLSanityError if rewards exceed 100k (likely unnormalized)."""
    if tensor.abs().max() > 1e5:
        msg = (
            f"\n[DEBUG FATAL] Step {step}: Reward scale is abnormally high (> 100,000)\n"
            f"Max reward: {tensor.abs().max().item()}\n"
            f"Cause: Env might be returning raw unnormalized rewards. Consider Reward Scaling."
          )
        raise RLSanityError(msg)
