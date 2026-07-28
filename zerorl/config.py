"""Configuration dataclasses for PPO training.

Provides PPOConfig (immutable hyperparameters), TrainConfig (mutable
training settings with computed fields), and WandbConfig (logging).
"""

import torch
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PPOConfig:
    """Immutable PPO hyperparameters.

    Frozen to prevent accidental modification during training.

    Attributes:
        lr: Adam optimizer learning rate.
        gamma: Discount factor for future rewards.
        gae_lambda: GAE lambda parameter (bias-variance tradeoff).
        clip_eps: Clipping range for the PPO surrogate loss.
        ent_coef: Entropy bonus coefficient.
        value_coef: Value loss coefficient.
    """
    lr: float = 3e-5
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_eps: float = 0.1
    ent_coef: float = 0.01
    value_coef: float = 0.5


@dataclass
class TrainConfig:
    """Training configuration with computed fields.

    model_path and num_update are derived in __post_init__(); do not
    pass them to the constructor.

    Attributes:
        model_name: Model name (used in the saved file path).
        model_saved_path: Directory for model checkpoints.
        device: PyTorch device string (auto-detects CUDA).
        model_path: Computed as "{model_saved_path}/{model_name}.pt".
        timestamp: Total environment timesteps for training.
        batch_size: Minibatch size for PPO updates.
        rollout_steps: Steps collected before each PPO update.
        num_update: Computed as timestamp // rollout_steps.
    """
    model_name: str
    model_saved_path: str
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path: str = field(init=False)
    timestamp: int = 1_000_000
    batch_size: int = 64
    rollout_steps: int = 2048
    num_update: int = field(init=False)

    def __post_init__(self) -> None:
        self.model_path = f"{self.model_saved_path}/{self.model_name}.pt"
        self.num_update = self.timestamp // self.rollout_steps
