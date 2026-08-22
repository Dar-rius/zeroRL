"""Configuration dataclasses for RL training.

Provides AlgoConfig (mutable hyperparameters) and TrainConfig (mutable
training settings with computed fields).
"""

import torch
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """Training configuration with computed fields.

    model_path and num_update are derived in __post_init__(); do not
    pass them to the constructor.

    Attributes:
        model_name: Model name (used in the saved file path).
        model_save_path: Directory for model checkpoints.
        device: PyTorch device (auto-detects CUDA).
        model_path: Computed as "{model_save_path}/{model_name}.pt".
        timestamp: Total environment timesteps for training.
        rollout_steps: Steps collected before each PPO update.
        num_update: Computed as timestamp // rollout_steps.
    """
    model_name: str
    project_name: str
    model_save_path: str = ".checkpoints"
    timestamp: int = 1_000_000
    rollout_steps: int = 2048
    num_envs: int = 1
    normalize: bool = False
    profile: bool = False

    device: torch.device = field(init=False)
    num_update: int = field(init=False)
    model_path: str = field(init=False)

    def __post_init__(self) -> None:
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = f"{self.model_save_path}/{self.model_name}.pt"
        self.num_update = self.timestamp // (self.rollout_steps * self.num_envs)
        if self.num_update <= 0:
            raise ValueError(f"num_update must be greater than 0, got {self.num_update}")


@dataclass(init=False)
class AlgoConfig:
    """Mutable algorithm hyperparameters for PPO and off-policy methods.

    Attributes:
        lr: Learning rate.
        gamma: Discount factor.
        batch_size: Minibatch size for PPO updates.
        gae_lambda: GAE lambda.
        clip_eps: PPO clipping range.
        ent_coef: Entropy bonus coefficient.
        value_coef: Value loss coefficient.
        epochs: PPO epochs per update.
        tau: Soft update coefficient (off-policy).
    """

    lr: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 64

    #For on-policy
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    value_coef: float = 0.5
    epochs: int = 10

    #For off-policy
    tau: float = 0.005

    def __init__(self, **kwargs):
        for key in self.__annotations__:
            setattr(self, key, getattr(self.__class__, key, None))

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> dict:
        """Return all hyperparameters as a dictionary."""
        return self.__dict__
