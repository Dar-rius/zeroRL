"""Configuration dataclasses for PPO training.

Provides PPOConfig (immutable hyperparameters), TrainConfig (mutable
training settings with computed fields), and WandbConfig (logging).
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
    timestamp: int = 1_000_000
    rollout_steps: int = 2048
    device: torch.device = field(init=False)

    num_update: int = field(init=False)
    model_path: str = field(init=False)

    def __post_init__(self) -> None:
        self.device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = f"{self.model_saved_path}/{self.model_name}.pt"
        self.num_update = self.timestamp // self.rollout_steps


@dataclass(init=False)
class AlgoConfig:
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

    def to_dict(self): self.__dict__
