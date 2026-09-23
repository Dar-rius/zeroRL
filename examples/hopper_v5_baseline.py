"""Short Gymnasium Hopper-v5 baseline — validates zeroRL PPO on a standard env."""

import torch

from zerorl.algorithms.ppo import easy_train_ppo
from zerorl.config import AlgoConfig, TrainConfig

config = TrainConfig(
    model_name="Hopper-v5-cpu",
    project_name="hopper_v5_baseline",
    timestamp=2_000_000,
    num_envs=8,
    rollout_steps=512,
    normalize=True,
    profile=True,
)
config.device = torch.device("cpu")
algo_config = AlgoConfig(
    lr=3e-4,
    ent_coef=0.0,
    epochs=10,
    batch_size=512,
    gae_lambda=0.95,
    clip_eps=0.2,
    value_coef=0.5,
)

if __name__ == "__main__":
    trainer = easy_train_ppo("Hopper-v5", config, algo_config, seed=42)
    trainer.train(use_wandb=True, use_tb=True, save_model=True)
    trainer.try_agent(iterations=2, gif_path="hopper_v5_baseline")
