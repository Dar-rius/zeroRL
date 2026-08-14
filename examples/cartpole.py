import torch
from zerorl.algorithms.ppo import easy_train_ppo
from zerorl.config import AlgoConfig, TrainConfig


config = TrainConfig(model_name="agent-carl", project_name="cartpole-example")
config.device = torch.device("cpu")
algo_config = AlgoConfig()

trainer = easy_train_ppo(config, algo_config, "CartPole-v1")
trainer.train(use_wandb=True)
