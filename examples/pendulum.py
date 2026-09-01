import torch
from zerorl.algorithms.ppo import easy_train_ppo
from zerorl.config import AlgoConfig, TrainConfig

config = TrainConfig(
    model_name="Pendulum",
    project_name="Pendulum-example",
    timestamp = 1_000_000,
    num_envs = 4
)
config.device = torch.device("cpu")
algo_config = AlgoConfig(ent_coef=0.0)

trainer = easy_train_ppo("Pendulum-v1", config, algo_config)
trainer.train(use_tb=True)
trainer.test()
