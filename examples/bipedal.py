import torch
from zerorl.algorithms.ppo import easy_train_ppo
from zerorl.config import AlgoConfig, TrainConfig

config = TrainConfig(
    model_name="Bipedal",
    project_name="Bipedal-example",
)
config.device = torch.device("cpu")
algo_config = AlgoConfig()

trainer = easy_train_ppo("BipedalWalker-v3", config, algo_config, hidden_layer=128)
#trainer.train(use_tb=True)
trainer.test()
