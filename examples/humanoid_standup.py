import torch
from zerorl.algorithms.ppo import easy_train_ppo
from zerorl.config import AlgoConfig, TrainConfig

config = TrainConfig(
    model_name="HumanoidStandup",
    project_name="HumanoidStandup",
    timestamp=10_000_000,
    rollout_steps=512,
    num_envs=20,
    normalize=True,
    profile=True,
)
config.device = torch.device("cpu")
algo_config = AlgoConfig(
    lr=2.55673e-5,
    batch_size=32,
    gae_lambda=0.9,
    clip_eps=0.3,
    ent_coef=0.0,
    value_coef=0.43,
    epochs=20,
)

trainer = easy_train_ppo("HumanoidStandup-v5", config, algo_config, hidden_layer=256)
torch.nn.init.constant_(trainer.agent.log_std, -2.0)
trainer.train(use_tb=True, save_model=True)
trainer.test(iterations=2, gif_path="humanoid_standup")
