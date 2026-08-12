import gymnasium as gym
import imageio.v2 as imageio
import torch
from zerorl.algorithms.ppo import easy_train_ppo
from zerorl.config import AlgoConfig, TrainConfig
from zerorl.factory import ActorCriticAgent
from zerorl.processing import NormMeanStd


def record_gif(agent, normalizer, device, path: str = "cartpole.gif",
               episodes: int = 3, max_steps: int = 500) -> None:
    """Roll out the trained agent and write an animated GIF."""
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    frames: list = []

    for _ in range(episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            state = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.inference_mode():
                action = agent.get_action(normalizer.normalize(state))["action"]
            obs, _, terminated, truncated, _ = env.step(int(action.item()))
            if terminated or truncated:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                break

    env.close()
    imageio.mimsave(path, frames, fps=30)
    print(f"GIF sauvegarde: {path} ({len(frames)} frames)")


def save_checkpoint(trainer, path: str) -> None:
    torch.save(
        {
            "agent": trainer.agent.state_dict(),
            "normalizer_mean": trainer.normalizer.mean.cpu(),
            "normalizer_var": trainer.normalizer.var.cpu(),
            "normalizer_count": trainer.normalizer.count,
        },
        path,
    )
    print(f"Checkpoint: {path}")


def load_and_record(ckpt_path: str, gif_path: str = "cartpole.gif") -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    agent = ActorCriticAgent(4, 2, is_discrete=True, hidden_dim=64)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    normalizer = NormMeanStd((4,), device=torch.device("cpu"))
    normalizer.mean = ckpt["normalizer_mean"]
    normalizer.var = ckpt["normalizer_var"]
    normalizer.count = ckpt["normalizer_count"]
    record_gif(agent, normalizer, torch.device("cpu"), path=gif_path)


config = TrainConfig(
    model_name="agent-carl",
    project_name="cartpole-example",
    model_save_path="checkpoints",
    timestamp=200_000,
)
config.device = torch.device("cpu")
algo_config = AlgoConfig()
ckpt_path = ".checkpoints/agent-carl.pt"

trainer = easy_train_ppo(config, algo_config, "CartPole-v1")
trainer.train(use_wandb=False, use_tb=True, save_model=False)
save_checkpoint(trainer, ckpt_path)
record_gif(trainer.agent, trainer.normalizer, config.device, path="cartpole.gif")

print("Courbes TensorBoard:")
print(f"  uv run tensorboard --logdir {config.model_save_path}/tensorboard")
print(f"Modele: {ckpt_path}")
print("GIF: cartpole.gif")
