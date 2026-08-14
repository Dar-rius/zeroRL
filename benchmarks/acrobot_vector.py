import time
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# BENCHMARK CONFIGURATION
# ==========================================
ENV_ID = "Acrobot-v1"
TOTAL_STEPS = 100_000
ROLLOUT_STEPS = 2048
NUM_ENVS = 4
BATCH_SIZE = 64
N_EPOCHS = 10
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

def sync_gpu():
    if DEVICE == "cuda":
        torch.cuda.synchronize()

# ==========================================
# 1. ZERO RL
# ==========================================
from zerorl.algorithms.ppo.easy_ppo import easy_train_ppo
from zerorl.config import TrainConfig, AlgoConfig

def benchmark_zerorl():
    config = TrainConfig(
        model_name="bench_zerorl",
        project_name="bench",
        model_save_path="/tmp/zerorl_bench",
        timestamp=TOTAL_STEPS,
        rollout_steps=ROLLOUT_STEPS,
        num_envs=NUM_ENVS
    )
    
    algo_config = AlgoConfig(
        lr=LR,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_eps=CLIP_RANGE,
        ent_coef=ENT_COEF,
        value_coef=VF_COEF,
        batch_size=BATCH_SIZE,
        epochs=N_EPOCHS,
        clip_vf = True
    )
    
    trainer = easy_train_ppo(
        config=config,
        algo_config=algo_config,
        env_id=ENV_ID,
        hidden_layer=64
    )

    obs, _ = trainer.env.reset(seed=SEED)
    trainer.state = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
    if trainer.state.dim() == 1:
        trainer.state = trainer.state.unsqueeze(0)

    # Warmup
    for _ in range(2):
        last_output = trainer.rollout_phase()
        trainer.update_weights(
            trainer.agent, trainer.buffer, trainer.scheduler,
            trainer.optimizer, last_output, trainer.algo_config
        )
        trainer.buffer.clear()

    wall_times = []
    timesteps = []
    rewards = []
    
    start_time = time.time()
    steps_done = 0
    steps_per_rollout = ROLLOUT_STEPS * NUM_ENVS

    while steps_done < TOTAL_STEPS:
        sync_gpu()
        
        last_output = trainer.rollout_phase()
        trainer.update_weights(
            trainer.agent, trainer.buffer, trainer.scheduler,
            trainer.optimizer, last_output, trainer.algo_config
        )
        trainer.buffer.clear()
        
        sync_gpu()
        
        if hasattr(trainer, "episode_rewards") and len(trainer.episode_rewards) > 0:
            mean_r = np.mean(trainer.episode_rewards[-10:])
        else:
            mean_r = -500.0
            
        current_time = time.time() - start_time
        steps_done += steps_per_rollout
        
        wall_times.append(current_time)
        timesteps.append(steps_done)
        rewards.append(mean_r)

    return {
        "wall_times": wall_times,
        "timesteps": timesteps,
        "rewards": rewards
    }


# ==========================================
# 2. STABLE BASELINES 3
# ==========================================
def benchmark_sb3():
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    # Création de 4 envs vectorisés pour SB3
    env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID)) for _ in range(NUM_ENVS)])
    
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=ROLLOUT_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LR,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        policy_kwargs=dict(
            net_arch=[64, 64],
            activation_fn=nn.Tanh,
            ortho_init=True
        ),
        verbose=0,
        device=DEVICE,
        seed=SEED
    )

    steps_per_rollout = ROLLOUT_STEPS * NUM_ENVS

    # Warmup
    model.learn(total_timesteps=steps_per_rollout * 2, reset_num_timesteps=False)

    wall_times = []
    timesteps = []
    rewards = []
    
    start_time = time.time()
    steps_done = 0

    while steps_done < TOTAL_STEPS:
        model.learn(total_timesteps=steps_per_rollout, reset_num_timesteps=False)
        sync_gpu()
        
        current_time = time.time() - start_time
        steps_done += steps_per_rollout
        
        if len(model.ep_info_buffer) > 0:
            recent = list(model.ep_info_buffer)[-10:]
            mean_r = np.mean([ep["r"] for ep in recent])
        else:
            mean_r = -500.0
            
        wall_times.append(current_time)
        timesteps.append(steps_done)
        rewards.append(mean_r)

    return {
        "wall_times": wall_times,
        "timesteps": timesteps,
        "rewards": rewards
    }


# ==========================================
# PLOTS (les deux)
# ==========================================
def plot_results(zerorl_data, sb3_data):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ----- Graphique 1 : Wall-clock Time -----
    axes[0].plot(zerorl_data["wall_times"], zerorl_data["rewards"], 
                 label='ZeroRL', color='#0052cc', linewidth=2.5)
    axes[0].plot(sb3_data["wall_times"], sb3_data["rewards"], 
                 label='Stable-Baselines3', color='#d62728', linewidth=2.5, linestyle='--')
    
    axes[0].set_title(f'PPO on {ENV_ID} ({NUM_ENVS} Env) — Wall-clock Time', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Wall-clock Time (seconds)', fontsize=12)
    axes[0].set_ylabel('Mean Episode Reward (last 10)', fontsize=12)
    axes[0].grid(True, linestyle=':', alpha=0.5)
    axes[0].legend(fontsize=11)

    # ----- Graphique 2 : Timesteps -----
    axes[1].plot(zerorl_data["timesteps"], zerorl_data["rewards"], 
                 label='ZeroRL', color='#0052cc', linewidth=2.5)
    axes[1].plot(sb3_data["timesteps"], sb3_data["rewards"], 
                 label='Stable-Baselines3', color='#d62728', linewidth=2.5, linestyle='--')
    
    axes[1].set_title(f'PPO on {ENV_ID} ({NUM_ENVS} Env) — Timesteps', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Timesteps', fontsize=12)
    axes[1].set_ylabel('Mean Episode Reward (last 10)', fontsize=12)
    axes[1].grid(True, linestyle=':', alpha=0.5)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    filename = f'benchmark_{ENV_ID.lower()}_{NUM_ENVS}envs.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nGraphique sauvegardé : {filename}")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print(f"Benchmark PPO — {ENV_ID} | {NUM_ENVS} Env | Device: {DEVICE.upper()} | Seed: {SEED}\n")
    
    print("→ Running ZeroRL...")
    z_data = benchmark_zerorl()
    print(f"  Finished in {z_data['wall_times'][-1]:.1f}s | Best reward: {max(z_data['rewards']):.1f}")
    
    print("\n→ Running Stable-Baselines3...")
    s_data = benchmark_sb3()
    print(f"  Finished in {s_data['wall_times'][-1]:.1f}s | Best reward: {max(s_data['rewards']):.1f}")
    
    plot_results(z_data, s_data)
