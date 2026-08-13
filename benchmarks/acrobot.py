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
BATCH_SIZE = 64
LR = 3e-4
GAMMA = 0.99
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def sync_gpu():
    if DEVICE == "cuda":
        torch.cuda.synchronize()

# ==========================================
# 1. ZERO RL
# ==========================================
from zerorl.algorithms.ppo.easy_ppo import easy_train_ppo
from zerorl.config import TrainConfig, AlgoConfig

def benchmark_zerorl():
    config = TrainConfig(model_name="bench", model_save_path="/tmp", project_name="bench", timestamp=TOTAL_STEPS)
    algo_config = AlgoConfig(lr=LR, gamma=GAMMA, batch_size=BATCH_SIZE, epochs=10)
    
    trainer = easy_train_ppo(config, algo_config, env_id=ENV_ID, hidden_layer=64)

    # Warmup
    state, _ = trainer.env.reset()
    trainer.rollout_phase(state)
    trainer.buffer.clear()

    times, rewards = [], []
    start_time = time.time()
    steps_done = 0
    
    while steps_done < TOTAL_STEPS:
        sync_gpu()
        last_output = trainer.rollout_phase(state)
        trainer.update_weights(trainer.agent, trainer.buffer, trainer.scheduler, trainer.optimizer, last_output, trainer.algo_config)
        trainer.buffer.clear()
        sync_gpu()
        
        times.append(time.time() - start_time)
        if len(trainer.episode_rewards) > 0:
            rewards.append(np.mean(trainer.episode_rewards[-10:]))
        else:
            rewards.append(-500)
        steps_done += ROLLOUT_STEPS
        
    return times, rewards


# ==========================================
# 2. STABLE BASELINES 3
# ==========================================
def benchmark_sb3():
    from stable_baselines3 import PPO
    
    env = gym.make(ENV_ID)
    model = PPO("MlpPolicy", env, n_steps=ROLLOUT_STEPS, batch_size=BATCH_SIZE, learning_rate=LR, gamma=GAMMA, 
                policy_kwargs=dict(net_arch=[64, 64], activation_fn=nn.Tanh), verbose=0, device=DEVICE)
    
    # Warmup
    model.learn(total_timesteps=ROLLOUT_STEPS*2)
    
    times, rewards = [], []
    start_time = time.time()
    steps_done = 0
    
    while steps_done < TOTAL_STEPS:
        model.learn(total_timesteps=ROLLOUT_STEPS)
        sync_gpu()
        times.append(time.time() - start_time)
        
        if len(model.ep_info_buffer) > 0:
            recent_episodes = list(model.ep_info_buffer)[-10:]
            rewards.append(np.mean([ep["r"] for ep in recent_episodes]))
        else:
            rewards.append(-500)
        steps_done += ROLLOUT_STEPS
        
    return times, rewards


# ==========================================
# PLOT GRAPH
# ==========================================
def plot_results(zerorl_data, sb3_data):
    plt.figure(figsize=(10, 6))
    
    if zerorl_data[0]:
        plt.plot(zerorl_data[0], zerorl_data[1], label='ZeroRL (Ours)', color='#0052cc', linewidth=3)
    if sb3_data[0]:
        plt.plot(sb3_data[0], sb3_data[1], label='Stable-Baselines3', color='#d62728', linewidth=3, linestyle='--')

    plt.title(f'PPO on {ENV_ID} : Time to Convergence', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Wall-clock Time (seconds)', fontsize=14)
    plt.ylabel('Mean Episode Reward', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12, loc='lower right', frameon=True, shadow=True)
    
    plt.tight_layout()
    filename = f'benchmark_{ENV_ID.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nGraphique sauvegardé sous '{filename}'")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"Démarrage du Benchmark sur {ENV_ID} ({DEVICE.upper()})...\n")
    
    print("1. Run of ZeroRL...")
    z_times, z_rewards = benchmark_zerorl()
    print(f" Finish in {z_times[-1]:.1f}s. Reward max: {max(z_rewards):.1f}")
    
    print("\n2. Run of Stable-Baselines3...")
    s_times, s_rewards = benchmark_sb3()
    print(f"    Finish in  {s_times[-1]:.1f}s. Reward max: {max(s_rewards):.1f}")
    
    plot_results((z_times, z_rewards), (s_times, s_rewards))
