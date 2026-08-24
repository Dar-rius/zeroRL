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
ENV_ID = "LunarLander-v3"
TOTAL_STEPS = 100_000
ROLLOUT_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE =  "cpu"
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
        project_name = "bench",
        model_save_path="/tmp/zerorl_bench",
        timestamp=TOTAL_STEPS,
        rollout_steps=ROLLOUT_STEPS,
        num_envs=1,
    )
    config.device = torch.device(DEVICE)
    
    algo_config = AlgoConfig(
        lr=LR,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_eps=CLIP_RANGE,
        ent_coef=ENT_COEF,
        value_coef=VF_COEF,
        batch_size=BATCH_SIZE,
        epochs=N_EPOCHS,
    )
    
    trainer = easy_train_ppo(
        env_spec=ENV_ID,
        config=config,
        algo_config=algo_config,
        hidden_layer=64
    )

    obs, _ = trainer.env.reset(seed=SEED)
    trainer.state = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
    print(trainer.env.action_space)
    print(trainer.buffer.data["action"].shape)

    # Warmup
    for _ in range(2):
        last_output = trainer.rollout_phase()
        trainer.update_weights(
            trainer.agent, trainer.buffer, trainer.scheduler,
            trainer.optimizer, last_output, trainer.algo_config)
        trainer.buffer.clear()

    wall_times = []
    timesteps = []
    rewards = []
    
    start_time = time.time()
    steps_done = 0

    while steps_done < TOTAL_STEPS:
        sync_gpu()
        last_output = trainer.rollout_phase()
        trainer.update_weights(
            trainer.agent, trainer.buffer, trainer.scheduler,
            trainer.optimizer, last_output, trainer.algo_config
        )
        trainer.buffer.clear()
        
        sync_gpu()
        
        # Reward
        if hasattr(trainer, "episode_rewards") and len(trainer.episode_rewards) > 0:
            mean_r = np.mean(trainer.episode_rewards[-10:])
        else:
            mean_r = -500.0
            
        current_time = time.time() - start_time
        steps_done += ROLLOUT_STEPS
        
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

    env = Monitor(gym.make(ENV_ID))
    
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
            ortho_init = True
        ),
        verbose=0,
        device=DEVICE,
        seed=SEED
    )

    # Warmup
    model.learn(total_timesteps=ROLLOUT_STEPS*2, reset_num_timesteps=False)

    wall_times = []
    timesteps = []
    rewards = []
    
    start_time = time.time()
    steps_done = 0
    total_target = ROLLOUT_STEPS

    while steps_done < TOTAL_STEPS:
        model.learn(total_timesteps=total_target, reset_num_timesteps=False)
        sync_gpu()
        
        current_time = time.time() - start_time
        steps_done += ROLLOUT_STEPS
        
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
# 3. CleanRL
# ==========================================
def benchmark_cleanrl():
    import random
    from torch.distributions.categorical import Categorical

    # --- Reproducibilité (Façon CleanRL) ---
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # --- Environnement ---
    env = gym.make(ENV_ID)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    hidden_dim = 64

    # --- Agent (Shared Layers ZeroRL + Initialisation CleanRL) ---
    class CleanRLAgent(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dim = hidden_dim

            # Feature Extractor partagé (Façon ZeroRL)
            self.extract_layer = nn.Sequential(
                nn.Linear(np.array(obs_shape).prod(), self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh()
            )
            # Têtes Actor et Critic
            self.actor = nn.Linear(self.hidden_dim, n_actions)
            self.critic = nn.Linear(self.hidden_dim, 1)

            self.apply(self._orthogonal_init)

        # Init façon CleanRL (qui correspond exactement aux gains de ZeroRL)
        def _orthogonal_init(self, module):
            if isinstance(module, nn.Linear):
                if module.out_features == self.hidden_dim:
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                elif module.out_features == 1:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                else:
                    nn.init.orthogonal_(module.weight, gain=0.01)

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        def get_value(self, x):
            x = self.extract_layer(x)
            return self.critic(x)

        def get_action_and_value(self, x, action=None):
            x = self.extract_layer(x)
            logits = self.actor(x)
            value = self.critic(x)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), value

    agent = CleanRLAgent().to(DEVICE)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    # --- Storage sur GPU ---
    obs_buf      = torch.zeros((ROLLOUT_STEPS,) + obs_shape, device=DEVICE)
    actions_buf  = torch.zeros(ROLLOUT_STEPS, dtype=torch.long, device=DEVICE)
    logprobs_buf = torch.zeros(ROLLOUT_STEPS, device=DEVICE)
    rewards_buf  = torch.zeros(ROLLOUT_STEPS, device=DEVICE)
    dones_buf    = torch.zeros(ROLLOUT_STEPS, device=DEVICE)
    values_buf   = torch.zeros(ROLLOUT_STEPS, device=DEVICE)

    # --- Init env ---
    next_obs, _ = env.reset(seed=SEED)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=DEVICE)
    next_done = torch.zeros(1, device=DEVICE)

    ep_rewards = []
    current_ep_reward = 0.0

    # --- Helper : une phase de rollout + update ---
    def rollout_and_update():
        nonlocal next_obs, next_done, current_ep_reward

        for step in range(ROLLOUT_STEPS):
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            next_obs_np, reward, terminated, truncated, _ = env.step(action.item())
            current_ep_reward += reward
            done = terminated or truncated
            if done:
                ep_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                next_obs_np, _ = env.reset()

            rewards_buf[step] = torch.tensor(reward, device=DEVICE).view(-1)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=DEVICE)
            next_done = torch.tensor(done, dtype=torch.float32, device=DEVICE).view(-1)

        # GAE (Façon stricte CleanRL)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            advantages = torch.zeros_like(rewards_buf).to(DEVICE)
            lastgaelam = 0
            for t in reversed(range(ROLLOUT_STEPS)):
                if t == ROLLOUT_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + GAMMA * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        # Flatten
        b_obs        = obs_buf.reshape((-1,) + obs_shape)
        b_actions    = actions_buf.reshape(-1)
        b_logprobs   = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns    = returns.reshape(-1)

        b_inds = np.arange(ROLLOUT_STEPS)
        for _ in range(N_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, ROLLOUT_STEPS, BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Normalisation des avantages (Default CleanRL)
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (CLIPPED - Default CleanRL)
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

    # --- Warmup ---
    for _ in range(2):
        rollout_and_update()

    # --- Benchmark loop ---
    wall_times = []
    timesteps = []
    rewards_data = []

    start_time = time.time()
    steps_done = 0

    while steps_done < TOTAL_STEPS:
        sync_gpu()
        rollout_and_update()
        sync_gpu()

        current_time = time.time() - start_time
        steps_done += ROLLOUT_STEPS

        if len(ep_rewards) > 0:
            mean_r = np.mean(ep_rewards[-10:])
        else:
            mean_r = -500.0

        wall_times.append(current_time)
        timesteps.append(steps_done)
        rewards_data.append(mean_r)

    env.close()

    return {
        "wall_times": wall_times,
        "timesteps": timesteps,
        "rewards": rewards_data
    }


# ==========================================
# PLOTS 
# ==========================================
def plot_results(zerorl_data, sb3_data, cleanrl_data):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ----- Graphic 1 : Wall-clock Time -----
    axes[0].plot(zerorl_data["wall_times"], zerorl_data["rewards"], 
                 label='ZeroRL', color='#0052cc', linewidth=2.5)
    axes[0].plot(sb3_data["wall_times"], sb3_data["rewards"], 
                 label='Stable-Baselines3', color='#d62728', linewidth=2.5, linestyle='--')
    axes[0].plot(cleanrl_data["wall_times"], cleanrl_data["rewards"], 
                 label='CleanRL', color='#2ca02c', linewidth=2.5, linestyle='-.')
    
    axes[0].set_title(f'PPO on {ENV_ID} — Wall-clock Time', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Wall-clock Time (seconds)', fontsize=12)
    axes[0].set_ylabel('Mean Episode Reward (last 10)', fontsize=12)
    axes[0].grid(True, linestyle=':', alpha=0.5)
    axes[0].legend(fontsize=11)

    # ----- Graphic 2 : Timesteps -----
    axes[1].plot(zerorl_data["timesteps"], zerorl_data["rewards"], 
                 label='ZeroRL', color='#0052cc', linewidth=2.5)
    axes[1].plot(sb3_data["timesteps"], sb3_data["rewards"], 
                 label='Stable-Baselines3', color='#d62728', linewidth=2.5, linestyle='--')
    axes[1].plot(cleanrl_data["timesteps"], cleanrl_data["rewards"], 
                 label='CleanRL', color='#2ca02c', linewidth=2.5, linestyle='-.')
    
    axes[1].set_title(f'PPO on {ENV_ID} — Timesteps', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Timesteps', fontsize=12)
    axes[1].set_ylabel('Mean Episode Reward (last 10)', fontsize=12)
    axes[1].grid(True, linestyle=':', alpha=0.5)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    filename = 'benchmark_lunar_single.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nGraphique sauvegardé : {filename}")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print(f"Benchmark PPO — {ENV_ID} | Device: {DEVICE.upper()} | Seed: {SEED}\n")
    
    print("→ Running ZeroRL...")
    z_data = benchmark_zerorl()
    print(f"  Finished in {z_data['wall_times'][-1]:.1f}s | Best reward: {max(z_data['rewards']):.1f}")

    print("\n→ Running CleanRL...")
    c_data = benchmark_cleanrl()
    print(f"  Finished in {c_data['wall_times'][-1]:.1f}s | Best reward: {max(c_data['rewards']):.1f}")
    
    print("\n→ Running Stable-Baselines3...")
    s_data = benchmark_sb3()
    print(f"  Finished in {s_data['wall_times'][-1]:.1f}s | Best reward: {max(s_data['rewards']):.1f}")
    
    plot_results(z_data, s_data, c_data)
