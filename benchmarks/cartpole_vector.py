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
# ==============g===========================
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
    total_target_timesteps = steps_per_rollout * 2

    # Warmup
    model.learn(total_timesteps=steps_per_rollout * 2, reset_num_timesteps=False)

    wall_times = []
    timesteps = []
    rewards = []
    
    start_time = time.time()
    steps_done = 0

    while steps_done < TOTAL_STEPS:
        total_target_timesteps += steps_per_rollout
        model.learn(total_timesteps=steps_done + steps_per_rollout, reset_num_timesteps=False)
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
# 3. CleanRL
# ==========================================
def benchmark_cleanrl():
    import random
    from torch.distributions.categorical import Categorical

    # TRY NOT TO MODIFY: seeding
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # env setup
    def make_env(env_id, seed):
        def thunk():
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.reset(seed=seed)
            return env
        return thunk

    envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, SEED + i) for i in range(NUM_ENVS)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    class Agent(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dim = 64
            self.extract_layer = nn.Sequential(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh()
            )
            self.actor = nn.Linear(self.hidden_dim, envs.single_action_space.n)
            self.critic = nn.Linear(self.hidden_dim, 1)

            self.apply(self._orthogonal_init)

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
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    agent = Agent().to(DEVICE)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((ROLLOUT_STEPS, NUM_ENVS) + envs.single_observation_space.shape).to(DEVICE)
    actions = torch.zeros((ROLLOUT_STEPS, NUM_ENVS) + envs.single_action_space.shape).to(DEVICE)
    logprobs = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(DEVICE)
    rewards = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(DEVICE)
    dones = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(DEVICE)
    values = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(DEVICE)

    # TRY NOT TO MODIFY: start the game
    next_obs, _ = envs.reset(seed=SEED)
    next_obs = torch.Tensor(next_obs).to(DEVICE)
    next_done = torch.zeros(NUM_ENVS).to(DEVICE)

    ep_rewards = []
    batch_size = ROLLOUT_STEPS * NUM_ENVS
    minibatch_size = BATCH_SIZE

    # --- Helper : une phase de rollout + update (Façon stricte CleanRL) ---
    def cleanrl_iteration():
        nonlocal next_obs, next_done

        for step in range(0, ROLLOUT_STEPS):
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(DEVICE).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(DEVICE), torch.Tensor(next_done).to(DEVICE)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ep_rewards.append(info["episode"]["r"])

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            advantages = torch.zeros_like(rewards).to(DEVICE)
            lastgaelam = 0
            for t in reversed(range(ROLLOUT_STEPS)):
                if t == ROLLOUT_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(N_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > CLIP_RANGE).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

    # --- Warmup ---
    for _ in range(2):
        cleanrl_iteration()

    # --- Benchmark loop ---
    wall_times = []
    timesteps = []
    rewards_data = []

    start_time = time.time()
    steps_done = 0
    steps_per_rollout = ROLLOUT_STEPS * NUM_ENVS

    while steps_done < TOTAL_STEPS:
        sync_gpu()
        cleanrl_iteration()
        sync_gpu()

        current_time = time.time() - start_time
        steps_done += steps_per_rollout

        if len(ep_rewards) > 0:
            mean_r = np.mean(ep_rewards[-10:])
        else:
            mean_r = -500.0

        wall_times.append(current_time)
        timesteps.append(steps_done)
        rewards_data.append(mean_r)

    envs.close()

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

    # ----- Graphique 1 : Wall-clock Time -----
    axes[0].plot(zerorl_data["wall_times"], zerorl_data["rewards"],
                 label='ZeroRL', color='#0052cc', linewidth=2.5)
    axes[0].plot(sb3_data["wall_times"], sb3_data["rewards"],
                 label='Stable-Baselines3', color='#d62728', linewidth=2.5, linestyle='--')
    axes[0].plot(cleanrl_data["wall_times"], cleanrl_data["rewards"],
                 label='CleanRL', color='#2ca02c', linewidth=2.5, linestyle='-.')

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
    axes[1].plot(cleanrl_data["timesteps"], cleanrl_data["rewards"],
                 label='CleanRL', color='#2ca02c', linewidth=2.5, linestyle='-.')

    axes[1].set_title(f'PPO on {ENV_ID} ({NUM_ENVS} Env) — Timesteps', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Timesteps', fontsize=12)
    axes[1].set_ylabel('Mean Episode Reward (last 10)', fontsize=12)
    axes[1].grid(True, linestyle=':', alpha=0.5)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    filename = f'benchmark_acrobot_vector.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nGraphique sauvegardé : {filename}")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print(f"Benchmark PPO — {ENV_ID} | {NUM_ENVS} Env | Device: {DEVICE.upper()} | Seed: {SEED}\n")

    print("→ Running CleanRL...")
    c_data = benchmark_cleanrl()
    print(f"  Finished in {c_data['wall_times'][-1]:.1f}s | Best reward: {max(c_data['rewards']):.1f}")

    print("→ Running ZeroRL...")
    z_data = benchmark_zerorl()
    print(f"  Finished in {z_data['wall_times'][-1]:.1f}s | Best reward: {max(z_data['rewards']):.1f}")

    print("\n→ Running Stable-Baselines3...")
    s_data = benchmark_sb3()
    print(f"  Finished in {s_data['wall_times'][-1]:.1f}s | Best reward: {max(s_data['rewards']):.1f}")

    plot_results(z_data, s_data, c_data)
