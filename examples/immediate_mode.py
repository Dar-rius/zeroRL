import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from zerorl.algorithms.ppo import ppo_func, gae_compute
from zerorl.helpers.factory import get_actor_critic_buffer, ActorCriticAgent
from zerorl.config import TrainConfig, AlgoConfig
from zerorl.logger import create_logger
from zerorl.functions import (processing_state,
                              parse_env_step,
                              to_env_action,
                              try_agent,
                              get_obs_act,
                              vectorize_env,
                              set_seed)


cfg = TrainConfig(model_name="Lunar-model", project_name="Lunar-example", num_envs=4)
cfg.device = torch.device("cpu")
algo_cfg = AlgoConfig(ent_coef=0.0)
seed = set_seed(42, cfg.num_envs)
env = vectorize_env("LunarLander-v3", num_envs = cfg.num_envs)
obs_dim, act_dim, obs_n, act_n, is_discrete = get_obs_act(env)
agent = ActorCriticAgent(obs_n, act_n, is_discrete)
buffer = get_actor_critic_buffer(obs_dim, act_dim, cfg)
optimizer = optim.Adam(agent.parameters(), lr=algo_cfg.lr, eps=1e-5)
scheduler = LambdaLR(optimizer, lambda step_: 1.0 - (step_ / cfg.num_update))
log = create_logger(cfg, algo_cfg, use_tb=True)
reward_tensor = torch.zeros(cfg.num_envs, device=cfg.device)
state, _ = env.reset(seed = seed)
state_tensor = processing_state(state)

for step in tqdm(range(cfg.num_update)):
    episodic_reward = []
    metrics = {}

    for _ in range(cfg.rollout_steps):
        with torch.inference_mode:
            outputs = agent.get_action(state_tensor)
        action  = to_env_action(outputs["action"], env)
        next_state, reward, terminated, truncated, _ = env.step(action)
        terminated = terminated | truncated
        outputs = {"next_state": next_state, "reward": reward,
                   "terminated": terminated, "truncated": truncated} 
        outputs = parse_env_step(outputs)
        next_state = outputs.pop("next_state")
        buffer.insert(state = state_tensor, **outputs)
        reward_tensor += outputs["reward"]
        finished = outputs["terminated"] > 0

        if finished.any():
            episodic_reward.extend(reward_tensor[finished].tolist())
            reward_tensor[finished] = 0.0
        state = next_state

    with torch.inference_mode():
        state = processing_state(state)
        last_output = agent.get_action(state)

    data = buffer.get_all()
    gae_compute(data["reward"], data["value"], last_output["value"], data["terminated"], buffer, algo_cfg)
    losses = ppo_func(agent, optimizer, buffer, algo_cfg, scheduler)
    if len(episodic_reward) > 0:
        recent = episodic_reward[-10:]
        mean_reward = float(np.mean(recent))
    else:
        mean_reward = 0.0
    metrics = {"train/mean_episodic_reward": mean_reward}
    for k, v in losses.items(): metrics[f"train/{k}"] = v
    log(metrics, step)
    buffer.clear()

env.close()
log.close()
try_agent(env, agent, cfg)
