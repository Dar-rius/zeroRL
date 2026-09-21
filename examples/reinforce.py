import torch
from zerorl.helpers.factory  import get_env, PolicyAgent, get_policy_buffer
from zerorl.config import TrainConfig, AlgoConfig
from zerorl.functions import get_obs_act
from zerorl.helpers.train import BaseTrain

#1. configure train and define agent
config = TrainConfig(project_name="reinforce_example", model_name="agent-reinforce", timestamp=1_000_000, num_envs=2, profile=True, debug=True)
config.device = torch.device("cpu")
algo_config = AlgoConfig()
env = get_env("MountainCar-v0", config.num_envs)
obs_dim, act_dim, obs_n, act_n, is_discrete = get_obs_act(env)
agent = PolicyAgent(obs_n, act_n, is_discrete).to(config.device)
buffer = get_policy_buffer(obs_dim, act_dim, config.rollout_steps, config.num_envs, config.device)

#2. Define your update function based on Reinforce algorithm
def reinforce_update(agent, buffer, optimizer, algo_config, scheduler, last_output=None): 
    data = buffer.get_all()
    rewards = data["reward"]
    total_size = rewards.shape[0]
    dones = data["terminated"]
    returns = torch.empty_like(rewards)
    mask = 1.0 - dones
    R = torch.zeros_like(rewards[0])
    for step in reversed(range(total_size)):
        R = rewards[step] + algo_config.gamma * mask[step] * R 
        returns[step] = R

    state = data["state"].flatten(0, 1)
    action = data["action"].flatten(0, 1)
    returns = returns.flatten(0, 1)
    global_losses = agent.get_action(state, action)
    loss = -(global_losses["log_prob"] * returns).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5) # Max grad norm
    optimizer.step()
    #scheduler.step()
    return {"loss": loss.detach()}

#3. Define Trainer and train agent
trainer = BaseTrain(agent=agent,
                    env=env,
                    buffer=buffer,
                    update_weights=reinforce_update,
                    config=config,
                    algo_config=algo_config)
trainer.train(use_tb=True)
trainer.try_agent()
