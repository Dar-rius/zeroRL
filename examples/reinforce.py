import torch
from zerorl.helpers.factory  import get_env, PolicyAgent, get_policy_buffer
from zerorl.config import TrainConfig, AlgoConfig
from zerorl.functions import get_obs_act
from zerorl.train import BaseTrain


#1. configure train and define agent
config = TrainConfig(project_name="reinforce_example", model_name="agent-reinforce", timestamp=1_000_000, num_envs=2, profile=True)
config.device = torch.device("cpu")
algo_config = AlgoConfig()
env = get_env("CartPole-v1", config.num_envs)
obs_dim, act_dim, obs_n, act_n, is_discrete = get_obs_act(env)
agent = PolicyAgent(obs_n, act_n, is_discrete)
buffer = get_policy_buffer(obs_dim, act_dim, config)


#2. Define your update function based on Reinforce algorithm
def reinforce_update(agent, buffer, optimizer, algo_config, scheduler=None, last_output=None): 
    data = buffer.get_all(reshape=True)
    rewards = data["reward"]
    total_size = rewards.shape[0]
    dones = data["done"]
    returns = torch.empty_like(rewards)
    mask = 1.0 - dones
    R = 0.0
    for step in reversed(range(total_size)):
        R = rewards[step] + algo_config.gamma * mask[step] * R 
        returns[step] = R
    
    global_losses = agent.get_action(data["state"], data["action"])
    loss = -(global_losses["log_prob"] * returns).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5) # Max grad norm
    optimizer.step()
    return {"loss": loss.detach()}


#3. Define Trainer and train agent
trainer = BaseTrain(agent=agent,
                    env=env,
                    buffer=buffer,
                    update_weights=reinforce_update,
                    config=config,
                    algo_config=algo_config)
trainer.train(use_tb=True)
trainer.test()
