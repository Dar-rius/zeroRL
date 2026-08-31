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
    data = buffer.get_all()
    rewards = data["reward"].squeeze()
    dones = data["done"].squeeze()
    returns = []
    R = 0.0
    for r, d in zip(reversed(rewards.tolist()), reversed(dones.tolist())):
        if d: R = 0.0
        R = r + algo_config.gamma * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns, device=agent.device)
    _, log_probs = agent.get_action(data["state"], data["action"])
    loss = -(log_probs * returns).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5) # Max grad norm
    optimizer.step()
    return {"loss": loss}


#3. Define Trainer and train agent
trainer = BaseTrain(agent=agent,
                    env=env,
                    buffer=buffer,
                    update_weights=reinforce_update,
                    config=config,
                    algo_config=algo_config)
trainer.train(use_tb=True)
