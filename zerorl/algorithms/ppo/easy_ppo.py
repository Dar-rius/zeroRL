from typing import Callable
from torch import optim
from zerorl.factory import get_env, get_actor_critic_buffer, ActorCriticAgent
from zerorl.train import BaseTrain
from zerorl.algorithms.ppo import ppo, gae_compute
from zerorl.config import TrainConfig, AlgoConfig
from zerorl.agent import BaseAgent
from zerorl.env import BaseEnv

def easy_train_ppo(config: TrainConfig,
                    algo_config: AlgoConfig,
                    env_id: str = "",
                    hidden_layer: int = 64,
                    render_mode: str | None = None,
                    base_agent: BaseAgent | None = None,
                    base_env: BaseEnv | None = None,
                    optimizer: optim.Optimizer | None = None,
                    schedule_func: Callable[[int], float] | None = None
                   ):
    if base_env is not None:
        env = base_env
    else:
        env = get_env(env_id, config.num_envs, render_mode)

    obs_dim = env.observation_space.shape[-1] #type: ignore
    act_dim = env.action_space.n #type: ignore[attr-defined]
    
    if base_agent is not None:
        agent = base_agent
    else: 
        agent = ActorCriticAgent(obs_dim, act_dim, hidden_layer)

    buffer = get_actor_critic_buffer(obs_dim, act_dim, config) #type: ignore

    #update weights function
    def easy_update_weights(agent, buffer, scheduler, optimizer, last_output, algo_config):
        data = buffer.get_all()
        gae_compute(data["reward"], data["value"], last_output["value"], data["done"], algo_config, buffer)
        return ppo(agent, optimizer, buffer, algo_config, scheduler, agent.device)

    train = BaseTrain(
            agent=agent,
            env=env,
            buffer=buffer,
            update_weights=easy_update_weights,
            config = config,
            algo_config = algo_config,
            optimizer = optimizer,
            schedule_func = schedule_func,
            render_mode = render_mode,
            )
    return train
