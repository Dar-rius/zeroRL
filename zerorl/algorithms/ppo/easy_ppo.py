from typing import Callable
from gymnasium import spaces
from torch import optim
from zerorl.factory import get_env, get_actor_critic_buffer, ActorCriticAgent
from zerorl.train import BaseTrain
from zerorl.algorithms.ppo import ppo_func, gae_compute
from zerorl.config import TrainConfig, AlgoConfig
from zerorl.agent import BaseAgent
from zerorl.helpers import BaseEnv
from zerorl.functions import get_obs_act

def easy_train_ppo(env_spec: str | Callable | BaseEnv, 
                    config: TrainConfig,
                    algo_config: AlgoConfig,
                    hidden_layer: int = 64,
                    render_mode: str | None = None,
                    base_agent: BaseAgent | None = None,
                    optimizer: optim.Optimizer | None = None,
                    schedule_func: Callable[[int], float] | None = None,
                   ):
    env = get_env(env_spec, config.num_envs, render_mode)
    obs_dim, act_dim = get_obs_act (env)
    is_discrete = isinstance(act_dim, spaces.Discrete)

    if is_discrete:
        act_n = act_dim.n #type: ignore
        act_shape = ()
    else:
        act_n = act_dim.shape[0] #type: ignore
        act_shape = (act_n, ) #type: ignore
    
    if base_agent is not None:
        agent = base_agent
    else:
        agent = ActorCriticAgent(obs_dim.shape[-1], act_n, is_discrete, hidden_layer) #type: ignore

    buffer = get_actor_critic_buffer(obs_dim.shape[-1], act_shape, config) #type: ignore

    #update weights function
    def easy_update_weights(agent, buffer, scheduler, optimizer, last_output, algo_config):
        data = buffer.get_all()
        gae_compute(data["reward"], data["value"], last_output["value"], data["done"], buffer, algo_config)
        return ppo_func(agent, optimizer, buffer, algo_config, scheduler, device=agent.device)

    train = BaseTrain(
            agent = agent,
            env = env,
            buffer = buffer,
            update_weights = easy_update_weights,
            config = config,
            algo_config = algo_config,
            optimizer = optimizer,
            schedule_func = schedule_func,
            render_mode = render_mode,
            )
    return train
