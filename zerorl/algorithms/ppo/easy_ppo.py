from typing import Callable
from gymnasium import spaces
from torch import optim
from zerorl.factory import get_env, get_actor_critic_buffer, ActorCriticAgent
from zerorl.train import BaseTrain
from zerorl.algorithms.ppo import ppo_func, gae_compute
from zerorl.config import TrainConfig, AlgoConfig
from zerorl.agent import BaseAgent
from zerorl.helpers import BaseEnv

def easy_train_ppo(config: TrainConfig,
                    algo_config: AlgoConfig,
                    env_id: str = "",
                    hidden_layer: int = 64,
                    render_mode: str | None = None,
                    base_agent: BaseAgent | None = None,
                    base_env: BaseEnv | None = None,
                    optimizer: optim.Optimizer | None = None,
                    schedule_func: Callable[[int], float] | None = None,
                   ):
    if base_env is not None:
        env = base_env
    else:
        env = get_env(env_id, config.num_envs, render_mode)
    
    if hasattr(env, "single_observation_space"):
        obs_dim = env.signle_observation_space
        act_dim = env.single_action_space
    else:
        obs_dim = env.observation_space
        act_dim = env.action_space
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
        gae_compute(data["reward"], data["value"], last_output["value"], data["done"], data["truncated"], data["final_value"], buffer, algo_config)
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
