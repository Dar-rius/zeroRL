"""One-call PPO quickstart.

Provides easy_train_ppo() which wires together agent, environment, buffer,
and PPO update into a ready-to-train BaseTrain instance.
"""

from typing import Callable
from torch import optim
from zerorl.helpers.factory import get_env, get_actor_critic_buffer, ActorCriticAgent
from zerorl.helpers.train import BaseTrain
from zerorl.algorithms.ppo import ppo_func, gae_compute
from zerorl.config import TrainConfig, AlgoConfig
from zerorl.helpers.agent import BaseAgent
from zerorl.helpers.env import BaseEnv
from zerorl.functions import get_obs_act

def easy_train_ppo(env_spec: str | Callable | BaseEnv, 
                    config: TrainConfig,
                    algo_config: AlgoConfig,
                    *,
                    base_agent: BaseAgent | None = None,
                    hidden_layer: int = 64,
                    optimizer: optim.Optimizer | None = None,
                    schedule_func: Callable[[int], float] | None = None,
                    seed: int = 22,
                    render_mode: str | None = None,
                   ):
    """Create and return a BaseTrain instance with PPO wiring.

    Automatically builds the environment, agent, buffer, and update function.

    Args:
        env_spec: Gymnasium env ID string, BaseEnv subclass/instance, or callable.
        config: Training configuration (device, paths, timesteps).
        algo_config: Algorithm hyperparameters (lr, gamma, clip_eps, etc.).
        hidden_layer: Hidden layer size for the default ActorCriticAgent.
        render_mode: Render mode for the environment.
        base_agent: Custom agent (overrides the default ActorCriticAgent).
        optimizer: Custom optimizer (overrides default Adam).
        schedule_func: Custom LR schedule function (overrides default linear decay).

    Returns:
        BaseTrain instance ready for .train() or .test().
    """
    env = get_env(env_spec, config.num_envs, render_mode)
    obs_dim, act_dim, n_obs, n_act, is_discrete = get_obs_act(env)

    if base_agent is not None:
        agent = base_agent
    else:
        agent = ActorCriticAgent(n_obs, n_act, is_discrete, hidden_layer) #type: ignore

    buffer = get_actor_critic_buffer(obs_dim, act_dim, config.rollout_steps, config.num_envs, config.device) #type: ignore

    def easy_update_weights(agent, buffer, scheduler, optimizer, last_output, algo_config):
        """Compute GAE advantages then run PPO update."""
        data = buffer.get_all()
        gae_compute(data["reward"], data["value"], last_output["value"], data["terminated"], buffer, algo_config)
        return ppo_func(agent, optimizer, buffer, algo_config, scheduler)

    train = BaseTrain(
            agent = agent,
            env = env,
            buffer = buffer,
            update_weights = easy_update_weights,
            config = config,
            algo_config = algo_config,
            optimizer = optimizer,
            schedule_func = schedule_func,
            seed = seed,
            render_mode = render_mode,
            )
    return train
