import gymnasium as gym
import numpy as np
from torch import Tensor
from zerorl.env import BaseEnv
from typing import Callable


class VectorEnv(BaseEnv):
    """Universal vectorized environment wrapper.
    
    Accept either a registered Gymnasium env_id (string) or a custom
    ZeroRL env class (callable)
    """
    def __init__(self, env_spec: str | Callable, num_envs: int = 1, render_mode: str | None = None):
        super().__init__()
        def make_env_fn(seed:int) -> Callable:
            def _init():
                if isinstance(env_spec, str):
                    env = gym.make(env_spec, render_mode = render_mode)
                elif callable(env_spec):
                    env = env_spec()
                else:
                    raise ValueError("env_spec must be a string (Gymnasium ID) or a callable (Env class)")

                env.reset(seed=seed)
                return env
            return _init
    
        self._env = gym.vector.SyncVectorEnv([make_env_fn(i) for i in range(num_envs)])
        self.observation_space = self._env.observation_space
        self.action_space = getattr(self._env, "single_action_space", self._env.action_space)


    @property
    def auto_reset(self): return True

    
    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)


    def step(self, action: np.ndarray | Tensor):
        return self._env.step(action)
    
    
    def close(self):
        self._env.close()
