import os
import time
import torch
import numpy as np
from typing import Callable
from fucntools import wraps
from dataclasses import dataclass
from torch import Tensor
from zerorl.config import AlgoConfig, TrainConfig

try:
    import wandb
except ImportError:
    wandb = None #type: ignore

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None #type: ignore


# Visualizer (Wandb and TensorBoard)
def create_logger(config: TrainConfig, algo_config: AlgoConfig, *, use_wandb:bool = False, use_tb:bool = False) -> Callable:
    os.path.join(config.model_save_path)
    if use_wandb:
        if wandb is None:
            raise ImportError("`Wandb` is not installed. Install it with: pip install wandb")
        wandb.init(project=config.project_name,
                   config={"Train Configs": algo_config .__dict__,
                           "Hyper Paramters": algo_config.__dict__})

    if use_tb:
        if SummaryWriter is None:
            raise ImportError("`TensorBoard` is not installed. Install it with: pip install wandb")
        tb_log_dir = os.path.join(config.model_save_path)
        tb_writer = SummaryWriter(tb_log_dir)
    clean_metrics: dict[str, float | np.ndarray] = {}


    def log(metrics:dict, step:int):
        for k, v in metrics.items():
            if isinstance(v, Tensor):
                clean_metrics[k] = float(v.item()) if v.numel() == 1 else v.detach().cpu().numpy()
            else:
                clean_metrics[k] = float(v)

        if use_wandb:
            wandb.log(clean_metrics, step=step)

        if use_tb:
            for key, value in clean_metrics.items():
                tb_writer.add_scalar(key, value, step)


    def finish():
        if wandb is not None: wandb.finish()
        if tb_writer is not None: tb_writer.close()

    log.finish =  finish 
    return log


# profiler
"""Profiling metrics captured during a training step."""
@dataclass
class ProfileMetrics:
    name: str
    duration_ms: float
    vram_peak_gb: float

# Profiler is a decorator
def profiler(name:str, is_cuda:bool = False):
    def profile_func(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs) -> tuple:
            if is_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t_start = time.perf_counter()
            result = func(*args, **kwargs)
            
            if is_cuda:
                torch.cuda.synchronize()
                t_end = time.perf_counter()
                vram_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            else:
                t_end = time.perf_counter()
                vram_peak_gb = 0.0

            duration_ms = (t_end - t_start) * 1000
            return (result, ProfileMetrics(name, duration_ms, vram_peak_gb))
        return wrapper
    return profile_func
