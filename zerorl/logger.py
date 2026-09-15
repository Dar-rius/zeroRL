import os
import warnings
import time
import torch
import numpy as np
from typing import Callable
from functools import wraps
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
    if use_wandb:
        if wandb is None:
            raise ImportError("`Wandb` is not installed. Install it with: pip install wandb")
        wandb.init(project=config.project_name,
                   config={"Train Configs": algo_config .__dict__,
                           "Hyper Paramters": algo_config.__dict__})

    if use_tb:
        if SummaryWriter is None:
            raise ImportError("`TensorBoard` is not installed. Install it with: pip install tensorboard")
        tb_log_dir = os.path.join(config.model_save_path, f"tensorboard_{config.model_name}")
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


    def close():
        if use_wandb: wandb.finish()
        if use_tb: tb_writer.close()

    log.close = close
    return log


# profiler
"""Profiling metrics captured during a training step."""
@dataclass
class ProfileMetrics:
    name: str
    duration_ms: float
    vram_peak_gb: float

# Profiler is a decorator
def profile(name:str, is_cuda:bool = False):
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


@dataclass
class PhaseMetrics:
    rollout_ms: float = 0.0
    update_ms: float = 0.0
    total_ms: float = 0.0
    fps: float = 0.0
    vram_peak_gb: float = 0.0
    vram_allocated_gb: float = 0.0
    ram_mb: float = 0.0

class PhaseProfiler:
    def __init__(self, config: TrainConfig, *, is_cuda: bool = False):
        self.is_cuda = is_cuda
        self.total_rollout = config.rollout_steps * config.num_envs
        self.metrics = PhaseMetrics()
        self._current_start = 0.0


    def start_phase(self):
        if self.is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._current_start = time.perf_counter()


    def track(self, phase_name: str):
        profiler = self
        class PhaseContext:
            def __enter__(self):
                if profiler.is_cuda: torch.cuda.synchronize()
                profiler._phase_start = time.perf_counter()

            def __exit__(self, exc_type, exc_val, exc_tb):
                if profiler.is_cuda: torch.cuda.synchronize()
                phase_end = time.perf_counter()
                duration = (phase_end - profiler._phase_start) * 1000
                setattr(profiler.metrics, f"{phase_name}_ms", duration)
        return PhaseContext()


    def end_phase(self):
        try:
            import psutil
            ram_bytes = psutil.Process().memory_info().rss
            self.metrics.ram_mb = ram_bytes / (1024 ** 2)
        except ImportError:
            warnings.warn("Profiles are running but they are unable to capture the state of ram, install psutil")

        if self.is_cuda:
            self.metrics.vram_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            self.metrics.vram_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        end_time = time.perf_counter()
        self.metrics.total_ms = (end_time - self._current_start) * 1000

            # FPS
        if self.metrics.rollout_ms > 0:
            self.metrics.fps = self.total_rollout / (self.metrics.total_ms / 1000)
        return self.metrics
