import os
import numpy as np
from torch import Tensor
from typing import Callable
from zerorl.config import AlgoConfig, TrainConfig

try:
    import wandb
except ImportError:
    wandb = None #type: ignore

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None #type: ignore


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
