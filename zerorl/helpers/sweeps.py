from typing import Callable

def run_wandb_sweep(sweep_config: dict, train_fn: Callable, project_name: str, num_trials: int = 10):
    try:
        import wandb
    except ImportError:
        raise ImportError("Wandb is not installted. Intall it with: pip install wandb")

    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
    wandb.agent(sweep_id, function = train_fn, count = num_trials)


def run_optuna_sweep(objective_fn:Callable,  direction: str = "maximize", num_trials: int = 10):
    try:
        import optuna
    except ImportError:
        raise ImportError("Optuna is not installed. Install it with: pip install optuna")

    study = optuna.create_study(direction = direction)
    study.optimize(objective_fn, n_trials = num_trials)
    return study.best_params
