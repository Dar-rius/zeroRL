"""Hyperparameter sweep helpers for wandb and optuna."""

from typing import Callable


def run_wandb_sweep(sweep_config: dict, train_fn: Callable, project_name: str, num_trials: int = 10):
    """Create a wandb sweep and run an agent for num_trials runs.

    Args:
        sweep_config: wandb sweep config dict (method, parameters, metric).
        train_fn: Train function invoked by the wandb agent per trial.
        project_name: wandb project name.
        num_trials: Number of agent runs.
    """
    try:
        import wandb
    except ImportError:
        raise ImportError("Wandb is not installted. Intall it with: pip install wandb")

    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
    wandb.agent(sweep_id, function = train_fn, count = num_trials)


def run_optuna_sweep(objective_fn:Callable,  direction: str = "maximize", num_trials: int = 10):
    """Run an optuna study optimizing objective_fn for num_trials trials.

    Args:
        objective_fn: Callable taking an optuna Trial and returning the
            metric to optimize (direction="maximize") or minimize.
        direction: "maximize" or "minimize".
        num_trials: Number of optimization trials.

    Returns:
        The study's best_params dict.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("Optuna is not installed. Install it with: pip install optuna")

    study = optuna.create_study(direction = direction)
    study.optimize(objective_fn, n_trials = num_trials)
    return study.best_params
