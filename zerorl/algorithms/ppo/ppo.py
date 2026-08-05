"""Proximal Policy Optimization (PPO) standalone functions.

Provides gae_compute(), ppo_loss(), and ppo() for computing GAE
advantages and running the clipped surrogate loss optimization.
Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

import torch
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from zerorl.common import Buffer
from zerorl.agent import BaseAgent, eval_action
from zerorl.config import AlgoConfig
from zerorl.function import get_buffer_params_model


def gae_compute(rewards: Tensor,
        values: Tensor,
        last_value: Tensor,
        dones: Tensor,
        hyper_params: AlgoConfig) -> tuple[Tensor, Tensor, Tensor]:
    """Compute Generalized Advantage Estimation.

    Works backwards through the trajectory, accumulating TD errors
    with exponentially decaying weights.

    Args:
        rewards: Rewards for each timestep, shape (T,).
        values: Value estimates for each timestep, shape (T,).
        last_value: Bootstrap value for the state after the last step.
        dones: Episode termination flags, shape (T,). 1.0 = done.

    Returns:
        Tuple of (returns, advantages, deltas), each shape (T,).
    """
    gae = torch.zeros(1, dtype=torch.float32, device=rewards.device)
    # Mask: 0.0 at episode boundaries (no bootstrapping across episodes)
    mask = 1.0 - dones
    next_values = torch.cat((values[1:], last_value), 0)
    total_size = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    delta = rewards + hyper_params.gamma * next_values * mask - values
    for step in reversed(range(total_size)):
        gae = delta[step] + hyper_params.gamma * hyper_params.gae_lambda * mask[step] * gae
        advantages[step] = gae

    returns = advantages + values
    return (returns, advantages, delta)


def ppo_loss(
        agent: BaseAgent,
        params: dict,
        buffers: dict,
        states: Tensor,
        actions: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
        returns: Tensor,
        hyper_params: AlgoConfig
        ) -> dict[str, Tensor]:
    """Compute PPO clipped surrogate loss, value loss, and entropy bonus.

    Args:
        agent: The policy network.
        params: Named parameters dict from get_buffer_params_model().
        buffers: Named buffers dict from get_buffer_params_model().
        states: Batch of observations.
        actions: Batch of actions taken.
        old_log_probs: Log probabilities from the old policy.
        advantages: GAE advantage estimates.
        returns: GAE return estimates.
        hyper_params: Algorithm hyperparameters.

    Returns:
        Dict with keys "loss", "policy_loss", "value_loss", "entropy_loss".
    """
    logits, new_values = torch.func.functional_call(agent, (params, buffers), (states,))
    dist = agent.build_distribution(logits)
    new_log_probs, dist_entropy = eval_action(dist, actions)

    idx_adv = advantages.view(-1)
    idx_return = returns.view(-1)

    logratio = new_log_probs - old_log_probs
    ratio = torch.exp(logratio)

    surr1 = ratio * idx_adv
    surr2 = torch.clamp(ratio,
                        1.0 - hyper_params.clip_eps,
                        1.0 + hyper_params.clip_eps) * idx_adv

    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = nn.functional.mse_loss(new_values.view(-1), idx_return)
    entropy_loss = dist_entropy.mean()
    loss = policy_loss + \
            (hyper_params.value_coef * value_loss) - \
            (hyper_params.ent_coef * entropy_loss)
    return {'loss': loss, 
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss':entropy_loss}


def ppo(agent: BaseAgent,
        optimizer: Optimizer,
        buffer: Buffer,
        hyper_params: AlgoConfig,
        scheduler: LambdaLR,
        batch_size: int,
        epochs: int,
        device: torch.device = torch.device("cpu")
        ) ->  dict[str, Tensor]:
    """Run a full PPO update on collected rollout data.

    Normalizes advantages and returns, runs minibatch SGD epochs with
    clipped surrogate loss and gradient clipping, then steps the scheduler.

    Args:
        agent: The policy network.
        optimizer: Optimizer for the agent parameters.
        buffer: Buffer containing rollout data with keys "state", "actions",
            "old_log_probs", "adv", "returns".
        hyper_params: Algorithm hyperparameters.
        scheduler: Learning rate scheduler (stepped once per call).
        batch_size: Minibatch size.
        epochs: Number of passes over the data.
        device: Torch device for computations.

    Returns:
        Dict of averaged loss metrics ("loss", "policy_loss", "value_loss", "entropy_loss").
    """
    params, buffers = get_buffer_params_model(agent)
    all_data = buffer.get_all()
    adv_norm = (all_data["adv"] - all_data["adv"].mean()) / (all_data["adv"].std() + 1e-8)
    returns = (all_data["returns"] - all_data["returns"].mean()) / (all_data["returns"].std() + 1e-8)
    dataset_size = all_data["actions"].size(0)
    final_metrics: dict[str, Tensor] = {}

    @torch.compile(mode="reduce-overhead")
    def ppo_backward(agent: BaseAgent,
            params: dict,
            buffers: dict,
            states: Tensor,
            actions: Tensor,
            old_log_probs: Tensor,
            advantages: Tensor,
            returns: Tensor,
            hyper_params: AlgoConfig) -> dict[str, Tensor]:
        global_losses = ppo_loss(agent, params, buffers, states, actions,
                            old_log_probs, advantages, returns, hyper_params)
        global_losses["loss"].backward()
        return global_losses

    def update() -> list[dict[str, Tensor]]:
        """Run a PPO update on collected rollout data.

        Normalizes advantages and returns, then runs multiple epochs of
        minibatch SGD with the clipped surrogate loss.
        """
        history = []
        for _ in range(epochs):
            shuffle_index = torch.randperm(dataset_size, device=device)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = shuffle_index[start:end]
                if idx.numel() == 0:
                    continue  # Skip empty batches
                
                optimizer.zero_grad(set_to_none=True)
                global_losses = ppo_backward(agent, params, buffers, all_data["state"][idx],
                                all_data["actions"][idx], all_data["old_log_probs"][idx], adv_norm[idx],
                                returns[idx], hyper_params)
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()
                history.append({k: v.detach() for k, v in global_losses.items()})
        return history

    # Compute losses and update weights
    history = update()
    scheduler.step()
    # Return average losses over all actual updates ([:index_loss] excludes
    # any unused pre-allocated entries)
    if history:
        keys = history[0].keys()
        for key in keys:
            stacked = torch.stack([h[key] for h in history])
            final_metrics[key] = stacked.mean()
    return final_metrics
