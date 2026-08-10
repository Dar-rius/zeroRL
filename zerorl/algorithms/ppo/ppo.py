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
from zerorl.errors import assert_agent_contract


def gae_compute(rewards: Tensor,
            values: Tensor,
            last_value: Tensor,
            dones: Tensor,
            hyper_params: AlgoConfig,
            buffer:Buffer | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Compute Generalized Advantage Estimation.

    Works backwards through the trajectory, accumulating TD errors
    with exponentially decaying weights.

    Args:
        rewards: Rewards for each timestep, shape (T,).
        values: Value estimates for each timestep, shape (T,).
        last_value: Bootstrap value for the state after the last step.
        dones: Episode termination flags, shape (T,). 1.0 = done.

    Returns:
        Tuple of (return, advantage, delta), each shape (T,).
    """
    rewards = rewards.reshape(rewards.shape[0], -1)
    values = values.reshape(values.shape[0], -1)
    dones = dones.reshape(dones.shape[0], -1)
    last_value = last_value.reshape(-1)
    num_envs = rewards.shape[1]
    gae = torch.zeros(num_envs, dtype=torch.float32, device=rewards.device)
    # Mask: 0.0 at episode boundaries (no bootstrapping across episodes)
    mask = 1.0 - dones
    next_values = torch.cat((values[1:], last_value.unsqueeze(0)), 0)
    total_size = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    delta = rewards + hyper_params.gamma * next_values * mask - values
    for step in reversed(range(total_size)):
        gae = delta[step] + hyper_params.gamma * hyper_params.gae_lambda * mask[step] * gae
        advantages[step] = gae
    returns = advantages + values
    if buffer is not None:
        buffer.data["advantage"][:buffer.size] = advantages
        buffer.data["return"][:buffer.size] = returns
    return (returns, advantages, delta)


def ppo_loss(
        agent: BaseAgent,
        params: dict,
        buffers: dict,
        states: Tensor,
        actions: Tensor,
        old_log_prob: Tensor,
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
        old_log_prob: Log probabilities from the old policy.
        advantages: GAE advantage estimates.
        returns: GAE return estimates.
        hyper_params: Algorithm hyperparameters.

    Returns:
        Dict with keys "loss", "policy_loss", "value_loss", "entropy_loss".
    """
    assert_agent_contract(agent,
                    {"foward": "Your agent should have the method `forward`",
                     "get_action": "Your agent should have the method `build_distribution`"})
    logits, new_values = torch.func.functional_call(agent, (params, buffers), (states,))
    dist = agent.build_distribution(logits) #type: ignore[operator]
    new_log_probs, dist_entropy = eval_action(dist, actions)

    idx_adv = advantages.view(-1)
    idx_return = returns.view(-1)

    logratio = new_log_probs - old_log_prob
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
        buffer: Buffer containing rollout data with keys "state", "action",
            "log_prob", "advantage", "return".
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
    flat_data = {key: tensor.reshape(-1, *tensor.shape[2:]) 
                for key, tensor in all_data.items()}
    adv_norm = (flat_data["advantage"] - flat_data["advantage"].mean()) / (flat_data["advantage"].std() + 1e-8)
    returns = (flat_data["return"] - flat_data["return"].mean()) / (flat_data["return"].std() + 1e-8)
    dataset_size = flat_data["action"].size(0)
    final_metrics: dict[str, Tensor] = {}

    @torch.compile(mode="reduce-overhead")
    def ppo_backward(agent: BaseAgent,
                    params: dict,
                    buffers: dict,
                    state: Tensor,
                    action: Tensor,
                    old_log_prob: Tensor,
                    advantage: Tensor,
                    return_: Tensor,
                    hyper_params: AlgoConfig) -> dict[str, Tensor]:
        global_losses = ppo_loss(agent, params, buffers, state, action,
                            old_log_prob, advantage, return_, hyper_params)
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
                
                torch.compiler.cudagraph_mark_step_begin()
                optimizer.zero_grad(set_to_none=True)
                global_losses = ppo_backward(agent, params, buffers, flat_data["state"][idx],
                                flat_data["action"][idx], flat_data["log_prob"][idx], adv_norm[idx],
                                returns[idx], hyper_params)
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()
                history.append({k: v.detach().clone() for k, v in global_losses.items()})
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
