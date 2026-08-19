"""Proximal Policy Optimization (PPO) standalone functions.

Provides gae_compute(), ppo_loss(), and ppo() for computing GAE
advantages and running the clipped surrogate loss optimization.
Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

import torch
from typing import Callable
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from zerorl.buffer import Buffer
from zerorl.agent import BaseAgent, eval_action
from zerorl.config import AlgoConfig
from zerorl.functions import get_buffer_params_model, fast_compile
from zerorl.errors import assert_agent_contract


def gae_compute(rewards: Tensor,
            values: Tensor,
            last_value: Tensor,
            dones: Tensor,
            buffer:Buffer,
            algo_config: AlgoConfig):
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
    delta = rewards + algo_config.gamma * next_values * mask - values
    advantages = torch.empty_like(delta)
    for step in reversed(range(total_size)):
        gae = delta[step] + algo_config.gamma * algo_config.gae_lambda * mask[step] * gae
        advantages[step] = gae
    returns = advantages + values
    buffer.data["advantage"][:buffer.size] = advantages
    buffer.data["return"][:buffer.size] = returns


def ppo_loss(
        agent: BaseAgent,
        params: dict,
        buffers: dict,
        states: Tensor,
        actions: Tensor,
        old_log_prob: Tensor,
        old_values: Tensor,
        advantages: Tensor,
        returns: Tensor,
        ent_coef: float,
        value_coef: float,
        clip_eps: float,
        clip_vf: float,
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
        algo_config: Algorithm configuration. 

    Returns:
        Dict with keys "loss", "policy_loss", "value_loss", "entropy_loss".
    """
    logits, new_values = torch.func.functional_call(agent, (params, buffers), (states,))
    dist = agent.build_distribution(logits) #type: ignore[operator]
    new_log_probs, dist_entropy = eval_action(dist, actions)

    idx_adv = advantages.view(-1)
    idx_return = returns.view(-1)
    new_values = new_values.view(-1)
    old_values = old_values.view(-1)
    old_log_prob = old_log_prob.view(-1)

    logratio = new_log_probs - old_log_prob
    ratio = torch.exp(logratio)

    clip_eps = clip_eps
    surr1 = ratio * idx_adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * idx_adv
    policy_loss = -torch.min(surr1, surr2).mean()
    
    clip_vf = clip_vf
    if clip_vf:
        value_pred_clipped = old_values + (new_values - old_values).clamp(-clip_eps, clip_eps)
        value_loss  = torch.max((idx_return - new_values).pow(2), (value_pred_clipped - idx_return).pow(2)).mean() 
    else:
        value_loss = 0.5 * nn.functional.mse_loss(new_values, idx_return)

    entropy_loss = dist_entropy.mean()

    loss = policy_loss + \
            (value_coef * value_loss) - \
            (ent_coef * entropy_loss)
    return {'loss': loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss':entropy_loss}


def ppo_func(agent: BaseAgent,
        optimizer: Optimizer,
        buffer: Buffer,
        algo_config: AlgoConfig,
        scheduler: LambdaLR,
        *,
        ppo_loss_func: Callable[[BaseAgent, dict, dict, Tensor, Tensor,
                                 Tensor, Tensor, Tensor, Tensor, float,
                                 float, float, float], dict[str, Tensor]] = ppo_loss,
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
        algo_config: Algorithm  Configuration.
        scheduler: Learning rate scheduler (stepped once per call).
        batch_size: Minibatch size.
        epochs: Number of passes over the data.
        device: Torch device for computations.

    Returns:
        Dict of averaged loss metrics ("loss", "policy_loss", "value_loss", "entropy_loss").
    """
    assert_agent_contract(agent,
         {"forward": "Your agent should have the method `forward`",
          "build_distribution": "Your agent should have the method `build_distribution`"})

    params, buffers = get_buffer_params_model(agent)
    all_data = buffer.get_all()
    flat_data = {key: tensor.reshape(-1, *tensor.shape[2:])
                for key, tensor in all_data.items()}

    mb_advantages = flat_data["advantage"]
    adv_norm = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
    returns = flat_data["return"]

    #coefficient and eps for clipping
    value_coef = algo_config.value_coef
    ent_coef = algo_config.ent_coef
    clip_eps = algo_config.clip_eps
    clip_vf = getattr(algo_config, "clip_vf", False)

    dataset_size = flat_data["action"].size(0)
    final_metrics: dict[str, Tensor] = {}

    @fast_compile(mode="reduce-overhead") #type: ignore
    def ppo_backward(agent: BaseAgent,
                    params: dict,
                    buffers: dict,
                    state: Tensor,
                    action: Tensor,
                    old_log_prob: Tensor,
                    old_values: Tensor,
                    advantage: Tensor,
                    return_: Tensor) -> dict[str, Tensor]:
        global_losses = ppo_loss_func(agent, params, buffers, state, action,
                                old_log_prob, old_values, advantage, return_,
                                value_coef, ent_coef, clip_eps, clip_vf)
        loss_tensor = global_losses["loss"]
        loss_tensor.backward()
        return global_losses

    def update() -> list[dict[str, Tensor]]:
        """Run a PPO update on collected rollout data.

        Normalizes advantages and returns, then runs multiple epochs of
        minibatch SGD with the clipped surrogate loss.
        """
        history = []
        for _ in range(algo_config.epochs):
            shuffle_index = torch.randperm(dataset_size, device=device)
            for start in range(0, dataset_size, algo_config.batch_size):
                end = start + algo_config.batch_size
                idx = shuffle_index[start:end]
                optimizer.zero_grad(set_to_none=True)
                torch.compiler.cudagraph_mark_step_begin()
                global_losses: dict[str, Tensor] = ppo_backward(agent, params, buffers, flat_data["state"][idx],
                                flat_data["action"][idx], flat_data["log_prob"][idx],
                                flat_data["value"][idx], adv_norm[idx], returns[idx])
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                history.append({k: v.clone().detach() for k, v in global_losses.items()})
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
