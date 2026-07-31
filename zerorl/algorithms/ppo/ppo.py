"""Proximal Policy Optimization (PPO) trainer implementation.

Provides PPOTrainer, which computes GAE advantages and runs the clipped
surrogate loss optimization with linear learning rate decay.
Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

import torch
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from ...common import Buffer
from ...agent import BaseAgent, eval_action
from ...config import PPOConfig
from ..general import get_buffer_params_model


@torch.compile
def gae(rewards: Tensor,
        values: Tensor,
        last_value: Tensor,
        dones: Tensor,
        ppo_config: PPOConfig) -> tuple[Tensor, Tensor, Tensor]:
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
    delta = rewards + ppo_config.gamma * next_values * mask - values
    for step in reversed(range(total_size)):
        gae = delta[step] + ppo_config.gamma * ppo_config.gae_lambda * mask[step] * gae
        advantages[step] = gae

    returns = advantages + values
    return (returns, advantages, delta)


def ppo_loss(model: BaseAgent,
        params: dict,
        buffers: dict,
        states: Tensor,
        actions: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
        returns: Tensor,
        ppo_config: PPOConfig) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    logits, new_values = torch.func.functional_call(model, (params, buffers), (states,))
    dist = model.build_distribution(logits)
    new_log_probs, dist_entropy = eval_action(dist, actions)

    idx_adv = advantages.view(-1)
    idx_return = returns.view(-1)

    logratio = new_log_probs - old_log_probs
    ratio = torch.exp(logratio)

    surr1 = ratio * idx_adv
    surr2 = torch.clamp(ratio,
                        1.0 - ppo_config.clip_eps,
                        1.0 + ppo_config.clip_eps) * idx_adv

    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = nn.functional.mse_loss(new_values.view(-1), idx_return)
    entropy_loss = dist_entropy.mean()
    loss = policy_loss + \
            (ppo_config.value_coef * value_loss) - \
            (ppo_config.ent_coef * entropy_loss)
    return (loss, policy_loss, value_loss, entropy_loss)


def ppo(model: BaseAgent,
        ppo_config: PPOConfig,
        optimizer: Optimizer,
        buffer: Buffer,
        step: int,
        num_update: int,
        batch_size: int = 64,
        epochs: int = 10,
        device: str = "cpu"
        ) ->  tuple[Tensor, Tensor, Tensor, Tensor]:
    lr_decay(ppo_config.lr, optimizer, step, num_update)
    params, buffers = get_buffer_params_model(model)
    all_data = buffer.get_all()
    adv_norm = (all_data["adv"] - all_data["adv"].mean()) / (all_data["adv"].std() + 1e-8)
    returns = (all_data["returns"] - all_data["returns"].mean()) / (all_data["returns"].std() + 1e-8)

    dataset_size = all_data["actions"].size(0)
    num_batches_per_epoch = (dataset_size + batch_size - 1) // batch_size
    size_total = num_batches_per_epoch * epochs

    #Create storage
    epoch_losses = torch.zeros((size_total), dtype=torch.float32, device=device)
    epoch_pi_losses = torch.zeros((size_total), dtype=torch.float32, device=device)
    epoch_v_losses = torch.zeros((size_total), dtype=torch.float32, device=device)
    epoch_entropies = torch.zeros((size_total), dtype=torch.float32, device=device)
    index_loss = 0

    @torch.compile(mode="reduce-overhead")
    def ppo_backward(model: BaseAgent,
            params: dict,
            buffers: dict,
            states: Tensor,
            actions: Tensor,
            old_log_probs: Tensor,
            advantages: Tensor,
            returns: Tensor,
            ppo_config: PPOConfig) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        loss, policy_loss, value_loss, ent_loss = ppo_loss(model, params, buffers, states, actions,
                    old_log_probs, advantages, returns, ppo_config)
        loss.backward()
        return loss, policy_loss, value_loss, ent_loss

    @torch.compile
    def update():
        """Run a PPO update on collected rollout data.

        Normalizes advantages and returns, then runs multiple epochs of
        minibatch SGD with the clipped surrogate loss.

        Args:
            memory: Buffer containing rollout data.
            total_steps: Total training steps (LR decay denominator).
            step: Current training step (LR decay numerator).
            batch_size: Minibatch size.
            epochs: Number of passes over the data.

        Returns:
            Tuple of (total_loss, policy_loss, value_loss, entropy),
            each averaged over all minibatch updates.
        """
        nonlocal index_loss
        for _ in range(epochs):
            shuffle_index = torch.randperm(dataset_size, device=device)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = shuffle_index[start:end]
                if idx.numel() == 0:
                    continue  # Skip empty batches
                
                optimizer.zero_grad(set_to_none=True)
                loss, policy_loss, value_loss, entropy_loss = ppo_backward(model, params, buffers, all_data["state"][idx],
                                all_data["actions"][idx], all_data["old_log_probs"][idx], adv_norm[idx],
                                returns[idx], ppo_config)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_losses[index_loss] = loss.detach()
                epoch_pi_losses[index_loss] = policy_loss.detach()
                epoch_v_losses[index_loss] = value_loss.detach()
                epoch_entropies[index_loss] = entropy_loss.detach()
                index_loss += 1

    #calcul loss and update weights
    update()
    # Return average losses over all actual updates ([:index_loss] excludes
    # any unused pre-allocated entries)
    return (epoch_losses[:index_loss].mean(),
            epoch_pi_losses[:index_loss].mean(),
            epoch_v_losses[:index_loss].mean(),
            epoch_entropies[:index_loss].mean())
