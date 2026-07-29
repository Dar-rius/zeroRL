"""Proximal Policy Optimization (PPO) trainer implementation.

Provides PPOTrainer, which computes GAE advantages and runs the clipped
surrogate loss optimization with linear learning rate decay.
Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

import torch
from torch import nn
from torch import Tensor, optim
from ...common import Buffer
from ...agent import BaseAgent
from ...config import PPOConfig


class PPOTrainer:
    """PPO trainer handling advantage computation and policy updates.

    Maintains an Adam optimizer and applies linear LR decay over training.

    Args:
        model: Neural network with a get_action(state, action) method.
        lr: Initial learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda (bias-variance tradeoff).
        clip_eps: PPO clipping parameter.
        value_coef: Value loss coefficient.
        ent_coef: Entropy bonus coefficient.
    """

    def __init__(self,
                 model: BaseAgent,
                 ppo_config: PPOConfig,
                 ):
       
        self.model = model
        self.ppo_config = ppo_config
        self.optimizer = optim.Adam(model.parameters(), lr=self.ppo_config.lr)
        self.mse_loss = nn.MSELoss()

    def compute_gae(self,
                    rewards: Tensor,
                    values: Tensor,
                    last_value: Tensor,
                    dones: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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
        gae = torch.zeros(1, dtype=torch.float32)
        # Mask: 0.0 at episode boundaries (no bootstrapping across episodes)
        mask = 1.0 - dones
        next_values = torch.cat((values[1:], last_value), 0)
        print(next_values.shape)
        total_size = rewards.shape[0]
        advantages = torch.zeros_like(rewards)

        delta = rewards + self.ppo_config.gamma * next_values * mask - values

        for step in reversed(range(total_size)):
            gae = delta[step] + self.ppo_config.gamma * self.ppo_config.gae_lambda * mask[step] * gae
            advantages[step] = gae
            print(gae)

        returns = advantages + values
        return (returns, advantages, delta)

    def lr_decay(self, lr: float, total_steps: int, step: int):
        """Apply linear learning rate decay from lr to 0.

        Args:
            lr: Base learning rate.
            total_steps: Total number of training steps.
            step: Current training step.
        """
        frac = 1.0 - (step / total_steps)
        current_lr = lr * frac
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

    def update(self, memory: Buffer, total_steps: int, step: int,
               batch_size: int = 64, epochs: int = 10, device:str="cpu") -> tuple:
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
        self.lr_decay(self.ppo_config.lr, total_steps, step)

        states, actions, old_log_probs, returns, adv, _, _, _, _ = memory.get_all(device)

        advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        dataset_size = actions.size(0)
        num_batches_per_epoch = (dataset_size + batch_size - 1) // batch_size
        size_total = num_batches_per_epoch * epochs

        #Create storage
        epoch_losses = torch.zeros((size_total), dtype=torch.float32, device=device)
        epoch_pi_losses = torch.zeros((size_total), dtype=torch.float32, device=device)
        epoch_v_losses = torch.zeros((size_total), dtype=torch.float32, device=device)
        epoch_entropies = torch.zeros((size_total), dtype=torch.float32, device=device)
        index_loss = 0

        for _ in range(epochs):
            shuffle_index = torch.randperm(dataset_size, device=device)

            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = shuffle_index[start:end]
                if idx.numel() == 0:
                    continue  # Skip empty batches

                idx_adv = advantages[idx].view(-1)
                idx_return = returns[idx].view(-1)
                
                _, new_log_probs, dist_entropy, new_values = self.model.get_action(
                    states[idx],
                    actions[idx]
                )

                logratio = new_log_probs - old_log_probs[idx]
                ratio = torch.exp(logratio)

                surr1 = ratio * idx_adv
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_config.clip_eps,
                                    1.0 + self.ppo_config.clip_eps) * idx_adv

                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = self.mse_loss(new_values.view(-1), idx_return)

                entropy_loss = dist_entropy.mean()

                loss = policy_loss + \
                        (self.ppo_config.value_coef * value_loss) - \
                        (self.ppo_config.ent_coef * entropy_loss)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_losses[index_loss] = loss.detach()
                epoch_pi_losses[index_loss] = policy_loss.detach()
                epoch_v_losses[index_loss] = value_loss.detach()
                epoch_entropies[index_loss] = entropy_loss.detach()
                index_loss += 1

        # Return average losses over all actual updates ([:index_loss] excludes
        # any unused pre-allocated entries)
        return (epoch_losses[:index_loss].mean().item(),
                epoch_pi_losses[:index_loss].mean().item(),
                epoch_v_losses[:index_loss].mean().item(),
                epoch_entropies[:index_loss].mean().item())
