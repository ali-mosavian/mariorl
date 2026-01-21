"""Dreamer V3 Learner for model-based RL.

Implements the Learner protocol for Dreamer V3:
1. World Model Training: reconstruction, dynamics, reward prediction
2. Behavior Learning: actor-critic trained on imagined trajectories

V3 simplifications vs V2:
- Categorical KL with simple free bits (no balancing, no scheduler)
- Symlog MSE for reconstruction (no SSIM, no spatial weighting)
- Symlog for reward/value predictions (scale invariance)
- Percentile-based return normalization
- No auxiliary collapse-prevention losses
"""

from typing import Any
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from mario_rl.models import DreamerModel
from mario_rl.models.dreamer import symexp
from mario_rl.models.dreamer import symlog
from mario_rl.mcts.protocols import ValueAdapter
from mario_rl.mcts.protocols import PolicyAdapter
from mario_rl.mcts.protocols import WorldModelAdapter


def categorical_kl_loss(
    posterior_logits: Tensor,
    prior_logits: Tensor | None = None,
    free_bits: float = 1.0,
) -> Tensor:
    """Compute KL divergence between categorical distributions with free bits.

    Args:
        posterior_logits: (batch, num_cat, num_classes) - encoder output
        prior_logits: (batch, num_cat, num_classes) - prior (uniform if None)
        free_bits: Minimum KL per categorical (prevents posterior collapse)

    Returns:
        KL loss (scalar)
    """
    posterior = F.softmax(posterior_logits, dim=-1)

    if prior_logits is None:
        # Uniform prior
        num_classes = posterior_logits.shape[-1]
        prior = torch.ones_like(posterior) / num_classes
    else:
        prior = F.softmax(prior_logits, dim=-1)

    # KL per categorical: sum over classes
    kl_per_cat = (posterior * (posterior.log() - prior.log() + 1e-8)).sum(dim=-1)

    # Free bits: only penalize KL above threshold per categorical
    kl_per_cat = torch.maximum(kl_per_cat, torch.tensor(free_bits, device=kl_per_cat.device))

    # Mean over batch and categoricals
    return kl_per_cat.mean()


def percentile_normalize(x: Tensor, low_pct: float = 5.0, high_pct: float = 95.0) -> Tensor:
    """Normalize tensor using percentiles (V3 style return normalization).

    Args:
        x: Input tensor
        low_pct: Lower percentile (default 5%)
        high_pct: Upper percentile (default 95%)

    Returns:
        Normalized tensor in approximately [0, 1] range
    """
    low = torch.quantile(x.flatten(), low_pct / 100.0)
    high = torch.quantile(x.flatten(), high_pct / 100.0)
    return (x - low) / (high - low + 1e-8)


@dataclass
class DreamerLearner:
    """Learner for Dreamer V3 model-based RL.

    Combines world model training with actor-critic behavior learning.
    Uses V3-style simplifications: symlog, categorical KL, free bits.
    """

    model: DreamerModel
    gamma: float = 0.99
    lambda_gae: float = 0.95
    imagination_horizon: int = 15

    # MCTS adapter (injected by factory, optional)
    mcts_adapter: PolicyAdapter | ValueAdapter | WorldModelAdapter | None = None

    # Loss weights (simplified from V2)
    recon_scale: float = 1.0
    dynamics_scale: float = 1.0
    reward_scale: float = 1.0
    continue_scale: float = 1.0
    actor_scale: float = 1.0
    critic_scale: float = 1.0
    entropy_scale: float = 0.001
    kl_scale: float = 0.1

    # KL configuration (V3 style - simple free bits)
    free_bits: float = 1.0

    def compute_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
        weights: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute combined world model and behavior loss."""
        wm_loss, wm_metrics = self.compute_world_model_loss(states, actions, rewards, next_states, dones)

        z = self.model.encode(states, deterministic=False)
        behavior_loss, behavior_metrics = self.compute_behavior_loss(z)

        total_loss = wm_loss + behavior_loss

        if weights is not None:
            total_loss = (total_loss * weights).mean()

        metrics: dict[str, Any] = {
            "loss": total_loss.item(),
            **wm_metrics,
            **behavior_metrics,
        }

        return total_loss, metrics

    def compute_world_model_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute world model loss with V3-style symlog reconstruction."""
        # Normalize to [0, 1]
        states_norm = states / 255.0

        # Encode with logits for KL
        z, posterior_logits = self.model.encode_with_logits(states)

        # 1. Reconstruction loss (symlog MSE)
        recon = self.model.decoder(z)
        recon_loss = F.mse_loss(recon, symlog(states_norm))

        # 2. KL loss with free bits (vs uniform prior)
        kl_loss = categorical_kl_loss(posterior_logits, None, self.free_bits)

        # 3. Dynamics loss
        z_next_target = self.model.encode(next_states, deterministic=True).detach()
        z_next_pred, _, dynamics_logits = self.model.dynamics(z, actions)

        # Dynamics KL (predicted distribution vs encoded next state)
        # Get target logits by encoding next_states
        _, target_logits = self.model.encode_with_logits(next_states)
        dynamics_kl = categorical_kl_loss(dynamics_logits, target_logits.detach(), self.free_bits)

        # Also MSE on flattened latents for direct supervision
        dynamics_mse = F.mse_loss(z_next_pred, z_next_target)
        dynamics_loss = dynamics_kl + dynamics_mse

        # 4. Reward prediction loss (symlog target)
        reward_pred = self.model.reward_pred(z_next_pred)
        reward_loss = F.mse_loss(reward_pred, symlog(rewards))

        # 5. Continue prediction loss
        continue_pred = self.model.continue_pred(z_next_pred)
        continue_target = 1.0 - dones.float()
        continue_loss = F.binary_cross_entropy(continue_pred, continue_target)

        # Total world model loss
        wm_loss = (
            self.recon_scale * recon_loss
            + self.kl_scale * kl_loss
            + self.dynamics_scale * dynamics_loss
            + self.reward_scale * reward_loss
            + self.continue_scale * continue_loss
        )

        metrics = {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
            "dynamics_kl": dynamics_kl.item(),
            "dynamics_mse": dynamics_mse.item(),
            "reward_loss": reward_loss.item(),
            "continue_loss": continue_loss.item(),
            "wm_loss": wm_loss.item(),
        }

        return wm_loss, metrics

    def compute_behavior_loss(
        self,
        z_start: Tensor,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute actor-critic loss on imagined trajectories."""
        # Imagine trajectory
        z_traj, rewards, conts, _ = self.model.imagine_trajectory(z_start, horizon=self.imagination_horizon)

        batch_size, horizon_plus_one, latent_dim = z_traj.shape

        # Get values for all states in trajectory
        z_flat = z_traj.view(-1, latent_dim)
        values_symlog = self.model.critic(z_flat)
        values_symlog = values_symlog.view(batch_size, horizon_plus_one)

        # Convert symlog predictions to actual values for return computation
        values = symexp(values_symlog)
        rewards_actual = symexp(rewards)

        # Compute lambda-returns
        returns = self._compute_lambda_returns(rewards_actual, values, conts)

        # Normalize returns (V3 style)
        returns_norm = percentile_normalize(returns)

        # Actor loss
        z_for_actions = z_traj[:, :-1]
        z_for_actions_flat = z_for_actions.reshape(-1, latent_dim)

        logits = self.model.actor(z_for_actions_flat)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Advantages (normalized returns - values)
        values_for_actions = values[:, :-1]
        advantages = (returns_norm - percentile_normalize(values_for_actions)).detach()
        advantages_flat = advantages.reshape(-1)

        # Policy gradient with entropy bonus
        entropy = -(probs * log_probs).sum(dim=-1)

        # Reinforce-style policy loss
        action_dist = torch.distributions.Categorical(probs=probs)
        actions_sampled = action_dist.sample()
        log_prob_actions = log_probs.gather(1, actions_sampled.unsqueeze(-1)).squeeze(-1)
        policy_loss = -(log_prob_actions * advantages_flat).mean()

        entropy_bonus = entropy.mean()
        actor_loss = self.actor_scale * policy_loss - self.entropy_scale * entropy_bonus

        # Critic loss (predict symlog returns)
        returns_symlog = symlog(returns)
        critic_values = values_symlog[:, :-1].reshape(-1)
        returns_flat = returns_symlog.reshape(-1)
        critic_loss = self.critic_scale * F.mse_loss(critic_values, returns_flat.detach())

        total_loss = actor_loss + critic_loss

        metrics = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy_bonus.item(),
            "behavior_loss": total_loss.item(),
            "value_mean": values.mean().item(),
            "return_mean": returns.mean().item(),
        }

        return total_loss, metrics

    def _compute_lambda_returns(
        self,
        rewards: Tensor,
        values: Tensor,
        conts: Tensor,
    ) -> Tensor:
        """Compute lambda-returns (GAE-style TD(λ))."""
        batch_size, horizon = rewards.shape

        returns = torch.zeros_like(rewards)
        last_gae = torch.zeros(batch_size, device=rewards.device)

        for t in reversed(range(horizon)):
            next_value = values[:, t + 1]
            delta = rewards[:, t] + self.gamma * next_value * conts[:, t] - values[:, t]
            last_gae = delta + self.gamma * self.lambda_gae * conts[:, t] * last_gae
            returns[:, t] = last_gae + values[:, t]

        return returns

    def update_targets(self, tau: float = 1.0) -> None:
        """Update target networks (no-op for basic Dreamer)."""
        pass
