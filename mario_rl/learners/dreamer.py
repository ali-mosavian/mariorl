"""Dreamer Learner for model-based RL.

Implements the Learner protocol for Dreamer:
1. World Model Training: reconstruction, dynamics, reward prediction
2. Behavior Learning: actor-critic trained on imagined trajectories

Training proceeds in two phases:
1. World model learns to predict future states and rewards
2. Actor-critic learns behavior by "imagining" trajectories
"""

from typing import Any
from dataclasses import field
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from mario_rl.models import DreamerModel


@dataclass
class DreamerLearner:
    """Learner for Dreamer model-based RL.

    Combines world model training with actor-critic behavior learning:
    - World model: predicts dynamics, rewards, and done flags
    - Actor-Critic: trained on imagined trajectories from world model

    The key insight is that the actor-critic can be trained purely on
    imagined experience, making learning very sample-efficient.
    """

    model: DreamerModel
    gamma: float = 0.99
    lambda_gae: float = 0.95
    imagination_horizon: int = 15
    
    # Loss weights
    dynamics_scale: float = 1.0
    reward_scale: float = 1.0
    actor_scale: float = 1.0
    critic_scale: float = 1.0
    entropy_scale: float = 0.001

    def compute_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
        weights: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute combined world model and behavior loss.

        Args:
            states: Current observations (batch, *obs_shape)
            actions: Actions taken (batch,)
            rewards: Rewards received (batch,)
            next_states: Next observations (batch, *obs_shape)
            dones: Episode termination flags (batch,)
            weights: Importance sampling weights for PER (batch,), None for uniform

        Returns:
            loss: Combined loss tensor
            metrics: Dict with all training metrics
        """
        # 1. World model loss on real experience
        wm_loss, wm_metrics = self.compute_world_model_loss(
            states, actions, rewards, next_states
        )

        # 2. Behavior loss on imagined trajectories
        z = self.model.encode(states, deterministic=True)
        behavior_loss, behavior_metrics = self.compute_behavior_loss(z)

        # Combine losses
        total_loss = wm_loss + behavior_loss

        # Apply importance sampling weights if provided (for PER)
        if weights is not None:
            total_loss = (total_loss * weights).mean()

        # Merge metrics
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
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute world model loss on real transitions.

        Losses:
        - Dynamics: predict next latent from current latent + action
        - Reward: predict reward from next latent

        Args:
            states: Current observations
            actions: Actions taken
            rewards: Rewards received
            next_states: Next observations

        Returns:
            loss: World model loss
            metrics: Dynamics and reward loss metrics
        """
        # Encode current and next states
        z = self.model.encode(states, deterministic=False)
        z_next_target = self.model.encode(next_states, deterministic=True).detach()

        # Predict next latent using dynamics
        z_next_pred, _, z_next_mu = self.model.dynamics(z, actions)

        # Dynamics loss: MSE between predicted and actual next latent
        dynamics_loss = F.mse_loss(z_next_mu, z_next_target)

        # Reward prediction loss
        reward_pred = self.model.reward_pred(z_next_pred)
        reward_loss = F.mse_loss(reward_pred, rewards)

        # Total world model loss
        wm_loss = (
            self.dynamics_scale * dynamics_loss
            + self.reward_scale * reward_loss
        )

        metrics = {
            "dynamics_loss": dynamics_loss.item(),
            "reward_loss": reward_loss.item(),
            "wm_loss": wm_loss.item(),
        }

        return wm_loss, metrics

    def compute_behavior_loss(
        self,
        z_start: Tensor,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute actor-critic loss on imagined trajectories.

        Uses lambda-returns (TD(λ)) for value estimation:
        - Imagine trajectory using current policy
        - Compute lambda-returns from imagined rewards and values
        - Update actor to maximize returns
        - Update critic to predict returns

        Args:
            z_start: Starting latent states (batch, latent_dim)

        Returns:
            loss: Actor + critic loss
            metrics: Actor and critic loss metrics
        """
        # Imagine trajectory using current policy
        z_traj, rewards, dones = self.model.imagine_trajectory(
            z_start, horizon=self.imagination_horizon
        )
        # z_traj: (batch, horizon+1, latent_dim)
        # rewards: (batch, horizon)
        # dones: (batch, horizon)

        # Compute values for all states in trajectory
        batch_size, horizon_plus_one, latent_dim = z_traj.shape
        z_flat = z_traj.view(-1, latent_dim)
        values_flat = self.model.critic(z_flat)
        values = values_flat.view(batch_size, horizon_plus_one)
        # values: (batch, horizon+1)

        # Compute lambda-returns using GAE-style TD(λ)
        returns = self._compute_lambda_returns(
            rewards=rewards,
            values=values,
            dones=dones,
        )
        # returns: (batch, horizon)

        # Actor loss: maximize returns (policy gradient)
        # Get action log-probs for imagined actions
        z_for_actions = z_traj[:, :-1]  # All but last (batch, horizon, latent_dim)
        z_for_actions_flat = z_for_actions.reshape(-1, latent_dim)

        logits = self.model.actor(z_for_actions_flat)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Policy gradient with baseline
        advantages = (returns - values[:, :-1]).detach()  # (batch, horizon)
        advantages_flat = advantages.reshape(-1)

        # We need to get the action that was taken during imagination
        # Since imagination samples from policy, we use the entropy-regularized objective
        # Maximize expected return + entropy
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch * horizon,)
        
        # Actor loss: -E[advantage * log_prob(action)]
        # We use reinforce-style gradient by weighting log_probs by advantage
        # Since we don't have explicit actions from imagination, use entropy-weighted objective
        policy_loss = -(log_probs.max(dim=-1).values * advantages_flat).mean()
        entropy_bonus = entropy.mean()
        
        actor_loss = self.actor_scale * policy_loss - self.entropy_scale * entropy_bonus

        # Critic loss: MSE between predicted values and returns
        critic_values = values[:, :-1].reshape(-1)  # Exclude bootstrap value
        returns_flat = returns.reshape(-1)
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
        dones: Tensor,
    ) -> Tensor:
        """Compute lambda-returns (GAE-style TD(λ)).

        Args:
            rewards: Predicted rewards (batch, horizon)
            values: Value estimates (batch, horizon+1)
            dones: Done predictions (batch, horizon)

        Returns:
            returns: Lambda-returns (batch, horizon)
        """
        batch_size, horizon = rewards.shape

        # Initialize with bootstrap value
        returns = torch.zeros_like(rewards)
        last_gae = torch.zeros(batch_size, device=rewards.device)

        # Compute returns backwards
        for t in reversed(range(horizon)):
            next_value = values[:, t + 1]
            delta = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t]) - values[:, t]
            last_gae = delta + self.gamma * self.lambda_gae * (1 - dones[:, t]) * last_gae
            returns[:, t] = last_gae + values[:, t]

        return returns

    def update_targets(self, tau: float = 1.0) -> None:
        """Update target networks (no-op for basic Dreamer).

        Dreamer doesn't use separate target networks by default.
        This method is provided for Learner protocol compatibility.
        """
        pass
