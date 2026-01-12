"""Dreamer Learner for model-based RL.

Implements the Learner protocol for Dreamer:
1. World Model Training: reconstruction, dynamics, reward prediction
2. Behavior Learning: actor-critic trained on imagined trajectories

Key improvements to prevent decoder collapse:
- SSIM as primary reconstruction loss
- Free bits in KL loss
- KL balancing between encoder/decoder
- Spatially-weighted MSE for stability
- Beta warmup schedule for KL weight
"""

from typing import Any
from dataclasses import field
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from mario_rl.models import DreamerModel
from mario_rl.models.dreamer import ssim


def kl_loss_with_free_bits(mu: Tensor, logvar: Tensor, free_bits: float = 1.0) -> Tensor:
    """
    KL divergence with free bits to prevent posterior collapse.
    
    Only penalizes KL above the free_bits threshold per dimension.
    This guarantees minimum information flow from encoder to decoder.
    """
    logvar_clamped = logvar.clamp(-10, 2)
    kl_per_dim = -0.5 * (1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())
    kl_per_dim = torch.maximum(kl_per_dim, torch.tensor(free_bits, device=kl_per_dim.device))
    return kl_per_dim.mean()


def kl_loss_balanced(
    mu: Tensor,
    logvar: Tensor,
    recon: Tensor,
    target: Tensor,
    alpha: float = 0.8,
) -> tuple[Tensor, dict[str, float]]:
    """
    KL balancing to prevent encoder-decoder imbalance.
    
    Splits KL loss between encoder and decoder:
    - Decoder gets less KL gradient (encourages better reconstruction)
    - Encoder still gets regularization
    """
    logvar_clamped = logvar.clamp(-10, 2)
    kl_full = -0.5 * torch.mean(
        1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp()
    )
    
    kl_encoder = kl_full
    kl_decoder = kl_full.detach() + (kl_full - kl_full.detach()) * (1 - alpha)
    kl_loss = alpha * kl_encoder + (1 - alpha) * kl_decoder
    
    metrics = {
        "kl_full": kl_full.item(),
        "kl_encoder": kl_encoder.item(),
        "kl_decoder": kl_decoder.item(),
    }
    
    return kl_loss, metrics


def compute_spatial_weights(frames: Tensor, percentile: float = 75) -> Tensor:
    """
    Create spatial weight map emphasizing dynamic regions.
    
    Regions with high variance across batch = important for RL.
    """
    variance = torch.var(frames, dim=0, keepdim=True)
    threshold = torch.quantile(variance.flatten(), percentile / 100.0)
    weights = torch.where(
        variance > threshold,
        torch.ones_like(variance) * 3.0,
        torch.ones_like(variance)
    )
    return weights


@dataclass
class BetaScheduler:
    """Warmup schedule for KL weight to prevent early collapse."""
    
    max_beta: float = 0.05
    warmup_steps: int = 50000
    start_beta: float = 0.0
    step_count: int = field(init=False, default=0)
    
    def get_beta(self) -> float:
        """Get current KL weight."""
        if self.step_count >= self.warmup_steps:
            return self.max_beta
        
        progress = self.step_count / self.warmup_steps
        return self.start_beta + (self.max_beta - self.start_beta) * progress
    
    def step(self) -> None:
        """Increment step counter."""
        self.step_count += 1


@dataclass
class DreamerLearner:
    """Learner for Dreamer model-based RL.

    Combines world model training with actor-critic behavior learning:
    - World model: predicts dynamics, rewards, and done flags
    - Actor-Critic: trained on imagined trajectories from world model
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
    recon_scale: float = 1.0
    mse_weight: float = 0.1  # Weight for MSE in combined loss (SSIM + mse_weight * MSE)
    
    # KL configuration
    kl_free_bits: float = 1.0
    kl_balance_alpha: float = 0.8
    use_kl_balancing: bool = True
    kl_imagination_scale: float = 0.1  # Regularize imagined latents
    
    # Beta warmup
    beta_scheduler: BetaScheduler = field(default_factory=lambda: BetaScheduler(
        max_beta=0.05,
        warmup_steps=50000,
        start_beta=0.0
    ))
    
    # Spatial weighting
    use_spatial_weighting: bool = True
    spatial_percentile: float = 75.0

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
        wm_loss, wm_metrics = self.compute_world_model_loss(
            states, actions, rewards, next_states
        )

        z = self.model.encode(states, deterministic=True)
        behavior_loss, behavior_metrics = self.compute_behavior_loss(z)

        total_loss = wm_loss + behavior_loss

        if weights is not None:
            total_loss = (total_loss * weights).mean()

        metrics: dict[str, Any] = {
            "loss": total_loss.item(),
            "kl_weight": self.beta_scheduler.get_beta(),
            **wm_metrics,
            **behavior_metrics,
        }
        
        self.beta_scheduler.step()

        return total_loss, metrics

    def compute_world_model_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute world model loss with SSIM-based reconstruction."""
        states_norm = states / 255.0
        next_states_norm = next_states / 255.0

        z_mu, z_logvar = self.model.encoder(states_norm)
        z = self.model.encoder.sample(z_mu, z_logvar)
        z_next_target = self.model.encode(next_states, deterministic=True).detach()

        # 1. Reconstruction with SSIM as primary loss
        recon = self.model.decoder(z)
        
        ssim_val = ssim(recon, states_norm)
        ssim_loss = 1.0 - ssim_val
        
        if self.use_spatial_weighting:
            weights_spatial = compute_spatial_weights(states_norm, self.spatial_percentile)
            mse_loss = torch.mean(weights_spatial * (recon - states_norm) ** 2)
        else:
            mse_loss = F.mse_loss(recon, states_norm)
        
        recon_loss = ssim_loss + self.mse_weight * mse_loss

        # 2. KL divergence with improvements
        if self.use_kl_balancing:
            kl_loss, kl_metrics = kl_loss_balanced(
                z_mu, z_logvar, recon, states_norm, alpha=self.kl_balance_alpha
            )
        else:
            kl_loss = kl_loss_with_free_bits(z_mu, z_logvar, self.kl_free_bits)
            kl_metrics = {"kl_full": kl_loss.item()}
        
        kl_weight = self.beta_scheduler.get_beta()

        # 3. Dynamics loss
        z_next_pred, _, z_next_mu = self.model.dynamics(z, actions)
        dynamics_loss = F.mse_loss(z_next_mu, z_next_target)

        # 4. Reward prediction loss
        reward_pred = self.model.reward_pred(z_next_pred)
        reward_loss = F.mse_loss(reward_pred, rewards)

        # Total world model loss
        wm_loss = (
            self.recon_scale * recon_loss
            + kl_weight * kl_loss
            + self.dynamics_scale * dynamics_loss
            + self.reward_scale * reward_loss
        )

        metrics = {
            "recon_loss": recon_loss.item(),
            "ssim": ssim_val.item(),
            "ssim_loss": ssim_loss.item(),
            "mse_loss": mse_loss.item(),
            "kl_loss": kl_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
            "reward_loss": reward_loss.item(),
            "wm_loss": wm_loss.item(),
            **kl_metrics,
        }

        return wm_loss, metrics

    def compute_behavior_loss(
        self,
        z_start: Tensor,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute actor-critic loss on imagined trajectories with KL regularization."""
        z_traj, rewards, dones, imagination_kl = self._imagine_trajectory_with_kl(
            z_start, horizon=self.imagination_horizon
        )

        batch_size, horizon_plus_one, latent_dim = z_traj.shape
        z_flat = z_traj.view(-1, latent_dim)
        values_flat = self.model.critic(z_flat)
        values = values_flat.view(batch_size, horizon_plus_one)

        returns = self._compute_lambda_returns(rewards, values, dones)

        z_for_actions = z_traj[:, :-1]
        z_for_actions_flat = z_for_actions.reshape(-1, latent_dim)

        logits = self.model.actor(z_for_actions_flat)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        advantages = (returns - values[:, :-1]).detach()
        advantages_flat = advantages.reshape(-1)

        entropy = -(probs * log_probs).sum(dim=-1)
        policy_loss = -(log_probs.max(dim=-1).values * advantages_flat).mean()
        entropy_bonus = entropy.mean()
        
        actor_loss = self.actor_scale * policy_loss - self.entropy_scale * entropy_bonus

        critic_values = values[:, :-1].reshape(-1)
        returns_flat = returns.reshape(-1)
        critic_loss = self.critic_scale * F.mse_loss(critic_values, returns_flat.detach())

        # Add KL regularization on imagined latents
        kl_imagination_loss = self.kl_imagination_scale * imagination_kl
        
        total_loss = actor_loss + critic_loss + kl_imagination_loss

        metrics = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy_bonus.item(),
            "kl_imagination": imagination_kl.item(),
            "behavior_loss": total_loss.item(),
            "value_mean": values.mean().item(),
            "return_mean": returns.mean().item(),
        }

        return total_loss, metrics

    def _imagine_trajectory_with_kl(
        self,
        z_start: Tensor,
        horizon: int = 15,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Imagine trajectory and compute KL loss on imagined latents.
        
        This prevents imagined latents from diverging to unrealistic values
        and keeps them in the same distribution as real latents.
        
        Returns:
            z_traj: Latent trajectory (batch, horizon+1, latent_dim)
            rewards: Predicted rewards (batch, horizon)
            dones: Predicted dones (batch, horizon)
            kl_imagination: KL divergence of imagined latents vs prior
        """
        batch_size = z_start.shape[0]
        device = z_start.device
        
        z_traj = [z_start]
        rewards = []
        dones = []
        kl_losses = []
        
        z = z_start
        h = None
        
        for _ in range(horizon):
            # Sample action from policy
            logits = self.model.actor(z)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            
            # Imagine step - get both sampled latent and distribution params
            # We need to access dynamics forward to get mu and logvar
            if h is None:
                h = torch.zeros(batch_size, self.model.dynamics.hidden_dim, device=device)
            
            # Embed action and get dynamics prediction
            a_embed = self.model.dynamics.action_embed(action)
            x = torch.cat([z, a_embed], dim=-1)
            h = self.model.dynamics.gru(x, h)
            
            out = self.model.dynamics.fc_out(h)
            mu = self.model.dynamics.fc_mu(out)
            logvar = self.model.dynamics.fc_logvar(out).clamp(-10, 2)
            
            # Sample next latent
            std = torch.exp(0.5 * logvar)
            z_next = mu + std * torch.randn_like(std)
            
            # Compute KL divergence: KL(N(mu, sigma^2) || N(0, 1))
            # KL = 0.5 * (sigma^2 + mu^2 - 1 - log(sigma^2))
            kl = 0.5 * (logvar.exp() + mu.pow(2) - 1 - logvar)
            kl_losses.append(kl.mean())  # Mean over batch and latent dims
            
            # Predict reward and done
            reward = self.model.reward_pred(z_next)
            done = self.model.done_pred(z_next)
            
            z_traj.append(z_next)
            rewards.append(reward)
            dones.append(done)
            
            z = z_next
        
        # Average KL over trajectory
        kl_imagination = torch.stack(kl_losses).mean()
        
        return (
            torch.stack(z_traj, dim=1),
            torch.stack(rewards, dim=1),
            torch.stack(dones, dim=1),
            kl_imagination
        )

    def _compute_lambda_returns(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
    ) -> Tensor:
        """Compute lambda-returns (GAE-style TD(λ))."""
        batch_size, horizon = rewards.shape

        returns = torch.zeros_like(rewards)
        last_gae = torch.zeros(batch_size, device=rewards.device)

        for t in reversed(range(horizon)):
            next_value = values[:, t + 1]
            delta = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t]) - values[:, t]
            last_gae = delta + self.gamma * self.lambda_gae * (1 - dones[:, t]) * last_gae
            returns[:, t] = last_gae + values[:, t]

        return returns

    def update_targets(self, tau: float = 1.0) -> None:
        """Update target networks (no-op for basic Dreamer)."""
        pass
