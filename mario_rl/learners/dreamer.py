"""Dreamer Learner for model-based RL.

Implements the Learner protocol for Dreamer:
1. World Model Training: reconstruction, dynamics, reward prediction
2. Behavior Learning: actor-critic trained on imagined trajectories

Key improvements to prevent encoder/decoder collapse:

ENCODER COLLAPSE PREVENTION:
- Contrastive diversity loss to push apart different latents
- Variance regularization to prevent constant outputs
- Aggressive free bits (5.0) to guarantee information flow

DECODER COLLAPSE PREVENTION (THE KEY FIX!):
- Decoder diversity loss: forces decoder to produce DIFFERENT outputs for
  different latent inputs (prevents decoder from ignoring latent)
- Decoder sensitivity loss: penalizes decoder if output doesn't change
  when latent is perturbed (forces decoder to USE the latent)

KL REGULARIZATION:
- Very gentle KL penalty (max_beta=0.0001)
- Short delay (5k steps) then slow ramp (100k steps)
- KL balancing between encoder/decoder
- Free bits to guarantee minimum information flow

RECONSTRUCTION:
- SSIM as primary loss + spatially-weighted MSE
"""

from typing import Any
from dataclasses import field
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from mario_rl.models import DreamerModel
from mario_rl.models.dreamer import ssim
from mario_rl.mcts.protocols import PolicyAdapter, ValueAdapter, WorldModelAdapter


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


def contrastive_latent_loss(z: Tensor, temperature: float = 0.1) -> Tensor:
    """
    Contrastive loss to maximize diversity between different inputs.
    
    Forces encoder to produce diverse latent codes for different observations.
    Minimizes cosine similarity between different samples in the batch.
    
    Args:
        z: (batch_size, latent_dim) - encoded latents
        temperature: Temperature for scaling similarities
    
    Returns:
        Loss encouraging low similarity between different samples
    """
    if z.size(0) < 2:
        return torch.tensor(0.0, device=z.device)
    
    # Normalize latents
    z_norm = F.normalize(z, dim=1)
    
    # Cosine similarity matrix (batch_size, batch_size)
    similarity = torch.matmul(z_norm, z_norm.t()) / temperature
    
    # Create mask for off-diagonal elements (different samples)
    mask = torch.eye(z.size(0), device=z.device).bool()
    
    # Off-diagonal similarities (these should be LOW for diversity)
    off_diag_sim = similarity.masked_select(~mask)
    
    # Penalize high off-diagonal similarity
    diversity_loss = off_diag_sim.pow(2).mean()
    
    return diversity_loss


def latent_diversity_loss(z: Tensor, target_std: float = 1.0) -> Tensor:
    """
    Encourage high variance across batch for each latent dimension.
    
    Prevents encoder from collapsing to constant outputs by explicitly
    maximizing the standard deviation of latents across the batch.
    
    Args:
        z: (batch_size, latent_dim)
        target_std: Target standard deviation per dimension
    
    Returns:
        Loss penalizing low variance in latents
    """
    if z.size(0) < 2:
        return torch.tensor(0.0, device=z.device)
    
    # Compute std per dimension across batch
    z_std = z.std(dim=0)  # (latent_dim,)
    
    # Penalize if std is too low (want high diversity)
    diversity_loss = F.relu(target_std - z_std).mean()
    
    return diversity_loss


def decoder_diversity_loss(decoder_outputs: Tensor, min_diff: float = 0.01) -> Tensor:
    """
    Force decoder to produce DIFFERENT outputs for different latent inputs.
    
    THIS IS THE KEY FIX FOR DECODER COLLAPSE!
    
    The decoder can collapse by ignoring its input and producing constant output.
    This loss penalizes when decoder outputs are too similar across the batch.
    
    Args:
        decoder_outputs: (batch_size, C, H, W) - reconstructed images
        min_diff: Minimum required difference between outputs
    
    Returns:
        Loss that penalizes similar decoder outputs
    """
    if decoder_outputs.size(0) < 2:
        return torch.tensor(0.0, device=decoder_outputs.device)
    
    batch_size = decoder_outputs.size(0)
    
    # Flatten to (batch, -1)
    flat = decoder_outputs.view(batch_size, -1)
    
    # Compute pairwise L1 differences
    # More efficient: use broadcasting
    # flat[i] - flat[j] for all i < j
    total_diff = 0.0
    count = 0
    for i in range(min(batch_size, 8)):  # Limit to avoid O(n^2) explosion
        for j in range(i + 1, min(batch_size, 8)):
            diff = torch.mean(torch.abs(flat[i] - flat[j]))
            total_diff = total_diff + diff
            count += 1
    
    if count == 0:
        return torch.tensor(0.0, device=decoder_outputs.device)
    
    mean_diff = total_diff / count
    
    # Penalize if outputs are too similar (below min_diff threshold)
    # Loss = max(0, min_diff - mean_diff)
    loss = F.relu(min_diff - mean_diff)
    
    return loss


def decoder_sensitivity_loss(
    decoder: torch.nn.Module,
    z: Tensor,
    perturbation_scale: float = 0.1,
    min_response: float = 0.001,
) -> Tensor:
    """
    Ensure decoder output CHANGES when latent input changes.
    
    Tests if decoder actually uses its input by measuring response to perturbation.
    If decoder is collapsed (ignoring input), perturbing z won't change output.
    
    Args:
        decoder: The decoder module
        z: Latent codes (batch_size, latent_dim)
        perturbation_scale: Size of perturbation to apply
        min_response: Minimum required response to perturbation
    
    Returns:
        Loss penalizing decoder that doesn't respond to input changes
    """
    # Get baseline output
    output_base = decoder(z)
    
    # Perturb latent
    perturbation = torch.randn_like(z) * perturbation_scale
    z_perturbed = z + perturbation
    
    # Get perturbed output
    output_perturbed = decoder(z_perturbed)
    
    # Measure response (should be non-zero if decoder uses input)
    response = torch.mean(torch.abs(output_base - output_perturbed))
    
    # Penalize if response is too small
    loss = F.relu(min_response - response)
    
    return loss


@dataclass
class BetaScheduler:
    """Warmup schedule for KL weight to prevent early collapse.
    
    Strategy: Delay KL regularization to let encoder/decoder learn first,
    then gradually introduce KL penalty over a long warmup period.
    """
    
    max_beta: float = 0.001  # Much lower than before (was 0.05)
    warmup_steps: int = 200000  # Much longer warmup (was 50000)
    start_beta: float = 0.0
    delay_steps: int = 20000  # NEW: No KL penalty for first 20k steps
    step_count: int = field(init=False, default=0)
    
    def get_beta(self) -> float:
        """Get current KL weight with delayed start."""
        # Phase 1: No KL penalty during delay period
        if self.step_count < self.delay_steps:
            return 0.0
        
        # Phase 2: Gradual warmup after delay
        effective_step = self.step_count - self.delay_steps
        if effective_step < self.warmup_steps:
            progress = effective_step / self.warmup_steps
            return self.start_beta + (self.max_beta - self.start_beta) * progress
        
        # Phase 3: Constant at max_beta
        return self.max_beta
    
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

    # MCTS adapter (injected by factory, optional)
    # For Dreamer, the adapter also implements WorldModelAdapter for imagined rollouts
    mcts_adapter: PolicyAdapter | ValueAdapter | WorldModelAdapter | None = None

    # Loss weights
    dynamics_scale: float = 1.0
    reward_scale: float = 1.0
    actor_scale: float = 1.0
    critic_scale: float = 1.0
    entropy_scale: float = 0.001
    recon_scale: float = 1.0
    mse_weight: float = 0.1  # Weight for MSE in combined loss (SSIM + mse_weight * MSE)
    
    # KL configuration
    kl_free_bits: float = 5.0  # INCREASED from 1.0 - allow much more info through bottleneck
    kl_balance_alpha: float = 0.8
    use_kl_balancing: bool = True
    kl_imagination_scale: float = 0.1  # Regularize imagined latents
    
    # Diversity loss to prevent encoder collapse
    diversity_scale: float = 0.1  # Contrastive loss weight
    variance_scale: float = 0.05  # Variance regularization weight
    target_latent_std: float = 1.0  # Target std for latent dimensions
    
    # DECODER COLLAPSE PREVENTION (THE KEY FIX!)
    decoder_diversity_scale: float = 10.0  # Force decoder to produce varied outputs
    decoder_sensitivity_scale: float = 5.0  # Force decoder to respond to latent changes
    decoder_min_diff: float = 0.01  # Minimum required difference between outputs
    decoder_min_response: float = 0.001  # Minimum response to latent perturbation
    
    # Beta warmup - reduced delay since decoder collapsed by 20k
    beta_scheduler: BetaScheduler = field(default_factory=lambda: BetaScheduler(
        max_beta=0.0001,     # DECREASED further - even gentler KL
        warmup_steps=100000, # Slower ramp
        start_beta=0.0,
        delay_steps=5000     # DECREASED from 20k - start KL earlier to prevent explosion
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

        # 2b. Diversity losses to prevent encoder collapse
        # Contrastive loss: push apart latents from different inputs
        diversity_loss_contrastive = contrastive_latent_loss(z, temperature=0.1)
        
        # Variance loss: ensure latents have sufficient spread
        diversity_loss_variance = latent_diversity_loss(z, target_std=self.target_latent_std)

        # 2c. DECODER COLLAPSE PREVENTION (THE KEY FIX!)
        # Force decoder to produce different outputs for different latents
        dec_diversity = decoder_diversity_loss(recon, min_diff=self.decoder_min_diff)
        
        # Force decoder to actually USE the latent input (respond to changes)
        dec_sensitivity = decoder_sensitivity_loss(
            self.model.decoder, z, 
            perturbation_scale=0.1, 
            min_response=self.decoder_min_response
        )

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
            + self.diversity_scale * diversity_loss_contrastive
            + self.variance_scale * diversity_loss_variance
            + self.decoder_diversity_scale * dec_diversity  # DECODER FIX
            + self.decoder_sensitivity_scale * dec_sensitivity  # DECODER FIX
            + self.dynamics_scale * dynamics_loss
            + self.reward_scale * reward_loss
        )

        metrics = {
            "recon_loss": recon_loss.item(),
            "ssim": ssim_val.item(),
            "ssim_loss": ssim_loss.item(),
            "mse_loss": mse_loss.item(),
            "kl_loss": kl_loss.item(),
            "kl_weight": kl_weight,
            "diversity_contrastive": diversity_loss_contrastive.item(),
            "diversity_variance": diversity_loss_variance.item(),
            "decoder_diversity": dec_diversity.item(),  # NEW: Monitor decoder diversity
            "decoder_sensitivity": dec_sensitivity.item(),  # NEW: Monitor decoder response
            "latent_std": z.std().item(),  # Monitor actual latent diversity
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
            
            # Project through pre_gru before GRU (matches Dynamics.forward)
            x = self.model.dynamics.pre_gru(x)
            h = self.model.dynamics.gru(x, h)
            h = self.model.dynamics.gru_norm(h)
            
            # Residual connection around GRU
            h = h + x
            
            # Post-GRU processing
            out = self.model.dynamics.post_gru(h)
            mu = self.model.dynamics.fc_mu(out)
            logvar = self.model.dynamics.fc_logvar(out).clamp(-10, 2)
            
            # Sample next latent with residual from current z
            std = torch.exp(0.5 * logvar)
            z_delta = mu + std * torch.randn_like(std)
            z_next = self.model.dynamics.z_skip(z) + z_delta
            
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
