"""
World Model for Super Mario Bros.

Learns latent representations and transition dynamics for level-agnostic learning.

Architecture Overview
=====================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                          WORLD MODEL                                    │
    │                                                                         │
    │   ┌──────────┐                              ┌──────────┐                │
    │   │  Frame   │   (4,64,64,1)                │  Frame   │                │
    │   │  Stack   │ ─────────────────┐      ┌───▶│  Stack'  │                │
    │   └──────────┘                  │      │    └──────────┘                │
    │                                 ▼      │                                │
    │                          ┌───────────┐ │                                │
    │                          │  ENCODER  │ │                                │
    │                          │           │ │                                │
    │                          │ CNN + FC  │ │                                │
    │                          └─────┬─────┘ │                                │
    │                                │       │                                │
    │                          ┌─────▼─────┐ │                                │
    │                          │   μ, σ    │ │                                │
    │                          │  Latent   │ │                                │
    │                          │  (z_t)    │ │                                │
    │                          └─────┬─────┘ │                                │
    │               ┌────────────────┼───────┴────────────┐                   │
    │               │                │                    │                   │
    │               ▼                ▼                    ▼                   │
    │        ┌───────────┐    ┌───────────┐        ┌───────────┐              │
    │        │  DECODER  │    │ DYNAMICS  │        │  REWARD   │              │
    │        │           │    │   MODEL   │        │ PREDICTOR │              │
    │        │ DeConv    │    │           │        │           │              │
    │        └─────┬─────┘    │ z + a → z'│        │  z → r    │              │
    │              │          └─────┬─────┘        └─────┬─────┘              │
    │              ▼                │                    │                    │
    │        ┌───────────┐          │                    ▼                    │
    │        │   Recon   │          │              ┌───────────┐              │
    │        │   Frame   │          │              │  Reward   │              │
    │        └───────────┘          │              │  (r̂)      │              │
    │                               ▼              └───────────┘              │
    │                         ┌───────────┐                                   │
    │                         │  z_{t+1}  │                                   │
    │                         │  (pred)   │                                   │
    │                         └───────────┘                                   │
    └─────────────────────────────────────────────────────────────────────────┘


Training Flow
=============

    Experience Buffer                    World Model
    ┌─────────────────┐                 ┌─────────────────┐
    │ s_t,a,r,s_{t+1} │ ──────────────▶ │   Encode s_t    │
    └─────────────────┘                 │   Encode s_{t+1}│
                                        │                 │
                                        │   Losses:       │
                                        │   - Recon MSE   │
                                        │   - Dynamics KL │
                                        │   - Reward MSE  │
                                        └────────┬────────┘
                                                 │
    Q-Network (Latent)                           │
    ┌─────────────────┐                          │
    │   z_t ──▶ Q(a)  │ ◀────── frozen encoder ──┘
    │                 │
    │   DuelingDDQN   │
    │   on latent z   │
    └─────────────────┘


Imagination Rollout (Future)
============================

    z_0 ──┬──▶ Dynamics ──▶ z_1 ──┬──▶ Dynamics ──▶ z_2 ──▶ ...
          │       ▲               │       ▲
          │       │               │       │
          │     action            │     action
          │                       │
          └──▶ Reward ──▶ r̂_0     └──▶ Reward ──▶ r̂_1


Components
==========
- FrameEncoder: Frames → latent (mu, logvar)
- FrameDecoder: Latent → reconstructed frames
- DynamicsModel: (z_t, action) → z_{t+1} with GRU
- RewardPredictor: Latent → reward
- MarioWorldModel: Combined model with imagination capabilities
"""

from typing import Tuple
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class WorldModelOutput(NamedTuple):
    """Output from a world model forward pass."""

    z: Tensor  # Current latent
    z_mu: Tensor  # Encoder mean
    z_logvar: Tensor  # Encoder log-variance
    frame_recon: Tensor  # Reconstructed current frame
    z_next: Tensor  # Predicted next latent
    z_next_mu: Tensor  # Predicted next latent mean
    z_next_logvar: Tensor  # Predicted next latent log-variance
    frame_next_pred: Tensor  # Predicted next frame
    reward_pred: Tensor  # Predicted reward


class WorldModelMetrics(NamedTuple):
    """Metrics from world model training."""

    total_loss: float
    recon_mse: float
    pred_mse: float
    ssim: float
    dynamics_loss: float
    reward_loss: float
    kl_loss: float


# =============================================================================
# SSIM Implementation (no external dependencies)
# =============================================================================


def _gaussian_window(size: int, sigma: float, device: torch.device) -> Tensor:
    """Create a 1D Gaussian window."""
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords -= size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def _create_ssim_window(window_size: int, channels: int, device: torch.device) -> Tensor:
    """Create a 2D Gaussian window for SSIM."""
    _1d_window = _gaussian_window(window_size, 1.5, device)
    _2d_window = _1d_window.unsqueeze(1) @ _1d_window.unsqueeze(0)
    return _2d_window.expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    x: Tensor,
    y: Tensor,
    window_size: int = 11,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute Structural Similarity Index (SSIM) between two images.

    Args:
        x: First image tensor (N, C, H, W) in [0, 1]
        y: Second image tensor (N, C, H, W) in [0, 1]
        window_size: Size of the Gaussian window
        reduction: 'none', 'mean', or 'sum'

    Returns:
        SSIM value(s) in [0, 1], higher is better
    """
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {x.dim()}D")

    channels = x.shape[1]
    window = _create_ssim_window(window_size, channels, x.device)

    # Constants for numerical stability
    C1 = 0.01**2
    C2 = 0.03**2

    padding = window_size // 2

    # Compute means
    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)

    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y

    # Compute variances and covariance
    sigma_x_sq = F.conv2d(x**2, window, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y**2, window, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy

    # SSIM formula
    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_map = numerator / denominator

    if reduction == "mean":
        return ssim_map.mean()
    elif reduction == "sum":
        return ssim_map.sum()
    return ssim_map


# =============================================================================
# Encoder
# =============================================================================


class FrameEncoder(nn.Module):
    """
    Encodes stacked frames into a compact latent representation.

    Input: (N, F, H, W, C) uint8 frames
    Output: (mu, logvar) each of shape (N, latent_dim)
    """

    def __init__(self, frame_shape: Tuple[int, ...], latent_dim: int = 128):
        super().__init__()
        frames, height, width, channels = frame_shape
        self.latent_dim = latent_dim
        self.frame_shape = frame_shape

        # Architecture designed for 64x64 input:
        # 64 -> 31 (k=4,s=2) -> 14 (k=4,s=2) -> 6 (k=3,s=2) -> 2 (k=3,s=2)
        self.conv = nn.Sequential(
            nn.Conv2d(frames * channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, frames * channels, height, width)
            flat_size = self.conv(dummy).shape[1]

        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_logvar = nn.Linear(flat_size, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Frames tensor (N, F, H, W, C) float32 in [0, 1]

        Returns:
            mu, logvar of latent distribution
        """
        # NxFxHxWxC -> NxFxCxHxW -> Nx(F*C)xHxW
        x = x.permute(0, 1, 4, 2, 3)
        x = x.flatten(1, 2)

        h = self.conv(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def sample(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for VAE sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: Tensor, deterministic: bool = True) -> Tensor:
        """Get latent representation."""
        mu, logvar = self.forward(x)
        return mu if deterministic else self.sample(mu, logvar)


# =============================================================================
# Decoder
# =============================================================================


class FrameDecoder(nn.Module):
    """
    Decodes latent representation back to frames.

    Input: (N, latent_dim)
    Output: (N, F, H, W, C) reconstructed frames in [0, 1]
    """

    def __init__(self, latent_dim: int, frame_shape: Tuple[int, ...]):
        super().__init__()
        self.frames, self.height, self.width, self.channels = frame_shape
        self.latent_dim = latent_dim
        out_channels = self.frames * self.channels

        # Match encoder: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: Latent tensor (N, latent_dim)

        Returns:
            Reconstructed frames (N, F, H, W, C) in [0, 1]
        """
        h = self.fc(z).view(-1, 256, 4, 4)
        x = self.deconv(h)

        # Resize to exact frame dimensions if needed
        if x.shape[2] != self.height or x.shape[3] != self.width:
            x = F.interpolate(x, size=(self.height, self.width), mode="bilinear", align_corners=False)

        # Nx(F*C)xHxW -> NxFxCxHxW -> NxFxHxWxC
        x = x.view(-1, self.frames, self.channels, self.height, self.width)
        return x.permute(0, 1, 3, 4, 2)


# =============================================================================
# Dynamics Model
# =============================================================================


class DynamicsModel(nn.Module):
    """
    Predicts next latent state given current latent and action.

    Uses GRU for temporal context to handle partial observability.

    Input: z_t (N, latent_dim), action (N,), h (N, hidden_dim) or None
    Output: z_next_mu, z_next_logvar, z_next_sample, h_next
    """

    def __init__(self, latent_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        # Action embedding
        self.action_embed = nn.Embedding(num_actions, 32)

        # GRU for temporal modeling
        self.gru = nn.GRUCell(latent_dim + 32, hidden_dim)

        # Predict next latent distribution
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        z: Tensor,
        action: Tensor,
        h: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            z: Current latent state (N, latent_dim)
            action: Action taken (N,) long tensor
            h: Previous hidden state (N, hidden_dim) or None

        Returns:
            z_next_mu, z_next_logvar, z_next_sample, h_next
        """
        batch_size = z.shape[0]

        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=z.device)

        # Embed action and concatenate with latent
        a_embed = self.action_embed(action)
        x = torch.cat([z, a_embed], dim=-1)

        # GRU update
        h_next = self.gru(x, h)

        # Predict next latent distribution
        out = self.fc_out(h_next)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        # Sample next latent
        std = torch.exp(0.5 * logvar)
        z_next = mu + std * torch.randn_like(std)

        return mu, logvar, z_next, h_next


# =============================================================================
# Reward Predictor
# =============================================================================


class RewardPredictor(nn.Module):
    """Predicts reward from latent state."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z).squeeze(-1)


# =============================================================================
# Terminal Predictor
# =============================================================================


class TerminalPredictor(nn.Module):
    """Predicts if episode terminates from latent state."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z).squeeze(-1)


# =============================================================================
# Complete World Model
# =============================================================================


class MarioWorldModel(nn.Module):
    """
    Complete world model for Super Mario Bros.

    Given current frame + action, predicts:
    1. Latent representation of current frame
    2. Next latent state
    3. Next frame (reconstruction)
    4. Reward
    5. Terminal flag

    This enables:
    - "Imagination" - planning without environment interaction
    - Learning from imagined trajectories
    - Level-agnostic representations
    """

    def __init__(
        self,
        frame_shape: Tuple[int, ...],
        num_actions: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.frame_shape = frame_shape
        self.latent_dim = latent_dim
        self.num_actions = num_actions

        # Core components
        self.encoder = FrameEncoder(frame_shape, latent_dim)
        self.decoder = FrameDecoder(latent_dim, frame_shape)
        self.dynamics = DynamicsModel(latent_dim, num_actions, hidden_dim)
        self.reward_pred = RewardPredictor(latent_dim)
        self.terminal_pred = TerminalPredictor(latent_dim)

    def encode(self, frames: Tensor, deterministic: bool = True) -> Tensor:
        """Encode frames to latent."""
        return self.encoder.encode(frames, deterministic)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent to frames."""
        return self.decoder(z)

    def imagine_step(
        self,
        z: Tensor,
        action: Tensor,
        h: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Imagine one step forward in latent space.

        Args:
            z: Current latent (N, latent_dim)
            action: Action to take (N,)
            h: Recurrent hidden state

        Returns:
            z_next, reward, terminal, frame_pred, h_next
        """
        # Predict next latent
        _, _, z_next, h_next = self.dynamics(z, action, h)

        # Predict reward and terminal from next state
        reward = self.reward_pred(z_next)
        terminal = self.terminal_pred(z_next)

        # Decode to frame (optional, mainly for visualization)
        frame_pred = self.decoder(z_next)

        return z_next, reward, terminal, frame_pred, h_next

    def imagine_trajectory(
        self,
        z_start: Tensor,
        actions: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Imagine a full trajectory without environment interaction.

        Args:
            z_start: Starting latent (N, latent_dim)
            actions: Sequence of actions (N, T)

        Returns:
            z_trajectory: (N, T+1, latent_dim)
            rewards: (N, T)
            terminals: (N, T)
        """
        horizon = actions.shape[1]

        z_traj = [z_start]
        rewards = []
        terminals = []

        z = z_start
        h = None

        for t in range(horizon):
            z, r, term, _, h = self.imagine_step(z, actions[:, t], h)
            z_traj.append(z)
            rewards.append(r)
            terminals.append(term)

        return (
            torch.stack(z_traj, dim=1),
            torch.stack(rewards, dim=1),
            torch.stack(terminals, dim=1),
        )

    def forward(
        self,
        frames: Tensor,
        actions: Tensor,
        frames_next: Tensor,
    ) -> WorldModelOutput:
        """
        Full forward pass for training.

        Args:
            frames: Current frames (N, F, H, W, C) float32 in [0, 1]
            actions: Actions taken (N,)
            frames_next: Next frames (N, F, H, W, C) float32 in [0, 1]

        Returns:
            WorldModelOutput with all predictions and latents
        """
        # Encode current frame
        z_mu, z_logvar = self.encoder(frames)
        z = self.encoder.sample(z_mu, z_logvar)

        # Reconstruct current frame
        frame_recon = self.decoder(z)

        # Predict next state from dynamics
        z_next_mu, z_next_logvar, z_next, _ = self.dynamics(z, actions)

        # Predict next frame from predicted latent
        frame_next_pred = self.decoder(z_next)

        # Predict reward
        reward_pred = self.reward_pred(z_next)

        return WorldModelOutput(
            z=z,
            z_mu=z_mu,
            z_logvar=z_logvar,
            frame_recon=frame_recon,
            z_next=z_next,
            z_next_mu=z_next_mu,
            z_next_logvar=z_next_logvar,
            frame_next_pred=frame_next_pred,
            reward_pred=reward_pred,
        )


# =============================================================================
# World Model Loss
# =============================================================================


class WorldModelLoss(nn.Module):
    """Combined loss for world model training with all metrics."""

    def __init__(
        self,
        beta_kl: float = 0.1,
        beta_dynamics: float = 1.0,
        beta_ssim: float = 0.1,
    ):
        super().__init__()
        self.beta_kl = beta_kl
        self.beta_dynamics = beta_dynamics
        self.beta_ssim = beta_ssim

    def forward(
        self,
        frames: Tensor,
        frames_next: Tensor,
        rewards: Tensor,
        output: WorldModelOutput,
        z_next_target: Tuple[Tensor, Tensor] | None = None,
    ) -> Tuple[Tensor, WorldModelMetrics]:
        """
        Compute world model losses.

        Args:
            frames: Current frames (N, F, H, W, C) float32 in [0, 1]
            frames_next: Next frames (N, F, H, W, C) float32 in [0, 1]
            rewards: True rewards (N,)
            output: WorldModelOutput from forward pass
            z_next_target: Optional (mu, logvar) from encoding frames_next

        Returns:
            total_loss, WorldModelMetrics
        """
        # 1. Reconstruction loss (current frame)
        recon_mse = F.mse_loss(output.frame_recon, frames)

        # 2. Prediction loss (next frame)
        pred_mse = F.mse_loss(output.frame_next_pred, frames_next)

        # 3. SSIM for perceptual quality (on predicted next frame)
        # Reshape for SSIM: (N, F, H, W, C) -> (N, F*C, H, W)
        pred_for_ssim = output.frame_next_pred.permute(0, 1, 4, 2, 3).flatten(1, 2)
        target_for_ssim = frames_next.permute(0, 1, 4, 2, 3).flatten(1, 2)
        ssim_val = ssim(pred_for_ssim, target_for_ssim)

        # 4. KL divergence for encoder regularization
        kl_loss = -0.5 * torch.mean(1 + output.z_logvar - output.z_mu.pow(2) - output.z_logvar.exp())

        # 5. Dynamics consistency loss
        if z_next_target is not None:
            z_target_mu, z_target_logvar = z_next_target
            dynamics_loss = self._kl_divergence(
                output.z_next_mu,
                output.z_next_logvar,
                z_target_mu.detach(),
                z_target_logvar.detach(),
            )
        else:
            # If no target encoding, use MSE between predicted and encoded next
            dynamics_loss = torch.tensor(0.0, device=frames.device)

        # 6. Reward prediction loss (Huber loss is more robust to outliers)
        reward_loss = F.huber_loss(output.reward_pred, rewards.clamp(-15, 15))

        # Total loss
        total_loss = (
            recon_mse
            + pred_mse
            + self.beta_kl * kl_loss
            + self.beta_dynamics * dynamics_loss
            + reward_loss
            - self.beta_ssim * ssim_val  # Negative because we want to maximize SSIM
        )

        metrics = WorldModelMetrics(
            total_loss=total_loss.item(),
            recon_mse=recon_mse.item(),
            pred_mse=pred_mse.item(),
            ssim=ssim_val.item(),
            dynamics_loss=dynamics_loss.item(),
            reward_loss=reward_loss.item(),
            kl_loss=kl_loss.item(),
        )

        return total_loss, metrics

    @staticmethod
    def _kl_divergence(
        mu1: Tensor,
        logvar1: Tensor,
        mu2: Tensor,
        logvar2: Tensor,
    ) -> Tensor:
        """KL(N(mu1, var1) || N(mu2, var2)) with clamping to prevent explosion."""
        # Clamp logvars to prevent numerical instability
        logvar1 = logvar1.clamp(-10, 10)
        logvar2 = logvar2.clamp(-10, 10)

        var1 = logvar1.exp()
        var2 = logvar2.exp()

        kl = 0.5 * (logvar2 - logvar1 + var1 / var2 + (mu2 - mu1).pow(2) / var2 - 1)
        # Clamp per-element KL to prevent explosion from outliers
        kl = kl.clamp(0, 100)
        return kl.mean()


# =============================================================================
# Latent Q-Network
# =============================================================================


class LatentQNetwork(nn.Module):
    """Q-network that operates on latent representations, not pixels."""

    def __init__(self, latent_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.q_net(z)


class LatentDuelingDQN(nn.Module):
    """Dueling DQN architecture on latent space."""

    def __init__(self, latent_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions

        # Shared feature layer
        self.feature = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
        )

        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        features = self.feature(z)
        advantage = self.advantage(features)
        value = self.value(features)

        # Dueling: Q = V + (A - mean(A))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class LatentDDQN(nn.Module):
    """Double DQN with online and target networks on latent space."""

    def __init__(self, latent_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.online = LatentDuelingDQN(latent_dim, num_actions, hidden_dim)
        self.target = LatentDuelingDQN(latent_dim, num_actions, hidden_dim)

        # Copy weights and freeze target
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, z: Tensor, model: str = "online") -> Tensor:
        if model == "online":
            return self.online(z)
        elif model == "target":
            return self.target(z)
        raise ValueError(f"Unknown model: {model}")

    def sync_target(self):
        """Copy online network weights to target network."""
        self.target.load_state_dict(self.online.state_dict())


class DreamerDDQN(nn.Module):
    """
    Dreamer-style DDQN: World Model encoder + Latent Q-network.
    
    Drop-in replacement for DoubleDQN that operates on learned latent space
    instead of raw pixels. Compatible with distributed DDQN training.
    
    Architecture:
        Frame (4,64,64) → Encoder → z (latent) → LatentDDQN → Q(a)
    
    Benefits over pixel-based DDQN:
    - Smaller Q-network (MLP vs CNN)
    - Learned representations can generalize across levels
    - Can optionally use imagination for planning
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_actions: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.freeze_encoder = freeze_encoder
        
        # Encoder: frames → latent
        self.encoder = FrameEncoder(input_shape, latent_dim)
        
        # Q-networks operating on latent space
        self.online = LatentDuelingDQN(latent_dim, num_actions, hidden_dim)
        self.target = LatentDuelingDQN(latent_dim, num_actions, hidden_dim)
        
        # Copy weights and freeze target
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False
        
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def encode(self, x: Tensor, deterministic: bool = True) -> Tensor:
        """Encode frames to latent representation."""
        return self.encoder.encode(x, deterministic)

    def forward(self, x: Tensor, network: str = "online") -> Tensor:
        """
        Forward pass: frames → latent → Q-values.
        
        Args:
            x: Observation tensor (N, C, H, W) normalized to [0, 1]
            network: "online" or "target"
        
        Returns:
            Q-values for each action (N, num_actions)
        """
        # Encode frames to latent (always deterministic for Q-learning)
        z = self.encoder.encode(x, deterministic=True)
        
        # Compute Q-values from latent
        if network == "online":
            return self.online(z)
        elif network == "target":
            return self.target(z)
        raise ValueError(f"Unknown network: {network}")

    def get_action(self, x: Tensor, epsilon: float = 0.0) -> Tensor:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            x: Observation tensor (N, C, H, W)
            epsilon: Exploration rate (0 = greedy, 1 = random)
        
        Returns:
            action: Selected action indices (N,)
        """
        import numpy as np
        
        if np.random.random() < epsilon:
            batch_size = x.shape[0]
            return torch.randint(0, self.num_actions, (batch_size,), device=x.device)
        else:
            with torch.no_grad():
                q_values = self.forward(x)
                return q_values.argmax(dim=1)

    def sync_target(self):
        """Copy online network weights to target network."""
        self.target.load_state_dict(self.online.state_dict())
    
    def soft_update_target(self, tau: float = 0.005):
        """Soft update target network: θ_target = τ*θ_online + (1-τ)*θ_target."""
        for target_param, online_param in zip(
            self.target.parameters(), self.online.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )

    def compute_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
        gamma: float = 0.99,
    ) -> Tuple[Tensor, dict]:
        """
        Compute Double DQN loss in latent space.
        
        Returns:
            loss: Scalar loss tensor
            metrics: Dict with q_mean, td_error, etc.
        """
        # Encode states to latent
        z = self.encoder.encode(states, deterministic=True)
        z_next = self.encoder.encode(next_states, deterministic=True)
        
        # Current Q-values
        q_values = self.online(z)
        q_current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use online network to select action, target to evaluate
        with torch.no_grad():
            q_next_online = self.online(z_next)
            best_actions = q_next_online.argmax(dim=1)
            
            q_next_target = self.target(z_next)
            q_next = q_next_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            
            # Target: r + γ * Q_target(s', argmax Q_online(s'))
            q_target = rewards + gamma * q_next * (1 - dones.float())
        
        # Huber loss (more stable than MSE for RL)
        loss = F.smooth_l1_loss(q_current, q_target)
        
        # Metrics
        td_error = (q_current - q_target).abs().mean().item()
        q_mean = q_values.mean().item()
        
        return loss, {"q_mean": q_mean, "td_error": td_error}
