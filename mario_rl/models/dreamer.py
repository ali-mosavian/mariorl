"""Dreamer Model for model-based RL.

Architecture:
    Frame → Encoder → z (latent)
                          ↓
              ┌───────────┼───────────┐
              ↓           ↓           ↓
           Actor       Critic     Dynamics
         (policy)     (value)    (z, a → z')
              ↓           ↓           ↓
           logits      value     z_next, r

Dreamer learns a world model that can imagine future trajectories,
then trains an actor-critic policy on those imagined experiences.

Key components:
- Encoder: CNN that maps frames to latent space
- Dynamics: GRU-based model that predicts next latent given current + action
- Reward predictor: MLP that predicts reward from latent
- Actor: MLP that outputs action logits from latent
- Critic: MLP that outputs value estimate from latent

Architecture improvements:
- Dropout everywhere for regularization
- LayerNorm between layers for stable gradients
- Residual connections for gradient flow
- Deeper CNN with smaller kernels (3x3) and MaxPool
"""

from dataclasses import dataclass

import torch
import numpy as np
from torch import nn
from torch import Tensor
import torch.nn.functional as F


def _layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer with orthogonal weights and constant bias."""
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.orthogonal_(layer.weight, std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)
    return layer


# =============================================================================
# ResNet Building Blocks
# =============================================================================


class ResidualMLPBlock(nn.Module):
    """Residual MLP block with LayerNorm and Dropout."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            _layer_init(nn.Linear(dim, dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            _layer_init(nn.Linear(dim, dim)),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class ResNetBasicBlock(nn.Module):
    """ResNet BasicBlock: two 3x3 convs with skip connection.

    Pre-activation style (GroupNorm -> GELU -> Conv) for better gradient flow.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Main path
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = _layer_init(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        )
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = _layer_init(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.dropout = nn.Dropout2d(dropout)

        # Skip connection (identity or projection)
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = _layer_init(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)

        out = self.norm1(x)
        out = F.gelu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        return out + identity


class ResNetBottleneck(nn.Module):
    """ResNet Bottleneck: 1x1 -> 3x3 -> 1x1 with skip connection.

    Pre-activation style for better gradient flow.
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        stride: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        out_channels = mid_channels * self.expansion

        # Main path: 1x1 reduce -> 3x3 -> 1x1 expand
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = _layer_init(nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False))

        self.norm2 = nn.GroupNorm(min(8, mid_channels), mid_channels)
        self.conv2 = _layer_init(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        )

        self.norm3 = nn.GroupNorm(min(8, mid_channels), mid_channels)
        self.conv3 = _layer_init(nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False))

        self.dropout = nn.Dropout2d(dropout)

        # Skip connection
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = _layer_init(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)

        out = self.norm1(x)
        out = F.gelu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = F.gelu(out)
        out = self.conv2(out)

        out = self.norm3(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.conv3(out)

        return out + identity


class ResNetUpsampleBlock(nn.Module):
    """ResNet block for decoder with upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Upsample first if needed
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest") if upsample else nn.Identity()

        # Main path
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = _layer_init(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))

        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = _layer_init(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))

        self.dropout = nn.Dropout2d(dropout)

        # Skip connection with channel adjustment
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = _layer_init(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        identity = self.skip(x)

        out = self.norm1(x)
        out = F.gelu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        return out + identity


@dataclass(frozen=True)
class DreamerConfig:
    """Configuration for DreamerModel."""

    input_shape: tuple[int, int, int]
    num_actions: int
    latent_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1


class Encoder(nn.Module):
    """ResNet-style encoder that stops at 32x32 spatial resolution.

    Architecture: ResNet blocks with stride-2 downsampling, stopping at 32x32.
    Uses pre-activation (GroupNorm -> GELU -> Conv) for better gradient flow.

    64x64 input -> 32x32 (with 128 channels) -> flatten -> latent
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        latent_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        c, h, w = input_shape
        self.latent_dim = latent_dim

        # Stem: 64x64, C -> 64 channels
        self.stem = nn.Sequential(
            _layer_init(nn.Conv2d(c, 64, kernel_size=7, stride=1, padding=3, bias=False)),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

        # Stage 1: 64x64 -> 32x32, 64 -> 64 channels (2 blocks, first with stride 2)
        self.stage1 = nn.Sequential(
            ResNetBasicBlock(64, 64, stride=2, dropout=dropout),  # downsample
            ResNetBasicBlock(64, 64, stride=1, dropout=dropout),
        )

        # Stage 2: 32x32, 64 -> 128 channels (2 blocks, no downsample)
        self.stage2 = nn.Sequential(
            ResNetBasicBlock(64, 128, stride=1, dropout=dropout),  # channel expansion
            ResNetBasicBlock(128, 128, stride=1, dropout=dropout),
        )

        # Final norm before flatten
        self.final_norm = nn.GroupNorm(8, 128)

        # Compute flattened size (should be 128 * 32 * 32 = 131072 for 64x64 input)
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            x = self.stem(dummy)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.final_norm(x)
            self.spatial_size = x.shape[2]  # Should be 32
            flat_size = x.flatten(1).shape[1]

        # Project to latent with residual MLP
        self.fc_pre = nn.Sequential(
            _layer_init(nn.Linear(flat_size, latent_dim * 4)),
            nn.LayerNorm(latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(latent_dim * 4, dropout),
            _layer_init(nn.Linear(latent_dim * 4, latent_dim * 2)),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # VAE-style outputs for stochastic latent
        self.fc_mu = _layer_init(nn.Linear(latent_dim * 2, latent_dim))
        self.fc_logvar = _layer_init(nn.Linear(latent_dim * 2, latent_dim))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode frames to latent distribution parameters.

        Args:
            x: Frames (batch, C, H, W) in [0, 1]

        Returns:
            mu, logvar: Mean and log-variance of latent distribution
        """
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.final_norm(h)
        h = F.gelu(h)
        h = h.flatten(1)
        h = self.fc_pre(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def sample(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for VAE sampling."""
        logvar = logvar.clamp(-10, 2)  # Clamp for numerical stability
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: Tensor, deterministic: bool = True) -> Tensor:
        """Get latent representation."""
        mu, logvar = self.forward(x)
        return mu if deterministic else self.sample(mu, logvar)


class Dynamics(nn.Module):
    """GRU-based dynamics model with residual connections and LayerNorm.

    Predicts next latent from current latent + action using a GRU with
    residual MLPs for better gradient flow.
    """

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        # Action embedding with LayerNorm
        self.action_embed = nn.Sequential(
            nn.Embedding(num_actions, 64),
            nn.LayerNorm(64),
        )

        # Pre-GRU projection with residual
        self.pre_gru = nn.Sequential(
            _layer_init(nn.Linear(latent_dim + 64, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # GRU for temporal modeling
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.gru_norm = nn.LayerNorm(hidden_dim)

        # Post-GRU residual MLP
        self.post_gru = nn.Sequential(
            ResidualMLPBlock(hidden_dim, dropout),
            ResidualMLPBlock(hidden_dim, dropout),
        )

        # Predict next latent distribution
        self.fc_mu = _layer_init(nn.Linear(hidden_dim, latent_dim))
        self.fc_logvar = _layer_init(nn.Linear(hidden_dim, latent_dim))

        # Residual skip: project z to match output for residual connection
        self.z_skip = nn.Sequential(
            _layer_init(nn.Linear(latent_dim, latent_dim)),
            nn.LayerNorm(latent_dim),
        )

    def forward(
        self,
        z: Tensor,
        action: Tensor,
        h: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Predict next latent given current latent and action.

        Args:
            z: Current latent (batch, latent_dim)
            action: Action taken (batch,)
            h: GRU hidden state (batch, hidden_dim) or None

        Returns:
            z_next: Predicted next latent (sampled)
            h_next: Next GRU hidden state
            z_mu: Mean of predicted distribution (for KL loss)
        """
        batch_size = z.shape[0]

        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=z.device)

        # Embed action and concatenate with latent
        a_embed = self.action_embed(action)
        x = torch.cat([z, a_embed], dim=-1)

        # Project to GRU hidden dim
        x = self.pre_gru(x)

        # GRU update with LayerNorm
        h_next = self.gru(x, h)
        h_next = self.gru_norm(h_next)

        # Residual connection around GRU
        h_next = h_next + x

        # Post-GRU residual processing
        out = self.post_gru(h_next)

        # Predict next latent distribution
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out).clamp(-10, 2)  # Clamp for stability

        # Sample next latent with residual from current z
        std = torch.exp(0.5 * logvar)
        z_delta = mu + std * torch.randn_like(std)

        # Residual connection: z_next = z + delta (helps learn incremental changes)
        z_next = self.z_skip(z) + z_delta

        return z_next, h_next, mu


class RewardPredictor(nn.Module):
    """MLP that predicts reward from latent state with residual connections."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            _layer_init(nn.Linear(latent_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Sequential(
            ResidualMLPBlock(hidden_dim, dropout),
            ResidualMLPBlock(hidden_dim, dropout),
        )
        self.output = _layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, z: Tensor) -> Tensor:
        """Predict reward from latent."""
        h = self.input_proj(z)
        h = self.residual(h)
        return self.output(h).squeeze(-1)


class DonePredictor(nn.Module):
    """MLP that predicts episode termination from latent state."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            _layer_init(nn.Linear(latent_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = ResidualMLPBlock(hidden_dim, dropout)
        self.output = _layer_init(nn.Linear(hidden_dim, 1))

    def forward(self, z: Tensor) -> Tensor:
        """Predict done probability from latent."""
        h = self.input_proj(z)
        h = self.residual(h)
        return torch.sigmoid(self.output(h)).squeeze(-1)


class Actor(nn.Module):
    """MLP that outputs action logits from latent state with residual connections."""

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            _layer_init(nn.Linear(latent_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Sequential(
            ResidualMLPBlock(hidden_dim, dropout),
            ResidualMLPBlock(hidden_dim, dropout),
            ResidualMLPBlock(hidden_dim, dropout),
        )
        self.output = _layer_init(nn.Linear(hidden_dim, num_actions), std=0.01)

    def forward(self, z: Tensor) -> Tensor:
        """Get action logits from latent."""
        h = self.input_proj(z)
        h = self.residual(h)
        return self.output(h)


class Critic(nn.Module):
    """MLP that outputs value estimate from latent state with residual connections."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            _layer_init(nn.Linear(latent_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Sequential(
            ResidualMLPBlock(hidden_dim, dropout),
            ResidualMLPBlock(hidden_dim, dropout),
            ResidualMLPBlock(hidden_dim, dropout),
        )
        self.output = _layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, z: Tensor) -> Tensor:
        """Get value estimate from latent."""
        h = self.input_proj(z)
        h = self.residual(h)
        return self.output(h).squeeze(-1)


class Decoder(nn.Module):
    """ResNet-style decoder that starts from 32x32 spatial resolution.

    Architecture mirrors the Encoder: starts at 32x32, upsamples to 64x64.
    Uses ResNet upsample blocks for gradient flow.

    latent -> 32x32 (128 channels) -> 64x64 -> output
    """

    def __init__(
        self,
        output_shape: tuple[int, int, int],
        latent_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_shape = output_shape
        c, h, w = output_shape

        # Start from 32x32 spatial size (matches encoder output)
        self.h_conv = 32
        self.w_conv = 32
        self.conv_channels = 128  # Match encoder output channels

        # Project latent to conv feature size with residual MLP
        flat_size = self.conv_channels * self.h_conv * self.w_conv  # 128 * 32 * 32 = 131072
        self.fc = nn.Sequential(
            _layer_init(nn.Linear(latent_dim, latent_dim * 2)),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            _layer_init(nn.Linear(latent_dim * 2, latent_dim * 4)),
            nn.LayerNorm(latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(latent_dim * 4, dropout),
            _layer_init(nn.Linear(latent_dim * 4, flat_size)),
            nn.LayerNorm(flat_size),
            nn.GELU(),
        )

        # Stage 1: 32x32, 128 -> 128 channels (2 ResNet blocks, no upsample)
        self.stage1 = nn.Sequential(
            ResNetUpsampleBlock(128, 128, upsample=False, dropout=dropout),
            ResNetUpsampleBlock(128, 128, upsample=False, dropout=dropout),
        )

        # Stage 2: 32x32 -> 64x64, 128 -> 64 channels
        self.stage2 = nn.Sequential(
            ResNetUpsampleBlock(128, 64, upsample=True, dropout=dropout),  # upsample
            ResNetUpsampleBlock(64, 64, upsample=False, dropout=dropout),
        )

        # Final output: 64x64, 64 -> C channels
        self.final_norm = nn.GroupNorm(8, 64)
        self.output = nn.Sequential(
            _layer_init(nn.Conv2d(64, 64, kernel_size=3, padding=1)),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            _layer_init(nn.Conv2d(64, c, kernel_size=7, padding=3)),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(self, z: Tensor) -> Tensor:
        """Decode latent to image.

        Args:
            z: Latent representation (batch, latent_dim)

        Returns:
            Reconstructed image (batch, C, H, W) in [0, 1]
        """
        batch_size = z.shape[0]

        # Project and reshape to conv feature map
        h = self.fc(z)
        h = h.view(batch_size, self.conv_channels, self.h_conv, self.w_conv)

        # Decode through ResNet stages
        h = self.stage1(h)
        h = self.stage2(h)

        # Final output
        h = self.final_norm(h)
        h = F.gelu(h)
        return self.output(h)


# =============================================================================
# SSIM Loss
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
    """Compute Structural Similarity Index (SSIM) between two images.

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


class DreamerModel(nn.Module):
    """Dreamer: World Model + Actor-Critic for model-based RL.

    The model can:
    1. Encode observations to latent space
    2. Imagine future trajectories using learned dynamics
    3. Evaluate actions using actor (policy) and critic (value)

    During training:
    - World model is trained on real experience (reconstruction + dynamics)
    - Actor-Critic is trained on imagined trajectories

    Architecture improvements:
    - Dropout everywhere for regularization
    - LayerNorm/GroupNorm for stable gradients
    - Residual connections for gradient flow
    - Deeper CNN with 3x3 kernels and MaxPool
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_actions: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.latent_dim = latent_dim

        # World model components
        self.encoder = Encoder(input_shape, latent_dim, dropout)
        self.decoder = Decoder(input_shape, latent_dim, dropout)
        self.dynamics = Dynamics(latent_dim, num_actions, hidden_dim, dropout)
        self.reward_pred = RewardPredictor(latent_dim, hidden_dim, dropout)
        self.done_pred = DonePredictor(latent_dim, hidden_dim, dropout)

        # Actor-Critic components
        self.actor = Actor(latent_dim, num_actions, hidden_dim, dropout)
        self.critic = Critic(latent_dim, hidden_dim, dropout)

    def encode(self, x: Tensor, deterministic: bool = True) -> Tensor:
        """Encode observations to latent space.

        Args:
            x: Observations (batch, C, H, W) in [0, 255] range
            deterministic: If True, use mean; if False, sample from distribution

        Returns:
            z: Latent representation (batch, latent_dim)
        """
        # Normalize from [0, 255] to [0, 1]
        x = x / 255.0
        return self.encoder.encode(x, deterministic)

    def imagine_step(
        self,
        z: Tensor,
        actions: Tensor,
        h: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Imagine one step forward.

        Args:
            z: Current latent (batch, latent_dim)
            actions: Actions to take (batch,)
            h: Optional GRU hidden state

        Returns:
            z_next: Next latent (batch, latent_dim)
            reward: Predicted reward (batch,)
            done: Predicted done probability (batch,)
        """
        z_next, _, _ = self.dynamics(z, actions, h)
        reward = self.reward_pred(z_next)
        done = self.done_pred(z_next)
        return z_next, reward, done

    def imagine_trajectory(
        self,
        z_start: Tensor,
        horizon: int = 15,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Imagine a trajectory by sampling actions from policy.

        Args:
            z_start: Starting latent (batch, latent_dim)
            horizon: Number of steps to imagine

        Returns:
            z_traj: Latent trajectory (batch, horizon+1, latent_dim)
            rewards: Predicted rewards (batch, horizon)
            dones: Predicted dones (batch, horizon)
        """
        batch_size = z_start.shape[0]
        device = z_start.device

        z_traj = [z_start]
        rewards = []
        dones = []

        z = z_start
        h = None

        for _ in range(horizon):
            # Sample action from policy
            logits = self.actor(z)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()

            # Imagine step
            z_next, h, _ = self.dynamics(z, action, h)
            reward = self.reward_pred(z_next)
            done = self.done_pred(z_next)

            z_traj.append(z_next)
            rewards.append(reward)
            dones.append(done)

            z = z_next

        return (
            torch.stack(z_traj, dim=1),
            torch.stack(rewards, dim=1),
            torch.stack(dones, dim=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: encode and return action logits.

        Args:
            x: Observations (batch, C, H, W)

        Returns:
            Action logits (batch, num_actions)
        """
        z = self.encode(x, deterministic=True)
        return self.actor(z)

    def select_action(self, x: Tensor, epsilon: float = 0.0) -> Tensor:
        """Get action using policy with optional exploration.

        Args:
            x: Observations (batch, C, H, W)
            epsilon: Random action probability

        Returns:
            actions: Selected actions (batch,)
        """
        if np.random.random() < epsilon:
            return torch.randint(0, self.num_actions, (x.shape[0],), device=x.device)

        with torch.no_grad():
            logits = self.forward(x)
            # Sample from policy distribution (not argmax - stochastic policy)
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample()
