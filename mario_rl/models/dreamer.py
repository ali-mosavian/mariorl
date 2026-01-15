"""Dreamer V3 Model for model-based RL.

Architecture:
    Frame → Encoder → z (categorical latent)
                          ↓
              ┌───────────┼───────────┐
              ↓           ↓           ↓
           Actor       Critic     Dynamics
         (policy)     (value)    (z, a → z')
              ↓           ↓           ↓
           logits      value     z_next, r

Dreamer V3 key features:
- Categorical latents (32 categoricals × 32 classes) instead of Gaussian
- Symlog transform for scale-invariant predictions
- Simple free bits KL instead of complex balancing
- No auxiliary collapse-prevention losses needed

Key components:
- Encoder: CNN that maps frames to categorical latent space
- Dynamics: GRU-based model that predicts next latent given current + action
- Reward predictor: MLP that predicts symlog(reward) from latent
- Actor: MLP that outputs action logits from latent
- Critic: MLP that outputs symlog(value) estimate from latent
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


# =============================================================================
# Symlog Transform (V3 key feature)
# =============================================================================


def symlog(x: Tensor) -> Tensor:
    """Symmetric logarithm: sign(x) * ln(|x| + 1).
    
    Compresses large values while preserving small values and sign.
    Used for scale-invariant predictions of rewards, values, and reconstruction.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# =============================================================================
# Helpers
# =============================================================================


def _layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer with orthogonal weights and constant bias."""
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.orthogonal_(layer.weight, std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)
    return layer


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class DreamerConfig:
    """Configuration for DreamerModel V3."""

    input_shape: tuple[int, int, int]
    num_actions: int
    
    # Categorical latent space (V3 style)
    num_categoricals: int = 32  # Number of categorical distributions
    num_classes: int = 32       # Classes per categorical
    
    # Network dimensions
    hidden_dim: int = 256
    
    # KL regularization
    free_bits: float = 1.0      # Minimum KL per categorical
    
    @property
    def latent_dim(self) -> int:
        """Effective latent dimension (for compatibility)."""
        return self.num_categoricals * self.num_classes


# =============================================================================
# Categorical Latent Space (V3 key feature)
# =============================================================================


class CategoricalEncoder(nn.Module):
    """CNN encoder that maps frames to categorical latent space.

    Uses Nature DQN-style CNN backbone.
    Output: 32 categorical distributions with 32 classes each.
    Sampling uses straight-through gradients.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_categoricals: int = 32,
        num_classes: int = 32,
    ) -> None:
        super().__init__()
        c, h, w = input_shape
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes

        # Nature DQN-style CNN backbone
        self.conv = nn.Sequential(
            _layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.LayerNorm([32, 15, 15]),  # After 64x64 -> 15x15
            nn.SiLU(),
            _layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.LayerNorm([64, 6, 6]),
            nn.SiLU(),
            _layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.LayerNorm([64, 4, 4]),
            nn.SiLU(),
            nn.Flatten(),
        )

        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_size = self.conv(dummy).shape[1]

        # Project to categorical logits
        self.fc = nn.Sequential(
            _layer_init(nn.Linear(flat_size, 512)),
            nn.LayerNorm(512),
            nn.SiLU(),
            _layer_init(nn.Linear(512, num_categoricals * num_classes)),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode frames to categorical logits.

        Args:
            x: Frames (batch, C, H, W) in [0, 1]

        Returns:
            logits: (batch, num_categoricals, num_classes)
        """
        h = self.conv(x)
        logits = self.fc(h)
        return logits.view(-1, self.num_categoricals, self.num_classes)

    def sample(self, logits: Tensor) -> Tensor:
        """Sample from categorical with straight-through gradients.
        
        Args:
            logits: (batch, num_categoricals, num_classes)
            
        Returns:
            z: One-hot samples (batch, num_categoricals, num_classes)
               Gradients flow through softmax probabilities.
        """
        probs = F.softmax(logits, dim=-1)
        # Straight-through: sample but pass gradients through probs
        samples = F.one_hot(probs.argmax(dim=-1), self.num_classes).float()
        # Add small uniform noise for exploration (unimix from V3)
        uniform = torch.ones_like(probs) / self.num_classes
        probs_mixed = 0.99 * probs + 0.01 * uniform
        return samples + probs_mixed - probs_mixed.detach()

    def encode(self, x: Tensor, deterministic: bool = True) -> Tensor:
        """Get flattened latent representation.
        
        Args:
            x: Frames (batch, C, H, W) in [0, 1]
            deterministic: If True, use mode; if False, sample
            
        Returns:
            z: Flattened latent (batch, num_categoricals * num_classes)
        """
        logits = self.forward(x)
        if deterministic:
            # Use mode (one-hot of argmax)
            z = F.one_hot(logits.argmax(dim=-1), self.num_classes).float()
        else:
            z = self.sample(logits)
        return z.view(-1, self.num_categoricals * self.num_classes)


class CategoricalDynamics(nn.Module):
    """GRU-based dynamics model with categorical latent predictions."""

    def __init__(
        self,
        num_categoricals: int,
        num_classes: int,
        num_actions: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.latent_dim = num_categoricals * num_classes
        self.hidden_dim = hidden_dim

        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Embedding(num_actions, 32),
            nn.LayerNorm(32),
        )

        # GRU for temporal modeling
        self.gru = nn.GRUCell(self.latent_dim + 32, hidden_dim)
        self.gru_norm = nn.LayerNorm(hidden_dim)

        # Predict next latent categorical logits
        self.fc_out = nn.Sequential(
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            _layer_init(nn.Linear(hidden_dim, num_categoricals * num_classes)),
        )

    def forward(
        self,
        z: Tensor,
        action: Tensor,
        h: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Predict next latent given current latent and action.

        Args:
            z: Current latent (batch, latent_dim) - flattened categorical
            action: Action taken (batch,)
            h: GRU hidden state (batch, hidden_dim) or None

        Returns:
            z_next: Predicted next latent (batch, latent_dim) - sampled
            h_next: Next GRU hidden state
            logits: Raw logits (batch, num_categoricals, num_classes) for KL
        """
        batch_size = z.shape[0]

        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=z.device)

        # Embed action and concatenate with latent
        a_embed = self.action_embed(action)
        x = torch.cat([z, a_embed], dim=-1)

        # GRU update
        h_next = self.gru(x, h)
        h_next = self.gru_norm(h_next)

        # Predict next latent logits
        logits = self.fc_out(h_next).view(-1, self.num_categoricals, self.num_classes)
        
        # Sample with straight-through
        probs = F.softmax(logits, dim=-1)
        samples = F.one_hot(probs.argmax(dim=-1), self.num_classes).float()
        z_next = samples + probs - probs.detach()
        z_next = z_next.view(-1, self.latent_dim)

        return z_next, h_next, logits


class RewardPredictor(nn.Module):
    """MLP that predicts symlog(reward) from latent state."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(latent_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            _layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Predict symlog(reward) from latent."""
        return self.net(z).squeeze(-1)


class ContinuePredictor(nn.Module):
    """MLP that predicts continuation probability from latent state."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(latent_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            _layer_init(nn.Linear(hidden_dim, 1)),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Predict continuation probability (1 - done) from latent."""
        return torch.sigmoid(self.net(z)).squeeze(-1)


class Actor(nn.Module):
    """MLP that outputs action logits from latent state."""

    def __init__(self, latent_dim: int, num_actions: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(latent_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            _layer_init(nn.Linear(hidden_dim, num_actions), std=0.01),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Get action logits from latent."""
        return self.net(z)


class Critic(nn.Module):
    """MLP that outputs symlog(value) estimate from latent state."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(latent_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            _layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Get symlog(value) estimate from latent."""
        return self.net(z).squeeze(-1)


class Decoder(nn.Module):
    """Transposed CNN decoder that reconstructs images from latent space."""

    def __init__(self, output_shape: tuple[int, int, int], latent_dim: int) -> None:
        super().__init__()
        self.output_shape = output_shape
        c, h, w = output_shape

        # Spatial dimensions for intermediate conv
        self.h_conv = 4
        self.w_conv = 4
        self.conv_channels = 64

        # Project latent to conv feature size
        self.fc = nn.Sequential(
            _layer_init(nn.Linear(latent_dim, 512)),
            nn.LayerNorm(512),
            nn.SiLU(),
            _layer_init(nn.Linear(512, self.conv_channels * self.h_conv * self.w_conv)),
            nn.LayerNorm(self.conv_channels * self.h_conv * self.w_conv),
            nn.SiLU(),
        )

        # Transposed convolutions (reverse of encoder)
        self.deconv = nn.Sequential(
            _layer_init(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1)),
            nn.LayerNorm([64, 6, 6]),
            nn.SiLU(),
            _layer_init(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, output_padding=1)),
            nn.LayerNorm([32, 15, 15]),
            nn.SiLU(),
            _layer_init(nn.ConvTranspose2d(32, c, kernel_size=8, stride=4)),
            # No sigmoid - output in symlog space for MSE loss
        )

    def forward(self, z: Tensor) -> Tensor:
        """Decode latent to image (in symlog space).

        Args:
            z: Latent representation (batch, latent_dim)

        Returns:
            Reconstructed image (batch, C, H, W) - NOT normalized to [0,1]
        """
        batch_size = z.shape[0]
        h = self.fc(z)
        h = h.view(batch_size, self.conv_channels, self.h_conv, self.w_conv)
        return self.deconv(h)


# =============================================================================
# Main Model
# =============================================================================


class DreamerModel(nn.Module):
    """Dreamer V3: World Model + Actor-Critic for model-based RL.

    Key V3 features:
    - Categorical latent space (32 × 32)
    - Symlog predictions for scale invariance
    - Simple architecture without collapse-prevention hacks
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_actions: int,
        latent_dim: int = 1024,  # For compatibility (= 32*32)
        hidden_dim: int = 256,
        num_categoricals: int = 32,
        num_classes: int = 32,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.latent_dim = num_categoricals * num_classes

        # World model components
        self.encoder = CategoricalEncoder(input_shape, num_categoricals, num_classes)
        self.decoder = Decoder(input_shape, self.latent_dim)
        self.dynamics = CategoricalDynamics(num_categoricals, num_classes, num_actions, hidden_dim)
        self.reward_pred = RewardPredictor(self.latent_dim, hidden_dim)
        self.continue_pred = ContinuePredictor(self.latent_dim, hidden_dim)

        # Actor-Critic components
        self.actor = Actor(self.latent_dim, num_actions, hidden_dim)
        self.critic = Critic(self.latent_dim, hidden_dim)

    def encode(self, x: Tensor, deterministic: bool = True) -> Tensor:
        """Encode observations to latent space.

        Args:
            x: Observations (batch, C, H, W) in [0, 255] range
            deterministic: If True, use mode; if False, sample

        Returns:
            z: Latent representation (batch, latent_dim)
        """
        x = x / 255.0
        return self.encoder.encode(x, deterministic)

    def encode_with_logits(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode and return both latent and logits (for KL computation).
        
        Args:
            x: Observations (batch, C, H, W) in [0, 255] range
            
        Returns:
            z: Sampled latent (batch, latent_dim)
            logits: Categorical logits (batch, num_categoricals, num_classes)
        """
        x = x / 255.0
        logits = self.encoder(x)
        z = self.encoder.sample(logits)
        return z.view(-1, self.latent_dim), logits

    def imagine_step(
        self,
        z: Tensor,
        actions: Tensor,
        h: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Imagine one step forward.

        Args:
            z: Current latent (batch, latent_dim)
            actions: Actions to take (batch,)
            h: Optional GRU hidden state

        Returns:
            z_next: Next latent (batch, latent_dim)
            h_next: Next hidden state
            reward: Predicted symlog(reward) (batch,)
            cont: Predicted continuation probability (batch,)
        """
        z_next, h_next, _ = self.dynamics(z, actions, h)
        reward = self.reward_pred(z_next)
        cont = self.continue_pred(z_next)
        return z_next, h_next, reward, cont

    def imagine_trajectory(
        self,
        z_start: Tensor,
        horizon: int = 15,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Imagine a trajectory by sampling actions from policy.

        Args:
            z_start: Starting latent (batch, latent_dim)
            horizon: Number of steps to imagine

        Returns:
            z_traj: Latent trajectory (batch, horizon+1, latent_dim)
            rewards: Predicted symlog rewards (batch, horizon)
            conts: Predicted continuation probs (batch, horizon)
            logits_traj: Dynamics logits for KL (batch, horizon, num_cat, num_class)
        """
        z_traj = [z_start]
        rewards = []
        conts = []
        logits_list = []

        z = z_start
        h = None

        for _ in range(horizon):
            # Sample action from policy
            action_logits = self.actor(z)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample()

            # Imagine step
            z_next, h, logits = self.dynamics(z, action, h)
            reward = self.reward_pred(z_next)
            cont = self.continue_pred(z_next)

            z_traj.append(z_next)
            rewards.append(reward)
            conts.append(cont)
            logits_list.append(logits)

            z = z_next

        return (
            torch.stack(z_traj, dim=1),
            torch.stack(rewards, dim=1),
            torch.stack(conts, dim=1),
            torch.stack(logits_list, dim=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: encode and return action logits."""
        z = self.encode(x, deterministic=True)
        return self.actor(z)

    def select_action(self, x: Tensor, epsilon: float = 0.0) -> Tensor:
        """Get action using policy with optional exploration."""
        if np.random.random() < epsilon:
            return torch.randint(0, self.num_actions, (x.shape[0],), device=x.device)

        with torch.no_grad():
            logits = self.forward(x)
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample()
