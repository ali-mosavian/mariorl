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
"""

from dataclasses import dataclass

import torch
import numpy as np
from torch import nn
from torch import Tensor
import torch.nn.functional as F


def _layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer with orthogonal weights and constant bias."""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)
    return layer


@dataclass(frozen=True)
class DreamerConfig:
    """Configuration for DreamerModel."""

    input_shape: tuple[int, int, int]
    num_actions: int
    latent_dim: int = 128
    hidden_dim: int = 256


class Encoder(nn.Module):
    """CNN encoder that maps frames to latent space with VAE-style output."""

    def __init__(self, input_shape: tuple[int, int, int], latent_dim: int) -> None:
        super().__init__()
        c, h, w = input_shape
        self.latent_dim = latent_dim

        # Nature DQN-style CNN backbone
        self.conv = nn.Sequential(
            _layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.GELU(),
            _layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.GELU(),
            _layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.GELU(),
            nn.Flatten(),
        )

        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_size = self.conv(dummy).shape[1]

        # VAE-style outputs for stochastic latent
        self.fc_mu = _layer_init(nn.Linear(flat_size, latent_dim))
        self.fc_logvar = _layer_init(nn.Linear(flat_size, latent_dim))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode frames to latent distribution parameters.

        Args:
            x: Frames (batch, C, H, W) in [0, 1]

        Returns:
            mu, logvar: Mean and log-variance of latent distribution
        """
        h = self.conv(x)
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
    """GRU-based dynamics model that predicts next latent from current + action."""

    def __init__(self, latent_dim: int, num_actions: int, hidden_dim: int = 256) -> None:
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
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.GELU(),
        )
        self.fc_mu = _layer_init(nn.Linear(hidden_dim, latent_dim))
        self.fc_logvar = _layer_init(nn.Linear(hidden_dim, latent_dim))

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

        # GRU update
        h_next = self.gru(x, h)

        # Predict next latent distribution
        out = self.fc_out(h_next)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out).clamp(-10, 2)  # Clamp for stability

        # Sample next latent
        std = torch.exp(0.5 * logvar)
        z_next = mu + std * torch.randn_like(std)

        return z_next, h_next, mu


class RewardPredictor(nn.Module):
    """MLP that predicts reward from latent state."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(latent_dim, hidden_dim)),
            nn.GELU(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.GELU(),
            _layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Predict reward from latent."""
        return self.net(z).squeeze(-1)


class DonePredictor(nn.Module):
    """MLP that predicts episode termination from latent state."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(latent_dim, hidden_dim)),
            nn.GELU(),
            _layer_init(nn.Linear(hidden_dim, 1)),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Predict done probability from latent."""
        return torch.sigmoid(self.net(z)).squeeze(-1)


class Actor(nn.Module):
    """MLP that outputs action logits from latent state."""

    def __init__(self, latent_dim: int, num_actions: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(latent_dim, hidden_dim)),
            nn.GELU(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.GELU(),
            _layer_init(nn.Linear(hidden_dim, num_actions), std=0.01),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Get action logits from latent."""
        return self.net(z)


class Critic(nn.Module):
    """MLP that outputs value estimate from latent state."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(latent_dim, hidden_dim)),
            nn.GELU(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.GELU(),
            _layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Get value estimate from latent."""
        return self.net(z).squeeze(-1)


class DreamerModel(nn.Module):
    """Dreamer: World Model + Actor-Critic for model-based RL.

    The model can:
    1. Encode observations to latent space
    2. Imagine future trajectories using learned dynamics
    3. Evaluate actions using actor (policy) and critic (value)

    During training:
    - World model is trained on real experience (reconstruction + dynamics)
    - Actor-Critic is trained on imagined trajectories
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_actions: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.latent_dim = latent_dim

        # World model components
        self.encoder = Encoder(input_shape, latent_dim)
        self.dynamics = Dynamics(latent_dim, num_actions, hidden_dim)
        self.reward_pred = RewardPredictor(latent_dim, hidden_dim)
        self.done_pred = DonePredictor(latent_dim, hidden_dim)

        # Actor-Critic components
        self.actor = Actor(latent_dim, num_actions, hidden_dim)
        self.critic = Critic(latent_dim, hidden_dim)

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
