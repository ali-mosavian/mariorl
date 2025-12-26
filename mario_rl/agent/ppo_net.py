"""
Actor-Critic network for PPO.

Shared CNN backbone with separate actor (policy) and critic (value) heads.

Includes:
- GELU activation (smoother than ReLU)
- LayerNorm for stability
- Orthogonal initialization (standard for PPO)
"""

from typing import Tuple

import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer with orthogonal weights and constant bias."""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOBackbone(nn.Module):
    """
    Shared CNN backbone for actor-critic (Nature DQN style).

    Takes frame-stacked observations (N, C, H, W) where C is the frame stack.
    Input should already be normalized to [0, 1] and in channels-first format.

    Improvements over vanilla:
    - GELU activation (smoother gradients)
    - Dropout2d between conv layers (drops entire feature maps)
    - LayerNorm after flatten (stabilizes training)
    - Orthogonal initialization (standard for PPO)
    """

    def __init__(self, input_shape: Tuple[int, ...], feature_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        # input_shape: (C, H, W) where C = num stacked frames
        c, h, w = input_shape

        # Conv layers with orthogonal init
        self.conv1 = layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            dummy = torch.nn.functional.gelu(self.conv1(dummy))
            dummy = torch.nn.functional.gelu(self.conv2(dummy))
            dummy = torch.nn.functional.gelu(self.conv3(dummy))
            flat_size = dummy.flatten(1).shape[1]

        # LayerNorm for stability
        self.layer_norm = nn.LayerNorm(flat_size)

        # Feature projection with orthogonal init
        self.fc = layer_init(nn.Linear(flat_size, feature_dim))
        self.feature_dim = feature_dim

        # Dropout for regularization
        # Dropout2d drops entire feature maps (better for CNNs)
        # Regular Dropout for FC layers
        self.dropout_rate = dropout
        self.conv_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.fc_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.gelu(self.conv1(x))
        x = self.conv_dropout(x)
        x = torch.nn.functional.gelu(self.conv2(x))
        x = self.conv_dropout(x)
        x = torch.nn.functional.gelu(self.conv3(x))
        x = x.flatten(1)
        x = self.layer_norm(x)
        x = self.fc_dropout(x)  # Dropout before FC
        x = torch.nn.functional.gelu(self.fc(x))
        return x


class ActorHead(nn.Module):
    """Policy head that outputs action logits."""

    def __init__(self, feature_dim: int, num_actions: int):
        super().__init__()
        # Small std (0.01) for near-uniform initial policy
        self.fc = layer_init(nn.Linear(feature_dim, num_actions), std=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class CriticHead(nn.Module):
    """Value head that outputs state value V(s)."""

    def __init__(self, feature_dim: int):
        super().__init__()
        # std=1.0 for value head
        self.fc = layer_init(nn.Linear(feature_dim, 1), std=1.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Shared backbone with separate heads for policy and value.

    Features:
    - GELU activation
    - LayerNorm for stability
    - Orthogonal initialization
    - Optional dropout (default 0.1)
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_actions: int,
        feature_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = PPOBackbone(input_shape, feature_dim, dropout=dropout)
        self.actor = ActorHead(feature_dim, num_actions)
        self.critic = CriticHead(feature_dim)
        self.num_actions = num_actions

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action logits and state value.

        Args:
            x: Observation tensor (N, C, H, W) where C is frame stack

        Returns:
            logits: Action logits (N, num_actions)
            value: State value (N,)
        """
        features = self.backbone(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
        temperature: float = 1.0,
        logit_clip: float = 20.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.

        Used during rollout collection and training.

        Args:
            x: Observation tensor (N, C, H, W) where C is frame stack
            action: Optional action to evaluate (N,). If None, samples new action.
            temperature: Softmax temperature (>1 = more exploration, <1 = more greedy)
            logit_clip: Clip logits to [-clip, +clip] to prevent saturation

        Returns:
            action: Selected action (N,)
            log_prob: Log probability of action (N,)
            entropy: Policy entropy (N,)
            value: State value (N,)
        """
        logits, value = self.forward(x)

        # Clip logits to prevent softmax saturation (entropy collapse prevention)
        logits = torch.clamp(logits, -logit_clip, logit_clip)

        # Apply temperature scaling (higher temp = more uniform distribution)
        scaled_logits = logits / temperature

        probs = Categorical(logits=scaled_logits)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action)
        entropy = probs.entropy()

        return action, log_prob, entropy, value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get state value only (for bootstrapping)."""
        features = self.backbone(x)
        return self.critic(features)

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get action probabilities (for debugging/visualization)."""
        logits, _ = self.forward(x)
        return torch.softmax(logits, dim=-1)
