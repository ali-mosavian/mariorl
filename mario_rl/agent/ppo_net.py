"""
Actor-Critic network for PPO.

Shared CNN backbone with separate actor (policy) and critic (value) heads.
"""

from typing import Tuple

import torch
from torch import nn
from torch.distributions import Categorical

from mario_rl.agent.neural import LambdaLayer


class PPOBackbone(nn.Module):
    """
    Shared CNN backbone for actor-critic.

    Takes frame-stacked observations (N, F, H, W, C) and outputs features.
    """

    def __init__(self, input_shape: Tuple[int, ...], feature_dim: int = 512):
        super().__init__()
        f, h, w, c = input_shape

        self.net = nn.Sequential(
            # Swap NxFxHxWxC -> NxFxCxHxW and normalize to [0, 1]
            LambdaLayer(lambda t: t.permute(0, 1, 4, 2, 3).float() / 255),
            # NxFxCxHxW -> (N*F)xCxHxW
            LambdaLayer(lambda t: t.view(t.shape[0] * f, c, h, w)),
            # Conv layers
            nn.LazyConv2d(out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            # Flatten and combine frames
            nn.Flatten(),
        )

        # Feature projection
        self.fc = nn.LazyLinear(feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x)
        return torch.relu(self.fc(features))


class ActorHead(nn.Module):
    """Policy head that outputs action logits."""

    def __init__(self, feature_dim: int, num_actions: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_actions)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class CriticHead(nn.Module):
    """Value head that outputs state value V(s)."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Shared backbone with separate heads for policy and value.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_actions: int,
        feature_dim: int = 512,
    ):
        super().__init__()
        self.backbone = PPOBackbone(input_shape, feature_dim)
        self.actor = ActorHead(feature_dim, num_actions)
        self.critic = CriticHead(feature_dim)
        self.num_actions = num_actions

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action logits and state value.

        Args:
            x: Observation tensor (N, F, H, W, C)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.

        Used during rollout collection and training.

        Args:
            x: Observation tensor (N, F, H, W, C)
            action: Optional action to evaluate (N,). If None, samples new action.

        Returns:
            action: Selected action (N,)
            log_prob: Log probability of action (N,)
            entropy: Policy entropy (N,)
            value: State value (N,)
        """
        logits, value = self.forward(x)
        probs = Categorical(logits=logits)

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
