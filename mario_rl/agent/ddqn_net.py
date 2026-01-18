"""
Dueling Double DQN network with PPO-inspired stability techniques.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      DDQNNet Architecture                        │
    │                                                                  │
    │   Input: (N, C, H, W) frames        C = 4 stacked grayscale     │
    │           ↓                                                      │
    │   ┌─────────────────────────────────────────────────────────┐   │
    │   │              CNN Backbone (Nature DQN style)             │   │
    │   │  Conv(C→32, 8x8, s=4) → GELU → Dropout2d                │   │
    │   │  Conv(32→64, 4x4, s=2) → GELU → Dropout2d               │   │
    │   │  Conv(64→64, 3x3, s=1) → GELU                           │   │
    │   │  Flatten → LayerNorm → Dropout → FC(→512) → GELU        │   │
    │   └───────────────────────┬─────────────────────────────────┘   │
    │                           │                                      │
    │              ┌────────────┴────────────┐                        │
    │              ↓                         ↓                        │
    │   ┌─────────────────────┐   ┌─────────────────────┐            │
    │   │   Value Stream      │   │  Advantage Stream   │            │
    │   │   FC(512→256)→GELU  │   │   FC(512→256)→GELU  │            │
    │   │   FC(256→1) = V(s)  │   │   FC(256→A) = A(s,a)│            │
    │   └─────────┬───────────┘   └─────────┬───────────┘            │
    │             │                         │                         │
    │             └────────────┬────────────┘                         │
    │                          ↓                                      │
    │           Q(s,a) = V(s) + (A(s,a) - mean(A))                   │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

Stability Techniques (from PPO):
- GELU activation (smoother gradients than ReLU)
- LayerNorm after flatten (stabilizes hidden representations)
- Orthogonal initialization (standard for RL)
- Dropout2d for conv layers (drops entire feature maps)
- Dropout for FC layers

Double DQN:
- Online network: selects actions (argmax Q_online)
- Target network: evaluates Q-values (Q_target[argmax Q_online])
- Soft target updates (polyak averaging) or hard sync

Dueling Architecture:
- Separates value V(s) and advantage A(s,a) estimation
- Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
- More stable value estimation when actions don't matter much
"""

from typing import Tuple

import torch
import numpy as np
from torch import nn


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer with orthogonal weights and constant bias."""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)
    return layer


class DDQNBackbone(nn.Module):
    """
    CNN backbone for DDQN (Nature DQN style with modern improvements).

    Takes frame-stacked observations (N, C, H, W) where C is the frame stack.
    Input should already be normalized to [0, 1] and in channels-first format.

    Improvements over vanilla Nature DQN:
    - GELU activation (smoother gradients)
    - Dropout2d between conv layers (drops entire feature maps)
    - LayerNorm after flatten (stabilizes training)
    - Orthogonal initialization
    
    Optional action history input:
    - If action_history_dim > 0, expects flattened action history as second input
    - Action history is concatenated with visual features before the final FC layer
    - This helps the network understand action effects (e.g., "I pressed jump last frame")
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        feature_dim: int = 512,
        dropout: float = 0.1,
        action_history_dim: int = 0,
    ):
        super().__init__()
        c, h, w = input_shape
        self.action_history_dim = action_history_dim

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
        # If action history is used, concatenate it before the FC layer
        fc_input_dim = flat_size + action_history_dim
        self.fc = layer_init(nn.Linear(fc_input_dim, feature_dim))
        self.feature_dim = feature_dim

        # Dropout for regularization
        self.dropout_rate = dropout
        self.conv_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.fc_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, action_history: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.nn.functional.gelu(self.conv1(x))
        x = self.conv_dropout(x)
        x = torch.nn.functional.gelu(self.conv2(x))
        x = self.conv_dropout(x)
        x = torch.nn.functional.gelu(self.conv3(x))
        x = x.flatten(1)
        x = self.layer_norm(x)
        x = self.fc_dropout(x)
        
        # Concatenate action history if provided
        if action_history is not None:
            # Flatten action history: (N, history_len, num_actions) -> (N, history_len * num_actions)
            if action_history.dim() == 3:
                action_history = action_history.flatten(1)
            x = torch.cat([x, action_history], dim=1)
        elif self.action_history_dim > 0:
            # If network expects action history but none provided, use zeros
            batch_size = x.shape[0]
            zeros = torch.zeros(batch_size, self.action_history_dim, device=x.device)
            x = torch.cat([x, zeros], dim=1)
        
        x = torch.nn.functional.gelu(self.fc(x))
        return x


class DuelingHead(nn.Module):
    """
    Dueling architecture head for Q-value estimation.

    Separates value V(s) and advantage A(s,a) streams:
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))

    This makes it easier to identify valuable states regardless
    of action choice, improving stability when many actions are similar.
    """

    def __init__(self, feature_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()

        # Value stream: estimates V(s)
        self.value_stream = nn.Sequential(
            layer_init(nn.Linear(feature_dim, hidden_dim)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        # Advantage stream: estimates A(s,a) for each action
        self.advantage_stream = nn.Sequential(
            layer_init(nn.Linear(feature_dim, hidden_dim)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_dim, num_actions), std=0.01),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.value_stream(features)  # (N, 1)
        advantages = self.advantage_stream(features)  # (N, num_actions)

        # Dueling aggregation: Q = V + (A - mean(A))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class DDQNNet(nn.Module):
    """
    Single Q-network for DDQN with dueling architecture.

    Features:
    - Nature DQN-style CNN backbone with modern improvements
    - Dueling architecture (separate value and advantage streams)
    - GELU activation, LayerNorm, Dropout, Orthogonal init
    - Raw linear Q-value output (unbounded)
    - Optional action history input for temporal context
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_actions: int,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        action_history_len: int = 4,
    ):
        """
        Args:
            input_shape: Shape of frame stack (C, H, W)
            num_actions: Number of actions in action space
            feature_dim: Dimension of backbone feature output
            hidden_dim: Dimension of dueling head hidden layers
            dropout: Dropout rate for regularization
            action_history_len: Length of action history to use (0 = disabled).
                               When enabled, expects action_history tensor of shape
                               (N, action_history_len, num_actions) as second input.
        """
        super().__init__()
        self.action_history_len = action_history_len
        self.num_actions = num_actions
        
        # Calculate action history dimension (flattened one-hot)
        action_history_dim = action_history_len * num_actions if action_history_len > 0 else 0
        
        self.backbone = DDQNBackbone(input_shape, feature_dim, dropout, action_history_dim)
        self.head = DuelingHead(feature_dim, num_actions, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass returning Q-values for all actions.

        Args:
            x: Observation tensor (N, C, H, W) where C is frame stack
            action_history: Optional action history tensor (N, history_len, num_actions)
                           One-hot encoded previous actions.

        Returns:
            q_values: Q-values for each action (N, num_actions)
        """
        features = self.backbone(x, action_history)
        q_values = self.head(features)

        # Raw linear output (no activation)
        return q_values

    def get_action(
        self,
        x: torch.Tensor,
        epsilon: float = 0.0,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Get action using epsilon-greedy policy.

        Args:
            x: Observation tensor (N, C, H, W)
            epsilon: Exploration rate (0 = greedy, 1 = random)
            action_history: Optional action history tensor

        Returns:
            action: Selected action indices (N,)
        """
        if np.random.random() < epsilon:
            # Random action
            batch_size = x.shape[0]
            return torch.randint(0, self.num_actions, (batch_size,), device=x.device)
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.forward(x, action_history)
                return q_values.argmax(dim=1)


class DoubleDQN(nn.Module):
    """
    Double DQN with online and target networks.

    Double DQN decouples action selection from Q-value estimation:
    - Action selection: argmax_a Q_online(s', a)
    - Q-value estimation: Q_target(s', argmax_a Q_online(s', a))

    This reduces overestimation bias common in standard DQN.

    Target network is updated via:
    - Soft update (polyak averaging): θ_target = τ * θ_online + (1-τ) * θ_target
    - Hard sync: copy online weights to target every N steps

    Features:
    - Raw linear Q-value output (unbounded)
    - Optional action history input for temporal context
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_actions: int,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        action_history_len: int = 4,
    ):
        super().__init__()
        self.online = DDQNNet(
            input_shape, num_actions, feature_dim, hidden_dim, dropout, action_history_len
        )
        self.target = DDQNNet(
            input_shape, num_actions, feature_dim, hidden_dim, dropout, action_history_len
        )
        self.num_actions = num_actions
        self.action_history_len = action_history_len

        # Copy online weights to target and freeze target
        self.sync_target()

    def forward(
        self,
        x: torch.Tensor,
        network: str = "online",
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through specified network.

        Args:
            x: Observation tensor (N, C, H, W)
            network: "online" or "target"
            action_history: Optional action history tensor (N, history_len, num_actions)

        Returns:
            q_values: Q-values for each action (N, num_actions)
        """
        if network == "online":
            return self.online(x, action_history)
        elif network == "target":
            return self.target(x, action_history)
        raise ValueError(f"Unknown network: {network}")

    def get_action(
        self,
        x: torch.Tensor,
        epsilon: float = 0.0,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get action using online network with epsilon-greedy."""
        return self.online.get_action(x, epsilon, action_history)

    def sync_target(self) -> None:
        """Hard sync: copy online weights to target network."""
        self.target.load_state_dict(self.online.state_dict())
        # Freeze target parameters
        for p in self.target.parameters():
            p.requires_grad = False

    def soft_update(self, tau: float = 0.005) -> None:
        """
        Soft update target network (polyak averaging).

        θ_target = τ * θ_online + (1-τ) * θ_target
        """
        for online_param, target_param in zip(
            self.online.parameters(),
            self.target.parameters(),
            strict=True,
        ):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        action_history: torch.Tensor | None = None,
        next_action_history: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Double DQN loss with Huber loss for robustness.

        Double DQN target:
            y = r + γ * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)

        Args:
            states: Current states (N, C, H, W)
            actions: Actions taken (N,)
            rewards: Rewards received (N,)
            next_states: Next states (N, C, H, W)
            dones: Episode done flags (N,)
            gamma: Discount factor
            action_history: Optional action history for current states (N, history_len, num_actions)
            next_action_history: Optional action history for next states

        Returns:
            loss: Huber loss
            info: Dict with metrics
        """
        states.shape[0]

        # Current Q-values for taken actions
        current_q = self.online(states, action_history)  # (N, num_actions)
        current_q_selected = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)  # (N,)

        # Double DQN target
        with torch.no_grad():
            # Select best actions using online network
            next_q_online = self.online(next_states, next_action_history)  # (N, num_actions)
            best_actions = next_q_online.argmax(dim=1)  # (N,)

            # Evaluate using target network
            next_q_target = self.target(next_states, next_action_history)  # (N, num_actions)
            next_q_selected = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)  # (N,)

            # TD target
            target_q = rewards + gamma * next_q_selected * (1.0 - dones)

        # Huber loss (more robust to outliers than MSE)
        loss = torch.nn.functional.huber_loss(current_q_selected, target_q, delta=1.0)

        # Compute TD errors for prioritized replay (absolute error)
        td_errors = (current_q_selected - target_q).abs().detach()

        # Metrics for logging
        info = {
            "loss": loss.item(),
            "q_mean": current_q_selected.mean().item(),
            "q_max": current_q_selected.max().item(),
            "td_error_mean": td_errors.mean().item(),
            "target_q_mean": target_q.mean().item(),
        }

        return loss, info
