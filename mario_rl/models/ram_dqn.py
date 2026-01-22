"""RAM-based DQN with simple MLP architecture.

Uses NES RAM (2048 bytes) as input instead of visual frames.
This is a simpler, faster alternative to CNN-based approaches.

Architecture:
    Input: (N, 2048) RAM bytes [0, 255]
           ↓
    Normalize to [0, 1]
           ↓
    MLP Backbone:
    Linear(2048→512) → LayerNorm → GELU → Dropout
           ↓
    Concat with action_history (flattened)
           ↓
    Linear(512+action_hist_dim→512) → GELU → Dropout
           ↓
    ┌──────┴──────┐
    ↓             ↓
    Value      Advantage
    Stream     Stream
    ↓             ↓
    └──────┬──────┘
           ↓
    Q(s,a) = V(s) + (A(s,a) - mean(A))

Advantages over CNN:
- Much faster (no convolutions)
- RAM contains precise game state (enemy positions, timer, etc.)
- More sample efficient (structured data vs raw pixels)

Disadvantages:
- Game-specific (RAM layout varies by game)
- May miss visual patterns that aren't in RAM
"""

from dataclasses import dataclass

import torch
import numpy as np
from torch import nn
from torch import Tensor


def _layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer with orthogonal weights and constant bias.

    Respects the global skip flag from ddqn_net to avoid CPU contention
    when multiple workers initialize in parallel.
    """
    from mario_rl.agent.ddqn_net import get_skip_weight_init

    if isinstance(layer, nn.Linear):
        if not get_skip_weight_init():
            nn.init.orthogonal_(layer.weight, std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)
    return layer


@dataclass(frozen=True)
class RAMDQNConfig:
    """Configuration for RAM-based DQN model."""

    ram_size: int = 2048
    num_actions: int = 7
    feature_dim: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    action_history_len: int = 4
    danger_prediction_bins: int = 0  # Not used in RAM model, for API compatibility


class RAMBackbone(nn.Module):
    """MLP backbone for RAM input with optional action history.

    Takes raw RAM bytes (N, 2048) and outputs feature vector (N, feature_dim).
    Input should be uint8 [0, 255] - normalized internally.

    Action history (if enabled) is concatenated after the first FC layer,
    allowing the model to learn how recent actions affect the current state.
    """

    def __init__(
        self,
        ram_size: int = 2048,
        feature_dim: int = 512,
        dropout: float = 0.1,
        action_history_dim: int = 0,
    ) -> None:
        super().__init__()
        self.action_history_dim = action_history_dim

        # First layer: RAM embedding
        self.fc1 = _layer_init(nn.Linear(ram_size, feature_dim))
        self.ln1 = nn.LayerNorm(feature_dim)

        # Second layer: combines RAM features with action history
        fc2_input_dim = feature_dim + action_history_dim
        self.fc2 = _layer_init(nn.Linear(fc2_input_dim, feature_dim))
        self.ln2 = nn.LayerNorm(feature_dim)

        self.feature_dim = feature_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, action_history: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x: RAM bytes (N, 2048), uint8 [0, 255] or float [0, 1]
            action_history: Optional action history (N, history_len, num_actions)

        Returns:
            Feature vector (N, feature_dim)
        """
        # Normalize to [0, 1] if uint8
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.0:
            x = x / 255.0

        # First layer: embed RAM
        x = torch.nn.functional.gelu(self.ln1(self.fc1(x)))
        x = self.dropout(x)

        # Concatenate action history if provided
        if action_history is not None:
            if action_history.dim() == 3:
                action_history = action_history.flatten(1)
            x = torch.cat([x, action_history], dim=1)
        elif self.action_history_dim > 0:
            # Pad with zeros if action history expected but not provided
            x = torch.cat([x, torch.zeros(x.shape[0], self.action_history_dim, device=x.device)], dim=1)

        # Second layer: combine features
        x = torch.nn.functional.gelu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        return x


class DuelingHead(nn.Module):
    """Dueling architecture head for Q-value estimation.

    Separates value V(s) and advantage A(s,a) streams:
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
    """

    def __init__(self, feature_dim: int, num_actions: int, hidden_dim: int = 256) -> None:
        super().__init__()

        # Value stream: estimates V(s)
        self.value_stream = nn.Sequential(
            _layer_init(nn.Linear(feature_dim, hidden_dim)),
            nn.GELU(),
            _layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        # Advantage stream: estimates A(s,a) for each action
        self.advantage_stream = nn.Sequential(
            _layer_init(nn.Linear(feature_dim, hidden_dim)),
            nn.GELU(),
            _layer_init(nn.Linear(hidden_dim, num_actions), std=0.01),
        )

    def forward(self, features: Tensor) -> Tensor:
        value = self.value_stream(features)  # (N, 1)
        advantages = self.advantage_stream(features)  # (N, num_actions)

        # Dueling aggregation: Q = V + (A - mean(A))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class RAMDQNNet(nn.Module):
    """Single Q-network for RAM-based DQN with dueling architecture.

    Features:
    - MLP backbone for 2048-byte RAM input
    - Optional action history for temporal context
    - Dueling architecture (separate value and advantage streams)
    - GELU activation, LayerNorm, Dropout
    - Raw linear Q-value output (unbounded)
    """

    def __init__(
        self,
        ram_size: int = 2048,
        num_actions: int = 7,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        action_history_len: int = 4,
    ) -> None:
        super().__init__()
        self.action_history_len = action_history_len
        self.num_actions = num_actions

        # Calculate action history dimension (flattened one-hot)
        action_history_dim = action_history_len * num_actions if action_history_len > 0 else 0

        self.backbone = RAMBackbone(ram_size, feature_dim, dropout, action_history_dim)
        self.head = DuelingHead(feature_dim, num_actions, hidden_dim)

    def forward(self, x: Tensor, action_history: Tensor | None = None) -> Tensor:
        """Forward pass returning Q-values for all actions.

        Args:
            x: RAM bytes (N, 2048)
            action_history: Optional action history (N, history_len, num_actions)

        Returns:
            q_values: Q-values for each action (N, num_actions)
        """
        features = self.backbone(x, action_history)
        q_values = self.head(features)
        return q_values

    def select_action(
        self,
        x: Tensor,
        epsilon: float = 0.0,
        action_history: Tensor | None = None,
    ) -> Tensor:
        """Get action using epsilon-greedy policy.

        Args:
            x: RAM bytes (N, 2048)
            epsilon: Exploration rate (0 = greedy, 1 = random)
            action_history: Optional action history tensor

        Returns:
            action: Selected action indices (N,)
        """
        if np.random.random() < epsilon:
            batch_size = x.shape[0]
            return torch.randint(0, self.num_actions, (batch_size,), device=x.device)
        with torch.no_grad():
            q_values = self.forward(x, action_history)
            return q_values.argmax(dim=1)


class RAMDoubleDQN(nn.Module):
    """Double DQN with online and target networks for RAM input.

    Double DQN decouples action selection from Q-value estimation:
    - Action selection: argmax_a Q_online(s', a)
    - Q-value estimation: Q_target(s', argmax_a Q_online(s', a))

    This reduces overestimation bias common in standard DQN.

    Supports optional action history for temporal context, matching
    the interface of the frame-based DoubleDQN.
    """

    def __init__(
        self,
        ram_size: int = 2048,
        num_actions: int = 7,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        action_history_len: int = 4,
        danger_prediction_bins: int = 0,  # Not used, for API compatibility
    ) -> None:
        super().__init__()
        self.online = RAMDQNNet(ram_size, num_actions, feature_dim, hidden_dim, dropout, action_history_len)
        self.target = RAMDQNNet(ram_size, num_actions, feature_dim, hidden_dim, dropout, action_history_len)
        self.num_actions = num_actions
        self.ram_size = ram_size
        self.action_history_len = action_history_len
        self.danger_prediction_bins = danger_prediction_bins

        # Copy online weights to target and freeze target
        self.sync_target()

    def forward(
        self,
        x: Tensor,
        network: str = "online",
        action_history: Tensor | None = None,
        return_danger: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass through specified network.

        Args:
            x: RAM bytes (N, 2048)
            network: "online" or "target"
            action_history: Optional action history (N, history_len, num_actions)
            return_danger: If True, returns (q_values, danger_pred) tuple.
                          Since RAM model has no danger head, returns zeros.

        Returns:
            q_values: Q-values for each action (N, num_actions)
            If return_danger=True: tuple of (q_values, danger_prediction)
        """
        match network:
            case "online":
                q_values = self.online(x, action_history)
            case "target":
                q_values = self.target(x, action_history)
            case _:
                raise ValueError(f"Unknown network: {network}")

        if return_danger:
            # No danger head in RAM model, return zeros for compatibility
            danger_pred = torch.zeros(x.shape[0], self.danger_prediction_bins or 16, device=x.device)
            return q_values, danger_pred
        return q_values

    def select_action(
        self,
        x: Tensor,
        epsilon: float = 0.0,
        action_history: Tensor | None = None,
    ) -> Tensor:
        """Get action using online network with epsilon-greedy."""
        return self.online.select_action(x, epsilon, action_history)

    def sync_target(self) -> None:
        """Hard sync: copy online weights to target network."""
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

    def soft_update(self, tau: float = 0.005) -> None:
        """Soft update target network (polyak averaging).

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
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
        gamma: float = 0.99,
        action_history: Tensor | None = None,
        next_action_history: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """Compute Double DQN loss with Huber loss.

        Args:
            states: Current RAM states (N, 2048)
            actions: Actions taken (N,)
            rewards: Rewards received (N,)
            next_states: Next RAM states (N, 2048)
            dones: Episode done flags (N,)
            gamma: Discount factor
            action_history: Optional action history for current states
            next_action_history: Optional action history for next states

        Returns:
            loss: Huber loss
            info: Dict with metrics
        """
        # Current Q-values for taken actions
        current_q = self.online(states, action_history)  # (N, num_actions)
        current_q_selected = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)  # (N,)

        # Double DQN target
        with torch.no_grad():
            # Select best actions using online network
            next_q_online = self.online(next_states, next_action_history)
            best_actions = next_q_online.argmax(dim=1)

            # Evaluate using target network
            next_q_target = self.target(next_states, next_action_history)
            next_q_selected = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            # TD target
            target_q = rewards + gamma * next_q_selected * (1.0 - dones)

        # Huber loss
        loss = torch.nn.functional.huber_loss(current_q_selected, target_q, delta=1.0)

        # TD errors for prioritized replay
        td_errors = (current_q_selected - target_q).abs().detach()

        info = {
            "loss": loss.item(),
            "q_mean": current_q_selected.mean().item(),
            "q_max": current_q_selected.max().item(),
            "td_error_mean": td_errors.mean().item(),
            "target_q_mean": target_q.mean().item(),
        }

        return loss, info
