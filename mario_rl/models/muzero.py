"""MuZero model with simple DDQN-style networks.

MuZero learns three functions in latent space:
1. Representation h(s) → z: Encodes observations to latent state
2. Dynamics g(z, a) → (z', r): Predicts next latent and reward
3. Prediction f(z) → (π, v): Predicts policy and value

Unlike Dreamer, MuZero doesn't reconstruct observations - it only needs
predictions useful for planning via MCTS.

Naming convention:
- s: observation/state (from environment)
- z: latent state (internal representation)
- a: action
- r: reward
- π: policy
- v: value

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                       MuZero Architecture                            │
    │                                                                      │
    │   REPRESENTATION h(s)     DYNAMICS g(z,a)      PREDICTION f(z)      │
    │   ──────────────────      ───────────────      ────────────────     │
    │   obs (4, 84, 84)         latent + action      latent                │
    │         ↓                       ↓                    ↓              │
    │   CNN Backbone            MLP (512+7→512)      Policy (→7)          │
    │   (Nature DQN)                  ↓              Value (→1)           │
    │         ↓                 (z', reward)         (π, v)               │
    │   latent z (512)                                                    │
    └─────────────────────────────────────────────────────────────────────┘

Training:
- Collect trajectories using MCTS for action selection
- Store (obs, actions, MCTS_policies, rewards, values) in replay buffer
- Train by unrolling dynamics K steps and matching predictions to targets
"""

from dataclasses import dataclass

import torch
import numpy as np
from torch import nn
from torch import Tensor
import torch.nn.functional as F

# =============================================================================
# Symlog Transform (from Dreamer V3)
# =============================================================================


def symlog(x: Tensor) -> Tensor:
    """Symmetric logarithm: sign(x) * ln(|x| + 1).

    Compresses large values while preserving small values and sign.
    Used for scale-invariant value and reward predictions.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# =============================================================================
# Initialization
# =============================================================================


def _layer_init(
    layer: nn.Module,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
) -> nn.Module:
    """Initialize layer with orthogonal weights and constant bias."""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)
    return layer


@dataclass(frozen=True)
class MuZeroConfig:
    """Configuration for MuZero model."""

    input_shape: tuple[int, int, int]  # (C, H, W) e.g., (4, 84, 84)
    num_actions: int  # e.g., 7 for Mario SIMPLE_MOVEMENT
    latent_dim: int = 512
    hidden_dim: int = 256
    proj_dim: int = 256  # Projection dimension for contrastive loss
    dropout: float = 0.0  # Usually 0 for MuZero
    num_simulations: int = 50  # MCTS simulations per action
    discount: float = 0.997
    unroll_steps: int = 5  # K steps to unroll during training
    td_steps: int = 10  # n-step returns for value targets
    support_size: int = 0  # 0 = scalar value, >0 = categorical
    # Latent grounding loss weights
    consistency_weight: float = 2.0  # SimSiam-style consistency loss
    contrastive_weight: float = 0.5  # InfoNCE contrastive loss
    contrastive_temp: float = 0.1  # Temperature for InfoNCE


class RepresentationNetwork(nn.Module):
    """h(s) → z: Encodes observation to latent state.

    Uses Nature DQN-style CNN backbone with modern improvements.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        latent_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c, h, w = input_shape

        # Conv layers (Nature DQN style)
        self.conv1 = _layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4))
        self.conv2 = _layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = _layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            dummy = F.gelu(self.conv1(dummy))
            dummy = F.gelu(self.conv2(dummy))
            dummy = F.gelu(self.conv3(dummy))
            flat_size = dummy.flatten(1).shape[1]

        # LayerNorm for stability
        self.layer_norm = nn.LayerNorm(flat_size)

        # Project to latent space
        self.fc = _layer_init(nn.Linear(flat_size, latent_dim))

        # Dropout
        self.conv_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.fc_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, s: Tensor) -> Tensor:
        """Encode observation to latent state.

        Args:
            s: (N, C, H, W) observation in [0, 255] range

        Returns:
            z: (N, latent_dim) normalized latent state
        """
        # Normalize input
        x = s / 255.0

        # CNN backbone
        x = F.gelu(self.conv1(x))
        x = self.conv_dropout(x)
        x = F.gelu(self.conv2(x))
        x = self.conv_dropout(x)
        x = F.gelu(self.conv3(x))

        # Flatten and project
        x = x.flatten(1)
        x = self.layer_norm(x)
        x = self.fc_dropout(x)
        z = self.fc(x)

        # Normalize latent state (helps dynamics model)
        z = F.normalize(z, dim=-1)
        return z


class DynamicsNetwork(nn.Module):
    """g(z, a) → (z', r): Predicts next latent state and reward.

    Simple MLP that takes latent state + one-hot action.
    """

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.latent_dim = latent_dim

        # State transition network
        self.transition = nn.Sequential(
            _layer_init(nn.Linear(latent_dim + num_actions, hidden_dim)),
            nn.GELU(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.GELU(),
            _layer_init(nn.Linear(hidden_dim, latent_dim)),
        )

        # Reward prediction network
        self.reward_head = nn.Sequential(
            _layer_init(nn.Linear(latent_dim + num_actions, hidden_dim)),
            nn.GELU(),
            _layer_init(nn.Linear(hidden_dim, 1), std=0.01),
        )

    def forward(self, z: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        """Predict next latent state and reward.

        Args:
            z: (N, latent_dim) current latent state
            a: (N,) action indices

        Returns:
            z_next: (N, latent_dim) normalized next latent state
            r: (N,) predicted reward
        """
        # One-hot encode action
        a_onehot = F.one_hot(a.long(), self.num_actions).float()
        x = torch.cat([z, a_onehot], dim=-1)

        # Predict next state (normalized)
        z_next = self.transition(x)
        z_next = F.normalize(z_next, dim=-1)

        # Predict reward
        r = self.reward_head(x).squeeze(-1)

        return z_next, r


class PredictionNetwork(nn.Module):
    """f(z) → (π, v): Predicts policy and value from latent state.

    Two-headed network similar to AlphaZero/MuZero.
    """

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        # Policy head (outputs logits, apply softmax for probabilities)
        self.policy_head = nn.Sequential(
            _layer_init(nn.Linear(latent_dim, hidden_dim)),
            nn.GELU(),
            _layer_init(nn.Linear(hidden_dim, num_actions), std=0.01),
        )

        # Value head (outputs scalar value estimate)
        self.value_head = nn.Sequential(
            _layer_init(nn.Linear(latent_dim, hidden_dim)),
            nn.GELU(),
            _layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Predict policy and value.

        Args:
            z: (N, latent_dim) latent state

        Returns:
            policy_logits: (N, num_actions) unnormalized log probabilities
            v: (N,) value estimate
        """
        policy_logits = self.policy_head(z)
        v = self.value_head(z).squeeze(-1)
        return policy_logits, v


class ProjectorNetwork(nn.Module):
    """Projects latent state to embedding space for contrastive learning.

    Maps z → e where similarity is computed for InfoNCE loss.
    Uses a smaller projection dimension to reduce computational cost.
    """

    def __init__(
        self,
        latent_dim: int,
        proj_dim: int = 256,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            _layer_init(nn.Linear(latent_dim, hidden_dim)),
            nn.GELU(),
            _layer_init(nn.Linear(hidden_dim, proj_dim)),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Project latent to embedding space.

        Args:
            z: (N, latent_dim) latent state

        Returns:
            e: (N, proj_dim) normalized embedding
        """
        e = self.projector(z)
        e = F.normalize(e, dim=-1)
        return e


class PredictorNetwork(nn.Module):
    """SimSiam-style predictor for consistency loss.

    Asymmetric architecture: only applied to predicted latent,
    not to encoded target. This asymmetry prevents collapse.

    pred(g(z, a)) should match sg(h(s'))
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            _layer_init(nn.Linear(latent_dim, hidden_dim)),
            nn.GELU(),
            _layer_init(nn.Linear(hidden_dim, latent_dim)),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Predict target latent representation.

        Args:
            z: (N, latent_dim) predicted latent from dynamics

        Returns:
            z_pred: (N, latent_dim) prediction to match encoded target
        """
        return self.predictor(z)


class MuZeroNetwork(nn.Module):
    """Complete MuZero network combining representation, dynamics, and prediction.

    Provides initial_inference (from observation) and recurrent_inference
    (from latent state + action) for use in MCTS planning.

    Includes projector and predictor for latent grounding:
    - Projector: for contrastive loss (InfoNCE)
    - Predictor: for consistency loss (SimSiam-style)
    """

    def __init__(self, config: MuZeroConfig) -> None:
        super().__init__()
        self.config = config

        self.representation = RepresentationNetwork(
            config.input_shape,
            config.latent_dim,
            config.dropout,
        )
        self.dynamics = DynamicsNetwork(
            config.latent_dim,
            config.num_actions,
            config.hidden_dim,
        )
        self.prediction = PredictionNetwork(
            config.latent_dim,
            config.num_actions,
            config.hidden_dim,
        )
        # Latent grounding networks
        self.projector = ProjectorNetwork(
            config.latent_dim,
            config.proj_dim,
            config.hidden_dim,
        )
        self.predictor = PredictorNetwork(
            config.latent_dim,
            config.hidden_dim,
        )

    @property
    def num_actions(self) -> int:
        return self.config.num_actions

    def initial_inference(self, s: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """First step: encode observation and predict policy/value.

        Used at the root of MCTS tree.

        Args:
            s: (N, C, H, W) observation in [0, 255] range

        Returns:
            z: (N, latent_dim) latent state
            policy_logits: (N, num_actions) policy logits
            v: (N,) value estimate
        """
        z = self.representation(s)
        policy_logits, v = self.prediction(z)
        return z, policy_logits, v

    def recurrent_inference(
        self,
        z: Tensor,
        a: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Subsequent steps: predict next latent, reward, policy, value.

        Used for expanding MCTS tree without real environment.

        Args:
            z: (N, latent_dim) current latent state
            a: (N,) action indices

        Returns:
            z_next: (N, latent_dim) next latent state
            r: (N,) predicted reward
            policy_logits: (N, num_actions) policy logits
            v: (N,) value estimate
        """
        z_next, r = self.dynamics(z, a)
        policy_logits, v = self.prediction(z_next)
        return z_next, r, policy_logits, v

    def get_policy(self, s: Tensor, temperature: float = 1.0) -> Tensor:
        """Get action probabilities from observation (without MCTS).

        Useful for fast action selection during evaluation.

        Args:
            s: (N, C, H, W) observation
            temperature: Softmax temperature (0 = greedy, 1 = proportional)

        Returns:
            policy: (N, num_actions) action probabilities
        """
        _, policy_logits, _ = self.initial_inference(s)

        if temperature == 0:
            # Greedy
            policy = F.one_hot(policy_logits.argmax(dim=-1), self.num_actions).float()
        else:
            policy = F.softmax(policy_logits / temperature, dim=-1)

        return policy

    def select_action(
        self,
        s: Tensor,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> Tensor:
        """Select action from policy (without MCTS).

        Args:
            s: (N, C, H, W) observation
            temperature: Softmax temperature
            greedy: If True, select argmax; otherwise sample

        Returns:
            a: (N,) selected action indices
        """
        policy = self.get_policy(s, temperature if not greedy else 0.0)

        if greedy:
            return policy.argmax(dim=-1)
        return torch.multinomial(policy, num_samples=1).squeeze(-1)

    def project(self, z: Tensor) -> Tensor:
        """Project latent to embedding space for contrastive loss.

        Args:
            z: (N, latent_dim) latent state

        Returns:
            e: (N, proj_dim) normalized embedding
        """
        return self.projector(z)

    def predict(self, z: Tensor) -> Tensor:
        """Predict target latent for consistency loss.

        Args:
            z: (N, latent_dim) predicted latent from dynamics

        Returns:
            z_pred: (N, latent_dim) prediction to match encoded target
        """
        return self.predictor(z)


def info_nce_loss(
    query: Tensor,
    positive: Tensor,
    temperature: float = 0.1,
) -> Tensor:
    """Compute InfoNCE contrastive loss with in-batch negatives.

    Uses other samples in the batch as negatives.

    Args:
        query: (N, D) query embeddings (from predicted z)
        positive: (N, D) positive embeddings (from encoded z)
        temperature: Softmax temperature τ

    Returns:
        loss: (N,) per-sample InfoNCE loss
    """
    # Normalize embeddings
    query = F.normalize(query, dim=-1)
    positive = F.normalize(positive, dim=-1)

    # Compute similarity matrix: (N, N)
    # sim[i, j] = query[i] · positive[j]
    sim_matrix = torch.mm(query, positive.t()) / temperature

    # Positive pairs are on the diagonal
    # Loss is -log(exp(sim[i,i]) / sum_j(exp(sim[i,j])))
    labels = torch.arange(query.size(0), device=query.device)
    loss = F.cross_entropy(sim_matrix, labels, reduction="none")

    return loss


class MuZeroModel(nn.Module):
    """MuZero with target network for stable training.

    Similar to Double DQN, uses a slowly-updated target network
    for computing value targets during training.
    """

    def __init__(self, config: MuZeroConfig) -> None:
        super().__init__()
        self.config = config

        self.online = MuZeroNetwork(config)
        self.target = MuZeroNetwork(config)

        # Copy online to target and freeze
        self.sync_target()

    @property
    def num_actions(self) -> int:
        return self.config.num_actions

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

    def initial_inference(self, s: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Initial inference using online network."""
        return self.online.initial_inference(s)

    def recurrent_inference(
        self,
        z: Tensor,
        a: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Recurrent inference using online network."""
        return self.online.recurrent_inference(z, a)

    def select_action(
        self,
        s: Tensor,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> Tensor:
        """Select action using online network."""
        return self.online.select_action(s, temperature, greedy)

    def project(self, z: Tensor) -> Tensor:
        """Project latent to embedding space using online network."""
        return self.online.project(z)

    def predict(self, z: Tensor) -> Tensor:
        """Predict target latent using online network."""
        return self.online.predict(z)

    def encode(self, s: Tensor) -> Tensor:
        """Encode observation to latent using online network."""
        return self.online.representation(s)

    def compute_loss(
        self,
        s: Tensor,
        actions: Tensor,
        target_policies: Tensor,
        target_values: Tensor,
        target_rewards: Tensor,
        importance_weights: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute MuZero loss by unrolling dynamics model.

        Args:
            s: (N, C, H, W) initial observations
            actions: (N, K) sequence of K actions taken
            target_policies: (N, K+1, num_actions) MCTS policy targets
            target_values: (N, K+1) value targets (from MCTS or n-step returns)
            target_rewards: (N, K) reward targets
            importance_weights: (N,) optional PER weights

        Returns:
            loss: Scalar loss
            info: Dict with loss components for logging
        """
        batch_size = s.shape[0]
        K = actions.shape[1]  # Unroll steps

        if importance_weights is None:
            importance_weights = torch.ones(batch_size, device=s.device)

        # Initial inference
        z, policy_logits, v = self.online.initial_inference(s)

        # Loss at root (step 0)
        policy_loss = F.cross_entropy(
            policy_logits,
            target_policies[:, 0],
            reduction="none",
        )
        value_loss = F.mse_loss(v, target_values[:, 0], reduction="none")
        reward_loss = torch.zeros_like(value_loss)

        # Unroll K steps
        for k in range(K):
            z, r, policy_logits, v = self.online.recurrent_inference(z, actions[:, k])

            # Accumulate losses
            policy_loss = policy_loss + F.cross_entropy(
                policy_logits,
                target_policies[:, k + 1],
                reduction="none",
            )
            value_loss = value_loss + F.mse_loss(
                v,
                target_values[:, k + 1],
                reduction="none",
            )
            reward_loss = reward_loss + F.mse_loss(
                r,
                target_rewards[:, k],
                reduction="none",
            )

        # Apply importance weights and average
        total_loss = (policy_loss + value_loss + reward_loss) * importance_weights
        loss = total_loss.mean()

        # Metrics
        info = {
            "loss": loss.item(),
            "policy_loss": (policy_loss * importance_weights).mean().item() / (K + 1),
            "value_loss": (value_loss * importance_weights).mean().item() / (K + 1),
            "reward_loss": (reward_loss * importance_weights).mean().item() / K if K > 0 else 0.0,
        }

        return loss, info
