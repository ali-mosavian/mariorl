"""DDQN Learner for Double DQN training.

Implements the Learner protocol for Double DQN:
- Computes TD loss using Double DQN targets
- Uses symlog transform for scale-invariant Q-learning
- Supports N-step returns with correct gamma discounting
- Uses importance sampling weights for PER correction
- Includes entropy regularization for exploration
- Supports soft and hard target updates
- Returns training metrics for logging
"""

from typing import Any
from dataclasses import dataclass
from dataclasses import field

import torch
from torch import Tensor
import torch.nn.functional as F

from mario_rl.models import DoubleDQN
from mario_rl.models.ddqn import symlog, symexp
from mario_rl.mcts.protocols import PolicyAdapter, ValueAdapter


@dataclass
class DDQNLearner:
    """Learner for Double DQN training.

    Computes Double DQN loss with N-step returns and symlog transform:
        target = symlog(r + γ^n * symexp(Q_target(s', a*)) * (1 - done))
        loss = mean(weights * huber_loss(Q_online(s, a), target))

    The network learns to output symlog-scaled Q-values directly. This means:
    - Network outputs are in a compressed range (approx ±10 instead of ±10000)
    - Action selection (argmax) still works since symlog is monotonic
    - For TD targets, we convert target Q back to real space, compute Bellman
      update, then convert back to symlog space

    Features:
    - Symlog transform for scale-invariant Q-learning (handles varying reward scales)
    - N-step returns with correct gamma^n discounting
    - Importance sampling weights for PER bias correction
    - Entropy regularization for exploration
    - Huber loss for robustness to outliers
    """

    model: DoubleDQN
    gamma: float = 0.99
    n_step: int = 1
    entropy_coef: float = 0.01

    # MCTS adapter (injected by factory, optional)
    mcts_adapter: PolicyAdapter | ValueAdapter | None = None

    # Pre-computed n-step gamma (computed in __post_init__)
    _n_step_gamma: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Pre-compute n-step gamma for efficiency."""
        self._n_step_gamma = self.gamma ** self.n_step

    def compute_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
        weights: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute Double DQN loss for a batch of transitions.

        The network outputs symlog-scaled Q-values. For TD targets:
        1. Get Q_target in symlog space from target network
        2. Convert to real space: Q_real = symexp(Q_target)
        3. Compute Bellman target: target_real = r + γ * Q_real * (1-done)
        4. Convert back to symlog: target = symlog(target_real)
        5. Loss = huber(Q_online, target) - both already in symlog space

        Args:
            states: Current observations (batch, *obs_shape)
            actions: Actions taken (batch,)
            rewards: Rewards received (batch,) - N-step cumulative rewards
            next_states: Next observations (batch, *obs_shape) - state after N steps
            dones: Episode termination flags (batch,)
            weights: Importance sampling weights for PER (batch,), None for uniform

        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dict with training metrics
        """
        # Current Q-values for taken actions (in symlog space)
        current_q = self.model(states, network="online")  # (batch, num_actions)
        current_q_selected = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch,)

        # Double DQN target:
        # 1. Select best action using online network (argmax works since symlog is monotonic)
        # 2. Evaluate that action using target network
        # 3. Convert to real space, apply Bellman, convert back to symlog
        with torch.no_grad():
            # Action selection with online network (symlog-scaled, but argmax is invariant)
            next_q_online = self.model(next_states, network="online")  # (batch, num_actions)
            best_actions = next_q_online.argmax(dim=1)  # (batch,)

            # Value evaluation with target network (symlog-scaled)
            next_q_target = self.model(next_states, network="target")  # (batch, num_actions)
            next_q_selected_symlog = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            # Convert to real space for Bellman update
            next_q_selected_real = symexp(next_q_selected_symlog)

            # TD target in real space: r + γ^n * Q_target(s', a*) * (1 - done)
            target_q_real = rewards + self._n_step_gamma * next_q_selected_real * (1.0 - dones.float())

            # Convert target back to symlog space
            target_q = symlog(target_q_real)

        # Compute element-wise Huber loss (both in symlog space)
        element_wise_loss = F.huber_loss(
            current_q_selected, target_q, reduction="none", delta=1.0
        )

        # Apply importance sampling weights for PER bias correction
        if weights is not None:
            td_loss = (weights * element_wise_loss).mean()
        else:
            td_loss = element_wise_loss.mean()

        # Entropy regularization: encourage exploration by penalizing low entropy
        # Q-values are in symlog space but softmax still gives valid distribution
        policy = F.softmax(current_q, dim=1)
        log_policy = F.log_softmax(current_q, dim=1)
        entropy = -(policy * log_policy).sum(dim=1).mean()

        # Total loss: TD loss - entropy bonus (we want to maximize entropy)
        loss = td_loss - self.entropy_coef * entropy

        # Compute TD errors for prioritized replay (in symlog space for consistency)
        td_errors = (current_q_selected - target_q).abs().detach()

        # Training metrics (convert to real space for interpretability)
        current_q_real = symexp(current_q_selected).detach()
        metrics: dict[str, Any] = {
            "loss": loss.item(),
            "td_loss": td_loss.item(),
            "q_mean": current_q_real.mean().item(),
            "q_max": current_q_real.max().item(),
            "td_error": td_errors.mean().item(),
            "target_q_mean": target_q_real.mean().item(),
            "entropy": entropy.item(),
        }

        return loss, metrics

    def update_targets(self, tau: float = 0.005) -> None:
        """Update target network weights.

        Args:
            tau: Interpolation coefficient.
                 tau=1.0 means hard copy (sync).
                 tau<1.0 means soft update (polyak averaging).
        """
        if tau >= 1.0:
            self.model.sync_target()
        else:
            self.model.soft_update(tau)
