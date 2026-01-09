"""DDQN Learner for Double DQN training.

Implements the Learner protocol for Double DQN:
- Computes TD loss using Double DQN targets
- Supports soft and hard target updates
- Returns training metrics for logging
"""

from typing import Any
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from mario_rl.models import DoubleDQN


@dataclass
class DDQNLearner:
    """Learner for Double DQN training.

    Computes Double DQN loss:
        target = r + γ * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)
        loss = huber_loss(Q_online(s, a), target)

    Uses Huber loss for robustness to outliers.
    """

    model: DoubleDQN
    gamma: float = 0.99

    def compute_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute Double DQN loss for a batch of transitions.

        Args:
            states: Current observations (batch, *obs_shape)
            actions: Actions taken (batch,)
            rewards: Rewards received (batch,)
            next_states: Next observations (batch, *obs_shape)
            dones: Episode termination flags (batch,)

        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dict with training metrics
        """
        # Current Q-values for taken actions
        current_q = self.model(states, network="online")  # (batch, num_actions)
        current_q_selected = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch,)

        # Double DQN target:
        # 1. Select best action using online network
        # 2. Evaluate that action using target network
        with torch.no_grad():
            # Action selection with online network
            next_q_online = self.model(next_states, network="online")  # (batch, num_actions)
            best_actions = next_q_online.argmax(dim=1)  # (batch,)

            # Value evaluation with target network
            next_q_target = self.model(next_states, network="target")  # (batch, num_actions)
            next_q_selected = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)  # (batch,)

            # TD target: r + γ * Q_target(s', a*) * (1 - done)
            target_q = rewards + self.gamma * next_q_selected * (1.0 - dones.float())

        # Huber loss (smooth L1) - more robust to outliers than MSE
        loss = F.huber_loss(current_q_selected, target_q, delta=1.0)

        # Compute TD errors for prioritized replay (absolute error)
        td_errors = (current_q_selected - target_q).abs().detach()

        # Training metrics
        metrics: dict[str, Any] = {
            "loss": loss.item(),
            "q_mean": current_q_selected.mean().item(),
            "q_max": current_q_selected.max().item(),
            "td_error": td_errors.mean().item(),  # Mean TD error for logging
            "target_q_mean": target_q.mean().item(),
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
