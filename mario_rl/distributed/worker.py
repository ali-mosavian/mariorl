"""Distributed Worker for gradient computation.

Workers run in parallel processes, each with their own copy of the model.
They:
1. Collect experience from environment
2. Compute gradients on batches
3. Send gradients to coordinator
4. Receive updated weights from coordinator
"""

from typing import Any
from dataclasses import dataclass

import torch
from torch import Tensor

from mario_rl.learners.base import Learner


@dataclass
class Worker:
    """Worker that computes gradients for distributed training.

    The worker wraps a Learner and provides:
    - Gradient computation from batches
    - Weight synchronization (get/apply)

    Works with any Model/Learner that implements the protocols.
    """

    learner: Learner

    @property
    def model(self):
        """Access the underlying model."""
        return self.learner.model

    def compute_gradients(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        """Compute gradients for a batch of transitions.

        Args:
            states: Current observations
            actions: Actions taken
            rewards: Rewards received
            next_states: Next observations
            dones: Episode termination flags

        Returns:
            gradients: Dict mapping parameter names to gradient tensors
            metrics: Training metrics from loss computation
        """
        # Zero existing gradients
        self.model.zero_grad()

        # Compute loss
        loss, metrics = self.learner.compute_loss(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )

        # Backpropagate
        loss.backward()

        # Collect gradients
        gradients = {
            name: param.grad.detach().clone()
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }

        return gradients, metrics

    def weights(self) -> dict[str, Tensor]:
        """Get current model weights.

        Returns:
            Dict mapping parameter names to weight tensors (detached)
        """
        return {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

    def apply_weights(self, weights: dict[str, Tensor]) -> None:
        """Apply new weights to model.

        Args:
            weights: Dict mapping parameter names to new weight tensors
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in weights:
                    param.copy_(weights[name])
