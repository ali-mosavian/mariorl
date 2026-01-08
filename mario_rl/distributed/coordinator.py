"""Distributed Coordinator for gradient aggregation and weight broadcasting.

The coordinator runs on the main process and:
1. Receives gradients from workers
2. Aggregates gradients (mean)
3. Applies updates using optimizer
4. Broadcasts new weights to workers
5. Manages target network updates
"""

from typing import Any
from dataclasses import field
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import Adam

from mario_rl.learners.base import Learner


@dataclass
class Coordinator:
    """Coordinator that aggregates gradients and manages central model.

    The coordinator wraps a Learner and provides:
    - Gradient aggregation from multiple workers
    - Optimizer step application
    - Weight broadcasting
    - Target network updates

    Works with any Model/Learner that implements the protocols.
    """

    learner: Learner
    lr: float = 0.0001
    optimizer: Any = field(init=False)

    def __post_init__(self) -> None:
        """Initialize optimizer after dataclass init."""
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    @property
    def model(self):
        """Access the underlying model."""
        return self.learner.model

    def aggregate_gradients(
        self,
        gradient_list: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        """Aggregate gradients from multiple workers by averaging.

        Args:
            gradient_list: List of gradient dicts from workers

        Returns:
            Averaged gradients

        Raises:
            ValueError: If gradient_list is empty
        """
        if not gradient_list:
            raise ValueError("Cannot aggregate empty gradient list")

        if len(gradient_list) == 1:
            return gradient_list[0]

        # Average gradients
        aggregated = {}
        num_workers = len(gradient_list)

        for name in gradient_list[0]:
            stacked = torch.stack([g[name] for g in gradient_list])
            aggregated[name] = stacked.mean(dim=0)

        return aggregated

    def apply_gradients(self, gradients: dict[str, Tensor]) -> None:
        """Apply gradients to model using optimizer.

        Args:
            gradients: Dict mapping parameter names to gradient tensors
        """
        # Set gradients on model parameters
        for name, param in self.model.named_parameters():
            if name in gradients:
                param.grad = gradients[name].clone()

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

    def weights(self) -> dict[str, Tensor]:
        """Get current model weights for broadcasting.

        Returns:
            Dict mapping parameter names to weight tensors (detached)
        """
        return {
            name: param.data.clone().detach()
            for name, param in self.model.named_parameters()
        }

    def update_targets(self, tau: float = 0.005) -> None:
        """Update target networks via learner.

        Args:
            tau: Soft update coefficient
        """
        self.learner.update_targets(tau)

    def training_step(
        self,
        gradient_list: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        """Execute a full training step.

        1. Aggregate gradients from workers
        2. Apply optimizer update
        3. Return new weights for broadcasting

        Args:
            gradient_list: Gradients from workers

        Returns:
            Updated model weights
        """
        # Aggregate
        aggregated = self.aggregate_gradients(gradient_list)

        # Apply
        self.apply_gradients(aggregated)

        # Return new weights
        return self.weights()
