"""Protocols for DQN models.

Defines the interface that all DQN-like models must implement,
enabling generic code to work with both CNN-based (frames) and
MLP-based (RAM) architectures.
"""

from typing import Iterator
from typing import Protocol
from typing import runtime_checkable

from torch import Tensor
from torch.nn import Parameter


@runtime_checkable
class DDQNModel(Protocol):
    """Protocol for Double DQN models.

    Both `DoubleDQN` (CNN for frames) and `RAMDoubleDQN` (MLP for RAM)
    implement this protocol, allowing learners and workers to be
    observation-type agnostic.

    Required attributes:
        num_actions: Number of discrete actions in action space

    Required methods:
        __call__: Forward pass returning Q-values
        sync_target: Hard copy online weights to target network
        soft_update: Polyak averaging update to target network
        parameters: Iterator over trainable parameters
        state_dict: Get model state for checkpointing
        load_state_dict: Load model state from checkpoint
    """

    num_actions: int

    def __call__(
        self,
        x: Tensor,
        network: str = "online",
        action_history: Tensor | None = None,
        return_danger: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass returning Q-values.

        Args:
            x: Observation tensor (batch, *obs_shape)
               - For frames: (batch, 4, 64, 64) uint8
               - For RAM: (batch, 2048) uint8
            network: Which network to use ("online" or "target")
            action_history: Optional one-hot action history (batch, history_len, num_actions)
            return_danger: If True, also return danger predictions

        Returns:
            Q-values for each action (batch, num_actions)
            If return_danger: tuple of (q_values, danger_predictions)
        """
        ...

    def sync_target(self) -> None:
        """Hard sync: copy online weights to target network."""
        ...

    def soft_update(self, tau: float) -> None:
        """Soft update target network (polyak averaging).

        θ_target = τ * θ_online + (1-τ) * θ_target

        Args:
            tau: Interpolation coefficient (0 < tau <= 1)
        """
        ...

    def parameters(self) -> Iterator[Parameter]:
        """Return iterator over model parameters."""
        ...

    def state_dict(self) -> dict:
        """Return model state for checkpointing."""
        ...

    def load_state_dict(self, state_dict: dict) -> None:
        """Load model state from checkpoint."""
        ...
