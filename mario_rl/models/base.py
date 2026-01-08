"""Base protocol for all models.

Models are pure network definitions with no training logic.
They must support forward pass, action selection, and state serialization.
"""

from typing import Protocol
from typing import runtime_checkable

from torch import Tensor


@runtime_checkable
class Model(Protocol):
    """Protocol defining the interface all models must implement.

    Models are pure network definitions with no training logic.
    They must support:
    - Forward pass returning action values/logits
    - Action selection (with optional exploration)
    - Weight state dict for checkpointing
    - Parameter access for gradient computation

    Examples of models that implement this protocol:
    - DoubleDQN: Dueling Double DQN with online/target networks
    - DreamerModel: World model with actor-critic heads
    """

    num_actions: int

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model.

        Args:
            x: Observation tensor (batch, *obs_shape)

        Returns:
            Action values/logits (batch, num_actions)
        """
        ...

    def select_action(self, x: Tensor, epsilon: float = 0.0) -> Tensor:
        """Select actions for given observations.

        Args:
            x: Observation tensor (batch, *obs_shape)
            epsilon: Exploration rate (0 = greedy, 1 = random)

        Returns:
            Selected actions (batch,)
        """
        ...

    def state_dict(self) -> dict:
        """Return model state for checkpointing."""
        ...

    def load_state_dict(self, state_dict: dict) -> None:
        """Load model state from checkpoint."""
        ...

    def parameters(self):
        """Return model parameters for optimization."""
        ...
