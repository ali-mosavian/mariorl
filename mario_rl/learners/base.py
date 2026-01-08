"""Base protocol for all learners.

Learners contain model-specific training logic:
- Loss computation
- Gradient computation
- Target network updates (if applicable)
"""

from typing import Any
from typing import Protocol
from typing import runtime_checkable

from torch import Tensor

from mario_rl.models.base import Model


@runtime_checkable
class Learner(Protocol):
    """Protocol defining the interface all learners must implement.

    Learners contain model-specific training logic:
    - Loss computation for a batch of transitions
    - Target network updates (soft or hard)

    The distributed infrastructure uses this protocol to:
    1. Call compute_loss() to get gradients
    2. Aggregate gradients across workers
    3. Apply updates on the coordinator
    4. Broadcast updated weights

    Examples of learners that implement this protocol:
    - DDQNLearner: Double DQN with TD loss and target sync
    - DreamerLearner: World model loss + actor-critic on imagined rollouts
    """

    model: Model

    def compute_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute training loss for a batch of transitions.

        Args:
            states: Current observations (batch, *obs_shape)
            actions: Actions taken (batch,)
            rewards: Rewards received (batch,)
            next_states: Next observations (batch, *obs_shape)
            dones: Episode termination flags (batch,)

        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dict with training metrics (loss value, Q-values, etc.)
        """
        ...

    def update_targets(self, tau: float = 1.0) -> None:
        """Update target networks (if applicable).

        Args:
            tau: Soft update coefficient (1.0 = hard update, 0.005 = soft update)
        """
        ...
