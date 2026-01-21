"""MetricCollector protocol definition."""

from typing import Any
from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class MetricCollector(Protocol):
    """Protocol for metric collectors.

    Collectors observe training events and extract metrics.
    This keeps metric extraction logic separate from core training.

    Methods:
        on_step: Called after each environment step
        on_episode_end: Called when an episode ends
        on_train_step: Called after a training step with loss/metrics
        flush: Flush accumulated metrics to logger/publisher
    """

    def on_step(self, info: dict[str, Any]) -> None:
        """Called after each environment step.

        Args:
            info: Environment step info dict (x_pos, state, etc.)
        """
        ...

    def on_episode_end(self, info: dict[str, Any]) -> None:
        """Called when an episode ends.

        Args:
            info: Final step info dict including terminal state
        """
        ...

    def on_train_step(self, metrics: dict[str, Any]) -> None:
        """Called after a training step.

        Args:
            metrics: Training metrics (loss, q_mean, td_error, etc.)
        """
        ...

    def flush(self) -> None:
        """Flush accumulated metrics to logger/publisher."""
        ...
