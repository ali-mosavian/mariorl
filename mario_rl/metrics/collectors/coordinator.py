"""Coordinator-side collectors for learner/coordinator metrics.

These collectors handle metrics on the coordinator side:
- Gradient reception and aggregation
- Learning updates
- Checkpoints
- Worker metrics aggregation
"""

import time
from typing import Any
from typing import Protocol
from dataclasses import field
from typing import TYPE_CHECKING
from dataclasses import dataclass
from typing import runtime_checkable

if TYPE_CHECKING:
    from mario_rl.metrics.logger import MetricLogger


@runtime_checkable
class CoordinatorCollector(Protocol):
    """Protocol for coordinator-side metric collectors.

    Different from worker collectors - handles gradient/update events.
    """

    def on_gradients_received(self, worker_id: int, packet: Any) -> None:
        """Called when gradients received from a worker.

        Args:
            worker_id: ID of the worker that sent gradients
            packet: Gradient packet containing grads, metrics, etc.
        """
        ...

    def on_update_applied(self, metrics: dict[str, Any]) -> None:
        """Called after optimizer step applied.

        Args:
            metrics: Update metrics (lr, grad_norm, weight_version, etc.)
        """
        ...

    def on_checkpoint_saved(self, path: str) -> None:
        """Called when checkpoint saved.

        Args:
            path: Path where checkpoint was saved
        """
        ...

    def on_worker_metrics(self, worker_id: int, snapshot: dict[str, Any]) -> None:
        """Called when worker metrics received via ZMQ.

        Args:
            worker_id: ID of the worker
            snapshot: Metrics snapshot from worker's logger
        """
        ...

    def flush(self) -> None:
        """Flush accumulated metrics."""
        ...


@dataclass
class GradientCollector:
    """Tracks gradient flow and coordination metrics.

    Metrics:
    - gradients_received (counter)
    - grads_per_sec (gauge)
    - total_timesteps (counter)
    """

    logger: "MetricLogger"

    _grads_since_log: int = field(init=False, default=0)
    _last_log_time: float = field(init=False, default_factory=time.time)

    def on_gradients_received(self, worker_id: int, packet: Any) -> None:
        """Track gradient reception."""
        self.logger.count("gradients_received")
        self._grads_since_log += 1

        # Calculate grads_per_sec
        now = time.time()
        elapsed = now - self._last_log_time
        if elapsed >= 1.0:
            grads_per_sec = self._grads_since_log / elapsed
            self.logger.gauge("grads_per_sec", grads_per_sec)
            self._grads_since_log = 0
            self._last_log_time = now

        # Track timesteps from packet
        timesteps = getattr(packet, "timesteps", 0) or packet.get("timesteps", 0) if isinstance(packet, dict) else 0
        if timesteps:
            self.logger.count("total_timesteps", n=timesteps)

    def on_update_applied(self, metrics: dict[str, Any]) -> None:
        """Track update metrics."""
        self.logger.count("update_count")

        if "weight_version" in metrics:
            self.logger.gauge("weight_version", metrics["weight_version"])

        if "lr" in metrics:
            self.logger.gauge("learning_rate", metrics["lr"])

    def on_checkpoint_saved(self, path: str) -> None:
        """Track checkpoint saves."""
        self.logger.count("checkpoints_saved")

    def on_worker_metrics(self, worker_id: int, snapshot: dict[str, Any]) -> None:
        """GradientCollector ignores worker metrics."""
        pass

    def flush(self) -> None:
        """Flush metrics to logger."""
        self.logger.flush()


@dataclass
class AggregatorCollector:
    """Aggregates metrics across all workers.

    Maintains per-worker snapshots and computes aggregates.
    """

    logger: "MetricLogger"

    _worker_snapshots: dict[int, dict[str, Any]] = field(init=False, default_factory=dict)

    def on_gradients_received(self, worker_id: int, packet: Any) -> None:
        """AggregatorCollector ignores gradient events."""
        pass

    def on_update_applied(self, metrics: dict[str, Any]) -> None:
        """AggregatorCollector ignores update events."""
        pass

    def on_checkpoint_saved(self, path: str) -> None:
        """AggregatorCollector ignores checkpoint events."""
        pass

    def on_worker_metrics(self, worker_id: int, snapshot: dict[str, Any]) -> None:
        """Update worker snapshot and recalculate aggregates."""
        self._worker_snapshots[worker_id] = snapshot
        self._update_aggregates()

    def _update_aggregates(self) -> None:
        """Compute and log aggregate metrics."""
        if not self._worker_snapshots:
            return

        snapshots = list(self._worker_snapshots.values())

        # Aggregate rolling metrics (average across workers)
        for metric in ["reward", "speed"]:
            values = [s.get(metric, 0) for s in snapshots if s.get(metric, 0) > 0]
            if values:
                self.logger.gauge(f"avg_{metric}", sum(values) / len(values))

        # Aggregate counters (sum across workers)
        for metric in ["deaths", "flags", "episodes", "steps"]:
            total = sum(s.get(metric, 0) for s in snapshots)
            self.logger.gauge(f"total_{metric}", total)

    def flush(self) -> None:
        """No-op - let composite handle flushing."""
        pass


@dataclass
class CoordinatorComposite:
    """Combines multiple coordinator collectors."""

    collectors: list[CoordinatorCollector] = field(default_factory=list)

    def on_gradients_received(self, worker_id: int, packet: Any) -> None:
        for c in self.collectors:
            c.on_gradients_received(worker_id, packet)

    def on_update_applied(self, metrics: dict[str, Any]) -> None:
        for c in self.collectors:
            c.on_update_applied(metrics)

    def on_checkpoint_saved(self, path: str) -> None:
        for c in self.collectors:
            c.on_checkpoint_saved(path)

    def on_worker_metrics(self, worker_id: int, snapshot: dict[str, Any]) -> None:
        for c in self.collectors:
            c.on_worker_metrics(worker_id, snapshot)

    def flush(self) -> None:
        for c in self.collectors:
            c.flush()
