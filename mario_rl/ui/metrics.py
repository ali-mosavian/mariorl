"""Metrics aggregation for training monitoring.

Collects and aggregates metrics from workers and learner
for display in the training UI.
"""

from typing import Any
from collections import deque
from dataclasses import field
from dataclasses import dataclass


@dataclass
class WorkerMetrics:
    """Per-worker metrics."""

    worker_id: int
    episodes: int = 0
    steps: int = 0
    reward: float = 0.0
    x_pos: int = 0
    best_x: int = 0
    epsilon: float = 1.0
    loss: float = 0.0
    deaths: int = 0
    flags: int = 0
    last_heartbeat: float = 0.0


@dataclass
class LearnerMetrics:
    """Learner/coordinator metrics."""

    update_count: int = 0
    total_steps: int = 0
    learning_rate: float = 0.0
    loss: float = 0.0
    gradients_per_sec: float = 0.0


@dataclass
class MetricsAggregator:
    """Aggregates metrics from workers and learner."""

    num_workers: int
    max_history: int = 100

    # Per-worker metrics
    worker_metrics: dict[int, WorkerMetrics] = field(init=False, repr=False)

    # Learner metrics
    learner_metrics: LearnerMetrics = field(init=False)

    # History for charts
    loss_history: deque[float] = field(init=False, repr=False)
    reward_history: deque[float] = field(init=False, repr=False)
    steps_history: deque[int] = field(init=False, repr=False)

    # Global stats
    total_episodes: int = field(init=False, default=0)
    total_flags: int = field(init=False, default=0)
    total_deaths: int = field(init=False, default=0)
    best_x_ever: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Initialize state."""
        self.worker_metrics = {
            i: WorkerMetrics(worker_id=i)
            for i in range(self.num_workers)
        }
        self.learner_metrics = LearnerMetrics()
        self.loss_history = deque(maxlen=self.max_history)
        self.reward_history = deque(maxlen=self.max_history)
        self.steps_history = deque(maxlen=self.max_history)

    def update_worker(
        self,
        worker_id: int,
        episodes: int | None = None,
        steps: int | None = None,
        reward: float | None = None,
        x_pos: int | None = None,
        best_x: int | None = None,
        epsilon: float | None = None,
        loss: float | None = None,
        deaths: int | None = None,
        flags: int | None = None,
        last_heartbeat: float | None = None,
    ) -> None:
        """Update metrics for a worker.

        Args:
            worker_id: Worker to update
            **kwargs: Metrics to update
        """
        if worker_id not in self.worker_metrics:
            return

        wm = self.worker_metrics[worker_id]

        # Convert to int for fields used with :d format specifiers
        if episodes is not None:
            episodes = int(episodes)
            self.total_episodes += episodes - wm.episodes
            wm.episodes = episodes
        if steps is not None:
            wm.steps = int(steps)
        if reward is not None:
            wm.reward = reward
        if x_pos is not None:
            wm.x_pos = int(x_pos)
        if best_x is not None:
            best_x = int(best_x)
            wm.best_x = best_x
            self.best_x_ever = max(self.best_x_ever, best_x)
        if epsilon is not None:
            wm.epsilon = epsilon
        if loss is not None:
            wm.loss = loss
        if deaths is not None:
            deaths = int(deaths)
            self.total_deaths += deaths - wm.deaths
            wm.deaths = deaths
        if flags is not None:
            flags = int(flags)
            self.total_flags += flags - wm.flags
            wm.flags = flags
        if last_heartbeat is not None:
            wm.last_heartbeat = last_heartbeat

    def update_learner(
        self,
        update_count: int | None = None,
        total_steps: int | None = None,
        learning_rate: float | None = None,
        loss: float | None = None,
        gradients_per_sec: float | None = None,
    ) -> None:
        """Update learner metrics.

        Args:
            **kwargs: Metrics to update
        """
        lm = self.learner_metrics

        # Convert to int for fields used with :d format specifiers
        if update_count is not None:
            lm.update_count = int(update_count)
        if total_steps is not None:
            lm.total_steps = int(total_steps)
            self.steps_history.append(lm.total_steps)
        if learning_rate is not None:
            lm.learning_rate = learning_rate
        if loss is not None:
            lm.loss = loss
            self.loss_history.append(loss)
        if gradients_per_sec is not None:
            lm.gradients_per_sec = gradients_per_sec

    def add_reward(self, reward: float) -> None:
        """Add reward to history."""
        self.reward_history.append(reward)

    def summary(self) -> dict[str, Any]:
        """Get summary of all metrics.

        Returns:
            Dict with aggregated metrics
        """
        return {
            "learner": {
                "update_count": self.learner_metrics.update_count,
                "total_steps": self.learner_metrics.total_steps,
                "learning_rate": self.learner_metrics.learning_rate,
                "loss": self.learner_metrics.loss,
                "gradients_per_sec": self.learner_metrics.gradients_per_sec,
            },
            "workers": {
                wid: {
                    "episodes": wm.episodes,
                    "steps": wm.steps,
                    "reward": wm.reward,
                    "x_pos": wm.x_pos,
                    "best_x": wm.best_x,
                    "epsilon": wm.epsilon,
                    "loss": wm.loss,
                }
                for wid, wm in self.worker_metrics.items()
            },
            "global": {
                "total_episodes": self.total_episodes,
                "total_flags": self.total_flags,
                "total_deaths": self.total_deaths,
                "best_x_ever": self.best_x_ever,
            },
            "history": {
                "loss": list(self.loss_history),
                "reward": list(self.reward_history),
                "steps": list(self.steps_history),
            },
        }

    def format_worker_status(self, worker_id: int) -> str:
        """Get status line for a worker.

        Args:
            worker_id: Worker to get status for

        Returns:
            Formatted status string
        """
        wm = self.worker_metrics.get(worker_id)
        if wm is None:
            return f"W{worker_id}: N/A"

        return (
            f"W{worker_id}: ep={wm.episodes:4d} "
            f"x={wm.x_pos:5d} best={wm.best_x:5d} "
            f"eps={wm.epsilon:.3f} loss={wm.loss:.4f}"
        )

    def format_learner_status(self) -> str:
        """Get status line for learner.

        Returns:
            Formatted status string
        """
        lm = self.learner_metrics
        return (
            f"Updates: {lm.update_count:6d} "
            f"Steps: {lm.total_steps:8d} "
            f"LR: {lm.learning_rate:.2e} "
            f"Loss: {lm.loss:.4f}"
        )
