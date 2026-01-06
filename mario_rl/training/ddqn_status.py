"""
Status collector for DDQN workers.

Gathers status data from DDQN-specific components into a WorkerStatus dataclass.
This separates data collection (algorithm-specific) from UI sending (generic).
"""

from typing import Any
from typing import Callable
from typing import Optional
from dataclasses import dataclass

from mario_rl.core.timing import TimingStats
from mario_rl.core.types import WorkerStatus
from mario_rl.core.episode import EpisodeState
from mario_rl.core.metrics import MetricsTracker
from mario_rl.core.weight_sync import WeightSync
from mario_rl.core.exploration import EpsilonGreedy
from mario_rl.buffers import PrioritizedReplayBuffer
from mario_rl.training.snapshot import SnapshotManager


@dataclass
class DDQNStatusCollector:
    """Collects status from DDQN worker components into WorkerStatus."""

    worker_id: int
    metrics: MetricsTracker
    episode: EpisodeState
    weights: WeightSync
    exploration: EpsilonGreedy
    timing: TimingStats
    buffer: PrioritizedReplayBuffer
    snapshots: Optional[SnapshotManager]
    get_level: Callable[[], str]
    batch_size: int = 64  # Batch size for training (from config)

    def collect(self, info: dict[str, Any]) -> WorkerStatus:
        """Collect current status into WorkerStatus dataclass."""
        buffer_size = len(self.buffer)
        buffer_capacity = self.buffer.capacity
        buffer_fill_pct = buffer_size / buffer_capacity * 100 if buffer_capacity > 0 else 0.0
        can_train = buffer_size >= self.batch_size

        return WorkerStatus(
            worker_id=self.worker_id,
            episode=self.metrics.episode_count,
            step=self.episode.length,
            reward=self.episode.reward,
            x_pos=info.get("x_pos", 0),
            game_time=info.get("time", 0),
            best_x=self.episode.best_x,
            best_x_ever=self.metrics.best_x_ever,
            deaths=self.metrics.deaths,
            flags=self.metrics.flags,
            epsilon=self.exploration.get_epsilon(self.metrics.total_steps),
            experiences=self.metrics.total_steps,
            weight_sync_count=self.weights.count,
            gradients_sent=self.metrics.gradients_sent,
            steps_per_sec=self.timing.steps_per_sec,
            snapshot_restores=self.snapshots.restore_count if self.snapshots else 0,
            restores_without_progress=self.snapshots.restores_without_progress if self.snapshots else 0,
            max_restores=self.snapshots.max_restores if self.snapshots else 3,
            current_level=self.get_level(),
            last_weight_sync=self.weights.last_sync,
            rolling_avg_reward=self.metrics.avg_reward,
            per_beta=self.buffer.current_beta,
            avg_speed=self.metrics.avg_speed,
            avg_x_at_death=self.metrics.avg_x_at_death,
            avg_time_to_flag=self.metrics.avg_time_to_flag,
            entropy=self.metrics.avg_entropy,
            last_action_time=self.timing.last_action_time,
            # Buffer diagnostics
            buffer_size=buffer_size,
            buffer_capacity=buffer_capacity,
            buffer_fill_pct=buffer_fill_pct,
            can_train=can_train,
        )
