"""
Generic UI reporter that sends WorkerStatus to the training UI.

This is algorithm-agnostic - it just sends WorkerStatus dataclass instances.
Each algorithm provides its own StatusCollector to gather the data.
"""

from typing import Any
import multiprocessing as mp
from typing import Optional
from dataclasses import field
from dataclasses import dataclass

from mario_rl.core.types import WorkerStatus


@dataclass
class UIReporter:
    """Sends WorkerStatus and log messages to UI queue."""

    worker_id: int
    queue: Optional[mp.Queue] = None

    # Lazy imports to avoid circular dependencies
    _UIMessage: Any = field(init=False, default=None, repr=False)
    _MessageType: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Import UI types lazily."""
        if self.queue is not None:
            try:
                from mario_rl.training.training_ui import UIMessage
                from mario_rl.training.training_ui import MessageType

                self._UIMessage = UIMessage
                self._MessageType = MessageType
            except ImportError:
                pass

    @property
    def enabled(self) -> bool:
        """Check if UI reporting is enabled."""
        return self.queue is not None and self._UIMessage is not None

    def send_status(self, status: WorkerStatus) -> None:
        """Send worker status to UI."""
        if not self.enabled:
            return

        try:
            # Convert frozen dataclass to dict for UI message
            data = {
                "episode": status.episode,
                "step": status.step,
                "reward": status.reward,
                "x_pos": status.x_pos,
                "game_time": status.game_time,
                "best_x": status.best_x,
                "best_x_ever": status.best_x_ever,
                "deaths": status.deaths,
                "flags": status.flags,
                "epsilon": status.epsilon,
                "experiences": status.experiences,
                "q_mean": 0.0,
                "q_max": 0.0,
                "weight_sync_count": status.weight_sync_count,
                "gradients_sent": status.gradients_sent,
                "steps_per_sec": status.steps_per_sec,
                "snapshot_restores": status.snapshot_restores,
                "restores_without_progress": status.restores_without_progress,
                "max_restores": status.max_restores,
                "current_level": status.current_level,
                "last_weight_sync": status.last_weight_sync,
                "rolling_avg_reward": status.rolling_avg_reward,
                "first_flag_time": status.avg_time_to_flag,
                "per_beta": status.per_beta,
                "avg_speed": status.avg_speed,
                "avg_x_at_death": status.avg_x_at_death,
                "avg_time_to_flag": status.avg_time_to_flag,
                "entropy": status.entropy,
                "last_action_time": status.last_action_time,
                # Buffer diagnostics
                "buffer_size": status.buffer_size,
                "buffer_capacity": status.buffer_capacity,
                "buffer_fill_pct": status.buffer_fill_pct,
                "can_train": status.can_train,
                # Elite buffer diagnostics
                "elite_size": status.elite_size,
                "elite_capacity": status.elite_capacity,
                "elite_min_quality": status.elite_min_quality,
                "elite_max_quality": status.elite_max_quality,
            }

            msg = self._UIMessage(
                msg_type=self._MessageType.WORKER_STATUS,
                source_id=self.worker_id,
                data=data,
            )
            self.queue.put_nowait(msg)  # type: ignore[union-attr]
        except Exception:
            pass

    def log(self, text: str) -> None:
        """Log message to UI or stdout."""
        if self.enabled:
            try:
                msg = self._UIMessage(
                    msg_type=self._MessageType.WORKER_LOG,
                    source_id=self.worker_id,
                    data={"text": text},
                )
                self.queue.put_nowait(msg)  # type: ignore[union-attr]
                return
            except Exception:
                pass

        # Fallback to stdout
        print(f"[W{self.worker_id}] {text}")
