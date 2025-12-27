"""
UI Reporter for sending worker status to the training UI.

Encapsulates all UI communication logic to keep workers clean.
"""

from typing import Any
import multiprocessing as mp
from typing import Optional
from dataclasses import field
from dataclasses import dataclass


@dataclass
class UIReporter:
    """Handles sending status updates and logs to the UI queue."""

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

    def send_status(
        self,
        *,
        # Episode info
        episode: int,
        step: int,
        reward: float,
        x_pos: int,
        game_time: int,
        best_x: int,
        best_x_ever: int,
        # Counters
        deaths: int,
        flags: int,
        experiences: int,
        gradients_sent: int,
        weight_sync_count: int,
        # Exploration
        epsilon: float,
        # Q-values
        q_mean: float = 0.0,
        q_max: float = 0.0,
        # Performance
        steps_per_sec: float,
        last_action_time: float,
        last_weight_sync: float,
        # Snapshots
        snapshot_restores: int,
        restores_without_progress: int,
        max_restores: int,
        # Level
        current_level: str,
        # Averages
        rolling_avg_reward: float,
        avg_speed: float,
        avg_x_at_death: float,
        avg_time_to_flag: float,
        entropy: float,
        per_beta: float,
    ) -> None:
        """Send worker status to UI."""
        if not self.enabled:
            return

        try:
            msg = self._UIMessage(
                msg_type=self._MessageType.WORKER_STATUS,
                source_id=self.worker_id,
                data={
                    "episode": episode,
                    "step": step,
                    "reward": reward,
                    "x_pos": x_pos,
                    "game_time": game_time,
                    "best_x": best_x,
                    "best_x_ever": best_x_ever,
                    "deaths": deaths,
                    "flags": flags,
                    "epsilon": epsilon,
                    "experiences": experiences,
                    "q_mean": q_mean,
                    "q_max": q_max,
                    "weight_sync_count": weight_sync_count,
                    "gradients_sent": gradients_sent,
                    "steps_per_sec": steps_per_sec,
                    "snapshot_restores": snapshot_restores,
                    "restores_without_progress": restores_without_progress,
                    "max_restores": max_restores,
                    "current_level": current_level,
                    "last_weight_sync": last_weight_sync,
                    "rolling_avg_reward": rolling_avg_reward,
                    "first_flag_time": avg_time_to_flag,
                    "per_beta": per_beta,
                    "avg_speed": avg_speed,
                    "avg_x_at_death": avg_x_at_death,
                    "avg_time_to_flag": avg_time_to_flag,
                    "entropy": entropy,
                    "last_action_time": last_action_time,
                },
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
