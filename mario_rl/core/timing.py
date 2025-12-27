"""
Timing and performance tracking.

Tracks steps per second and action timing for monitoring.
"""

import time
from dataclasses import field
from dataclasses import dataclass


@dataclass
class TimingStats:
    """
    Timing and performance tracking for workers.

    Tracks steps per second and last action time for
    detecting stuck workers.
    """

    steps_per_sec: float = 0.0
    last_entropy: float = 0.0
    _last_time: float = field(init=False)
    _last_action_time: float = field(init=False)

    def __post_init__(self) -> None:
        """Initialize timing to current time."""
        now = time.time()
        self._last_time = now
        self._last_action_time = now

    def update_speed(self, steps: int) -> None:
        """
        Update steps per second calculation.

        Args:
            steps: Number of steps taken since last update
        """
        now = time.time()
        elapsed = now - self._last_time
        self.steps_per_sec = steps / elapsed if elapsed > 0 else 0
        self._last_time = now

    def record_action(self) -> None:
        """Record that an action was just taken."""
        self._last_action_time = time.time()

    @property
    def last_action_time(self) -> float:
        """Time of last action (for stuck detection)."""
        return self._last_action_time

    @property
    def seconds_since_action(self) -> float:
        """Seconds since last action was taken."""
        return time.time() - self._last_action_time
