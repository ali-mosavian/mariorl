"""
Metrics tracking for training.

Consolidates all the rolling history lists and counters
that were scattered across the worker.
"""

from typing import List
from dataclasses import field
from dataclasses import dataclass

import numpy as np


@dataclass
class MetricsTracker:
    """
    Tracks training metrics with rolling history.

    Consolidates episode/step counters and rolling averages
    into a single component.
    """

    max_history: int = 100

    # Counters
    episode_count: int = 0
    total_steps: int = 0
    deaths: int = 0
    timeouts: int = 0
    flags: int = 0
    best_x_ever: int = 0
    gradients_sent: int = 0
    weight_sync_count: int = 0

    # Rolling histories
    reward_history: List[float] = field(default_factory=list)
    speed_history: List[float] = field(default_factory=list)
    entropy_history: List[float] = field(default_factory=list)
    x_at_death_history: List[int] = field(default_factory=list)
    time_to_flag_history: List[int] = field(default_factory=list)

    def add_reward(self, reward: float) -> None:
        """Add episode reward to history."""
        self.reward_history.append(reward)
        if len(self.reward_history) > self.max_history:
            self.reward_history.pop(0)

    def add_speed(self, speed: float) -> None:
        """Add episode speed to history."""
        self.speed_history.append(speed)
        if len(self.speed_history) > 20:
            self.speed_history.pop(0)

    def add_entropy(self, entropy: float) -> None:
        """Add entropy sample to history."""
        self.entropy_history.append(entropy)
        if len(self.entropy_history) > self.max_history:
            self.entropy_history.pop(0)

    def add_death(self, x_pos: int) -> None:
        """Record death at position."""
        self.deaths += 1
        self.x_at_death_history.append(x_pos)
        if len(self.x_at_death_history) > 20:
            self.x_at_death_history.pop(0)

    def add_timeout(self) -> None:
        """Record timeout (ran out of game time)."""
        self.timeouts += 1

    def add_flag(self, game_time: int) -> None:
        """Record flag capture."""
        self.flags += 1
        self.time_to_flag_history.append(game_time)
        if len(self.time_to_flag_history) > 20:
            self.time_to_flag_history.pop(0)

    def update_best_x(self, x_pos: int) -> None:
        """Update best x position ever seen."""
        if x_pos > self.best_x_ever:
            self.best_x_ever = x_pos

    @property
    def avg_reward(self) -> float:
        """Average episode reward."""
        return float(np.mean(self.reward_history)) if self.reward_history else 0.0

    @property
    def avg_speed(self) -> float:
        """Average episode speed."""
        return float(np.mean(self.speed_history)) if self.speed_history else 0.0

    @property
    def avg_entropy(self) -> float:
        """Average action entropy."""
        return float(np.mean(self.entropy_history)) if self.entropy_history else 0.0

    @property
    def avg_x_at_death(self) -> float:
        """Average x position at death."""
        return float(np.mean(self.x_at_death_history)) if self.x_at_death_history else 0.0

    @property
    def avg_time_to_flag(self) -> float:
        """Average time remaining when reaching flag."""
        return float(np.mean(self.time_to_flag_history)) if self.time_to_flag_history else 0.0
