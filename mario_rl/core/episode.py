"""
Episode state tracking.

Mutable state that gets reset at the start of each episode.
"""

from dataclasses import dataclass


@dataclass
class EpisodeState:
    """
    Mutable state for current episode.

    This is reset at the start of each new episode.
    """

    reward: float = 0.0
    length: int = 0
    best_x: int = 0
    start_time: int = 400  # Mario starts with 400 time

    def reset(self) -> None:
        """Reset for new episode."""
        self.reward = 0.0
        self.length = 0
        self.best_x = 0
        # start_time stays at 400

    def step(self, reward: float, x_pos: int) -> None:
        """Update state after a step."""
        self.reward += reward
        self.length += 1
        if x_pos > self.best_x:
            self.best_x = x_pos
