"""
Exploration policies for reinforcement learning.

Provides epsilon-greedy and other exploration strategies.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EpsilonGreedy:
    """
    Epsilon-greedy exploration policy with linear decay.

    Epsilon decreases linearly from start to end over decay_steps.
    """

    start: float = 1.0
    end: float = 0.01
    decay_steps: int = 100_000

    def get_epsilon(self, step: int) -> float:
        """
        Get epsilon value for given step.

        Args:
            step: Current training step

        Returns:
            Epsilon value between end and start
        """
        progress = min(1.0, step / self.decay_steps)
        return self.start + (self.end - self.start) * progress

    def should_explore(self, step: int) -> bool:
        """
        Decide whether to explore (random action) or exploit.

        Args:
            step: Current training step

        Returns:
            True if should take random action
        """
        return float(np.random.random()) < self.get_epsilon(step)

    def select_action(self, step: int, num_actions: int, greedy_action: int) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            step: Current training step
            num_actions: Number of possible actions
            greedy_action: Action to take if not exploring

        Returns:
            Selected action index
        """
        if self.should_explore(step):
            return int(np.random.randint(0, num_actions))
        return greedy_action
