"""
N-step buffer for computing multi-step returns.

Accumulates transitions and computes discounted N-step rewards
for better credit assignment in reinforcement learning.
"""

from typing import List
from typing import Tuple
from dataclasses import field
from dataclasses import dataclass

import numpy as np


@dataclass
class NStepBuffer:
    """
    Buffer for computing N-step returns.

    Accumulates transitions and computes discounted N-step rewards.
    """

    n_step: int
    gamma: float
    buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = field(init=False, default_factory=list)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Tuple[np.ndarray, int, float, np.ndarray, bool] | None:
        """
        Add transition and return N-step transition if ready.

        Args:
            state: Current observation
            action: Action taken
            reward: Reward received
            next_state: Next observation
            done: Whether episode ended

        Returns:
            N-step transition tuple or None if not enough steps yet
        """
        self.buffer.append((state.copy(), action, reward, next_state.copy(), done))

        if len(self.buffer) < self.n_step:
            return None

        # Compute N-step return
        n_step_reward = 0.0
        for i, (_, _, r, _, d) in enumerate(self.buffer):
            n_step_reward += (self.gamma**i) * r
            if d:
                # Episode ended early - use actual final state
                result = (
                    self.buffer[0][0],
                    self.buffer[0][1],
                    n_step_reward,
                    self.buffer[i][3],
                    True,
                )
                self.buffer.pop(0)
                return result

        # Full N-step - use state from N steps ahead
        result = (
            self.buffer[0][0],
            self.buffer[0][1],
            n_step_reward,
            self.buffer[-1][3],
            self.buffer[-1][4],
        )
        self.buffer.pop(0)
        return result

    def flush(self) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        """
        Flush remaining transitions at episode end.

        Returns:
            List of remaining N-step transitions
        """
        transitions = []
        while len(self.buffer) > 0:
            n_step_reward = 0.0
            last_idx = len(self.buffer) - 1

            for i, (_, _, r, _, d) in enumerate(self.buffer):
                n_step_reward += (self.gamma**i) * r
                if d:
                    last_idx = i
                    break

            transitions.append(
                (
                    self.buffer[0][0],
                    self.buffer[0][1],
                    n_step_reward,
                    self.buffer[last_idx][3],
                    self.buffer[last_idx][4],
                )
            )
            self.buffer.pop(0)

        return transitions

    def reset(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
