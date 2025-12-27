"""
N-step buffer for computing multi-step returns.

Accumulates transitions and computes discounted N-step rewards
for better credit assignment in reinforcement learning.
"""

from typing import List
from dataclasses import field
from dataclasses import dataclass

from mario_rl.core.types import Transition


@dataclass
class NStepBuffer:
    """
    Buffer for computing N-step returns.

    Accumulates transitions and computes discounted N-step rewards.
    """

    n_step: int
    gamma: float
    buffer: List[Transition] = field(init=False, default_factory=list)

    def add(self, transition: Transition) -> Transition | None:
        """
        Add transition and return N-step transition if ready.

        Args:
            transition: The experience transition to add

        Returns:
            N-step transition or None if not enough steps yet
        """
        # Store with copied arrays
        self.buffer.append(
            Transition(
                state=transition.state.copy(),
                action=transition.action,
                reward=transition.reward,
                next_state=transition.next_state.copy(),
                done=transition.done,
            )
        )

        if len(self.buffer) < self.n_step:
            return None

        # Compute N-step return
        n_step_reward = 0.0
        for i, t in enumerate(self.buffer):
            n_step_reward += (self.gamma**i) * t.reward
            if t.done:
                # Episode ended early - use actual final state
                result = Transition(
                    state=self.buffer[0].state,
                    action=self.buffer[0].action,
                    reward=n_step_reward,
                    next_state=self.buffer[i].next_state,
                    done=True,
                )
                self.buffer.pop(0)
                return result

        # Full N-step - use state from N steps ahead
        result = Transition(
            state=self.buffer[0].state,
            action=self.buffer[0].action,
            reward=n_step_reward,
            next_state=self.buffer[-1].next_state,
            done=self.buffer[-1].done,
        )
        self.buffer.pop(0)
        return result

    def flush(self) -> List[Transition]:
        """
        Flush remaining transitions at episode end.

        Returns:
            List of remaining N-step transitions
        """
        transitions: List[Transition] = []
        while len(self.buffer) > 0:
            n_step_reward = 0.0
            last_idx = len(self.buffer) - 1

            for i, t in enumerate(self.buffer):
                n_step_reward += (self.gamma**i) * t.reward
                if t.done:
                    last_idx = i
                    break

            transitions.append(
                Transition(
                    state=self.buffer[0].state,
                    action=self.buffer[0].action,
                    reward=n_step_reward,
                    next_state=self.buffer[last_idx].next_state,
                    done=self.buffer[last_idx].done,
                )
            )
            self.buffer.pop(0)

        return transitions

    def reset(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
