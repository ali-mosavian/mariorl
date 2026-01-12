"""
MCTS tree node implementation.
"""

from __future__ import annotations

from typing import Optional
from dataclasses import field
from dataclasses import dataclass

import numpy as np


@dataclass
class MCTSNode:
    """
    Node in the MCTS tree.

    Stores state information, visit statistics, and tree structure.
    Used for both real MCTS (with emulator) and imagined MCTS (with world model).

    Attributes:
        state_snapshot: Emulator state for restore (numpy array from dump_state).
        obs: Observation array for network input (C, H, W).
        parent: Parent node in tree (None for root).
        action: Action that led to this node from parent.
        visits: Number of times this node has been visited.
        total_value: Sum of all backpropagated values through this node.
        children: List of child nodes.
        terminal: Whether this is a terminal state (death/flag/timeout).
        prior: Prior probability from policy network (for PUCT).
    """

    state_snapshot: np.ndarray
    obs: np.ndarray
    parent: Optional[MCTSNode] = None
    action: Optional[int] = None
    visits: int = 0
    total_value: float = 0.0
    children: list[MCTSNode] = field(default_factory=list)
    terminal: bool = False
    prior: float = 0.0

    @property
    def value(self) -> float:
        """Average value of this node (Q-value estimate)."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def ucb_score(self, exploration: float = 1.41) -> float:
        """
        Calculate UCB1 score for node selection.

        UCB = Q + c * sqrt(ln(N_parent) / N)

        Args:
            exploration: Exploration constant (c)

        Returns:
            UCB score (higher = more promising to explore)
        """
        if self.visits == 0:
            return float("inf")
        if self.parent is None:
            return self.value

        exploit = self.value
        explore = exploration * float(np.sqrt(np.log(self.parent.visits) / self.visits))
        return exploit + explore

    def puct_score(self, exploration: float = 1.41, prior_weight: float = 1.0) -> float:
        """
        Calculate PUCT score (Predictor + UCB) like AlphaZero.

        PUCT = Q + c * P * sqrt(N_parent) / (1 + N)

        Args:
            exploration: Exploration constant
            prior_weight: Weight for prior probability

        Returns:
            PUCT score
        """
        if self.parent is None:
            return self.value

        exploit = self.value
        explore = exploration * self.prior * prior_weight * float(np.sqrt(self.parent.visits)) / (1 + self.visits)
        return exploit + explore

    def is_fully_expanded(self, num_actions: int) -> bool:
        """Check if all actions have been tried from this node."""
        return len(self.children) >= num_actions

    def best_child(
        self,
        exploration: float = 1.41,
        use_puct: bool = False,
        prior_weight: float = 1.0,
    ) -> MCTSNode:
        """
        Select best child using UCB or PUCT.

        Args:
            exploration: Exploration constant
            use_puct: Whether to use PUCT (requires prior probabilities)
            prior_weight: Weight for prior in PUCT

        Returns:
            Child node with highest score
        """
        if not self.children:
            raise ValueError("No children to select from")

        if use_puct:
            return max(
                self.children,
                key=lambda c: c.puct_score(exploration, prior_weight),
            )
        return max(self.children, key=lambda c: c.ucb_score(exploration))

    def most_visited_child(self) -> MCTSNode:
        """
        Get child with most visits (for final action selection).

        This is more robust than using value alone because
        visit count reflects confidence in the estimate.
        """
        if not self.children:
            raise ValueError("No children to select from")
        return max(self.children, key=lambda c: c.visits)

    def get_untried_actions(self, num_actions: int) -> list[int]:
        """Get actions not yet expanded from this node."""
        tried = {c.action for c in self.children}
        return [a for a in range(num_actions) if a not in tried]

    def __repr__(self) -> str:
        return (
            f"MCTSNode(action={self.action}, visits={self.visits}, "
            f"value={self.value:.3f}, children={len(self.children)}, "
            f"terminal={self.terminal})"
        )
