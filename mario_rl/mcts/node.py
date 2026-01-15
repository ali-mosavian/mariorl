"""MCTS tree node implementation with generic state type.

Supports both real MCTS (emulator snapshots) and imagined MCTS (latent states).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import numpy as np
import torch


@dataclass
class MCTSNode[S]:
    """Node in the MCTS tree, generic over state type S.

    S can be:
    - np.ndarray: Emulator snapshot for real MCTS
    - torch.Tensor: Latent state for MuZero-style imagined MCTS

    Attributes:
        state: The state representation (emulator snapshot or latent tensor).
        obs: Observation array for network input (C, H, W).
        parent: Parent node in tree (None for root).
        action: Action that led to this node from parent.
        visits: Number of times this node has been visited.
        total_value: Sum of all backpropagated values through this node.
        children: List of child nodes.
        terminal: Whether this is a terminal state (death/flag/timeout).
        prior: Prior probability from policy network (for PUCT).
        reward: Predicted/actual reward for reaching this node.
    """

    state: S
    obs: np.ndarray
    parent: MCTSNode[S] | None = None
    action: int | None = None
    visits: int = 0
    total_value: float = 0.0
    children: list[MCTSNode[S]] = field(default_factory=list)
    terminal: bool = False
    prior: float = 0.0
    reward: float = 0.0

    @property
    def value(self) -> float:
        """Average value of this node (Q-value estimate)."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def ucb_score(self, exploration: float = 1.41) -> float:
        """Calculate UCB1 score for node selection.

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
        """Calculate PUCT score (Predictor + UCB) like AlphaZero/MuZero.

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
        explore = (
            exploration
            * self.prior
            * prior_weight
            * float(np.sqrt(self.parent.visits))
            / (1 + self.visits)
        )
        return exploit + explore

    def is_fully_expanded(self, num_actions: int) -> bool:
        """Check if all actions have been tried from this node."""
        return len(self.children) >= num_actions

    def best_child(
        self,
        exploration: float = 1.41,
        use_puct: bool = False,
        prior_weight: float = 1.0,
    ) -> MCTSNode[S]:
        """Select best child using UCB or PUCT.

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

    def most_visited_child(self) -> MCTSNode[S]:
        """Get child with most visits (for final action selection).

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

    def get_policy_target(self, num_actions: int, temperature: float = 1.0) -> np.ndarray:
        """Get policy target from visit counts (for MuZero training).

        Args:
            num_actions: Total number of possible actions.
            temperature: Temperature for softening the distribution.
                         0 = argmax, 1 = proportional to visits.

        Returns:
            Probability distribution over actions based on visit counts.
        """
        visits = np.zeros(num_actions, dtype=np.float32)
        for child in self.children:
            if child.action is not None:
                visits[child.action] = child.visits

        if temperature == 0:
            # Argmax
            policy = np.zeros_like(visits)
            policy[visits.argmax()] = 1.0
        else:
            # Softmax with temperature
            visits_temp = visits ** (1.0 / temperature)
            total = visits_temp.sum()
            policy = visits_temp / total if total > 0 else np.ones_like(visits) / num_actions

        return policy

    def __repr__(self) -> str:
        return (
            f"MCTSNode(action={self.action}, visits={self.visits}, "
            f"value={self.value:.3f}, children={len(self.children)}, "
            f"terminal={self.terminal}, reward={self.reward:.3f})"
        )


# Type aliases for different MCTS modes
type EmulatorNode = MCTSNode[np.ndarray]
type LatentNode = MCTSNode[torch.Tensor]
