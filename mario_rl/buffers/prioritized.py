"""
Prioritized Experience Replay buffer using SumTree.

Samples experiences proportional to their TD-error priority.
Uses importance sampling weights to correct for bias.

Reference: Schaul et al. "Prioritized Experience Replay" (2015)
"""

from typing import Tuple
from dataclasses import field
from dataclasses import dataclass

import numpy as np

from mario_rl.buffers.sum_tree import SumTree


@dataclass
class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer using SumTree.

    Samples experiences proportional to their TD-error priority.
    Uses importance sampling weights to correct for bias.
    """

    capacity: int
    obs_shape: Tuple[int, ...]
    alpha: float = 0.6  # Priority exponent (0 = uniform, 1 = full prioritization)
    beta_start: float = 0.4  # Initial importance sampling exponent
    beta_end: float = 1.0  # Final importance sampling exponent
    epsilon: float = 1e-6  # Small constant to ensure non-zero priorities

    # Storage arrays (initialized in __post_init__)
    tree: SumTree = field(init=False, repr=False)
    states: np.ndarray = field(init=False, repr=False)
    actions: np.ndarray = field(init=False, repr=False)
    rewards: np.ndarray = field(init=False, repr=False)
    next_states: np.ndarray = field(init=False, repr=False)
    dones: np.ndarray = field(init=False, repr=False)

    # Tracking
    size: int = field(init=False, default=0)
    max_priority: float = field(init=False, default=1.0)
    current_beta: float = field(init=False, default=0.4)

    def __post_init__(self) -> None:
        """Initialize tree and storage arrays."""
        self.tree = SumTree(capacity=self.capacity)
        self.states = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.size = 0
        self.max_priority = 1.0
        self.current_beta = self.beta_start

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add a transition with max priority (will be updated after first sample).

        Args:
            state: Current observation
            action: Action taken
            reward: Reward received
            next_state: Next observation
            done: Whether episode ended
        """
        # Get data index from tree pointer
        data_idx = self.tree.data_pointer

        # Store transition
        self.states[data_idx] = state
        self.actions[data_idx] = action
        self.rewards[data_idx] = reward
        self.next_states[data_idx] = next_state
        self.dones[data_idx] = float(done)

        # Add with max priority (ensures new samples are seen at least once)
        priority = self.max_priority**self.alpha
        self.tree.add(priority)

        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        beta: float | None = None,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Sample a batch of transitions proportional to their priorities.

        Args:
            batch_size: Number of transitions to sample
            beta: Importance sampling exponent (uses current_beta if None)

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        if beta is None:
            beta = self.current_beta

        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        data_indices = np.zeros(batch_size, dtype=np.int64)

        # Divide priority range into segments for stratified sampling
        total_priority = self.tree.total
        segment_size = total_priority / batch_size

        for i in range(batch_size):
            # Sample uniformly within segment
            low = segment_size * i
            high = segment_size * (i + 1)
            value = np.random.uniform(low, high)

            leaf_idx, priority, data_idx = self.tree.get(value)
            indices[i] = leaf_idx
            priorities[i] = priority
            data_indices[i] = data_idx

        # Compute importance sampling weights
        # w_i = (N * P(i))^(-beta) / max_w
        probabilities = priorities / total_priority
        weights = (self.size * probabilities) ** (-beta)
        weights = weights / weights.max()  # Normalize

        return (
            self.states[data_indices],
            self.actions[data_indices],
            self.rewards[data_indices],
            self.next_states[data_indices],
            self.dones[data_indices],
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray,
    ) -> None:
        """
        Update priorities based on TD errors.

        Args:
            indices: Leaf indices from sampling
            td_errors: Absolute TD errors for each transition
        """
        for idx, td_error in zip(indices, td_errors, strict=False):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, abs(td_error) + self.epsilon)

    def update_beta(self, progress: float) -> None:
        """
        Update beta based on training progress.

        Args:
            progress: Training progress from 0 to 1
        """
        self.current_beta = self.beta_start + progress * (self.beta_end - self.beta_start)

    def __len__(self) -> int:
        return self.size
