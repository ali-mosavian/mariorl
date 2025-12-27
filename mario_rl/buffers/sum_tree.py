"""
Sum Tree data structure for O(log n) priority sampling.

Used by Prioritized Experience Replay to efficiently sample
transitions proportional to their priority.
"""

from typing import Tuple
from dataclasses import field
from dataclasses import dataclass

import numpy as np


@dataclass
class SumTree:
    """
    Sum Tree data structure for O(log n) priority sampling.

    A binary tree where each parent is the sum of its children.
    Leaf nodes store priorities, internal nodes store sums.

    Structure (capacity=4):
                    [sum]
                   /     \\
              [sum]       [sum]
             /    \\      /    \\
           [p0]  [p1]  [p2]  [p3]  <- priorities (leaves)
    """

    capacity: int
    tree: np.ndarray = field(init=False, repr=False)
    data_pointer: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Initialize tree array with zeros."""
        # Tree has 2*capacity - 1 nodes (capacity leaves + capacity-1 internal)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.data_pointer = 0

    @property
    def total(self) -> float:
        """Return the root node (sum of all priorities)."""
        return float(self.tree[0])

    def add(self, priority: float) -> int:
        """
        Add a new priority and return the leaf index.

        Args:
            priority: Priority value for the new sample

        Returns:
            Leaf index where priority was stored
        """
        leaf_idx = self.data_pointer + self.capacity - 1
        self.update(leaf_idx, priority)

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        return leaf_idx

    def update(self, leaf_idx: int, priority: float) -> None:
        """
        Update priority at leaf_idx and propagate change up the tree.

        Args:
            leaf_idx: Index in the tree array (not data index)
            priority: New priority value
        """
        change = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority

        # Propagate change up to root
        parent = leaf_idx
        while parent != 0:
            parent = (parent - 1) // 2
            self.tree[parent] += change

    def get(self, value: float) -> Tuple[int, float, int]:
        """
        Find leaf node for a given cumulative value.

        Args:
            value: Cumulative priority value to search for

        Returns:
            Tuple of (leaf_idx, priority, data_idx)
        """
        parent = 0

        while True:
            left = 2 * parent + 1
            right = left + 1

            # Reached leaf
            if left >= len(self.tree):
                leaf_idx = parent
                break

            # Go left or right based on value
            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                parent = right

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx
