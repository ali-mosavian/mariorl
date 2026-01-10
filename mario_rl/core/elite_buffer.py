"""Elite Buffer for preserving successful experiences.

This buffer maintains a protected set of high-quality transitions that
never get overwritten by normal buffer cycling. It addresses the
"forgetting success" problem where agents lose the ability to replicate
successful behaviors as the main buffer fills with failure experiences.

Quality is computed as: max_x + flag_bonus + reward_bonus
- max_x: The furthest X position reached in the episode
- flag_bonus: 1000 points for capturing the flag
- reward_bonus: 0.1 * total_episode_reward

Usage:
    elite_buffer = EliteBuffer(capacity=1000)
    
    # Add transitions from a successful episode
    for transition in episode_transitions:
        elite_buffer.add(transition, episode_quality)
    
    # Sample from elite buffer (10-20% of batch)
    elite_samples = elite_buffer.sample(batch_size // 5)
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from mario_rl.core.types import Transition


@dataclass(frozen=True, slots=True)
class EliteTransition:
    """A transition with associated quality score."""

    transition: Transition
    quality: float
    episode_id: int


@dataclass
class EliteBuffer:
    """Protected buffer for high-quality experiences.

    Maintains a fixed-size buffer of the best transitions, replacing
    lowest-quality entries only when new higher-quality ones arrive.

    Attributes:
        capacity: Maximum number of transitions to store.
        min_quality_to_add: Minimum quality threshold for new transitions.
    """

    capacity: int = 1000
    min_quality_to_add: float = 0.0

    # Storage
    _buffer: list[EliteTransition] = field(default_factory=list, init=False)
    _min_quality: float = field(default=float("-inf"), init=False)
    _episode_counter: int = field(default=0, init=False)
    _total_added: int = field(default=0, init=False)

    # Cached arrays for sampling (invalidated on add)
    _states_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _actions_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _rewards_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _next_states_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _dones_cache: np.ndarray | None = field(default=None, init=False, repr=False)

    def add_episode(
        self,
        transitions: list[Transition],
        max_x: int,
        flag_captured: bool,
        episode_reward: float,
    ) -> int:
        """Add all transitions from an episode with computed quality.

        Args:
            transitions: List of transitions from the episode.
            max_x: Maximum X position reached in the episode.
            flag_captured: Whether the flag was captured.
            episode_reward: Total reward for the episode.

        Returns:
            Number of transitions actually added to the buffer.
        """
        quality = self.compute_quality(max_x, flag_captured, episode_reward)

        if quality < self.min_quality_to_add:
            return 0

        added = 0
        self._episode_counter += 1
        episode_id = self._episode_counter

        for transition in transitions:
            if self._add_single(transition, quality, episode_id):
                added += 1

        return added

    def add(self, transition: Transition, quality: float, episode_id: int = -1) -> bool:
        """Add a single transition with explicit quality.

        Args:
            transition: The transition to add.
            quality: Quality score for this transition.
            episode_id: Optional episode identifier.

        Returns:
            True if the transition was added, False if rejected.
        """
        if episode_id == -1:
            self._episode_counter += 1
            episode_id = self._episode_counter

        return self._add_single(transition, quality, episode_id)

    def _add_single(
        self, transition: Transition, quality: float, episode_id: int
    ) -> bool:
        """Internal method to add a single transition."""
        if quality < self.min_quality_to_add:
            return False

        elite = EliteTransition(transition=transition, quality=quality, episode_id=episode_id)

        if len(self._buffer) < self.capacity:
            self._buffer.append(elite)
            self._invalidate_cache()
            self._update_min_quality()
            self._total_added += 1
            return True

        if quality > self._min_quality:
            # Find and replace the lowest quality entry
            min_idx = min(range(len(self._buffer)), key=lambda i: self._buffer[i].quality)
            self._buffer[min_idx] = elite
            self._invalidate_cache()
            self._update_min_quality()
            self._total_added += 1
            return True

        return False

    def _update_min_quality(self) -> None:
        """Update cached minimum quality."""
        if self._buffer:
            self._min_quality = min(e.quality for e in self._buffer)
        else:
            self._min_quality = float("-inf")

    def _invalidate_cache(self) -> None:
        """Invalidate cached arrays."""
        self._states_cache = None
        self._actions_cache = None
        self._rewards_cache = None
        self._next_states_cache = None
        self._dones_cache = None

    def _build_cache(self) -> None:
        """Build cached arrays for efficient sampling."""
        if not self._buffer:
            return

        n = len(self._buffer)
        first_state = self._buffer[0].transition.state

        self._states_cache = np.zeros((n, *first_state.shape), dtype=np.float32)
        self._actions_cache = np.zeros(n, dtype=np.int64)
        self._rewards_cache = np.zeros(n, dtype=np.float32)
        self._next_states_cache = np.zeros((n, *first_state.shape), dtype=np.float32)
        self._dones_cache = np.zeros(n, dtype=np.float32)

        for i, elite in enumerate(self._buffer):
            t = elite.transition
            self._states_cache[i] = t.state
            self._actions_cache[i] = t.action
            self._rewards_cache[i] = t.reward
            self._next_states_cache[i] = t.next_state
            self._dones_cache[i] = float(t.done)

    def sample(self, batch_size: int, device: str = "cpu") -> dict[str, Tensor] | None:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.
            device: Device for output tensors.

        Returns:
            Dictionary with states, actions, rewards, next_states, dones tensors,
            or None if buffer is empty.
        """
        if not self._buffer:
            return None

        # Build cache if needed
        if self._states_cache is None:
            self._build_cache()

        actual_size = min(batch_size, len(self._buffer))
        indices = np.random.choice(len(self._buffer), size=actual_size, replace=False)

        return {
            "states": torch.from_numpy(self._states_cache[indices]).to(device),
            "actions": torch.from_numpy(self._actions_cache[indices]).to(device),
            "rewards": torch.from_numpy(self._rewards_cache[indices]).to(device),
            "next_states": torch.from_numpy(self._next_states_cache[indices]).to(device),
            "dones": torch.from_numpy(self._dones_cache[indices]).to(device),
            "weights": torch.ones(actual_size, device=device),  # No importance sampling
        }

    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self._buffer) >= batch_size

    @staticmethod
    def compute_quality(max_x: int, flag_captured: bool, episode_reward: float) -> float:
        """Compute quality score for an episode.

        Quality = max_x + flag_bonus + reward_bonus

        Args:
            max_x: Maximum X position reached.
            flag_captured: Whether the flag was captured.
            episode_reward: Total episode reward.

        Returns:
            Quality score.
        """
        flag_bonus = 1000.0 if flag_captured else 0.0
        reward_bonus = episode_reward * 0.1
        return float(max_x) + flag_bonus + reward_bonus

    def get_stats(self) -> dict[str, Any]:
        """Get buffer statistics."""
        if not self._buffer:
            return {
                "size": 0,
                "capacity": self.capacity,
                "fill_pct": 0.0,
                "min_quality": 0.0,
                "max_quality": 0.0,
                "mean_quality": 0.0,
                "total_added": self._total_added,
                "unique_episodes": 0,
            }

        qualities = [e.quality for e in self._buffer]
        episode_ids = {e.episode_id for e in self._buffer}

        return {
            "size": len(self._buffer),
            "capacity": self.capacity,
            "fill_pct": len(self._buffer) / self.capacity * 100,
            "min_quality": min(qualities),
            "max_quality": max(qualities),
            "mean_quality": sum(qualities) / len(qualities),
            "total_added": self._total_added,
            "unique_episodes": len(episode_ids),
        }

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"EliteBuffer(size={len(self)}, capacity={self.capacity}, min_q={self._min_quality:.1f})"
