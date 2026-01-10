"""Unified Replay Buffer with N-step returns and optional PER.

Combines:
- Circular buffer storage
- N-step return computation
- Prioritized Experience Replay (optional, alpha=0 disables)
- Batch sampling with tensor conversion
"""

from typing import Any
from dataclasses import field
from dataclasses import dataclass

import torch
import numpy as np
from torch import Tensor

from mario_rl.core.types import Transition
from mario_rl.buffers.nstep import NStepBuffer
from mario_rl.buffers.sum_tree import SumTree


@dataclass(frozen=True, slots=True)
class Batch:
    """Sampled batch from replay buffer."""

    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_states: Tensor
    dones: Tensor
    indices: Tensor | None = None
    weights: Tensor | None = None


@dataclass
class ReplayBuffer:
    """Unified replay buffer with N-step returns and optional PER.

    Features:
    - N-step return computation (n_step >= 1)
    - Prioritized sampling (alpha > 0) or uniform (alpha = 0)
    - Tensor output for training
    - Device placement support
    - Flag capture priority boost (asymmetric priority)
    """

    capacity: int
    obs_shape: tuple[int, ...]
    n_step: int = 1
    gamma: float = 0.99
    alpha: float = 0.0  # 0 = uniform, 0.6 = typical PER
    beta_start: float = 0.4
    beta_end: float = 1.0
    epsilon: float = 1e-6
    flag_priority_multiplier: float = 50.0  # Priority boost for flag captures

    # Storage (initialized in __post_init__)
    _states: np.ndarray = field(init=False, repr=False)
    _actions: np.ndarray = field(init=False, repr=False)
    _rewards: np.ndarray = field(init=False, repr=False)
    _next_states: np.ndarray = field(init=False, repr=False)
    _dones: np.ndarray = field(init=False, repr=False)

    # For PER
    _tree: SumTree | None = field(init=False, repr=False, default=None)

    # Tracking
    _size: int = field(init=False, default=0)
    _ptr: int = field(init=False, default=0)
    _max_priority: float = field(init=False, default=1.0)
    _current_beta: float = field(init=False, default=0.4)

    # Metadata storage for priority boost
    _flag_gets: np.ndarray = field(init=False, repr=False)

    # N-step buffer
    _nstep_buffer: NStepBuffer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize storage arrays."""
        # Preallocate arrays
        self._states = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self._actions = np.zeros(self.capacity, dtype=np.int64)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._next_states = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self._dones = np.zeros(self.capacity, dtype=np.float32)
        self._flag_gets = np.zeros(self.capacity, dtype=np.bool_)

        # Initialize tracking
        self._size = 0
        self._ptr = 0
        self._max_priority = 1.0
        self._current_beta = self.beta_start

        # PER tree (only if alpha > 0)
        if self.alpha > 0:
            self._tree = SumTree(capacity=self.capacity)

        # N-step buffer
        self._nstep_buffer = NStepBuffer(n_step=self.n_step, gamma=self.gamma)

    def add(self, transition: Transition) -> None:
        """Add a transition to the buffer.

        With N-step > 1, transitions are accumulated and processed
        to compute N-step returns before storage.
        """
        if self.n_step == 1:
            # Direct storage
            self._store(transition)
        else:
            # Process through N-step buffer
            nstep_transition = self._nstep_buffer.add(transition)
            if nstep_transition is not None:
                self._store(nstep_transition)

            # If episode ended, flush remaining
            if transition.done:
                for t in self._nstep_buffer.flush():
                    self._store(t)

    def _store(self, transition: Transition) -> None:
        """Store a single transition in the circular buffer."""
        idx = self._ptr

        self._states[idx] = transition.state
        self._actions[idx] = transition.action
        self._rewards[idx] = transition.reward
        self._next_states[idx] = transition.next_state
        self._dones[idx] = float(transition.done)
        self._flag_gets[idx] = transition.flag_get

        # Update PER tree with max priority (apply flag bonus)
        if self._tree is not None:
            base_priority = self._max_priority ** self.alpha
            # Apply asymmetric priority: flag captures get boosted
            if transition.flag_get and self.flag_priority_multiplier > 1.0:
                priority = base_priority * self.flag_priority_multiplier
            else:
                priority = base_priority
            self._tree.add(priority)

        # Update pointers
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def flush(self) -> None:
        """Flush remaining N-step transitions (call at episode end)."""
        for t in self._nstep_buffer.flush():
            self._store(t)

    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self._size >= batch_size

    def sample(
        self,
        batch_size: int,
        device: str = "cpu",
        beta: float | None = None,
    ) -> Batch:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample
            device: Device for output tensors
            beta: Importance sampling exponent (for PER)

        Returns:
            Batch of transitions as tensors

        Raises:
            ValueError: If not enough samples in buffer
        """
        if not self.can_sample(batch_size):
            raise ValueError(
                f"Not enough samples: {self._size} < {batch_size}"
            )

        if self._tree is not None and self.alpha > 0:
            return self._sample_per(batch_size, device, beta)
        return self._sample_uniform(batch_size, device)

    def _sample_uniform(self, batch_size: int, device: str) -> Batch:
        """Sample uniformly from buffer."""
        indices = np.random.randint(0, self._size, size=batch_size)

        return Batch(
            states=torch.from_numpy(self._states[indices]).to(device),
            actions=torch.from_numpy(self._actions[indices]).to(device),
            rewards=torch.from_numpy(self._rewards[indices]).to(device),
            next_states=torch.from_numpy(self._next_states[indices]).to(device),
            dones=torch.from_numpy(self._dones[indices]).to(device),
            indices=torch.from_numpy(indices).to(device),
            weights=torch.ones(batch_size, device=device),
        )

    def _sample_per(self, batch_size: int, device: str, beta: float | None) -> Batch:
        """Sample with prioritized experience replay."""
        if beta is None:
            beta = self._current_beta

        assert self._tree is not None

        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        data_indices = np.zeros(batch_size, dtype=np.int64)

        # Stratified sampling
        total_priority = self._tree.total
        segment_size = total_priority / batch_size

        for i in range(batch_size):
            low = segment_size * i
            high = segment_size * (i + 1)
            value = np.random.uniform(low, high)

            node = self._tree.get(value)
            indices[i] = node.leaf_idx
            priorities[i] = node.priority
            data_indices[i] = node.data_idx

        # Importance sampling weights
        probabilities = priorities / total_priority
        weights = (self._size * probabilities) ** (-beta)
        weights = weights / weights.max()

        return Batch(
            states=torch.from_numpy(self._states[data_indices]).to(device),
            actions=torch.from_numpy(self._actions[data_indices]).to(device),
            rewards=torch.from_numpy(self._rewards[data_indices]).to(device),
            next_states=torch.from_numpy(self._next_states[data_indices]).to(device),
            dones=torch.from_numpy(self._dones[data_indices]).to(device),
            indices=torch.from_numpy(indices).to(device),
            weights=torch.from_numpy(weights.astype(np.float32)).to(device),
        )

    def update_priorities(self, indices: np.ndarray | Tensor, td_errors: np.ndarray | Tensor) -> None:
        """Update priorities based on TD errors.

        Maintains asymmetric priority: flag captures keep their boost multiplier.
        """
        if self._tree is None:
            return  # Uniform sampling, no priorities

        if isinstance(indices, Tensor):
            indices = indices.cpu().numpy()
        if isinstance(td_errors, Tensor):
            td_errors = td_errors.cpu().numpy()

        for leaf_idx, td_error in zip(indices, td_errors, strict=False):
            base_priority = (abs(td_error) + self.epsilon) ** self.alpha

            # Maintain flag priority boost (convert leaf_idx to data_idx)
            data_idx = int(leaf_idx) - self.capacity + 1
            if 0 <= data_idx < self.capacity and self._flag_gets[data_idx]:
                priority = base_priority * self.flag_priority_multiplier
            else:
                priority = base_priority

            self._tree.update(int(leaf_idx), priority)
            self._max_priority = max(self._max_priority, abs(td_error) + self.epsilon)

    def update_beta(self, progress: float) -> None:
        """Update beta based on training progress (0 to 1)."""
        self._current_beta = self.beta_start + progress * (self.beta_end - self.beta_start)

    def reset(self) -> None:
        """Clear the buffer."""
        self._size = 0
        self._ptr = 0
        self._max_priority = 1.0
        self._flag_gets.fill(False)
        self._nstep_buffer.reset()

        if self._tree is not None:
            self._tree = SumTree(capacity=self.capacity)

    def __len__(self) -> int:
        return self._size
