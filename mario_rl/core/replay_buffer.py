"""Unified Replay Buffer with N-step returns and optional PER.

Combines:
- Circular buffer storage
- N-step return computation
- Prioritized Experience Replay (optional, alpha=0 disables)
- Batch sampling with tensor conversion
"""

from dataclasses import field
from dataclasses import dataclass

import torch
import numpy as np
from torch import Tensor

from mario_rl.core.types import Transition
from mario_rl.core.types import TreeNode


# =============================================================================
# N-Step Buffer (inlined from buffers/nstep.py)
# =============================================================================


@dataclass
class NStepBuffer:
    """Buffer for computing N-step returns."""

    n_step: int
    gamma: float
    buffer: list[Transition] = field(init=False, default_factory=list)

    def add(self, transition: Transition) -> Transition | None:
        """Add transition and return N-step transition if ready."""
        self.buffer.append(
            Transition(
                state=transition.state.copy(),
                action=transition.action,
                reward=transition.reward,
                next_state=transition.next_state.copy(),
                done=transition.done,
                flag_get=transition.flag_get,
                max_x=transition.max_x,
            )
        )

        if len(self.buffer) < self.n_step:
            return None

        # Compute N-step return
        n_step_reward = 0.0
        for i, t in enumerate(self.buffer):
            n_step_reward += (self.gamma**i) * t.reward
            if t.done:
                result = Transition(
                    state=self.buffer[0].state,
                    action=self.buffer[0].action,
                    reward=n_step_reward,
                    next_state=self.buffer[i].next_state,
                    done=True,
                    flag_get=self.buffer[0].flag_get,
                    max_x=self.buffer[0].max_x,
                )
                self.buffer.pop(0)
                return result

        result = Transition(
            state=self.buffer[0].state,
            action=self.buffer[0].action,
            reward=n_step_reward,
            next_state=self.buffer[-1].next_state,
            done=self.buffer[-1].done,
            flag_get=self.buffer[0].flag_get,
            max_x=self.buffer[0].max_x,
        )
        self.buffer.pop(0)
        return result

    def flush(self) -> list[Transition]:
        """Flush remaining transitions at episode end."""
        transitions: list[Transition] = []
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
                    flag_get=self.buffer[0].flag_get,
                    max_x=self.buffer[0].max_x,
                )
            )
            self.buffer.pop(0)

        return transitions

    def reset(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()


# =============================================================================
# Sum Tree for PER (inlined from buffers/sum_tree.py)
# =============================================================================


@dataclass
class SumTree:
    """Sum Tree data structure for O(log n) priority sampling."""

    capacity: int
    tree: np.ndarray = field(init=False, repr=False)
    data_pointer: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Initialize tree array with zeros."""
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.data_pointer = 0

    @property
    def total(self) -> float:
        """Return the root node (sum of all priorities)."""
        return float(self.tree[0])

    def add(self, priority: float) -> int:
        """Add a new priority and return the leaf index."""
        leaf_idx = self.data_pointer + self.capacity - 1
        self.update(leaf_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        return leaf_idx

    def update(self, leaf_idx: int, priority: float) -> None:
        """Update priority at leaf_idx and propagate change up the tree."""
        change = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority

        parent = leaf_idx
        while parent != 0:
            parent = (parent - 1) // 2
            self.tree[parent] += change

    def get(self, value: float) -> TreeNode:
        """Find leaf node for a given cumulative value."""
        parent = 0

        while True:
            left = 2 * parent + 1
            right = left + 1

            if left >= len(self.tree):
                leaf_idx = parent
                break

            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                parent = right

        data_idx = leaf_idx - self.capacity + 1
        return TreeNode(leaf_idx=leaf_idx, priority=float(self.tree[leaf_idx]), data_idx=data_idx)


# =============================================================================
# Replay Buffer
# =============================================================================


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


# =============================================================================
# MuZero Trajectory Replay Buffer
# =============================================================================


@dataclass(frozen=True, slots=True)
class TrajectoryBatch:
    """Sampled batch of trajectory segments for MuZero training."""

    obs: Tensor  # (N, C, H, W) initial observations
    actions: Tensor  # (N, K) action sequences
    rewards: Tensor  # (N, K) reward sequences
    target_policies: Tensor  # (N, K+1, num_actions) MCTS policy targets
    target_values: Tensor  # (N, K+1) MCTS value targets
    next_obs: Tensor  # (N, K, C, H, W) next observations for grounding
    dones: Tensor  # (N, K+1) done flags
    indices: Tensor | None = None
    weights: Tensor | None = None


@dataclass
class MuZeroReplayBuffer:
    """Replay buffer for MuZero that stores trajectory segments.

    MuZero requires storing:
    - Initial observation
    - K consecutive actions
    - K rewards
    - K+1 MCTS policy targets (visit count distributions)
    - K+1 MCTS value estimates (root values from MCTS)
    - K+1 done flags
    - K next observations (for latent grounding losses)

    Unlike standard replay buffers that store (s, a, r, s', done) tuples,
    this stores full trajectory segments for K-step unrolling during training.
    """

    capacity: int
    obs_shape: tuple[int, ...]
    num_actions: int
    unroll_steps: int  # K
    gamma: float = 0.997
    td_steps: int = 10  # n-step returns for value targets
    alpha: float = 0.0  # PER alpha (0 = uniform)
    epsilon: float = 1e-6

    # Storage arrays
    _obs: np.ndarray = field(init=False, repr=False)
    _actions: np.ndarray = field(init=False, repr=False)
    _rewards: np.ndarray = field(init=False, repr=False)
    _policies: np.ndarray = field(init=False, repr=False)
    _values: np.ndarray = field(init=False, repr=False)
    _next_obs: np.ndarray = field(init=False, repr=False)
    _dones: np.ndarray = field(init=False, repr=False)

    # PER support
    _tree: SumTree | None = field(init=False, repr=False, default=None)
    _max_priority: float = field(init=False, default=1.0)

    # Tracking
    _size: int = field(init=False, default=0)
    _ptr: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Initialize storage arrays."""
        K = self.unroll_steps

        # Preallocate arrays
        self._obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self._actions = np.zeros((self.capacity, K), dtype=np.int64)
        self._rewards = np.zeros((self.capacity, K), dtype=np.float32)
        self._policies = np.zeros((self.capacity, K + 1, self.num_actions), dtype=np.float32)
        self._values = np.zeros((self.capacity, K + 1), dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, K, *self.obs_shape), dtype=np.float32)
        self._dones = np.zeros((self.capacity, K + 1), dtype=np.float32)

        # PER tree (only if alpha > 0)
        if self.alpha > 0:
            self._tree = SumTree(capacity=self.capacity)

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add a trajectory segment to the buffer.

        Args:
            obs: Initial observation (C, H, W)
            actions: K actions taken (K,)
            rewards: K rewards received (K,)
            policies: K+1 MCTS policy targets (K+1, num_actions)
            values: K+1 MCTS value estimates (K+1,)
            next_obs: K next observations (K, C, H, W)
            dones: K+1 done flags (K+1,)
        """
        idx = self._ptr

        self._obs[idx] = obs
        self._actions[idx] = actions
        self._rewards[idx] = rewards
        self._policies[idx] = policies
        self._values[idx] = values
        self._next_obs[idx] = next_obs
        self._dones[idx] = dones

        # Update PER tree with max priority
        if self._tree is not None:
            priority = self._max_priority ** self.alpha
            self._tree.add(priority)

        # Update pointers
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self._size >= batch_size

    def sample(
        self,
        batch_size: int,
        device: str = "cpu",
    ) -> TrajectoryBatch:
        """Sample a batch of trajectory segments.

        Args:
            batch_size: Number of trajectories to sample
            device: Device for output tensors

        Returns:
            TrajectoryBatch with tensors ready for training
        """
        if not self.can_sample(batch_size):
            raise ValueError(f"Not enough samples: {self._size} < {batch_size}")

        if self._tree is not None and self.alpha > 0:
            return self._sample_per(batch_size, device)
        return self._sample_uniform(batch_size, device)

    def _sample_uniform(self, batch_size: int, device: str) -> TrajectoryBatch:
        """Sample uniformly from buffer."""
        indices = np.random.randint(0, self._size, size=batch_size)

        return TrajectoryBatch(
            obs=torch.from_numpy(self._obs[indices]).to(device),
            actions=torch.from_numpy(self._actions[indices]).to(device),
            rewards=torch.from_numpy(self._rewards[indices]).to(device),
            target_policies=torch.from_numpy(self._policies[indices]).to(device),
            target_values=torch.from_numpy(self._values[indices]).to(device),
            next_obs=torch.from_numpy(self._next_obs[indices]).to(device),
            dones=torch.from_numpy(self._dones[indices]).to(device),
            indices=torch.from_numpy(indices).to(device),
            weights=torch.ones(batch_size, device=device),
        )

    def _sample_per(self, batch_size: int, device: str) -> TrajectoryBatch:
        """Sample with prioritized experience replay."""
        assert self._tree is not None

        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        data_indices = np.zeros(batch_size, dtype=np.int64)

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
        weights = (self._size * probabilities) ** (-0.4)  # beta = 0.4
        weights = weights / weights.max()

        return TrajectoryBatch(
            obs=torch.from_numpy(self._obs[data_indices]).to(device),
            actions=torch.from_numpy(self._actions[data_indices]).to(device),
            rewards=torch.from_numpy(self._rewards[data_indices]).to(device),
            target_policies=torch.from_numpy(self._policies[data_indices]).to(device),
            target_values=torch.from_numpy(self._values[data_indices]).to(device),
            next_obs=torch.from_numpy(self._next_obs[data_indices]).to(device),
            dones=torch.from_numpy(self._dones[data_indices]).to(device),
            indices=torch.from_numpy(indices).to(device),
            weights=torch.from_numpy(weights.astype(np.float32)).to(device),
        )

    def update_priorities(self, indices: np.ndarray | Tensor, td_errors: np.ndarray | Tensor) -> None:
        """Update priorities based on TD errors."""
        if self._tree is None:
            return

        if isinstance(indices, Tensor):
            indices = indices.cpu().numpy()
        if isinstance(td_errors, Tensor):
            td_errors = td_errors.cpu().numpy()

        for leaf_idx, td_error in zip(indices, td_errors, strict=False):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self._tree.update(int(leaf_idx), priority)
            self._max_priority = max(self._max_priority, abs(td_error) + self.epsilon)

    def reset(self) -> None:
        """Clear the buffer."""
        self._size = 0
        self._ptr = 0
        self._max_priority = 1.0

        if self._tree is not None:
            self._tree = SumTree(capacity=self.capacity)

    def __len__(self) -> int:
        return self._size
