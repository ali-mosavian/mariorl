"""Unified Replay Buffer with N-step returns and optional PER.

Combines:
- Circular buffer storage
- N-step return computation
- Prioritized Experience Replay (optional, alpha=0 disables)
- Batch sampling with tensor conversion
- LazyFrames for memory-efficient compressed state storage
"""

from dataclasses import field
from dataclasses import dataclass

import torch
import numpy as np
from torch import Tensor

from mario_rl.core.types import Transition
from mario_rl.core.types import TreeNode
from mario_rl.environment.frame_stack import LazyFrames


# =============================================================================
# N-Step Buffer (inlined from buffers/nstep.py)
# =============================================================================


def _copy_state(state):
    """Copy state if numpy array, return as-is if LazyFrames (immutable)."""
    if isinstance(state, LazyFrames):
        return state  # LazyFrames is frozen/immutable, no copy needed
    return state.copy() if hasattr(state, 'copy') else state


@dataclass
class NStepBuffer:
    """Buffer for computing N-step returns.
    
    Single responsibility: Accumulate transitions and compute discounted N-step returns.
    """

    n_step: int
    gamma: float
    _buffer: list[Transition] = field(init=False, default_factory=list)

    def add(self, transition: Transition) -> Transition | None:
        """Add transition and return N-step transition if buffer is full."""
        # Store a copy to prevent mutation
        self._buffer.append(self._copy_transition(transition))

        if len(self._buffer) < self.n_step:
            return None

        return self._pop_nstep_transition()

    def flush(self) -> list[Transition]:
        """Flush remaining transitions at episode end."""
        transitions: list[Transition] = []
        while self._buffer:
            transitions.append(self._pop_nstep_transition())
        return transitions

    def reset(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
    
    def _pop_nstep_transition(self) -> Transition:
        """Compute and pop one N-step transition from front of buffer."""
        # Compute discounted reward, stopping at done
        n_step_reward = 0.0
        last_idx = len(self._buffer) - 1
        
        for i, t in enumerate(self._buffer):
            n_step_reward += (self.gamma ** i) * t.reward
            if t.done:
                last_idx = i
                break
        
        first = self._buffer[0]
        last = self._buffer[last_idx]
        
        result = Transition(
            state=first.state,
            action=first.action,
            reward=n_step_reward,
            next_state=last.next_state,
            done=last.done,
            flag_get=first.flag_get,
            max_x=first.max_x,
            action_history=first.action_history,
            next_action_history=last.next_action_history,
            danger_target=first.danger_target,
            level_id=first.level_id,
            x_pos=first.x_pos,
        )
        self._buffer.pop(0)
        return result
    
    @staticmethod
    def _copy_transition(t: Transition) -> Transition:
        """Create a copy of transition with copied arrays."""
        return Transition(
            state=_copy_state(t.state),
            action=t.action,
            reward=t.reward,
            next_state=_copy_state(t.next_state),
            done=t.done,
            flag_get=t.flag_get,
            max_x=t.max_x,
            action_history=t.action_history.copy() if t.action_history is not None else None,
            next_action_history=t.next_action_history.copy() if t.next_action_history is not None else None,
            danger_target=t.danger_target.copy() if t.danger_target is not None else None,
            level_id=t.level_id,
            x_pos=t.x_pos,
        )


# =============================================================================
# Episode Collector (context manager for episode-scoped transitions)
# =============================================================================


class EpisodeCollector:
    """Context manager for episode-scoped transition collection.
    
    Single responsibility: Handle one episode's worth of transitions.
    - N-step return computation
    - Episode history tracking  
    - Flag history capture on episode end
    
    Single-use: After context exits, this collector is finished.
    Create a new one via buffer.episode() for the next episode.
    
    Usage:
        with buffer.episode() as ep:
            for step in range(max_steps):
                ep.add(transition)
                if done:
                    break
        # Automatically finalizes episode on exit
    """
    
    __slots__ = (
        '_buffer', '_flag_history_len', '_episode_history',
        '_nstep_buffer', '_flag_captured', '_active'
    )
    
    def __init__(
        self, 
        buffer: "ReplayBuffer", 
        n_step: int = 1,
        gamma: float = 0.99,
        flag_history_len: int = 10,
    ):
        self._buffer = buffer
        self._flag_history_len = flag_history_len
        self._episode_history: list[Transition] = []
        self._nstep_buffer = NStepBuffer(n_step=n_step, gamma=gamma) if n_step > 1 else None
        self._flag_captured = False
        self._active = True  # False after context exits
    
    def __enter__(self) -> "EpisodeCollector":
        if not self._active:
            raise RuntimeError("EpisodeCollector is single-use. Create a new one via buffer.episode()")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._finalize()
        self._active = False  # Mark as done - cannot be reused
    
    def add(self, transition: Transition) -> None:
        """Add a transition (will be N-step processed before storage)."""
        if not self._active:
            raise RuntimeError("EpisodeCollector is finished. Create a new one via buffer.episode()")
        
        if transition.flag_get:
            self._flag_captured = True
        
        # N-step processing
        if self._nstep_buffer is not None:
            processed = self._nstep_buffer.add(transition)
            if processed is not None:
                self._store(processed)
            # Flush remaining on episode end
            if transition.done:
                for t in self._nstep_buffer.flush():
                    self._store(t)
        else:
            self._store(transition)
    
    def _store(self, transition: Transition) -> None:
        """Store processed transition to buffer and history."""
        self._episode_history.append(transition)
        self._buffer._store(transition)
    
    def _finalize(self) -> None:
        """Finalize episode - store flag history if captured."""
        if self._flag_captured and self._buffer.positive_reward_ratio > 0:
            # Store last N transitions to positive buffer
            start = max(0, len(self._episode_history) - self._flag_history_len)
            for t in self._episode_history[start:]:
                self._buffer._positive.store(t)
        self._episode_history.clear()
    
    def end(self) -> None:
        """Finalize episode for training loops that span multiple episodes.
        
        After calling end(), this collector cannot be reused - create a new one.
        
        Training loop pattern:
            ep = buffer.episode()
            for step in range(num_steps):
                ep.add(transition)
                if done:
                    ep.end()
                    ep = buffer.episode()  # New collector for next episode
        """
        if not self._active:
            return  # Already ended
        self._finalize()
        self._active = False
    
    @property
    def history(self) -> list[Transition]:
        """Current episode history (N-step processed transitions)."""
        return self._episode_history.copy()


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
# Experience Buffer (base storage for all buffer types)
# =============================================================================


@dataclass
class ExperienceBuffer:
    """Circular buffer for storing transitions with LazyFrames compression.
    
    Stores LazyFrames directly (already LZ4 compressed by env) for memory efficiency.
    Only decompresses to float32 numpy arrays on sampling.
    
    Used as the base storage for:
    - Main replay buffer
    - Protected buffers (death, flag, difficult success)
    """
    
    capacity: int
    obs_shape: tuple[int, ...]
    action_history_shape: tuple[int, ...] | None = None
    danger_target_bins: int = 0
    
    # State storage (LazyFrames from env, already compressed)
    _states: list[LazyFrames | None] = field(init=False, repr=False)
    _next_states: list[LazyFrames | None] = field(init=False, repr=False)
    
    # Regular storage arrays (small, no compression needed)
    _actions: np.ndarray = field(init=False, repr=False)
    _rewards: np.ndarray = field(init=False, repr=False)
    _dones: np.ndarray = field(init=False, repr=False)
    _action_histories: np.ndarray | None = field(init=False, repr=False, default=None)
    _next_action_histories: np.ndarray | None = field(init=False, repr=False, default=None)
    _danger_targets: np.ndarray | None = field(init=False, repr=False, default=None)
    
    # Tracking
    _size: int = field(init=False, default=0)
    _ptr: int = field(init=False, default=0)
    
    def __post_init__(self) -> None:
        """Initialize storage arrays."""
        # State storage - preallocate list with None
        self._states = [None] * self.capacity
        self._next_states = [None] * self.capacity
        
        # Regular arrays for small data
        self._actions = np.zeros(self.capacity, dtype=np.int64)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._dones = np.zeros(self.capacity, dtype=np.float32)
        self._size = 0
        self._ptr = 0
        
        if self.action_history_shape is not None:
            self._action_histories = np.zeros((self.capacity, *self.action_history_shape), dtype=np.float32)
            self._next_action_histories = np.zeros((self.capacity, *self.action_history_shape), dtype=np.float32)
        
        if self.danger_target_bins > 0:
            self._danger_targets = np.zeros((self.capacity, self.danger_target_bins), dtype=np.float32)
    
    @property
    def size(self) -> int:
        """Current number of stored transitions."""
        return self._size
    
    def _to_numpy(self, state: LazyFrames | np.ndarray) -> np.ndarray:
        """Convert state to numpy array. No normalization - network handles that."""
        if isinstance(state, LazyFrames):
            # LazyFrames.__array__() decompresses and stacks
            return np.array(state)
        return state
    
    def store(self, transition: Transition) -> int:
        """Store a transition. Returns the index where it was stored.
        
        States should be LazyFrames (from env) for memory efficiency.
        """
        idx = self._ptr
        
        # Store states directly (LazyFrames are already compressed)
        self._states[idx] = transition.state
        self._next_states[idx] = transition.next_state
        
        # Store regular data
        self._actions[idx] = transition.action
        self._rewards[idx] = transition.reward
        self._dones[idx] = float(transition.done)
        
        if self._action_histories is not None and transition.action_history is not None:
            self._action_histories[idx] = transition.action_history
        if self._next_action_histories is not None and transition.next_action_history is not None:
            self._next_action_histories[idx] = transition.next_action_history
        if self._danger_targets is not None and transition.danger_target is not None:
            self._danger_targets[idx] = transition.danger_target
        
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        return idx
    
    def sample(self, batch_size: int) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray | None, np.ndarray | None, np.ndarray | None
    ]:
        """Sample random transitions.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones,
                     action_histories, next_action_histories, danger_targets)
        """
        if self._size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        indices = np.random.randint(0, self._size, size=batch_size)
        return self.get_by_indices(indices)
    
    def get_by_indices(self, indices: np.ndarray) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray | None, np.ndarray | None, np.ndarray | None
    ]:
        """Get transitions by indices, converting states to numpy."""
        # Convert LazyFrames to numpy (decompresses on access)
        states = np.stack([self._to_numpy(self._states[i]) for i in indices])
        next_states = np.stack([self._to_numpy(self._next_states[i]) for i in indices])
        
        action_histories = self._action_histories[indices] if self._action_histories is not None else None
        next_action_histories = self._next_action_histories[indices] if self._next_action_histories is not None else None
        danger_targets = self._danger_targets[indices] if self._danger_targets is not None else None
        
        return (
            states,
            self._actions[indices],
            self._rewards[indices],
            next_states,
            self._dones[indices],
            action_histories,
            next_action_histories,
            danger_targets,
        )
    
    def reset(self) -> None:
        """Clear the buffer."""
        self._size = 0
        self._ptr = 0
        # Clear state storage to free memory
        self._states = [None] * self.capacity
        self._next_states = [None] * self.capacity


# =============================================================================
# Replay Buffer
# =============================================================================


class _SampleCollector:
    """Accumulates samples from multiple buffers for batch construction.
    
    Internal helper - not part of public API.
    """
    __slots__ = (
        'states', 'actions', 'rewards', 'next_states', 'dones',
        'action_histories', 'next_action_histories', 'danger_targets', 'weights'
    )
    
    def __init__(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[np.ndarray] = []
        self.next_states: list[np.ndarray] = []
        self.dones: list[np.ndarray] = []
        self.action_histories: list[np.ndarray] = []
        self.next_action_histories: list[np.ndarray] = []
        self.danger_targets: list[np.ndarray] = []
        self.weights: list[np.ndarray] = []
    
    def add(
        self,
        s: np.ndarray, a: np.ndarray, r: np.ndarray,
        ns: np.ndarray, d: np.ndarray,
        ah: np.ndarray | None, nah: np.ndarray | None, dt: np.ndarray | None,
        weight: float | np.ndarray, count: int,
    ) -> None:
        """Add sampled data to collector."""
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.next_states.append(ns)
        self.dones.append(d)
        
        # Convert scalar weight to array
        if isinstance(weight, (int, float)):
            self.weights.append(np.full(count, weight, dtype=np.float32))
        else:
            self.weights.append(weight)
        
        if ah is not None:
            self.action_histories.append(ah)
            self.next_action_histories.append(nah)
        if dt is not None:
            self.danger_targets.append(dt)


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
    action_histories: Tensor | None = None
    next_action_histories: Tensor | None = None
    danger_targets: Tensor | None = None


@dataclass
class ReplayBuffer:
    """Unified replay buffer with N-step returns and optional PER.

    Features:
    - N-step return computation (n_step >= 1)
    - Prioritized sampling (alpha > 0) or uniform (alpha = 0)
    - Tensor output for training
    - Device placement support
    - Flag capture priority boost (asymmetric priority)
    - Optional action history storage
    - Four-way sampling with specialized buffers:
      * Main buffer: Normal circular eviction with PER
      * Negative buffer (death): Protected
      * Positive buffer (flag): Protected
      * Difficult buffer: Protected
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
    death_priority_multiplier: float = 10.0  # Priority boost for death transitions
    action_history_shape: tuple[int, ...] | None = None  # (history_len, num_actions) if used
    danger_target_bins: int = 16  # Number of bins for auxiliary danger prediction
    negative_reward_ratio: float = 0.25  # Ratio from negative buffer (death)
    positive_reward_ratio: float = 0.25  # Ratio from positive buffer (flag)
    difficult_success_ratio: float = 0.25  # Ratio from difficult buffer
    protected_capacity: int = 25000  # Capacity for each protected buffer

    # All four buffers use the same ExperienceBuffer class
    _main: ExperienceBuffer = field(init=False, repr=False)
    _negative: ExperienceBuffer = field(init=False, repr=False)
    _positive: ExperienceBuffer = field(init=False, repr=False)
    _difficult: ExperienceBuffer = field(init=False, repr=False)

    # For PER (main buffer only)
    _tree: SumTree | None = field(init=False, repr=False, default=None)
    _max_priority: float = field(init=False, default=1.0)
    _current_beta: float = field(init=False, default=0.4)

    # Metadata for main buffer (priority boost and difficulty tracking)
    _flag_gets: np.ndarray = field(init=False, repr=False)
    _level_ids: list = field(init=False, repr=False)
    _x_positions: np.ndarray = field(init=False, repr=False)
    
    # Flag history length for EpisodeCollector
    flag_history_len: int = 10  # Number of transitions to capture before flag
    
    # Difficulty ranges per level: {level_id: [(start, end), ...]}
    _difficulty_ranges: dict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize all four experience buffers."""
        # Common buffer config
        buf_config = {
            "obs_shape": self.obs_shape,
            "action_history_shape": self.action_history_shape,
            "danger_target_bins": self.danger_target_bins,
        }
        
        # Create all four buffers
        self._main = ExperienceBuffer(capacity=self.capacity, **buf_config)
        self._negative = ExperienceBuffer(capacity=self.protected_capacity, **buf_config)
        self._positive = ExperienceBuffer(capacity=self.protected_capacity, **buf_config)
        self._difficult = ExperienceBuffer(capacity=self.protected_capacity, **buf_config)

        # Metadata for main buffer (used for PER priority boosting)
        self._flag_gets = np.zeros(self.capacity, dtype=np.bool_)
        self._level_ids = [None] * self.capacity
        self._x_positions = np.zeros(self.capacity, dtype=np.int32)

        # PER tracking
        self._max_priority = 1.0
        self._current_beta = self.beta_start
        self._tree = SumTree(capacity=self.capacity) if self.alpha > 0 else None

        # Difficulty ranges for detecting difficult areas
        self._difficulty_ranges = {}

    def episode(self) -> "EpisodeCollector":
        """Create an episode collector - the only way to add transitions.
        
        MUST be used as a context manager:
            with buffer.episode() as ep:
                ep.add(transition)
                ep.add(transition)
            # Automatically handles N-step, flag history on exit
        """
        return EpisodeCollector(
            self, 
            n_step=self.n_step,
            gamma=self.gamma,
            flag_history_len=self.flag_history_len,
        )

    def _store(self, transition: Transition) -> None:
        """Store a transition in main buffer and optionally in protected buffers."""
        # Store in main buffer (returns the index used)
        idx = self._main.store(transition)
        
        # Store metadata for priority boosting
        self._flag_gets[idx] = transition.flag_get
        self._level_ids[idx] = transition.level_id
        self._x_positions[idx] = transition.x_pos

        # Also store in protected negative buffer (death transitions)
        if transition.reward < 0 and self.negative_reward_ratio > 0:
            self._negative.store(transition)

        # Note: Flag history is tracked by EpisodeCollector, not here

        # Also store in protected difficult success buffer
        if (
            not transition.done
            and transition.level_id is not None
            and self.difficult_success_ratio > 0
            and self._is_in_difficult_range(transition.level_id, transition.x_pos)
        ):
            self._difficult.store(transition)

        # Update PER tree with max priority (apply priority boosts)
        if self._tree is not None:
            base_priority = self._max_priority ** self.alpha
            priority = base_priority
            
            # Apply asymmetric priority boosts
            if transition.flag_get and self.flag_priority_multiplier > 1.0:
                priority = priority * self.flag_priority_multiplier
            
            # Death transitions get priority boost to ensure learning from rare deaths
            if transition.done and self.death_priority_multiplier > 1.0:
                priority = priority * self.death_priority_multiplier
            
            self._tree.add(priority)

    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self._main.size >= batch_size

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
                f"Not enough samples: {self._main.size} < {batch_size}"
            )

        if self._tree is not None and self.alpha > 0:
            return self._sample_per(batch_size, device, beta)
        return self._sample_uniform(batch_size, device)

    def _compute_batch_sizes(self, batch_size: int) -> tuple[int, int, int, int]:
        """Compute batch sizes for each buffer based on availability and ratios."""
        neg = int(batch_size * self.negative_reward_ratio) if self._negative.size > 0 else 0
        pos = int(batch_size * self.positive_reward_ratio) if self._positive.size > 0 else 0
        diff = int(batch_size * self.difficult_success_ratio) if self._difficult.size > 0 else 0
        main = batch_size - neg - pos - diff
        return neg, pos, diff, main

    def _collect_samples(
        self,
        buffer: ExperienceBuffer,
        size: int,
        collector: "_SampleCollector",
        weight: float | np.ndarray = 1.0,
    ) -> None:
        """Sample from a buffer and add to collector."""
        if size <= 0:
            return
        s, a, r, ns, d, ah, nah, dt = buffer.sample(size)
        collector.add(s, a, r, ns, d, ah, nah, dt, weight, size)

    def _build_batch(
        self, collector: "_SampleCollector", batch_size: int, device: str,
        indices: np.ndarray | None = None,
    ) -> Batch:
        """Build final Batch from collected samples."""
        # Concatenate or use empty defaults
        states = np.concatenate(collector.states) if collector.states else np.zeros((batch_size, *self.obs_shape), dtype=np.uint8)
        actions = np.concatenate(collector.actions) if collector.actions else np.zeros(batch_size, dtype=np.int64)
        rewards = np.concatenate(collector.rewards) if collector.rewards else np.zeros(batch_size, dtype=np.float32)
        next_states = np.concatenate(collector.next_states) if collector.next_states else np.zeros((batch_size, *self.obs_shape), dtype=np.uint8)
        dones = np.concatenate(collector.dones) if collector.dones else np.zeros(batch_size, dtype=np.float32)
        weights = np.concatenate(collector.weights) if collector.weights else np.ones(batch_size, dtype=np.float32)

        # Shuffle to mix samples from different sources
        shuffle_idx = np.random.permutation(batch_size)
        
        # Apply shuffle to main arrays
        states, actions = states[shuffle_idx], actions[shuffle_idx]
        rewards, next_states = rewards[shuffle_idx], next_states[shuffle_idx]
        dones, weights = dones[shuffle_idx], weights[shuffle_idx]
        
        # Handle optional arrays
        action_histories = next_action_histories = danger_targets = None
        if collector.action_histories:
            action_histories = torch.from_numpy(np.concatenate(collector.action_histories)[shuffle_idx]).to(device)
            next_action_histories = torch.from_numpy(np.concatenate(collector.next_action_histories)[shuffle_idx]).to(device)
        if collector.danger_targets:
            danger_targets = torch.from_numpy(np.concatenate(collector.danger_targets)[shuffle_idx]).to(device)

        # Handle indices (for PER priority updates)
        batch_indices = None
        if indices is not None:
            batch_indices = torch.from_numpy(indices[shuffle_idx]).to(device)

        return Batch(
            states=torch.from_numpy(states).to(device),
            actions=torch.from_numpy(actions).to(device),
            rewards=torch.from_numpy(rewards).to(device),
            next_states=torch.from_numpy(next_states).to(device),
            dones=torch.from_numpy(dones).to(device),
            indices=batch_indices,
            weights=torch.from_numpy(weights).to(device),
            action_histories=action_histories,
            next_action_histories=next_action_histories,
            danger_targets=danger_targets,
        )

    def _sample_uniform(self, batch_size: int, device: str) -> Batch:
        """Sample uniformly with four-way split from protected buffers."""
        neg_size, pos_size, diff_size, main_size = self._compute_batch_sizes(batch_size)
        
        collector = _SampleCollector()
        self._collect_samples(self._negative, neg_size, collector)
        self._collect_samples(self._positive, pos_size, collector)
        self._collect_samples(self._difficult, diff_size, collector)
        self._collect_samples(self._main, main_size, collector)
        
        return self._build_batch(collector, batch_size, device)

    def _sample_per(self, batch_size: int, device: str, beta: float | None) -> Batch:
        """Sample with PER from main buffer, uniform from protected buffers."""
        if beta is None:
            beta = self._current_beta
        assert self._tree is not None

        neg_size, pos_size, diff_size, per_size = self._compute_batch_sizes(batch_size)
        protected_count = neg_size + pos_size + diff_size
        
        collector = _SampleCollector()
        
        # Sample protected buffers (weight=1)
        self._collect_samples(self._negative, neg_size, collector)
        self._collect_samples(self._positive, pos_size, collector)
        self._collect_samples(self._difficult, diff_size, collector)
        
        # PER sampling from main buffer
        per_tree_indices = np.zeros(per_size, dtype=np.int64)
        if per_size > 0:
            per_priorities = np.zeros(per_size, dtype=np.float64)
            per_data_indices = np.zeros(per_size, dtype=np.int64)
            
            total = self._tree.total
            segment = total / per_size
            
            for i in range(per_size):
                value = np.random.uniform(segment * i, segment * (i + 1))
                node = self._tree.get(value)
                per_tree_indices[i] = node.leaf_idx
                per_priorities[i] = node.priority
                per_data_indices[i] = node.data_idx
            
            # Get samples by indices
            s, a, r, ns, d, ah, nah, dt = self._main.get_by_indices(per_data_indices)
            
            # Compute importance weights
            probs = per_priorities / total
            per_weights = (self._main.size * probs) ** (-beta)
            per_weights = (per_weights / per_weights.max()).astype(np.float32)
            
            collector.add(s, a, r, ns, d, ah, nah, dt, per_weights, per_size)
        
        # Build indices: -1 for protected, tree indices for PER
        indices = np.concatenate([
            -np.ones(protected_count, dtype=np.int64),
            per_tree_indices
        ])
        
        return self._build_batch(collector, batch_size, device, indices)
    
    @property
    def negative_reward_count(self) -> int:
        """Number of transitions in protected negative buffer (death)."""
        return self._negative.size

    @property
    def positive_reward_count(self) -> int:
        """Number of transitions in protected positive buffer (flag)."""
        return self._positive.size

    @property
    def difficult_success_count(self) -> int:
        """Number of transitions in protected difficult buffer."""
        return self._difficult.size

    @property
    def buffer_stats(self) -> dict[str, int]:
        """Statistics about all buffer sizes."""
        return {
            "main_size": self._main.size,
            "main_capacity": self.capacity,
            "negative_size": self._negative.size,
            "negative_capacity": self.protected_capacity,
            "positive_size": self._positive.size,
            "positive_capacity": self.protected_capacity,
            "difficult_size": self._difficult.size,
            "difficult_capacity": self.protected_capacity,
        }

    def set_difficulty_ranges(self, ranges: dict[str, list[tuple[int, int]]]) -> None:
        """Set difficulty ranges for tracking difficult success transitions.
        
        Args:
            ranges: Dict mapping level_id -> list of (start_x, end_x) tuples.
                   Transitions within these ranges (that don't die) are tracked.
        """
        self._difficulty_ranges = ranges

    def _is_in_difficult_range(self, level_id: str, x_pos: int) -> bool:
        """Check if position is within a known difficult range.
        
        Args:
            level_id: Level identifier
            x_pos: X position to check
            
        Returns:
            True if position is in a difficult range for this level.
        """
        ranges = self._difficulty_ranges.get(level_id, [])
        for start, end in ranges:
            if start <= x_pos <= end:
                return True
        return False

    def update_priorities(self, indices: np.ndarray | Tensor, td_errors: np.ndarray | Tensor) -> None:
        """Update priorities based on TD errors.

        Maintains asymmetric priority: flag captures keep their boost multiplier.
        Skips indices that are -1 (from negative reward buffer sampling).
        """
        if self._tree is None:
            return  # Uniform sampling, no priorities

        if isinstance(indices, Tensor):
            indices = indices.cpu().numpy()
        if isinstance(td_errors, Tensor):
            td_errors = td_errors.cpu().numpy()

        for leaf_idx, td_error in zip(indices, td_errors, strict=False):
            # Skip negative samples (they use index -1 and bypass PER)
            if leaf_idx < 0:
                continue
                
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
        """Clear all buffers (main and protected)."""
        self._main.reset()
        self._negative.reset()
        self._positive.reset()
        self._difficult.reset()
        
        self._max_priority = 1.0
        self._flag_gets.fill(False)
        self._level_ids = [None] * self.capacity
        self._x_positions.fill(0)

        if self._tree is not None:
            self._tree = SumTree(capacity=self.capacity)

    def __len__(self) -> int:
        return self._main.size


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
