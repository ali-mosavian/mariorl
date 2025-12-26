"""
Distributed DDQN Worker with local gradient computation.

Workers collect experiences in a local replay buffer, compute DQN loss locally,
and send only gradients to the learner. This is similar to Gorila DQN.

Features
========
- Prioritized Experience Replay (PER) with SumTree
- N-step returns for better credit assignment
- Double DQN for reduced overestimation
- Importance sampling weights for unbiased gradients

Architecture
============

                    ┌─────────────────────────┐
                    │    LEARNER (GPU/MPS)    │
                    │  ┌───────────────────┐  │
                    │  │  Global Network   │  │
                    │  │    (weights θ)    │  │
                    │  └───────────────────┘  │
                    │           │             │
                    │  ┌────────▼────────┐    │
                    │  │    Optimizer    │    │
                    │  │ (gradient step) │    │
                    │  └────────┬────────┘    │
                    │           │             │
                    │  ┌────────▼────────┐    │
                    │  │   weights.pt    │    │
                    └──┴────────┬────────┴────┘
                                │
           ┌────────────────────┼────────────────────┐
           │                    │                    │
           ▼                    ▼                    ▼
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │  WORKER 0   │      │  WORKER 1   │      │  WORKER N   │
    │ ┌─────────┐ │      │ ┌─────────┐ │      │ ┌─────────┐ │
    │ │  Local  │ │      │ │  Local  │ │      │ │  Local  │ │
    │ │ Network │ │      │ │ Network │ │      │ │ Network │ │
    │ │ + PER   │ │      │ │ + PER   │ │      │ │ + PER   │ │
    │ └────┬────┘ │      │ └────┬────┘ │      │ └────┬────┘ │
    │      │      │      │      │      │      │      │      │
    │ 1.Collect   │      │ 1.Collect   │      │ 1.Collect   │
    │   N steps   │      │   N steps   │      │   N steps   │
    │      │      │      │      │      │      │      │      │
    │ 2.PER sample│      │ 2.PER sample│      │ 2.PER sample│
    │   (priority)│      │   (priority)│      │   (priority)│
    │      │      │      │      │      │      │      │      │
    │ 3.Compute   │      │ 3.Compute   │      │ 3.Compute   │
    │   DQN loss  │      │   DQN loss  │      │   DQN loss  │
    │      │      │      │      │      │      │      │      │
    │ 4.Update    │      │ 4.Update    │      │ 4.Update    │
    │  priorities │      │  priorities │      │  priorities │
    │      │      │      │      │      │      │      │      │
    │ 5.Backward  │      │ 5.Backward  │      │ 5.Backward  │
    │   (grads)   │      │   (grads)   │      │   (grads)   │
    │      │      │      │      │      │      │      │      │
    └──────┼──────┘      └──────┼──────┘      └──────┼──────┘
           │                    │                    │
           │    GRADIENT QUEUE (small ~2MB each)     │
           └────────────────────┼────────────────────┘
                                │
                                ▼
                          ┌──────────┐
                          │ LEARNER  │
                          │ applies  │
                          │ gradients│
                          └──────────┘

Data Flow
=========

Standard DQN:   Worker → [experiences ~8MB] → Learner (compute loss + backward)
Distributed:    Worker (compute loss + backward) → [gradients ~2MB] → Learner

Benefits:
- ~4x less data through IPC
- Distributed computation across worker CPUs/GPUs
- Reduced learner bottleneck
- Each worker maintains diverse local buffer with PER
"""

import os
import time
import multiprocessing as mp

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from typing import Any
from typing import Dict
from typing import List
from pathlib import Path
from typing import Tuple
from typing import Literal
from typing import Optional
from dataclasses import field
from dataclasses import dataclass

import torch
import numpy as np
from torch import nn
from gymnasium.spaces import Box
from nes_py.wrappers import JoypadSpace
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import TransformObservation
from gymnasium.wrappers import FrameStackObservation
from gym_super_mario_bros import actions as smb_actions

from mario_rl.agent.ddqn_net import DoubleDQN
from mario_rl.environment.wrappers import SkipFrame
from mario_rl.environment.wrappers import ResizeObservation
from mario_rl.environment.mariogym import SuperMarioBrosMultiLevel

LevelType = Literal["sequential", "random"] | tuple[Literal[1, 2, 3, 4, 5, 6, 7, 8], Literal[1, 2, 3, 4]]


def create_env(level: LevelType = (1, 1), render_frames: bool = False):
    """Create wrapped Mario environment."""
    if render_frames:
        try:
            from pyglet.window import key
            import nes_py._image_viewer as _iv

            _iv.key = key
        except Exception:
            pass

    base_env = SuperMarioBrosMultiLevel(level=level)
    env = JoypadSpace(base_env, actions=smb_actions.COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4, render_frames=render_frames)
    env = GrayscaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=64)
    env = TransformObservation(
        env,
        func=lambda x: x / 255.0,
        observation_space=Box(low=0.0, high=1.0, shape=(64, 64, 1), dtype=np.float32),
    )
    env = FrameStackObservation(env, stack_size=4)
    return env, base_env


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


@dataclass
class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer using SumTree.

    Samples experiences proportional to their TD-error priority.
    Uses importance sampling weights to correct for bias.

    Reference: Schaul et al. "Prioritized Experience Replay" (2015)
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


@dataclass
class DDQNWorker:
    """
    DDQN Worker that computes gradients locally and sends them to the learner.

    Each worker:
    1. Collects experiences using epsilon-greedy
    2. Stores in local PER buffer
    3. Samples batch with priority weighting
    4. Computes Double DQN loss with importance sampling
    5. Updates priorities based on TD errors
    6. Sends gradients to learner
    7. Periodically syncs weights from learner
    """

    # Required fields
    worker_id: int
    weights_path: Path
    gradient_queue: mp.Queue

    # Configuration
    level: LevelType = (1, 1)
    n_step: int = 3  # N-step returns
    gamma: float = 0.99
    render_frames: bool = False
    weight_sync_interval: float = 5.0  # Seconds between weight syncs

    # Local buffer settings (PER)
    local_buffer_size: int = 10_000
    batch_size: int = 32
    steps_per_collection: int = 64
    train_steps: int = 4

    # PER hyperparameters
    per_alpha: float = 0.6  # Priority exponent
    per_beta_start: float = 0.4  # Initial IS exponent
    per_beta_end: float = 1.0  # Final IS exponent

    # Exploration (different per worker)
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 100_000

    # Gradient clipping
    max_grad_norm: float = 10.0

    # Device
    device: Optional[str] = None

    # UI
    ui_queue: Optional[mp.Queue] = None

    # Private fields
    env: Any = field(init=False, repr=False)
    base_env: Any = field(init=False, repr=False)
    net: Any = field(init=False, repr=False)
    buffer: PrioritizedReplayBuffer = field(init=False, repr=False)
    n_step_buffer: NStepBuffer = field(init=False, repr=False)
    action_dim: int = field(init=False)

    # Tracking
    episode_count: int = field(init=False, default=0)
    total_steps: int = field(init=False, default=0)
    episode_reward: float = field(init=False, default=0.0)
    episode_length: int = field(init=False, default=0)
    best_x: int = field(init=False, default=0)
    best_x_ever: int = field(init=False, default=0)
    flags: int = field(init=False, default=0)
    deaths: int = field(init=False, default=0)
    reward_history: List[float] = field(init=False, default_factory=list)
    last_weight_sync: float = field(init=False, default=0.0)
    weight_version: int = field(init=False, default=0)
    weight_sync_count: int = field(init=False, default=0)
    gradients_sent: int = field(init=False, default=0)
    steps_per_sec: float = field(init=False, default=0.0)
    _last_time: float = field(init=False, default=0.0)
    current_epsilon: float = field(init=False, default=1.0)

    # Additional metrics
    x_at_death_history: List[int] = field(init=False, default_factory=list)
    time_to_flag_history: List[int] = field(init=False, default_factory=list)
    speed_history: List[float] = field(init=False, default_factory=list)
    episode_start_time: int = field(init=False, default=400)  # Mario starts with 400 time

    def __post_init__(self) -> None:
        """Initialize environment, network, and buffer."""
        # Auto-detect best device
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Create environment
        self.env, self.base_env = create_env(
            level=self.level,
            render_frames=self.render_frames,
        )
        self.action_dim = self.env.action_space.n

        # Create network
        state_dim = (4, 64, 64)
        self.net = DoubleDQN(
            input_shape=state_dim,
            num_actions=self.action_dim,
            feature_dim=512,
            hidden_dim=256,
            dropout=0.1,
        ).to(self.device)

        # Create local PER buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity=self.local_buffer_size,
            obs_shape=state_dim,
            alpha=self.per_alpha,
            beta_start=self.per_beta_start,
            beta_end=self.per_beta_end,
        )

        # Create N-step buffer
        self.n_step_buffer = NStepBuffer(n_step=self.n_step, gamma=self.gamma)

        # Compute n-step gamma for TD target
        self.n_step_gamma = self.gamma**self.n_step

        # Initialize tracking
        self.episode_count = 0
        self.total_steps = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.best_x = 0
        self.best_x_ever = 0
        self.flags = 0
        self.deaths = 0
        self.reward_history = []
        self.last_weight_sync = 0.0
        self.weight_version = 0
        self.weight_sync_count = 0
        self.gradients_sent = 0
        self._last_time = time.time()
        self.current_epsilon = self.eps_start

        # Load initial weights
        self._load_weights()

    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        """Convert state from (4, 64, 64, 1) to (4, 64, 64)."""
        if state.ndim == 4 and state.shape[-1] == 1:
            state = np.squeeze(state, axis=-1)
        return state

    def _load_weights(self) -> bool:
        """Load latest weights from disk with retry on failure."""
        if not self.weights_path.exists():
            return False

        # Retry a few times in case learner is writing
        for attempt in range(3):
            try:
                checkpoint = torch.load(
                    self.weights_path,
                    map_location=self.device,
                    weights_only=True,
                )
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self.net.load_state_dict(checkpoint["state_dict"])
                    self.weight_version = checkpoint.get("version", 0)
                else:
                    self.net.load_state_dict(checkpoint)
                # Sync target network with online
                self.net.sync_target()
                self.last_weight_sync = time.time()
                self.weight_sync_count += 1
                return True
            except Exception:
                if attempt < 2:
                    time.sleep(0.1)  # Brief delay before retry
                continue
        return False

    def _maybe_sync_weights(self) -> None:
        """Sync weights if enough time has passed."""
        if time.time() - self.last_weight_sync >= self.weight_sync_interval:
            self._load_weights()

    def _get_epsilon(self) -> float:
        """Get current epsilon based on decay schedule."""
        progress = min(1.0, self.total_steps / self.eps_decay_steps)
        return self.eps_start + (self.eps_end - self.eps_start) * progress

    @torch.no_grad()
    def _get_action(self, state: np.ndarray) -> int:
        """Get action using epsilon-greedy policy."""
        self.current_epsilon = self._get_epsilon()

        if np.random.random() < self.current_epsilon:
            return int(np.random.randint(0, self.action_dim))

        state = self._preprocess_state(state)
        state_tensor = torch.from_numpy(np.expand_dims(state, 0)).float().to(self.device)
        q_values = self.net.online(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def collect_steps(self, num_steps: int) -> int:
        """
        Collect experiences for num_steps and store in local PER buffer.

        Returns number of episodes completed.
        """
        state, _ = self.env.reset()
        episodes_completed = 0

        for _ in range(num_steps):
            # Get action
            action = self._get_action(state)

            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Preprocess states
            state_processed = self._preprocess_state(state)
            next_state_processed = self._preprocess_state(next_state)

            # Add to N-step buffer and get N-step transition
            n_step_transition = self.n_step_buffer.add(state_processed, action, reward, next_state_processed, done)
            if n_step_transition is not None:
                self.buffer.add(*n_step_transition)

            # Update tracking
            self.episode_reward += reward
            self.episode_length += 1
            self.total_steps += 1

            x_pos = info.get("x_pos", 0)
            if x_pos > self.best_x:
                self.best_x = x_pos
            if x_pos > self.best_x_ever:
                self.best_x_ever = x_pos

            if info.get("flag_get", False):
                self.flags += 1

            # Update PER beta based on progress
            progress = min(1.0, self.total_steps / self.eps_decay_steps)
            self.buffer.update_beta(progress)

            # Send UI status periodically
            if self.total_steps % 50 == 0:
                self._send_ui_status(info)

            if done:
                # Flush remaining N-step transitions
                for transition in self.n_step_buffer.flush():
                    self.buffer.add(*transition)

                # Track episode stats
                self.episode_count += 1
                episodes_completed += 1
                self.reward_history.append(self.episode_reward)
                if len(self.reward_history) > 100:
                    self.reward_history.pop(0)

                # Calculate speed (x_pos / time_elapsed)
                game_time = info.get("time", 0)
                time_elapsed = self.episode_start_time - game_time
                if time_elapsed > 0:
                    speed = x_pos / time_elapsed
                    self.speed_history.append(speed)
                    if len(self.speed_history) > 20:
                        self.speed_history.pop(0)

                # Track death/flag metrics
                is_dead = info.get("is_dead", False) or info.get("is_dying", False)
                flag_get = info.get("flag_get", False)

                if flag_get:
                    self.time_to_flag_history.append(game_time)
                    if len(self.time_to_flag_history) > 20:
                        self.time_to_flag_history.pop(0)
                elif is_dead:
                    self.deaths += 1
                    self.x_at_death_history.append(x_pos)
                    if len(self.x_at_death_history) > 20:
                        self.x_at_death_history.pop(0)

                self._send_ui_status(info)

                # Reset
                state, _ = self.env.reset()
                self.episode_reward = 0.0
                self.episode_length = 0
                self.best_x = 0
            else:
                state = next_state

        return episodes_completed

    def compute_and_send_gradients(self) -> Dict[str, float]:
        """
        Sample from local PER buffer, compute DQN loss, update priorities,
        and send gradients to learner.

        Returns metrics dictionary.
        """
        if len(self.buffer) < self.batch_size:
            return {}

        # Sample batch from PER buffer
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights,
        ) = self.buffer.sample(self.batch_size)

        # Convert to tensors
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)
        weights_t = torch.from_numpy(weights).to(self.device)

        # Compute Double DQN loss
        self.net.train()

        # Current Q-values
        current_q = self.net.online(states_t)
        current_q_selected = current_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_q_online = self.net.online(next_states_t)
            best_actions = next_q_online.argmax(dim=1)
            next_q_target = self.net.target(next_states_t)
            next_q_selected = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.n_step_gamma * next_q_selected * (1.0 - dones_t)

        # Compute TD errors for priority update
        td_errors = (current_q_selected - target_q).detach()

        # Weighted Huber loss (importance sampling)
        element_wise_loss = torch.nn.functional.huber_loss(current_q_selected, target_q, reduction="none", delta=1.0)
        loss = (weights_t * element_wise_loss).mean()

        # Backward
        self.net.online.zero_grad()
        loss.backward()

        # Clip gradients
        nn.utils.clip_grad_norm_(self.net.online.parameters(), self.max_grad_norm)

        # Update priorities in PER buffer
        self.buffer.update_priorities(indices, td_errors.abs().cpu().numpy())

        # Extract gradients (move to CPU for IPC)
        grads = {
            name: param.grad.cpu().clone()
            for name, param in self.net.online.named_parameters()
            if param.grad is not None
        }

        # Metrics
        metrics = {
            "loss": loss.item(),
            "q_mean": current_q_selected.mean().item(),
            "q_max": current_q_selected.max().item(),
            "td_error": td_errors.abs().mean().item(),
            "per_beta": self.buffer.current_beta,
        }

        # Send gradients to learner
        gradient_packet = {
            "grads": grads,
            "timesteps": self.batch_size,
            "episodes": self.episode_count,
            "worker_id": self.worker_id,
            "weight_version": self.weight_version,
            "metrics": metrics,
        }

        try:
            self.gradient_queue.put(gradient_packet, timeout=5.0)
            self.gradients_sent += 1
        except Exception as e:
            self._log(f"Failed to send gradients: {e}")

        return metrics

    def _send_ui_status(self, info: dict) -> None:
        """Send status to UI queue."""
        if self.ui_queue is None:
            return

        try:
            from mario_rl.training.training_ui import UIMessage
            from mario_rl.training.training_ui import MessageType

            rolling_avg = np.mean(self.reward_history) if self.reward_history else 0.0
            level_str = self.base_env.current_level

            # Compute average metrics
            avg_speed = np.mean(self.speed_history) if self.speed_history else 0.0
            avg_x_at_death = np.mean(self.x_at_death_history) if self.x_at_death_history else 0.0
            avg_time_to_flag = np.mean(self.time_to_flag_history) if self.time_to_flag_history else 0.0

            msg = UIMessage(
                msg_type=MessageType.WORKER_STATUS,
                source_id=self.worker_id,
                data={
                    "episode": self.episode_count,
                    "step": self.episode_length,
                    "reward": self.episode_reward,
                    "x_pos": info.get("x_pos", 0),
                    "game_time": info.get("time", 0),
                    "best_x": self.best_x,
                    "best_x_ever": self.best_x_ever,
                    "deaths": self.deaths,
                    "flags": self.flags,
                    "epsilon": self.current_epsilon,
                    "experiences": self.total_steps,
                    "q_mean": 0.0,
                    "q_max": 0.0,
                    "weight_sync_count": self.weight_sync_count,
                    "gradients_sent": self.gradients_sent,
                    "steps_per_sec": self.steps_per_sec,
                    "snapshot_restores": 0,
                    "current_level": level_str,
                    "last_weight_sync": self.last_weight_sync,
                    "rolling_avg_reward": rolling_avg,
                    "first_flag_time": avg_time_to_flag,
                    "per_beta": self.buffer.current_beta,
                    # Additional metrics
                    "avg_speed": avg_speed,
                    "avg_x_at_death": avg_x_at_death,
                    "avg_time_to_flag": avg_time_to_flag,
                },
            )
            self.ui_queue.put_nowait(msg)
        except Exception:
            pass

    def _log(self, text: str) -> None:
        """Log message to UI queue or stdout."""
        if self.ui_queue is not None:
            try:
                from mario_rl.training.training_ui import UIMessage
                from mario_rl.training.training_ui import MessageType

                msg = UIMessage(
                    msg_type=MessageType.WORKER_LOG,
                    source_id=self.worker_id,
                    data={"text": text},
                )
                self.ui_queue.put_nowait(msg)
            except Exception:
                pass
        else:
            print(text)

    def run(self) -> None:
        """Main worker loop."""
        self._log(
            f"Worker {self.worker_id} started "
            f"(level={self.level}, device={self.device}, ε_end={self.eps_end:.4f}, PER)"
        )

        while True:
            # Maybe sync weights from learner
            self._maybe_sync_weights()

            # Collect experiences
            self.collect_steps(self.steps_per_collection)

            # Calculate speed
            now = time.time()
            elapsed = now - self._last_time
            self.steps_per_sec = self.steps_per_collection / elapsed if elapsed > 0 else 0
            self._last_time = now

            # Compute gradients and send to learner
            if len(self.buffer) >= self.batch_size:
                for _ in range(self.train_steps):
                    self.compute_and_send_gradients()


def run_ddqn_worker(
    worker_id: int,
    weights_path: Path,
    gradient_queue: mp.Queue,
    level: LevelType = (1, 1),
    ui_queue: Optional[mp.Queue] = None,
    **kwargs: Any,
) -> None:
    """Entry point for worker process."""
    worker = DDQNWorker(
        worker_id=worker_id,
        weights_path=weights_path,
        gradient_queue=gradient_queue,
        level=level,
        ui_queue=ui_queue,
        **kwargs,
    )
    worker.run()
