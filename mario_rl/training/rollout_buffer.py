"""
Rollout buffer for PPO with GAE computation.

Stores trajectories from workers and computes advantages using
Generalized Advantage Estimation (GAE).
"""

from typing import Any
from typing import Generator
from dataclasses import field
from typing import NamedTuple
from dataclasses import dataclass

import torch
import numpy as np


class RolloutBatch(NamedTuple):
    """A batch of rollout data for training."""

    states: torch.Tensor  # (batch_size, F, H, W, C)
    actions: torch.Tensor  # (batch_size,)
    old_log_probs: torch.Tensor  # (batch_size,)
    advantages: torch.Tensor  # (batch_size,)
    returns: torch.Tensor  # (batch_size,)
    old_values: torch.Tensor  # (batch_size,)


@dataclass
class RolloutBuffer:
    """
    Buffer for storing rollouts and computing GAE.

    Supports multiple parallel environments and minibatch sampling.
    """

    buffer_size: int  # Total steps to store (n_steps * n_envs)
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Storage (initialized in __post_init__)
    states: np.ndarray = field(init=False, repr=False)
    actions: np.ndarray = field(init=False, repr=False)
    rewards: np.ndarray = field(init=False, repr=False)
    dones: np.ndarray = field(init=False, repr=False)
    values: np.ndarray = field(init=False, repr=False)
    log_probs: np.ndarray = field(init=False, repr=False)

    # Computed after rollout
    advantages: np.ndarray = field(init=False, repr=False)
    returns: np.ndarray = field(init=False, repr=False)

    # State
    ptr: int = field(init=False, default=0)
    full: bool = field(init=False, default=False)
    state_shape: tuple | None = field(init=False, default=None)

    def __post_init__(self):
        self.ptr = 0
        self.full = False
        self.state_shape = None
        # Arrays will be initialized on first add

    def _init_storage(self, state_shape: tuple):
        """Initialize storage arrays with proper shapes."""
        self.state_shape = state_shape
        self.states = np.zeros((self.buffer_size, *state_shape), dtype=np.float32)
        self.actions = np.zeros(self.buffer_size, dtype=np.int64)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """Add a single transition to the buffer."""
        if self.state_shape is None:
            self._init_storage(state.shape)

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
    ) -> None:
        """Add a batch of transitions (from multiple envs)."""
        batch_size = len(states)
        if self.state_shape is None:
            self._init_storage(states[0].shape)

        end_ptr = min(self.ptr + batch_size, self.buffer_size)
        n_to_add = end_ptr - self.ptr

        self.states[self.ptr : end_ptr] = states[:n_to_add]
        self.actions[self.ptr : end_ptr] = actions[:n_to_add]
        self.rewards[self.ptr : end_ptr] = rewards[:n_to_add]
        self.dones[self.ptr : end_ptr] = dones[:n_to_add].astype(np.float32)
        self.values[self.ptr : end_ptr] = values[:n_to_add]
        self.log_probs[self.ptr : end_ptr] = log_probs[:n_to_add]

        self.ptr = end_ptr
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_gae(self, last_value: float, last_done: bool) -> None:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            last_value: Value of the last state (for bootstrapping)
            last_done: Whether the last state was terminal
        """
        size = self.ptr if not self.full else self.buffer_size

        # Compute GAE
        last_gae = 0.0
        for t in reversed(range(size)):
            if t == size - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        # Compute returns (advantage + value)
        self.returns[:size] = self.advantages[:size] + self.values[:size]

    def get_batches(
        self,
        batch_size: int,
        device: str = "cpu",
    ) -> Generator[RolloutBatch, None, None]:
        """
        Generate minibatches for training.

        Args:
            batch_size: Size of each minibatch
            device: Device to move tensors to

        Yields:
            RolloutBatch for each minibatch
        """
        size = self.ptr if not self.full else self.buffer_size
        indices = np.random.permutation(size)

        # Normalize advantages
        advantages = self.advantages[:size]
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        normalized_advantages = (advantages - adv_mean) / adv_std

        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            batch_indices = indices[start:end]

            yield RolloutBatch(
                states=torch.from_numpy(self.states[batch_indices]).to(device),
                actions=torch.from_numpy(self.actions[batch_indices]).to(device),
                old_log_probs=torch.from_numpy(self.log_probs[batch_indices]).to(device),
                advantages=torch.from_numpy(normalized_advantages[batch_indices]).to(device),
                returns=torch.from_numpy(self.returns[batch_indices]).to(device),
                old_values=torch.from_numpy(self.values[batch_indices]).to(device),
            )

    def reset(self) -> None:
        """Reset buffer for new rollout."""
        self.ptr = 0
        self.full = False

    def is_full(self) -> bool:
        """Check if buffer has collected enough data."""
        return self.full

    def size(self) -> int:
        """Get current number of transitions stored."""
        return self.buffer_size if self.full else self.ptr


@dataclass
class SharedRolloutBuffer:
    """
    Thread-safe rollout buffer for distributed PPO.

    Workers push rollouts, learner pulls and processes them.
    Uses a simple list-based approach with locks for thread safety.
    """

    max_rollouts: int = 100

    # Storage
    rollouts: list = field(init=False, default_factory=list)
    _lock: Any = field(init=False, repr=False, default=None)

    def __post_init__(self):
        import threading

        self.rollouts = []
        self._lock = threading.Lock()

    def push_rollout(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
        last_value: float,
        last_done: bool,
    ) -> bool:
        """
        Push a complete rollout from a worker.

        Returns True if successful, False if buffer is full.
        """
        with self._lock:
            if len(self.rollouts) >= self.max_rollouts:
                return False

            self.rollouts.append(
                {
                    "states": states.copy(),
                    "actions": actions.copy(),
                    "rewards": rewards.copy(),
                    "dones": dones.copy(),
                    "values": values.copy(),
                    "log_probs": log_probs.copy(),
                    "last_value": last_value,
                    "last_done": last_done,
                }
            )
            return True

    def pull_rollouts(self, max_rollouts: int = -1) -> list:
        """
        Pull all available rollouts for training.

        Args:
            max_rollouts: Maximum rollouts to pull (-1 for all)

        Returns:
            List of rollout dictionaries
        """
        with self._lock:
            if max_rollouts < 0:
                rollouts = self.rollouts
                self.rollouts = []
            else:
                rollouts = self.rollouts[:max_rollouts]
                self.rollouts = self.rollouts[max_rollouts:]
            return rollouts

    def num_rollouts(self) -> int:
        """Get number of available rollouts."""
        with self._lock:
            return len(self.rollouts)

    def clear(self) -> None:
        """Clear all rollouts."""
        with self._lock:
            self.rollouts = []
