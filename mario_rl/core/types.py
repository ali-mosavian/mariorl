"""
Core data types as frozen dataclasses.

Using frozen dataclasses with slots for:
- Named field access (clearer than tuple indexing)
- Memory efficiency (slots)
- Immutability (frozen)
- Built-in type hints
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Transition:
    """A single experience transition."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass(frozen=True, slots=True)
class GameSnapshot:
    """Saved game state for checkpoint restore."""

    observation: np.ndarray
    frame_queue: tuple  # Tuple of np.ndarray frames (immutable)
    nes_state: np.ndarray


@dataclass(frozen=True, slots=True)
class TreeNode:
    """Result from SumTree lookup."""

    leaf_idx: int
    priority: float
    data_idx: int


@dataclass(frozen=True, slots=True)
class PERBatch:
    """Sampled batch from Prioritized Experience Replay buffer."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    indices: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True, slots=True)
class WorkerStatus:
    """Status update from worker to UI."""

    worker_id: int
    episode: int
    step: int
    reward: float
    x_pos: int
    game_time: int
    best_x: int
    best_x_ever: int
    deaths: int
    flags: int
    epsilon: float
    experiences: int
    weight_sync_count: int
    gradients_sent: int
    steps_per_sec: float
    snapshot_restores: int
    restores_without_progress: int
    max_restores: int
    current_level: str
    last_weight_sync: float
    rolling_avg_reward: float
    per_beta: float
    avg_speed: float
    avg_x_at_death: float
    avg_time_to_flag: float
    entropy: float
    last_action_time: float
    # Buffer diagnostics
    buffer_size: int = 0
    buffer_capacity: int = 0
    buffer_fill_pct: float = 0.0
    can_train: bool = False


@dataclass(frozen=True, slots=True)
class GradientMetrics:
    """Metrics computed during gradient calculation."""

    loss: float
    q_mean: float
    q_max: float
    td_error: float
    per_beta: float
    entropy: float
    avg_reward: float
    avg_speed: float
    total_deaths: int
    total_flags: int
    best_x_ever: int


@dataclass(frozen=True, slots=True)
class GradientPacket:
    """Gradient data sent from worker to learner."""

    grads: dict  # name -> tensor
    timesteps: int
    episodes: int
    worker_id: int
    weight_version: int
    metrics: GradientMetrics
