"""
Core components for Mario RL training.

This module provides configuration, types, and common utilities.
"""

from mario_rl.core.types import PERBatch
from mario_rl.core.types import TreeNode
from mario_rl.core.config import LevelType
from mario_rl.core.types import Transition
from mario_rl.core.types import GameSnapshot
from mario_rl.core.types import WorkerStatus
from mario_rl.core.config import BufferConfig
from mario_rl.core.config import WorkerConfig
from mario_rl.core.config import LearnerConfig
from mario_rl.core.device import detect_device, get_gpu_count, assign_device, get_device_assignment_summary
from mario_rl.core.types import GradientPacket
from mario_rl.core.config import SnapshotConfig
from mario_rl.core.config import TrainingConfig
from mario_rl.core.types import GradientMetrics
from mario_rl.core.config import ExplorationConfig
from mario_rl.core.replay_buffer import MuZeroReplayBuffer
from mario_rl.core.replay_buffer import TrajectoryBatch

__all__ = [
    # Config
    "LevelType",
    "BufferConfig",
    "WorkerConfig",
    "LearnerConfig",
    "TrainingConfig",
    "SnapshotConfig",
    "ExplorationConfig",
    # Device utilities
    "detect_device",
    "get_gpu_count",
    "assign_device",
    "get_device_assignment_summary",
    # Types
    "Transition",
    "GameSnapshot",
    "TreeNode",
    "PERBatch",
    "WorkerStatus",
    "GradientPacket",
    "GradientMetrics",
    # MuZero replay buffer
    "MuZeroReplayBuffer",
    "TrajectoryBatch",
]
