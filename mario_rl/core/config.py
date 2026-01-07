"""
Configuration dataclasses for training.

All configuration is immutable (frozen) to prevent accidental modification.
"""

from typing import Tuple
from typing import Literal
from dataclasses import dataclass

# Type for level specification
LevelType = Literal["sequential", "random"] | Tuple[Literal[1, 2, 3, 4, 5, 6, 7, 8], Literal[1, 2, 3, 4]]


@dataclass(frozen=True)
class BufferConfig:
    """Configuration for replay buffers."""

    capacity: int = 10_000
    batch_size: int = 32
    n_step: int = 3
    gamma: float = 0.99

    # PER hyperparameters
    alpha: float = 0.6  # Priority exponent (0 = uniform, 1 = full prioritization)
    beta_start: float = 0.4  # Initial importance sampling exponent
    beta_end: float = 1.0  # Final importance sampling exponent


@dataclass(frozen=True)
class ExplorationConfig:
    """Configuration for exploration strategy."""

    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    decay_steps: int = 100_000


@dataclass(frozen=True)
class SnapshotConfig:
    """Configuration for game state snapshots."""

    enabled: bool = True
    slots: int = 10  # Number of rotating snapshot slots
    interval: int = 5  # Save checkpoint every N seconds of Mario time
    max_restores_without_progress: int = 3  # End episode after N restores with no x progress


@dataclass(frozen=True)
class WorkerConfig:
    """Configuration for a DDQN worker."""

    worker_id: int
    level: LevelType = (1, 1)
    render_frames: bool = False

    # Training
    steps_per_collection: int = 64
    train_steps: int = 4
    weight_sync_interval: float = 5.0  # Seconds between weight syncs
    max_grad_norm: float = 10.0

    # Reward normalization (clips rewards to [-reward_clip, +reward_clip])
    # Set to 0 to disable clipping. Default 1.0 matches DQN paper.
    reward_clip: float = 1.0

    # Device (None = auto-detect)
    device: str | None = None

    # Sub-configs
    buffer: BufferConfig = BufferConfig()
    exploration: ExplorationConfig = ExplorationConfig()
    snapshot: SnapshotConfig = SnapshotConfig()

    def with_epsilon(self, epsilon_end: float) -> "WorkerConfig":
        """Create a new config with different epsilon_end (for per-worker exploration)."""
        return WorkerConfig(
            worker_id=self.worker_id,
            level=self.level,
            render_frames=self.render_frames,
            steps_per_collection=self.steps_per_collection,
            train_steps=self.train_steps,
            weight_sync_interval=self.weight_sync_interval,
            max_grad_norm=self.max_grad_norm,
            reward_clip=self.reward_clip,
            device=self.device,
            buffer=self.buffer,
            exploration=ExplorationConfig(
                epsilon_start=self.exploration.epsilon_start,
                epsilon_end=epsilon_end,
                decay_steps=self.exploration.decay_steps,
            ),
            snapshot=self.snapshot,
        )


@dataclass(frozen=True)
class LearnerConfig:
    """Configuration for the DDQN learner."""

    # Optimizer
    learning_rate: float = 0.00025
    lr_end: float = 1e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 10.0

    # Target network
    tau: float = 0.005  # Soft update rate
    gamma: float = 0.99

    # Scheduling
    lr_schedule_steps: int = 500_000
    checkpoint_interval: int = 500  # Updates between checkpoints

    # Device (None = auto-detect)
    device: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    """Top-level training configuration."""

    num_workers: int = 4
    level: LevelType = (1, 1)

    # Worker settings
    worker: WorkerConfig = WorkerConfig(worker_id=0)  # Template for workers
    learner: LearnerConfig = LearnerConfig()

    # Queue settings
    gradient_queue_size_multiplier: int = 8

    # Exploration spread (for diverse exploration across workers)
    epsilon_base: float = 0.4

    def create_worker_configs(self) -> list["WorkerConfig"]:
        """Create worker configs with spread epsilon values."""
        configs = []
        for i in range(self.num_workers):
            # Worker 0: most exploration, Worker N-1: least exploration
            eps_end = self.epsilon_base ** (1 + (i + 1) / self.num_workers)
            config = WorkerConfig(
                worker_id=i,
                level=self.level,
                render_frames=self.worker.render_frames,
                steps_per_collection=self.worker.steps_per_collection,
                train_steps=self.worker.train_steps,
                weight_sync_interval=self.worker.weight_sync_interval,
                max_grad_norm=self.worker.max_grad_norm,
                reward_clip=self.worker.reward_clip,
                device=self.worker.device,
                buffer=self.worker.buffer,
                exploration=ExplorationConfig(
                    epsilon_start=self.worker.exploration.epsilon_start,
                    epsilon_end=eps_end,
                    decay_steps=self.worker.exploration.decay_steps,
                ),
                snapshot=self.worker.snapshot,
            )
            configs.append(config)
        return configs
