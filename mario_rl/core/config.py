"""
Configuration dataclasses for training.

All configuration is immutable (frozen) to prevent accidental modification.
"""

import os

from typing import Tuple
from typing import Literal
from dataclasses import dataclass

# Default to number of CPU cores (fallback to 4 if detection fails)
_DEFAULT_NUM_WORKERS = os.cpu_count() or 4

# Type for level specification
LevelType = Literal["sequential", "random"] | Tuple[Literal[1, 2, 3, 4, 5, 6, 7, 8], Literal[1, 2, 3, 4]]


@dataclass(frozen=True)
class BufferConfig:
    """Configuration for replay buffers."""

    capacity: int = 10_000
    batch_size: int = 32
    n_step: int = 10
    gamma: float = 0.99

    # PER hyperparameters
    alpha: float = 0.6  # Priority exponent (0 = uniform, 1 = full prioritization)
    beta_start: float = 0.4  # Initial importance sampling exponent
    beta_end: float = 1.0  # Final importance sampling exponent

    # Asymmetric priority: flag captures get boosted priority
    flag_priority_multiplier: float = 50.0

    # Elite buffer: preserves best experiences to prevent "forgetting success"
    elite_capacity: int = 1000  # Number of elite transitions to keep
    elite_sample_ratio: float = 0.15  # Fraction of batch to sample from elite buffer


@dataclass(frozen=True)
class ExplorationConfig:
    """Configuration for exploration strategy."""

    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    decay_steps: int = 1_000_000  # 1M steps for gradual exploration decay


@dataclass(frozen=True)
class SnapshotConfig:
    """Configuration for game state snapshots."""

    enabled: bool = True
    slots: int = 10  # Number of rotating snapshot slots
    interval: int = 5  # Save checkpoint every N seconds of Mario time
    max_restores_without_progress: int = 3  # End episode after N restores with no x progress


@dataclass(frozen=True)
class MCTSConfig:
    """Configuration for MCTS exploration integration."""

    enabled: bool = False  # Whether to use MCTS at all

    # MCTS parameters
    num_simulations: int = 50  # MCTS simulations per exploration
    max_rollout_depth: int = 20  # Max depth of rollouts
    exploration_constant: float = 1.41  # UCB exploration constant
    discount: float = 0.99  # Discount factor for rollouts

    # Rollout policy: "random", "policy", "mixed"
    rollout_policy: str = "mixed"
    policy_mix_ratio: float = 0.7  # 70% policy, 30% random

    # Value estimation: "rollout", "network", "mixed"
    value_source: str = "mixed"

    # When to use MCTS (hybrid mode triggers)
    use_when_stuck: bool = True  # Use MCTS when no x progress
    stuck_threshold: int = 500  # Steps without x progress to trigger MCTS

    use_periodically: bool = True  # Use MCTS every N episodes
    periodic_interval: int = 10  # Every N episodes, use MCTS for one episode


@dataclass(frozen=True)
class WorkerConfig:
    """Configuration for a DDQN worker."""

    worker_id: int
    level: LevelType = (1, 1)
    render_frames: bool = False

    # Network architecture
    use_dreamer: bool = False  # Use Dreamer-style encoder + latent Q-network
    latent_dim: int = 128  # Latent dimension for Dreamer network

    # Training
    steps_per_collection: int = 64
    train_steps: int = 4
    weight_sync_interval: float = 5.0  # Seconds between weight syncs
    max_grad_norm: float = 10.0

    # Reward normalization mode: "none" (use env rewards), "scale" (fixed), or "running"
    # - "none": use environment rewards directly (recommended - env already normalizes)
    # - "scale": multiply by reward_scale factor
    # - "running": normalize by running mean/std (unstable with sparse penalties)
    reward_norm: str = "none"  # Env already normalizes to ~[-2, +2] per frame
    reward_scale: float = 1.0  # Used when reward_norm="scale"
    reward_clip: float = 0.0  # Clip rewards (0 to disable)

    # Entropy regularization (encourages exploration, prevents policy collapse)
    entropy_coef: float = 0.01  # Coefficient for entropy bonus in loss

    # Stability settings
    loss_threshold: float = 1000.0  # Skip gradient if loss exceeds this

    # Device (None = auto-detect)
    device: str | None = None

    # Resume support: initial step count for epsilon decay
    initial_steps: int = 0  # Steps already collected (for --resume)

    # Sub-configs
    buffer: BufferConfig = BufferConfig()
    exploration: ExplorationConfig = ExplorationConfig()
    snapshot: SnapshotConfig = SnapshotConfig()
    mcts: MCTSConfig = MCTSConfig()  # MCTS exploration (disabled by default)

    def with_epsilon(self, epsilon_end: float) -> "WorkerConfig":
        """Create a new config with different epsilon_end (for per-worker exploration)."""
        return WorkerConfig(
            worker_id=self.worker_id,
            level=self.level,
            render_frames=self.render_frames,
            use_dreamer=self.use_dreamer,
            latent_dim=self.latent_dim,
            steps_per_collection=self.steps_per_collection,
            train_steps=self.train_steps,
            weight_sync_interval=self.weight_sync_interval,
            max_grad_norm=self.max_grad_norm,
            reward_norm=self.reward_norm,
            reward_scale=self.reward_scale,
            reward_clip=self.reward_clip,
            entropy_coef=self.entropy_coef,
            loss_threshold=self.loss_threshold,
            device=self.device,
            initial_steps=self.initial_steps,
            buffer=self.buffer,
            exploration=ExplorationConfig(
                epsilon_start=self.exploration.epsilon_start,
                epsilon_end=epsilon_end,
                decay_steps=self.exploration.decay_steps,
            ),
            snapshot=self.snapshot,
            mcts=self.mcts,
        )


@dataclass(frozen=True)
class LearnerConfig:
    """Configuration for the DDQN learner."""

    # Network architecture
    use_dreamer: bool = False  # Use Dreamer-style encoder + latent Q-network
    latent_dim: int = 128  # Latent dimension for Dreamer network

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

    num_workers: int = _DEFAULT_NUM_WORKERS
    level: LevelType = (1, 1)

    # Worker settings
    worker: WorkerConfig = WorkerConfig(worker_id=0)  # Template for workers
    learner: LearnerConfig = LearnerConfig()

    # Queue settings
    gradient_queue_size_multiplier: int = 8

    # Exploration spread (for diverse exploration across workers)
    # Epsilon base for per-worker spread: ε = base^(1 + i/N)
    # 0.15 gives floor of ~2-9% across workers (vs 16-32% with 0.4)
    epsilon_base: float = 0.15

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
                use_dreamer=self.worker.use_dreamer,
                latent_dim=self.worker.latent_dim,
                steps_per_collection=self.worker.steps_per_collection,
                train_steps=self.worker.train_steps,
                weight_sync_interval=self.worker.weight_sync_interval,
                max_grad_norm=self.worker.max_grad_norm,
                reward_norm=self.worker.reward_norm,
                reward_scale=self.worker.reward_scale,
                reward_clip=self.worker.reward_clip,
                entropy_coef=self.worker.entropy_coef,
                loss_threshold=self.worker.loss_threshold,
                device=self.worker.device,
                initial_steps=self.worker.initial_steps,
                buffer=self.worker.buffer,
                exploration=ExplorationConfig(
                    epsilon_start=self.worker.exploration.epsilon_start,
                    epsilon_end=eps_end,
                    decay_steps=self.worker.exploration.decay_steps,
                ),
                snapshot=self.worker.snapshot,
                mcts=self.worker.mcts,
            )
            configs.append(config)
        return configs
