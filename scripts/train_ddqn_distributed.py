#!/usr/bin/env python3
"""
Distributed DDQN training with async gradient updates.

Workers compute gradients locally and send them to the learner.
Similar to Gorila DQN / APPO architecture.

Architecture
============

┌─────────────────────────────────────────────────────────────────────────┐
│                           MAIN PROCESS                                   │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      Gradient Queue                              │   │
│   │              (workers → learner, ~2MB per packet)                │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
          │                    │                    │
          │                    │                    │
          ▼                    ▼                    ▼
   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
   │  WORKER 0   │      │  WORKER 1   │      │  WORKER N   │
   │             │      │             │      │             │
   │ ε = 0.4^1   │      │ ε = 0.4^2   │      │ ε = 0.4^N   │
   │             │      │             │      │             │
   │ 1. Collect  │      │ 1. Collect  │      │ 1. Collect  │
   │    steps    │      │    steps    │      │    steps    │
   │             │      │             │      │             │
   │ 2. Sample   │      │ 2. Sample   │      │ 2. Sample   │
   │    batch    │      │    batch    │      │    batch    │
   │             │      │             │      │             │
   │ 3. Compute  │      │ 3. Compute  │      │ 3. Compute  │
   │    DQN loss │      │    DQN loss │      │    DQN loss │
   │             │      │             │      │             │
   │ 4. Backward │      │ 4. Backward │      │ 4. Backward │
   │    pass     │      │    pass     │      │    pass     │
   │             │      │             │      │             │
   │ 5. Send     │      │ 5. Send     │      │ 5. Send     │
   │    gradients│      │    gradients│      │    gradients│
   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
          │                    │                    │
          │     GRADIENT QUEUE (small, ~2MB each)   │
          └────────────────────┼────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      LEARNER        │
                    │                     │
                    │  1. Receive grads   │
                    │  2. Accumulate      │
                    │  3. Average         │
                    │  4. Clip            │
                    │  5. Optimizer step  │
                    │  6. Soft update     │
                    │  7. Save weights    │
                    └─────────────────────┘

Comparison with Standard DDQN
=============================

    Standard DDQN:
    Single process: collect → store → sample → forward → backward → step

    Distributed DDQN:
    Worker: collect → sample → forward → backward → send grads (async)
    Learner: receive grads → accumulate → step → soft update

    Benefits:
    - ~Nx faster with N workers
    - Distributed computation (workers do forward+backward)
    - ~4x less data through IPC (grads ~2MB vs experiences ~8MB)
    - GPU utilization for training
    - Diverse exploration (different epsilons per worker)

Usage:
    uv run mario-train-ddqn-dist --workers 4 --level random
    uv run mario-train-ddqn-dist --workers 8 --level 1,1 --accumulate-grads 4
    uv run mario-train-ddqn-dist --workers 4 --no-ui  # Disable ncurses UI
"""

import os
import sys
import signal
from typing import cast
from pathlib import Path
from typing import Literal
from datetime import datetime
from multiprocessing import Queue
from multiprocessing import Process

# Ensure unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

from mario_rl.core.config import BufferConfig
from mario_rl.core.config import WorkerConfig
from mario_rl.core.config import SnapshotConfig
from mario_rl.core.config import ExplorationConfig
from mario_rl.training.training_ui import TrainingUI
from mario_rl.training.ddqn_worker import run_ddqn_worker
from mario_rl.training.ddqn_learner import run_ddqn_learner
from mario_rl.core.config import LevelType as ConfigLevelType

LevelType = Literal["sequential", "random"] | tuple[int, int]


def parse_level(level_str: str) -> LevelType:
    """Parse level string into LevelType."""
    if level_str == "random":
        return "random"
    elif level_str == "sequential":
        return "sequential"
    else:
        parts = level_str.split(",")
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
        raise ValueError(f"Invalid level format: {level_str}")


def run_training_ui(num_workers: int, ui_queue: Queue) -> None:
    """Run the training UI in a separate process."""
    ui = TrainingUI(num_workers=num_workers, ui_queue=ui_queue, use_ppo=False)
    ui.run()


def run_worker_silent(
    config: WorkerConfig,
    weights_path: Path,
    gradient_queue: "Queue[object]",
    ui_queue: "Queue[object] | None" = None,
) -> None:
    """Run worker with stdout/stderr suppressed (for UI mode)."""
    import os
    import sys
    import warnings

    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # Redirect stdout/stderr to /dev/null
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull

    # Now run the actual worker
    run_ddqn_worker(config, weights_path, gradient_queue, ui_queue)


def run_learner_silent(
    weights_path: Path,
    save_dir: Path,
    gradient_queue: Queue,
    restore_snapshot: bool = False,
    snapshot_path: Path | None = None,
    **kwargs,
) -> None:
    """Run learner with stdout/stderr suppressed (for UI mode)."""
    import os
    import sys
    import warnings

    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # Redirect stdout/stderr to /dev/null
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull

    # Now run the actual learner
    run_ddqn_learner(
        weights_path,
        save_dir,
        gradient_queue,
        restore_snapshot=restore_snapshot,
        snapshot_path=snapshot_path,
        **kwargs,
    )


@click.command()
@click.option("--workers", "-w", default=4, help="Number of worker processes")
@click.option("--level", "-l", default="random", help="Level: 'random', 'sequential', or 'W,S'")
@click.option("--save-dir", default="checkpoints", help="Directory for saving checkpoints")
# Learning hyperparameters
@click.option("--lr", default=2.5e-4, help="Initial learning rate")
@click.option("--lr-end", default=1e-5, help="Final learning rate")
@click.option("--gamma", default=0.99, help="Discount factor")
@click.option("--n-step", default=3, help="N-step returns")
@click.option("--tau", default=0.005, help="Soft update coefficient")
# Worker settings
@click.option("--local-buffer-size", default=10_000, help="Local replay buffer per worker")
@click.option("--batch-size", default=32, help="Training batch size per worker")
@click.option("--collect-steps", default=64, help="Steps to collect per cycle")
@click.option("--train-steps", default=4, help="Gradient computations per cycle")
# Learner settings
@click.option("--accumulate-grads", default=1, help="Gradients to accumulate before update")
# Epsilon settings (per worker)
@click.option("--eps-base", default=0.4, help="Base for per-worker epsilon (ε = base^(1+i/N))")
@click.option("--eps-decay-steps", default=100_000, help="Steps for epsilon decay per worker")
# Other
@click.option("--total-steps", default=2_000_000, help="Total training steps (for LR schedule)")
@click.option("--max-grad-norm", default=10.0, help="Maximum gradient norm")
@click.option("--weight-decay", default=1e-4, help="L2 regularization")
@click.option("--no-ui", is_flag=True, help="Disable ncurses UI")
@click.option("--resume", is_flag=True, help="Resume from latest checkpoint")
@click.option("--restore-snapshot", type=str, default=None, help="Restore from specific snapshot file")
@click.option("--no-game-snapshots", is_flag=True, help="Disable game state snapshots (save/restore on death)")
def main(
    workers: int,
    level: str,
    save_dir: str,
    lr: float,
    lr_end: float,
    gamma: float,
    n_step: int,
    tau: float,
    local_buffer_size: int,
    batch_size: int,
    collect_steps: int,
    train_steps: int,
    accumulate_grads: int,
    eps_base: float,
    eps_decay_steps: int,
    total_steps: int,
    max_grad_norm: float,
    weight_decay: float,
    no_ui: bool,
    resume: bool,
    restore_snapshot: str | None,
    no_game_snapshots: bool,
) -> None:
    """Train Mario using Distributed DDQN with async gradient updates."""
    # Parse level
    level_type = parse_level(level)

    # Create save directory
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = Path(save_dir) / f"ddqn_dist_{timestamp}"

    # Parse snapshot path if provided
    snapshot_path: Path | None = None
    if restore_snapshot:
        snapshot_path = Path(restore_snapshot)
        # If it's a directory, look for snapshot.pt inside
        if snapshot_path.is_dir():
            snapshot_path = snapshot_path / "snapshot.pt"
        # Use the parent directory as run_dir
        run_dir = snapshot_path.parent
        print(f"Restoring from snapshot: {snapshot_path}")

    # Check for resume (only if not restoring from specific snapshot)
    elif resume:
        checkpoints = list(Path(save_dir).glob("ddqn_dist_*/weights.pt"))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            run_dir = latest.parent
            # Check if snapshot exists
            if (run_dir / "snapshot.pt").exists():
                snapshot_path = run_dir / "snapshot.pt"
                print(f"Resuming from snapshot: {snapshot_path}")
            else:
                print(f"Resuming from: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "weights.pt"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Print config (will be hidden by UI if enabled)
    if no_ui:
        print("=" * 70)
        print("Distributed DDQN Training (Async Gradients)")
        print("=" * 70)
        print(f"  Workers: {workers}")
        print(f"  Level: {level}")
        print(f"  Save dir: {run_dir}")
        print(f"  LR: {lr} → {lr_end} (cosine)")
        print(f"  Gamma: {gamma}, N-step: {n_step}")
        print(f"  Tau: {tau}")
        print(f"  Local buffer: {local_buffer_size:,}, Batch: {batch_size}")
        print(f"  Collect steps: {collect_steps}, Train steps: {train_steps}")
        print(f"  Accumulate grads: {accumulate_grads}")
        print(f"  Per-worker epsilon: {eps_base}^(1+i/N)")
        print(f"  Total steps: {total_steps:,}")
        print("=" * 70)

    # Create queues
    # Larger queue to reduce worker blocking
    # Workers send train_steps (4) gradients per cycle, so need more buffer
    gradient_queue: Queue = Queue(maxsize=workers * 8)
    ui_queue: Queue | None = Queue() if not no_ui else None

    # Start UI process
    ui_process = None
    if not no_ui:
        ui_process = Process(
            target=run_training_ui,
            args=(workers, ui_queue),
            daemon=True,
        )
        ui_process.start()

    # Choose worker/learner targets based on UI mode
    worker_target = run_worker_silent if not no_ui else run_ddqn_worker
    learner_target = run_learner_silent if not no_ui else run_ddqn_learner

    # Create buffer config (shared by all workers)
    buffer_config = BufferConfig(
        capacity=local_buffer_size,
        batch_size=batch_size,
        n_step=n_step,
        gamma=gamma,
    )

    # Create snapshot config
    snapshot_config = SnapshotConfig(
        enabled=not no_game_snapshots,
    )

    # Start worker processes with different epsilons
    worker_processes = []
    for i in range(workers):
        # Each worker has different exploration rate
        # Worker 0: most exploration, Worker N-1: least exploration
        worker_eps_end = eps_base ** (1 + (i + 1) / workers)

        # Create exploration config for this worker
        exploration_config = ExplorationConfig(
            epsilon_start=1.0,
            epsilon_end=worker_eps_end,
            decay_steps=eps_decay_steps,
        )

        # Create worker config
        worker_config = WorkerConfig(
            worker_id=i,
            level=cast(ConfigLevelType, level_type),
            steps_per_collection=collect_steps,
            train_steps=train_steps,
            max_grad_norm=max_grad_norm,
            buffer=buffer_config,
            exploration=exploration_config,
            snapshot=snapshot_config,
        )

        p = Process(
            target=worker_target,
            args=(worker_config, weights_path, gradient_queue),
            kwargs={"ui_queue": ui_queue},
            daemon=True,
        )
        p.start()
        worker_processes.append(p)
        if no_ui:
            print(f"Started worker {i} (ε_end = {worker_eps_end:.4f})")

    # Start learner process
    learner_process = Process(
        target=learner_target,
        args=(weights_path, run_dir, gradient_queue),
        kwargs={
            "learning_rate": lr,
            "lr_end": lr_end,
            "total_timesteps": total_steps,
            "tau": tau,
            "accumulate_grads": accumulate_grads,
            "max_grad_norm": max_grad_norm,
            "weight_decay": weight_decay,
            "ui_queue": ui_queue,
            "restore_snapshot": snapshot_path is not None,
            "snapshot_path": snapshot_path,
        },
        daemon=True,
    )
    learner_process.start()
    if no_ui:
        print("Started learner")

    # Set up signal handler
    def signal_handler(sig, frame):
        print("\nShutting down...")
        for p in worker_processes:
            p.terminate()
        learner_process.terminate()
        if ui_process:
            ui_process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Wait for processes
    try:
        learner_process.join()
    except KeyboardInterrupt:
        print("\nShutting down...")
        for p in worker_processes:
            p.terminate()
        learner_process.terminate()
        if ui_process:
            ui_process.terminate()


if __name__ == "__main__":
    main()
