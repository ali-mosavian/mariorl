#!/usr/bin/env python3
"""
Custom distributed PPO training script.

Uses our own PPO implementation with distributed workers.

Usage:
    uv run python scripts/train_ppo_custom.py --workers 4 --level random
    uv run mario-train-ppo-custom --workers 8 --level 1,1
"""

import os
import sys
import signal
from pathlib import Path
from typing import Literal
from datetime import datetime
from multiprocessing import Queue
from multiprocessing import Process

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

from mario_rl.training.training_ui import TrainingUI
from mario_rl.training.ppo_worker import run_ppo_worker
from mario_rl.training.ppo_learner import run_ppo_learner

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
    ui = TrainingUI(num_workers=num_workers, ui_queue=ui_queue, use_ppo=True)
    ui.run()


@click.command()
@click.option("--workers", "-w", default=4, help="Number of worker processes")
@click.option("--level", "-l", default="random", help="Level: 'random', 'sequential', or 'W,S'")
@click.option("--save-dir", default="checkpoints", help="Directory for saving checkpoints")
@click.option("--n-steps", default=128, help="Steps per rollout per worker")
@click.option("--n-epochs", default=4, help="Epochs per PPO update")
@click.option("--batch-size", default=64, help="Minibatch size")
@click.option("--lr", default=2.5e-4, help="Learning rate")
@click.option("--gamma", default=0.99, help="Discount factor")
@click.option("--gae-lambda", default=0.95, help="GAE lambda")
@click.option("--clip-range", default=0.2, help="PPO clip range")
@click.option("--ent-coef", default=0.01, help="Entropy coefficient")
@click.option("--vf-coef", default=0.5, help="Value function coefficient")
@click.option("--no-ui", is_flag=True, help="Disable ncurses UI")
@click.option("--resume", is_flag=True, help="Resume from latest checkpoint")
def main(
    workers: int,
    level: str,
    save_dir: str,
    n_steps: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    no_ui: bool,
    resume: bool,
) -> None:
    """Train Mario using custom distributed PPO."""
    # Parse level
    level_type = parse_level(level)

    # Create save directory
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = Path(save_dir) / f"ppo_custom_{timestamp}"

    # Check for resume
    if resume:
        checkpoints = list(Path(save_dir).glob("ppo_custom_*/weights.pt"))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            run_dir = latest.parent
            print(f"Resuming from: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "weights.pt"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Custom Distributed PPO Training")
    print("=" * 60)
    print(f"  Workers: {workers}")
    print(f"  Level: {level}")
    print(f"  Save dir: {run_dir}")
    print(f"  N steps: {n_steps}")
    print(f"  N epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Gamma: {gamma}")
    print(f"  GAE lambda: {gae_lambda}")
    print(f"  Clip range: {clip_range}")
    print(f"  Entropy coef: {ent_coef}")
    print(f"  Value coef: {vf_coef}")
    print("=" * 60)

    # Create queues
    rollout_queue: Queue = Queue(maxsize=workers * 4)
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

    # Start worker processes
    worker_processes = []
    for i in range(workers):
        p = Process(
            target=run_ppo_worker,
            args=(i, weights_path, rollout_queue),
            kwargs={
                "level": level_type,
                "n_steps": n_steps,
                "ui_queue": ui_queue,
            },
            daemon=True,
        )
        p.start()
        worker_processes.append(p)
        print(f"Started worker {i}")

    # Start learner process
    learner_process = Process(
        target=run_ppo_learner,
        args=(weights_path, run_dir, rollout_queue),
        kwargs={
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "learning_rate": lr,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "ui_queue": ui_queue,
        },
        daemon=True,
    )
    learner_process.start()
    print("Started learner")

    # Handle Ctrl+C
    def signal_handler(signum, frame):
        print("\nShutting down...")
        for p in worker_processes:
            p.terminate()
        learner_process.terminate()
        if ui_process:
            ui_process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Wait for processes
    try:
        learner_process.join()
    except KeyboardInterrupt:
        signal_handler(None, None)

    print("Training complete!")


if __name__ == "__main__":
    main()
