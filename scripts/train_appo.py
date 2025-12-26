#!/usr/bin/env python3
"""
APPO (Asynchronous PPO) training script.

Hybrid of PPO and A3C - workers compute gradients locally and send only
gradients to the learner, reducing IPC overhead by ~4x.

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
   │ 1. Collect  │      │ 1. Collect  │      │ 1. Collect  │
   │    rollout  │      │    rollout  │      │    rollout  │
   │             │      │             │      │             │
   │ 2. Compute  │      │ 2. Compute  │      │ 2. Compute  │
   │    PPO loss │      │    PPO loss │      │    PPO loss │
   │             │      │             │      │             │
   │ 3. Backward │      │ 3. Backward │      │ 3. Backward │
   │    pass     │      │    pass     │      │    pass     │
   │             │      │             │      │             │
   │ 4. Send     │      │ 4. Send     │      │ 4. Send     │
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
                    │  6. Save weights    │
                    └─────────────────────┘

Comparison with Standard PPO
============================

    Standard PPO:
    Worker → [rollout ~8MB] → Learner (forward + backward)

    APPO:
    Worker (forward + backward) → [gradients ~2MB] → Learner (apply)

    Benefits:
    - ~4x less data through IPC
    - Distributed computation
    - Lower learner CPU load

Usage:
    uv run mario-train-appo --workers 4 --level random
    uv run mario-train-appo --workers 8 --level 1,1 --lr 3e-4
"""

import os
import sys
import signal
from pathlib import Path
from typing import Literal
from datetime import datetime
from multiprocessing import Queue
from multiprocessing import Process

# Ensure unbuffered output for multiprocessing child processes
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

from mario_rl.training.training_ui import TrainingUI
from mario_rl.training.appo_worker import run_appo_worker
from mario_rl.training.appo_learner import run_appo_learner

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
@click.option("--n-epochs", default=4, help="PPO epochs per rollout (in worker)")
@click.option("--minibatch-size", default=32, help="Minibatch size for PPO updates (in worker)")
@click.option("--lr", default=2.5e-4, help="Learning rate")
@click.option("--gamma", default=0.99, help="Discount factor")
@click.option("--gae-lambda", default=0.95, help="GAE lambda")
@click.option("--clip-range", default=0.2, help="PPO clip range")
@click.option("--ent-coef", default=0.05, help="Entropy coefficient (higher = more exploration)")
@click.option("--vf-coef", default=0.5, help="Value function coefficient")
@click.option("--max-grad-norm", default=0.5, help="Maximum gradient norm")
@click.option("--accumulate-grads", default=1, help="Number of gradients to accumulate before update")
@click.option("--no-ui", is_flag=True, help="Disable ncurses UI")
@click.option("--resume", is_flag=True, help="Resume from latest checkpoint")
def main(
    workers: int,
    level: str,
    save_dir: str,
    n_steps: int,
    n_epochs: int,
    minibatch_size: int,
    lr: float,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    accumulate_grads: int,
    no_ui: bool,
    resume: bool,
) -> None:
    """Train Mario using APPO (Asynchronous PPO)."""
    # Parse level
    level_type = parse_level(level)

    # Create save directory
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = Path(save_dir) / f"appo_{timestamp}"

    # Check for resume
    if resume:
        checkpoints = list(Path(save_dir).glob("appo_*/weights.pt"))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            run_dir = latest.parent
            print(f"Resuming from: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "weights.pt"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("APPO (Asynchronous PPO) Training")
    print("=" * 60)
    print(f"  Workers: {workers}")
    print(f"  Level: {level}")
    print(f"  Save dir: {run_dir}")
    print(f"  N steps: {n_steps}")
    print(f"  N epochs (per worker): {n_epochs}")
    print(f"  Minibatch size: {minibatch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Gamma: {gamma}")
    print(f"  GAE lambda: {gae_lambda}")
    print(f"  Clip range: {clip_range}")
    print(f"  Entropy coef: {ent_coef}")
    print(f"  Value coef: {vf_coef}")
    print(f"  Max grad norm: {max_grad_norm}")
    print(f"  Accumulate grads: {accumulate_grads}")
    print("=" * 60)

    # Create queues
    # Gradient queue is smaller than rollout queue since gradients are ~4x smaller
    gradient_queue: Queue = Queue(maxsize=workers * 2)
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
            target=run_appo_worker,
            args=(i, weights_path, gradient_queue),
            kwargs={
                "level": level_type,
                "n_steps": n_steps,
                "n_epochs": n_epochs,
                "minibatch_size": minibatch_size,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_range": clip_range,
                "ent_coef": ent_coef,
                "vf_coef": vf_coef,
                "max_grad_norm": max_grad_norm,
                "ui_queue": ui_queue,
            },
            daemon=True,
        )
        p.start()
        worker_processes.append(p)
        print(f"Started worker {i}")

    # Start learner process
    learner_process = Process(
        target=run_appo_learner,
        args=(weights_path, run_dir, gradient_queue),
        kwargs={
            "learning_rate": lr,
            "max_grad_norm": max_grad_norm,
            "accumulate_grads": accumulate_grads,
            "ui_queue": ui_queue,
        },
        daemon=True,
    )
    learner_process.start()
    print("Started learner")

    # Handle Ctrl+C
    shutdown_initiated = False

    def signal_handler(signum, frame):
        nonlocal shutdown_initiated
        if shutdown_initiated:
            return
        shutdown_initiated = True

        print("\nShutting down...")

        # Cancel queue join threads
        gradient_queue.cancel_join_thread()
        if ui_queue is not None:
            ui_queue.cancel_join_thread()

        # Terminate all processes
        for p in worker_processes:
            if p.is_alive():
                p.terminate()
        if learner_process.is_alive():
            learner_process.terminate()
        if ui_process and ui_process.is_alive():
            ui_process.terminate()

        # Wait for processes to finish
        for p in worker_processes:
            p.join(timeout=1.0)
        learner_process.join(timeout=1.0)
        if ui_process:
            ui_process.join(timeout=1.0)

        # Close queues
        gradient_queue.close()
        if ui_queue is not None:
            ui_queue.close()

        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Wait for processes
    try:
        learner_process.join()
    except KeyboardInterrupt:
        signal_handler(None, None)

    print("Training complete!")


if __name__ == "__main__":
    main()
