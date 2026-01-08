#!/usr/bin/env python3
"""Modular distributed training script supporting multiple models.

This script demonstrates the new modular architecture with:
- Model/Learner protocols
- Generic Worker/Coordinator
- Support for DDQN and Dreamer models

Usage:
    uv run python scripts/train_modular.py --model ddqn --workers 4
    uv run python scripts/train_modular.py --model dreamer --workers 2
"""

from __future__ import annotations

import os
import click
import torch
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from mario_rl.models import DoubleDQN
from mario_rl.models import DreamerModel
from mario_rl.learners import DDQNLearner
from mario_rl.learners import DreamerLearner
from mario_rl.distributed import Worker
from mario_rl.distributed import Coordinator


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training."""

    model_type: str
    num_workers: int
    input_shape: tuple[int, int, int]
    num_actions: int
    lr: float
    gamma: float
    batch_size: int
    imagination_horizon: int = 15  # For Dreamer


def create_model_and_learner(config: TrainingConfig):
    """Create model and learner based on config.

    Returns:
        (model, learner) tuple
    """
    match config.model_type:
        case "ddqn":
            model = DoubleDQN(
                input_shape=config.input_shape,
                num_actions=config.num_actions,
            )
            learner = DDQNLearner(model=model, gamma=config.gamma)
        case "dreamer":
            model = DreamerModel(
                input_shape=config.input_shape,
                num_actions=config.num_actions,
            )
            learner = DreamerLearner(
                model=model,
                gamma=config.gamma,
                imagination_horizon=config.imagination_horizon,
            )
        case _:
            raise ValueError(f"Unknown model type: {config.model_type}")

    return model, learner


def create_worker(config: TrainingConfig) -> Worker:
    """Create a worker for distributed training."""
    _, learner = create_model_and_learner(config)
    return Worker(learner=learner)


def create_coordinator(config: TrainingConfig) -> Coordinator:
    """Create a coordinator for distributed training."""
    _, learner = create_model_and_learner(config)
    return Coordinator(learner=learner, lr=config.lr)


@click.command()
@click.option(
    "--model",
    type=click.Choice(["ddqn", "dreamer"]),
    default="ddqn",
    help="Model type to train",
)
@click.option(
    "--workers",
    default=os.cpu_count() or 4,
    help="Number of parallel workers",
)
@click.option(
    "--lr",
    default=0.0001,
    help="Learning rate",
)
@click.option(
    "--gamma",
    default=0.99,
    help="Discount factor",
)
@click.option(
    "--batch-size",
    default=32,
    help="Batch size",
)
@click.option(
    "--horizon",
    default=15,
    help="Imagination horizon for Dreamer",
)
def main(
    model: str,
    workers: int,
    lr: float,
    gamma: float,
    batch_size: int,
    horizon: int,
) -> None:
    """Train a model using the modular distributed architecture."""
    config = TrainingConfig(
        model_type=model,
        num_workers=workers,
        input_shape=(4, 64, 64),
        num_actions=12,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        imagination_horizon=horizon,
    )

    print(f"Training {config.model_type.upper()} with {config.num_workers} workers")
    print(f"  Learning rate: {config.lr}")
    print(f"  Gamma: {config.gamma}")
    print(f"  Batch size: {config.batch_size}")
    if config.model_type == "dreamer":
        print(f"  Imagination horizon: {config.imagination_horizon}")

    # Create coordinator (central learner)
    coordinator = create_coordinator(config)
    print(f"\nCoordinator created with {sum(p.numel() for p in coordinator.model.parameters())} parameters")

    # Demo: create a worker and show gradient computation
    worker = create_worker(config)
    print(f"Worker created with {sum(p.numel() for p in worker.model.parameters())} parameters")

    # Sync worker weights from coordinator
    worker.apply_weights(coordinator.get_weights())

    # Demo batch
    demo_batch = {
        "states": torch.randn(config.batch_size, *config.input_shape),
        "actions": torch.randint(0, config.num_actions, (config.batch_size,)),
        "rewards": torch.randn(config.batch_size),
        "next_states": torch.randn(config.batch_size, *config.input_shape),
        "dones": torch.zeros(config.batch_size),
    }

    # Compute gradients
    grads, metrics = worker.compute_gradients(**demo_batch)
    print(f"\nGradient computation:")
    print(f"  Num gradient tensors: {len(grads)}")
    print(f"  Total gradient elements: {sum(g.numel() for g in grads.values())}")
    print(f"  Metrics: {metrics}")

    # Training step
    new_weights = coordinator.training_step([grads])
    print(f"\nTraining step completed")
    print(f"  New weights count: {len(new_weights)}")

    # Target update (for DDQN)
    coordinator.update_targets(tau=0.005)
    print(f"  Target update completed")

    print("\n✓ Modular training infrastructure working correctly!")


if __name__ == "__main__":
    main()
