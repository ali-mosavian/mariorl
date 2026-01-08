"""Full Training Coordinator with gradient accumulation and scheduling.

Combines:
- Gradient polling from shared memory pool
- Gradient aggregation and application
- Learning rate scheduling (cosine annealing)
- Checkpointing (model + optimizer state)
- Target network updates
"""

from typing import Any
from pathlib import Path
from dataclasses import field
from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.optim import Adam

from mario_rl.learners.base import Learner
from mario_rl.distributed.shm_gradient_pool import SharedGradientPool
from mario_rl.training.shared_gradient_tensor import GradientPacket


@dataclass
class TrainingCoordinator:
    """Full training coordinator with gradient accumulation and scheduling.

    Coordinates the training loop:
    1. Poll gradients from worker buffers
    2. Aggregate and apply gradients
    3. Update learning rate
    4. Update target networks
    5. Save checkpoints
    """

    learner: Learner
    num_workers: int
    shm_dir: Path
    checkpoint_dir: Path

    # Training params
    learning_rate: float = 1e-4
    lr_min: float = 1e-5
    lr_decay_steps: int = 1_000_000
    weight_decay: float = 1e-4
    max_grad_norm: float = 10.0

    # Update intervals
    target_update_interval: int = 100
    checkpoint_interval: int = 10_000

    # Whether to create shm files (False = attach to existing)
    create_shm: bool = False

    # Internal state
    gradient_pool: SharedGradientPool = field(init=False, repr=False)
    optimizer: Adam = field(init=False, repr=False)
    _update_count: int = field(init=False, default=0)
    _total_steps: int = field(init=False, default=0)
    _last_checkpoint: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Initialize gradient pool, optimizer, and directories."""
        self.shm_dir = Path(self.shm_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create/attach gradient pool
        self.gradient_pool = SharedGradientPool(
            num_workers=self.num_workers,
            model=self.model,
            shm_dir=self.shm_dir,
            create=self.create_shm,
        )

        # Create optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    @property
    def model(self):
        """Access the underlying model."""
        return self.learner.model

    @property
    def update_count(self) -> int:
        """Number of gradient updates performed."""
        return self._update_count

    def poll_gradients(self) -> list[GradientPacket]:
        """Poll all available gradients from workers.

        Returns:
            List of gradient packets
        """
        return self.gradient_pool.read_all()

    def aggregate_gradients(
        self,
        gradients_list: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        """Average multiple gradient batches.

        Args:
            gradients_list: List of gradient dicts

        Returns:
            Averaged gradient dict
        """
        if not gradients_list:
            return {}

        averaged: dict[str, Tensor] = {}
        for name in gradients_list[0]:
            stacked = torch.stack([g[name] for g in gradients_list])
            averaged[name] = stacked.mean(dim=0)

        return averaged

    def apply_gradients(self, gradients_list: list[dict[str, Tensor]]) -> None:
        """Apply gradients to model.

        Args:
            gradients_list: List of gradient dicts to average and apply
        """
        if not gradients_list:
            return

        # Aggregate
        averaged = self.aggregate_gradients(gradients_list)

        # Zero existing grads
        self.optimizer.zero_grad()

        # Apply averaged gradients (move to correct device)
        for name, param in self.model.named_parameters():
            if name in averaged:
                param.grad = averaged[name].to(param.device)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm,
        )

        # Optimizer step
        self.optimizer.step()
        self._update_count += 1

    def current_lr(self) -> float:
        """Get current learning rate based on schedule.

        Returns:
            Current learning rate
        """
        if self._total_steps >= self.lr_decay_steps:
            return self.lr_min

        # Cosine annealing
        progress = self._total_steps / self.lr_decay_steps
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.lr_min + (self.learning_rate - self.lr_min) * cosine

    def update_lr(self) -> None:
        """Update optimizer learning rate."""
        lr = self.current_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def maybe_update_targets(self) -> None:
        """Update target networks if at interval."""
        if self._update_count > 0 and self._update_count % self.target_update_interval == 0:
            self.learner.update_targets()

    def save_checkpoint(self) -> Path:
        """Save checkpoint to disk.

        Returns:
            Path to checkpoint file
        """
        ckpt_path = self.checkpoint_dir / f"checkpoint_{self._update_count}.pt"

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "update_count": self._update_count,
                "total_steps": self._total_steps,
            },
            ckpt_path,
        )

        self._last_checkpoint = self._update_count
        return ckpt_path

    def load_latest_checkpoint(self) -> bool:
        """Load most recent checkpoint.

        Returns:
            True if checkpoint was loaded
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
            reverse=True,
        )

        if not checkpoints:
            return False

        ckpt = torch.load(checkpoints[0], weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._update_count = ckpt["update_count"]
        self._total_steps = ckpt["total_steps"]

        return True

    def save_weights(self) -> Path:
        """Save current weights for workers to sync.

        Returns:
            Path to weights file
        """
        weights_path = self.checkpoint_dir / "weights.pt"
        torch.save(self.model.state_dict(), weights_path)
        return weights_path

    def training_step(self) -> dict[str, Any]:
        """Run one training step.

        Polls gradients, aggregates, applies, and updates LR/targets.

        Returns:
            Step info dict
        """
        # Poll gradients
        packets = self.poll_gradients()

        if not packets:
            return {
                "update_count": self._update_count,
                "total_steps": self._total_steps,
                "gradients_processed": 0,
            }

        # Extract gradients and sum timesteps
        grads_list = [p.grads for p in packets]
        timesteps = sum(p.timesteps for p in packets)

        # Apply gradients
        self.apply_gradients(grads_list)

        # Update tracking
        self._total_steps += timesteps

        # Update LR
        self.update_lr()

        # Maybe update targets
        self.maybe_update_targets()

        # Maybe checkpoint
        if self._update_count - self._last_checkpoint >= self.checkpoint_interval:
            self.save_checkpoint()

        # Save weights for workers
        self.save_weights()

        return {
            "update_count": self._update_count,
            "total_steps": self._total_steps,
            "gradients_processed": len(packets),
            "lr": self.current_lr(),
        }

    def close(self) -> None:
        """Release resources."""
        if hasattr(self, "gradient_pool"):
            self.gradient_pool.close()
