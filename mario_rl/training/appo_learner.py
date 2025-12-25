"""
APPO (Asynchronous PPO) Learner for distributed training.

Receives gradients from workers and applies them to the global network.
This is more efficient than receiving full rollouts because gradients are smaller.

Architecture
============

    GRADIENT QUEUE
    (from workers)
          │
          │  ┌────────────────────────────────────┐
          │  │         APPO LEARNER               │
          ▼  │                                    │
    ┌─────────────────────────────────────────┐   │
    │     Receive Gradients                   │   │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
    │  │ W0 grad │ │ W1 grad │ │ WN grad │   │   │
    │  └────┬────┘ └────┬────┘ └────┬────┘   │   │
    │       │           │           │         │   │
    │       └───────────┼───────────┘         │   │
    │                   ▼                     │   │
    │         ┌─────────────────┐             │   │
    │         │   Accumulate    │             │   │
    │         │   (sum grads)   │             │   │
    │         └────────┬────────┘             │   │
    │                  │                      │   │
    │                  ▼                      │   │
    │         ┌─────────────────┐             │   │
    │         │    Average      │             │   │
    │         │  (÷ num_grads)  │             │   │
    │         └────────┬────────┘             │   │
    │                  │                      │   │
    │                  ▼                      │   │
    │         ┌─────────────────┐             │   │
    │         │  Clip Gradients │             │   │
    │         │  (max_grad_norm)│             │   │
    │         └────────┬────────┘             │   │
    │                  │                      │   │
    │                  ▼                      │   │
    │         ┌─────────────────┐             │   │
    │         │ Optimizer Step  │             │   │
    │         │    (AdamW)      │             │   │
    │         └────────┬────────┘             │   │
    │                  │                      │   │
    └──────────────────┼──────────────────────┘   │
                       │                          │
                       ▼                          │
              ┌─────────────────┐                 │
              │   weights.pt    │◄────────────────┘
              │  (versioned)    │
              └────────┬────────┘
                       │
                       │  Workers read
                       │  periodically
                       ▼
              ┌─────────────────┐
              │    WORKERS      │
              │ (sync weights)  │
              └─────────────────┘

Gradient Accumulation
=====================

Instead of applying each gradient immediately (noisy), we can:
1. Collect N gradients from workers
2. Average them
3. Apply single update

This reduces noise while maintaining async benefits.

    Time ─────────────────────────────────────────────────▶

    W0: ──[grad]────────[grad]────────[grad]────────
    W1: ────[grad]────────[grad]────────[grad]──────
    W2: ──────[grad]────────[grad]────────[grad]────

    Learner:    ▼           ▼           ▼
              [acc]       [acc]       [acc]
              [avg]       [avg]       [avg]
              [step]      [step]      [step]
              [save]      [save]      [save]
"""

import os
import csv
import time
import multiprocessing as mp

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from typing import Any
from typing import Dict
from typing import List
from pathlib import Path
from typing import Optional
from dataclasses import field
from dataclasses import dataclass

import torch
import numpy as np
from torch import nn

from mario_rl.agent.ppo_net import ActorCritic


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class APPOLearner:
    """
    APPO Learner that receives gradients from workers and applies updates.

    Key differences from PPO Learner:
    - Receives gradients instead of rollouts
    - No forward/backward pass needed
    - Just gradient accumulation and optimizer step
    """

    # Required fields
    weights_path: Path
    save_dir: Path
    gradient_queue: mp.Queue

    # Hyperparameters
    learning_rate: float = 2.5e-4
    max_grad_norm: float = 0.5
    accumulate_grads: int = 1  # Number of gradients to accumulate before update

    # Scheduling
    lr_schedule: str = "linear"  # "linear", "constant"
    total_timesteps: int = 10_000_000

    # Other
    save_every: int = 10  # Save weights every N updates
    log_every: int = 10  # Log metrics every N updates
    device: Optional[str] = None
    ui_queue: Optional[mp.Queue] = None

    # Private fields
    net: Any = field(init=False, repr=False)
    optimizer: Any = field(init=False, repr=False)
    update_count: int = field(init=False, default=0)
    total_timesteps_collected: int = field(init=False, default=0)
    weight_version: int = field(init=False, default=0)
    _metrics_csv: Path = field(init=False, repr=False)

    # Tracking
    last_policy_loss: float = field(init=False, default=0.0)
    last_value_loss: float = field(init=False, default=0.0)
    last_entropy: float = field(init=False, default=0.0)
    last_kl: float = field(init=False, default=0.0)
    last_clip_fraction: float = field(init=False, default=0.0)
    grads_per_sec: float = field(init=False, default=0.0)
    _last_time: float = field(init=False, default=0.0)

    def __post_init__(self):
        """Initialize network and optimizer."""
        if self.device is None:
            self.device = best_device()

        # Create network
        state_dim = (4, 64, 64)
        action_dim = 12  # COMPLEX_MOVEMENT
        self.net = ActorCritic(
            input_shape=state_dim,
            num_actions=action_dim,
            feature_dim=512,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.learning_rate,
            eps=1e-5,
        )

        # Initialize tracking
        self.update_count = 0
        self.total_timesteps_collected = 0
        self.weight_version = 0
        self._last_time = time.time()

        # Load existing weights or save initial
        if self.weights_path.exists():
            try:
                checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=True)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self.net.load_state_dict(checkpoint["state_dict"])
                    self.weight_version = checkpoint.get("version", 0)
                else:
                    self.net.load_state_dict(checkpoint)
                print(f"Resumed from checkpoint: {self.weights_path} (v{self.weight_version})")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                self.save_weights()
        else:
            self.save_weights()

        # Initialize CSV logging
        self._metrics_csv = self.save_dir / "appo_metrics.csv"
        if not self._metrics_csv.exists():
            with open(self._metrics_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "update",
                        "timesteps",
                        "policy_loss",
                        "value_loss",
                        "entropy",
                        "kl",
                        "clip_fraction",
                        "lr",
                        "grads_per_sec",
                    ]
                )

    def save_weights(self) -> None:
        """Save network weights with version for workers to sync."""
        self.weight_version += 1
        checkpoint = {
            "state_dict": self.net.state_dict(),
            "version": self.weight_version,
        }
        torch.save(checkpoint, self.weights_path)

    def get_scheduled_lr(self) -> float:
        """Get learning rate based on schedule."""
        if self.lr_schedule == "constant":
            return self.learning_rate

        # Linear decay
        progress = min(1.0, self.total_timesteps_collected / self.total_timesteps)
        return self.learning_rate * (1.0 - progress)

    def update_lr(self) -> None:
        """Update learning rate in optimizer."""
        lr = self.get_scheduled_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def apply_gradients(self, gradient_packets: List[Dict]) -> Dict[str, float]:
        """
        Apply accumulated gradients to the network.

        Args:
            gradient_packets: List of gradient packets from workers

        Returns:
            Dictionary of metrics
        """
        if not gradient_packets:
            return {}

        # Zero gradients
        self.optimizer.zero_grad()

        # Accumulate gradients from all workers
        for packet in gradient_packets:
            grads = packet["grads"]
            for name, param in self.net.named_parameters():
                if name in grads:
                    grad = grads[name].to(self.device)
                    if param.grad is None:
                        param.grad = grad.clone()
                    else:
                        param.grad += grad

        # Average gradients
        num_packets = len(gradient_packets)
        for param in self.net.parameters():
            if param.grad is not None:
                param.grad /= num_packets

        # Clip gradients
        grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)

        # Update learning rate
        self.update_lr()
        current_lr = self.optimizer.param_groups[0]["lr"]

        # Optimizer step
        self.optimizer.step()

        # Update counts
        self.update_count += 1
        for packet in gradient_packets:
            self.total_timesteps_collected += packet["timesteps"]

        # Average metrics from workers
        avg_metrics = {}
        for key in ["policy_loss", "value_loss", "entropy", "kl", "clip_fraction"]:
            values = [p["metrics"].get(key, 0.0) for p in gradient_packets if "metrics" in p]
            avg_metrics[key] = np.mean(values) if values else 0.0

        # Update tracking
        self.last_policy_loss = avg_metrics["policy_loss"]
        self.last_value_loss = avg_metrics["value_loss"]
        self.last_entropy = avg_metrics["entropy"]
        self.last_kl = avg_metrics["kl"]
        self.last_clip_fraction = avg_metrics["clip_fraction"]

        # Calculate speed
        now = time.time()
        elapsed = now - self._last_time
        self.grads_per_sec = num_packets / elapsed if elapsed > 0 else 0
        self._last_time = now

        # Save weights periodically
        if self.update_count % self.save_every == 0:
            self.save_weights()

        # Log metrics periodically
        if self.update_count % self.log_every == 0:
            self._log_metrics(current_lr)

        # Send UI update
        self._send_ui_status(current_lr)

        return {
            "policy_loss": self.last_policy_loss,
            "value_loss": self.last_value_loss,
            "entropy": self.last_entropy,
            "kl": self.last_kl,
            "clip_fraction": self.last_clip_fraction,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": current_lr,
        }

    def _log_metrics(self, lr: float) -> None:
        """Log metrics to CSV."""
        with open(self._metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    time.time(),
                    self.update_count,
                    self.total_timesteps_collected,
                    self.last_policy_loss,
                    self.last_value_loss,
                    self.last_entropy,
                    self.last_kl,
                    self.last_clip_fraction,
                    lr,
                    self.grads_per_sec,
                ]
            )

    def _send_ui_status(self, lr: float) -> None:
        """Send status to UI queue."""
        if self.ui_queue is None:
            return

        try:
            from mario_rl.training.training_ui import UIMessage
            from mario_rl.training.training_ui import MessageType

            msg = UIMessage(
                msg_type=MessageType.PPO_STATUS,
                source_id=0,
                data={
                    "step": self.update_count,
                    "timesteps": self.total_timesteps_collected,
                    "policy_loss": self.last_policy_loss,
                    "value_loss": self.last_value_loss,
                    "entropy": self.last_entropy,
                    "clip_fraction": self.last_clip_fraction,
                    "kl": self.last_kl,
                    "lr": lr,
                    "ent_coef": 0.01,  # Fixed for now
                    "steps_per_sec": self.grads_per_sec,
                    "elapsed_time": time.time() - self._last_time,
                },
            )
            self.ui_queue.put_nowait(msg)
        except Exception:
            pass

    def run(self, max_updates: int = -1) -> None:
        """
        Main learner loop.

        Args:
            max_updates: Maximum number of updates (-1 for unlimited)
        """
        print(f"APPO Learner started on {self.device}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Accumulate grads: {self.accumulate_grads}")
        print(f"  Max grad norm: {self.max_grad_norm}")

        while max_updates < 0 or self.update_count < max_updates:
            # Collect gradients from workers
            gradient_packets = []

            try:
                # Wait for first gradient
                packet = self.gradient_queue.get(timeout=5.0)
                gradient_packets.append(packet)

                # Collect additional gradients if accumulating
                while len(gradient_packets) < self.accumulate_grads:
                    try:
                        packet = self.gradient_queue.get_nowait()
                        gradient_packets.append(packet)
                    except Exception:
                        break

                # Apply gradients
                self.apply_gradients(gradient_packets)

                # Log progress
                if self.update_count % self.log_every == 0:
                    print(
                        f"Update {self.update_count} | "
                        f"Steps: {self.total_timesteps_collected:,} | "
                        f"π_loss: {self.last_policy_loss:.4f} | "
                        f"v_loss: {self.last_value_loss:.4f} | "
                        f"H: {self.last_entropy:.4f} | "
                        f"KL: {self.last_kl:.4f} | "
                        f"v{self.weight_version}"
                    )

            except Exception as e:
                if "Empty" not in str(type(e).__name__):
                    print(f"Learner error: {e}")
                continue

        # Final save
        self.save_weights()
        print(f"Training complete. Total updates: {self.update_count}")


def run_appo_learner(
    weights_path: Path,
    save_dir: Path,
    gradient_queue: mp.Queue,
    ui_queue: Optional[mp.Queue] = None,
    **kwargs,
) -> None:
    """Entry point for learner process."""
    learner = APPOLearner(
        weights_path=weights_path,
        save_dir=save_dir,
        gradient_queue=gradient_queue,
        ui_queue=ui_queue,
        **kwargs,
    )
    learner.run()
