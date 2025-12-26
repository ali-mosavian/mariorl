"""
Distributed DDQN Learner with async gradient updates.

Receives gradients from workers and applies them to the global network.
Similar to Gorila DQN architecture.

Architecture
============

    GRADIENT QUEUE
    (from workers)
          │
          │  ┌────────────────────────────────────────┐
          │  │           DDQN LEARNER                 │
          ▼  │                                        │
    ┌─────────────────────────────────────────┐       │
    │     Receive Gradients                   │       │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐   │       │
    │  │ W0 grad │ │ W1 grad │ │ WN grad │   │       │
    │  └────┬────┘ └────┬────┘ └────┬────┘   │       │
    │       │           │           │         │       │
    │       └───────────┼───────────┘         │       │
    │                   ▼                     │       │
    │         ┌─────────────────┐             │       │
    │         │   Accumulate    │             │       │
    │         │   (sum grads)   │             │       │
    │         └────────┬────────┘             │       │
    │                  │                      │       │
    │                  ▼                      │       │
    │         ┌─────────────────┐             │       │
    │         │    Average      │             │       │
    │         │  (÷ num_grads)  │             │       │
    │         └────────┬────────┘             │       │
    │                  │                      │       │
    │                  ▼                      │       │
    │         ┌─────────────────┐             │       │
    │         │  Clip Gradients │             │       │
    │         │  (max_grad_norm)│             │       │
    │         └────────┬────────┘             │       │
    │                  │                      │       │
    │                  ▼                      │       │
    │         ┌─────────────────┐             │       │
    │         │ Optimizer Step  │             │       │
    │         │    (AdamW)      │             │       │
    │         └────────┬────────┘             │       │
    │                  │                      │       │
    │                  ▼                      │       │
    │         ┌─────────────────┐             │       │
    │         │ Soft Update     │             │       │
    │         │ Target Network  │             │       │
    │         └────────┬────────┘             │       │
    │                  │                      │       │
    └──────────────────┼──────────────────────┘       │
                       │                              │
                       ▼                              │
              ┌─────────────────┐                     │
              │   weights.pt    │◄────────────────────┘
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
              [soft]      [soft]      [soft]
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from mario_rl.agent.ddqn_net import DoubleDQN


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class DDQNLearner:
    """
    DDQN Learner that receives gradients from workers and applies updates.

    Key differences from experience-based learner:
    - Receives gradients, not experiences
    - No local replay buffer needed
    - Just gradient accumulation and optimizer step
    - Soft updates target network
    """

    # Required fields
    weights_path: Path
    save_dir: Path
    gradient_queue: mp.Queue

    # Hyperparameters
    learning_rate: float = 2.5e-4
    lr_end: float = 1e-5
    tau: float = 0.005  # Soft update coefficient
    max_grad_norm: float = 10.0
    weight_decay: float = 1e-4
    accumulate_grads: int = 1  # Number of gradients to accumulate before update

    # Scheduling
    total_timesteps: int = 2_000_000

    # Other
    save_every: int = 100  # Save weights every N updates
    log_every: int = 10  # Log metrics every N updates
    device: Optional[str] = None
    ui_queue: Optional[mp.Queue] = None

    # Private fields
    net: Any = field(init=False, repr=False)
    optimizer: Any = field(init=False, repr=False)
    scheduler: Any = field(init=False, repr=False)
    update_count: int = field(init=False, default=0)
    total_timesteps_collected: int = field(init=False, default=0)
    worker_episodes: Dict[int, int] = field(init=False, default_factory=dict)
    weight_version: int = field(init=False, default=0)
    _metrics_csv: Path = field(init=False, repr=False)

    # Tracking
    last_loss: float = field(init=False, default=0.0)
    last_q_mean: float = field(init=False, default=0.0)
    last_td_error: float = field(init=False, default=0.0)
    grads_per_sec: float = field(init=False, default=0.0)
    gradients_received: int = field(init=False, default=0)
    _last_time: float = field(init=False, default=0.0)

    def __post_init__(self):
        """Initialize network and optimizer."""
        if self.device is None:
            self.device = best_device()

        # Create network
        state_dim = (4, 64, 64)
        action_dim = 12  # COMPLEX_MOVEMENT
        self.net = DoubleDQN(
            input_shape=state_dim,
            num_actions=action_dim,
            feature_dim=512,
            hidden_dim=256,
            dropout=0.1,
        ).to(self.device)

        self.optimizer = AdamW(
            self.net.online.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # LR scheduler
        estimated_updates = max(1, self.total_timesteps // 100)  # Rough estimate
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=estimated_updates,
            eta_min=self.lr_end,
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
        self._metrics_csv = self.save_dir / "ddqn_metrics.csv"
        if not self._metrics_csv.exists():
            with open(self._metrics_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "update",
                        "timesteps",
                        "loss",
                        "q_mean",
                        "td_error",
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
            for name, param in self.net.online.named_parameters():
                if name in grads:
                    grad = grads[name].to(self.device)
                    if param.grad is None:
                        param.grad = grad.clone()
                    else:
                        param.grad += grad

        # Average gradients
        num_packets = len(gradient_packets)
        for param in self.net.online.parameters():
            if param.grad is not None:
                param.grad /= num_packets

        # Clip gradients
        grad_norm = nn.utils.clip_grad_norm_(self.net.online.parameters(), self.max_grad_norm)

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]["lr"]

        # Soft update target network
        self.net.soft_update(self.tau)

        # Update counts
        self.update_count += 1
        self.gradients_received += num_packets
        for packet in gradient_packets:
            self.total_timesteps_collected += packet["timesteps"]
            worker_id = packet.get("worker_id", 0)
            episodes = packet.get("episodes", 0)
            self.worker_episodes[worker_id] = episodes

        # Average metrics from workers
        avg_metrics = {}
        for key in ["loss", "q_mean", "td_error"]:
            values = [p["metrics"].get(key, 0.0) for p in gradient_packets if "metrics" in p]
            avg_metrics[key] = np.mean(values) if values else 0.0

        # Update tracking
        self.last_loss = avg_metrics.get("loss", 0.0)
        self.last_q_mean = avg_metrics.get("q_mean", 0.0)
        self.last_td_error = avg_metrics.get("td_error", 0.0)

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
            "loss": self.last_loss,
            "q_mean": self.last_q_mean,
            "td_error": self.last_td_error,
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
                    self.last_loss,
                    self.last_q_mean,
                    self.last_td_error,
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

            total_episodes = sum(self.worker_episodes.values())
            msg = UIMessage(
                msg_type=MessageType.LEARNER_STATUS,
                source_id=0,
                data={
                    "step": self.update_count,
                    "timesteps": self.total_timesteps_collected,
                    "total_episodes": total_episodes,
                    "loss": self.last_loss,
                    "q_mean": self.last_q_mean,
                    "td_error": self.last_td_error,
                    "lr": lr,
                    "grads_per_sec": self.grads_per_sec,
                    "gradients_received": self.gradients_received,
                    "weight_version": self.weight_version,
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
        print(f"DDQN Learner started on {self.device}")
        print(f"  Learning rate: {self.learning_rate} → {self.lr_end}")
        print(f"  Tau: {self.tau}")
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
                    total_eps = sum(self.worker_episodes.values())
                    print(
                        f"Update {self.update_count:>6} | "
                        f"Steps: {self.total_timesteps_collected:>8,} | "
                        f"Eps: {total_eps:>5} | "
                        f"Loss: {self.last_loss:>6.3f} | "
                        f"Q: {self.last_q_mean:>6.2f} | "
                        f"TD: {self.last_td_error:>5.2f} | "
                        f"v{self.weight_version}"
                    )

            except Exception as e:
                if "Empty" not in str(type(e).__name__):
                    print(f"Learner error: {e}")
                continue

        # Final save
        self.save_weights()
        print(f"Training complete. Total updates: {self.update_count}")


def run_ddqn_learner(
    weights_path: Path,
    save_dir: Path,
    gradient_queue: mp.Queue,
    ui_queue: Optional[mp.Queue] = None,
    **kwargs,
) -> None:
    """Entry point for learner process."""
    learner = DDQNLearner(
        weights_path=weights_path,
        save_dir=save_dir,
        gradient_queue=gradient_queue,
        ui_queue=ui_queue,
        **kwargs,
    )
    learner.run()
