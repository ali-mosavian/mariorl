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
import sys
import csv
import time
import signal
import traceback
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
from mario_rl.training.shared_gradient_tensor import SharedGradientTensorPool
from mario_rl.training.shared_gradient_tensor import GradientPacket


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
    shm_dir: Path  # Directory containing SharedGradientTensor files
    num_workers: int = 1  # Number of workers (for creating tensor pool)
    
    # Gradient pool - created in __post_init__ (after network)
    gradient_pool: Optional[SharedGradientTensorPool] = field(init=False, default=None)

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
    last_q_max: float = field(init=False, default=0.0)
    last_td_error: float = field(init=False, default=0.0)
    last_grad_norm: float = field(init=False, default=0.0)
    last_num_packets: int = field(init=False, default=0)
    grads_per_sec: float = field(init=False, default=0.0)
    gradients_received: int = field(init=False, default=0)
    _last_time: float = field(init=False, default=0.0)
    _resumed_from_checkpoint: bool = field(init=False, default=False)

    # Aggregated worker metrics
    _worker_avg_rewards: Dict[int, float] = field(init=False, default_factory=dict)
    _worker_avg_speeds: Dict[int, float] = field(init=False, default_factory=dict)
    _worker_avg_time_to_flag: Dict[int, float] = field(init=False, default_factory=dict)
    _worker_deaths: Dict[int, int] = field(init=False, default_factory=dict)
    _worker_flags: Dict[int, int] = field(init=False, default_factory=dict)
    _worker_best_x: Dict[int, int] = field(init=False, default_factory=dict)
    _worker_entropy: Dict[int, float] = field(init=False, default_factory=dict)

    # Snapshot settings
    snapshot_interval: int = 500  # Save full snapshot every N updates
    _snapshot_path: Path = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize network and optimizer."""
        if self.device is None:
            self.device = best_device()

        # Create network first (needed for gradient tensor pool)
        state_dim = (4, 64, 64)
        action_dim = 12  # COMPLEX_MOVEMENT
        self.net = DoubleDQN(
            input_shape=state_dim,
            num_actions=action_dim,
            feature_dim=512,
            hidden_dim=256,
            dropout=0.1,
        ).to(self.device)

        # Create gradient pool by attaching to existing shared memory files
        # (files were created by main process, we just attach to them)
        self.gradient_pool = SharedGradientTensorPool(
            num_workers=self.num_workers,
            model=self.net.online,
            shm_dir=self.shm_dir,
            num_slots=8,  # 8 slots per worker for good async buffering
            create=False,  # Attach to existing files
        )

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
        self._weights_loaded_version: int | None = None
        if self.weights_path.exists():
            try:
                checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=True)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self.net.load_state_dict(checkpoint["state_dict"])
                    self.weight_version = checkpoint.get("version", 0)
                    self._weights_loaded_version = self.weight_version
                else:
                    self.net.load_state_dict(checkpoint)
                # Log will be sent in run() after _log is available
                self._resumed_from_checkpoint = True
            except Exception:
                # Will create new weights
                self._resumed_from_checkpoint = False
                self.save_weights()
        else:
            self.save_weights()

        # Initialize snapshot path
        self._snapshot_path = self.save_dir / "snapshot.pt"

        # Initialize CSV logging - NEVER truncate existing data
        self._metrics_csv = self.save_dir / "ddqn_metrics.csv"
        csv_headers = [
            "timestamp",
            "update",
            "timesteps",
            "total_episodes",
            "loss",
            "q_mean",
            "q_max",
            "td_error",
            "grad_norm",
            "lr",
            "grads_per_sec",
            "gradients_received",
            "weight_version",
            "num_packets",
            "avg_reward",
            "avg_speed",
            "avg_time_to_flag",
            "avg_entropy",
            "total_deaths",
            "total_flags",
            "global_best_x",
        ]
        if not self._metrics_csv.exists():
            with open(self._metrics_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(csv_headers)
        else:
            # Check if file has data (not just headers) - warn if resuming with empty CSV
            csv_size = self._metrics_csv.stat().st_size
            self._csv_had_data = csv_size > 200  # Headers are ~150 bytes

    def save_weights(self) -> None:
        """Save network weights with version for workers to sync."""
        try:
            self.weight_version += 1
            checkpoint = {
                "state_dict": self.net.state_dict(),
                "version": self.weight_version,
            }
            torch.save(checkpoint, self.weights_path)
        except Exception as e:
            self._log(f"ERROR saving weights: {e}")
            self.weight_version -= 1  # Rollback version on failure

    def save_snapshot(self) -> None:
        """
        Save full training snapshot for resuming training.

        Includes: network, optimizer, scheduler, training state.
        """
        try:
            snapshot = {
                "net_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "update_count": self.update_count,
                "total_timesteps_collected": self.total_timesteps_collected,
                "weight_version": self.weight_version,
                "gradients_received": self.gradients_received,
                "timestamp": time.time(),
            }
            # Atomic write to prevent corruption
            tmp_path = self._snapshot_path.with_suffix(".pt.tmp")
            torch.save(snapshot, tmp_path)
            tmp_path.rename(self._snapshot_path)
            self._log(f"Snapshot saved: step={self.update_count}, timesteps={self.total_timesteps_collected:,}")
        except Exception as e:
            self._log(f"ERROR saving snapshot: {e}")

    def restore_snapshot(self, snapshot_path: Path | None = None) -> bool:
        """
        Restore training from a snapshot.

        Args:
            snapshot_path: Path to snapshot file. Uses default if None.

        Returns:
            True if restoration succeeded, False otherwise.
        """
        path = snapshot_path or self._snapshot_path
        if not path.exists():
            return False

        try:
            snapshot = torch.load(path, map_location=self.device, weights_only=False)

            # Restore network
            self.net.load_state_dict(snapshot["net_state_dict"])

            # Restore optimizer
            self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])

            # Restore scheduler
            self.scheduler.load_state_dict(snapshot["scheduler_state_dict"])

            # Restore training state
            self.update_count = snapshot["update_count"]
            self.total_timesteps_collected = snapshot["total_timesteps_collected"]
            self.weight_version = snapshot["weight_version"]
            self.gradients_received = snapshot.get("gradients_received", 0)

            self._log(f"Restored from snapshot: step={self.update_count}, v{self.weight_version}, timesteps={self.total_timesteps_collected:,}")

            # Save weights for workers to sync
            self.save_weights()
            return True

        except Exception as e:
            self._log(f"Failed to restore snapshot: {e}")
            return False

    def apply_gradients(self, gradient_packets: List[GradientPacket]) -> Dict[str, float]:
        """
        Apply accumulated gradients to the network.

        Args:
            gradient_packets: List of GradientPacket instances from workers

        Returns:
            Dictionary of metrics
        """
        if not gradient_packets:
            return {}

        # Zero gradients
        self.optimizer.zero_grad()

        # Accumulate gradients from all workers (zero-copy from shared memory)
        for packet in gradient_packets:
            for name, param in self.net.online.named_parameters():
                if name in packet.grads:
                    grad = packet.grads[name].to(self.device)
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
            self.total_timesteps_collected += packet.timesteps
            self.worker_episodes[packet.worker_id] = packet.episodes
            # Update worker-specific metrics for aggregation
            self._worker_avg_rewards[packet.worker_id] = packet.avg_reward
            self._worker_avg_speeds[packet.worker_id] = packet.avg_speed
            self._worker_entropy[packet.worker_id] = packet.entropy
            self._worker_deaths[packet.worker_id] = packet.deaths
            self._worker_flags[packet.worker_id] = packet.flags
            self._worker_best_x[packet.worker_id] = packet.best_x

        # Average worker-computed metrics from gradient packets
        if gradient_packets:
            self.last_loss = sum(p.loss for p in gradient_packets) / len(gradient_packets)
            self.last_q_mean = sum(p.q_mean for p in gradient_packets) / len(gradient_packets)
            self.last_td_error = sum(p.td_error for p in gradient_packets) / len(gradient_packets)

        self.last_grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        self.last_num_packets = num_packets

        # Calculate speed
        now = time.time()
        elapsed = now - self._last_time
        self.grads_per_sec = num_packets / elapsed if elapsed > 0 else 0
        self._last_time = now

        # Save weights periodically
        if self.update_count % self.save_every == 0:
            self.save_weights()

        # Save full snapshot periodically
        if self.update_count % self.snapshot_interval == 0:
            self.save_snapshot()

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
        try:
            total_episodes = sum(self.worker_episodes.values())
            avg_reward = np.mean(list(self._worker_avg_rewards.values())) if self._worker_avg_rewards else 0.0
            avg_speed = np.mean(list(self._worker_avg_speeds.values())) if self._worker_avg_speeds else 0.0
            # Only average non-zero values for time to flag (0 means no flags captured yet)
            flag_times = [t for t in self._worker_avg_time_to_flag.values() if t > 0]
            avg_time_to_flag = np.mean(flag_times) if flag_times else 0.0
            avg_entropy = np.mean(list(self._worker_entropy.values())) if self._worker_entropy else 0.0
            total_deaths = sum(self._worker_deaths.values())
            total_flags = sum(self._worker_flags.values())
            global_best_x = max(self._worker_best_x.values()) if self._worker_best_x else 0

            with open(self._metrics_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.time(),
                    self.update_count,
                    self.total_timesteps_collected,
                    total_episodes,
                    self.last_loss,
                    self.last_q_mean,
                    self.last_q_max,
                    self.last_td_error,
                    self.last_grad_norm,
                    lr,
                    self.grads_per_sec,
                    self.gradients_received,
                    self.weight_version,
                    self.last_num_packets,
                    avg_reward,
                    avg_speed,
                    avg_time_to_flag,
                    avg_entropy,
                    total_deaths,
                    total_flags,
                    global_best_x,
                ])
        except Exception as e:
            self._log(f"ERROR logging metrics: {e}")

    def _send_ui_status(self, lr: float) -> None:
        """Send status to UI queue."""
        if self.ui_queue is None:
            return

        try:
            from mario_rl.training.training_ui import UIMessage
            from mario_rl.training.training_ui import MessageType

            total_episodes = sum(self.worker_episodes.values())

            # Compute aggregated metrics from all workers
            avg_reward = np.mean(list(self._worker_avg_rewards.values())) if self._worker_avg_rewards else 0.0
            avg_speed = np.mean(list(self._worker_avg_speeds.values())) if self._worker_avg_speeds else 0.0
            flag_times = [t for t in self._worker_avg_time_to_flag.values() if t > 0]
            avg_time_to_flag = np.mean(flag_times) if flag_times else 0.0
            avg_entropy = np.mean(list(self._worker_entropy.values())) if self._worker_entropy else 0.0
            total_deaths = sum(self._worker_deaths.values())
            total_flags = sum(self._worker_flags.values())
            global_best_x = max(self._worker_best_x.values()) if self._worker_best_x else 0

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
                    # Aggregated worker metrics for graphs
                    "avg_reward": avg_reward,
                    "avg_speed": avg_speed,
                    "avg_time_to_flag": avg_time_to_flag,
                    "avg_entropy": avg_entropy,
                    "total_deaths": total_deaths,
                    "total_flags": total_flags,
                    "global_best_x": global_best_x,
                },
            )
            self.ui_queue.put_nowait(msg)
        except Exception:
            pass

    def _log(self, text: str) -> None:
        """Log message to UI queue or stdout."""
        if self.ui_queue is not None:
            try:
                from mario_rl.training.training_ui import UIMessage
                from mario_rl.training.training_ui import MessageType

                msg = UIMessage(
                    msg_type=MessageType.LEARNER_LOG,
                    source_id=-1,
                    data={"text": text},
                )
                self.ui_queue.put_nowait(msg)
            except Exception:
                pass
        else:
            print(text)

    def run(self, max_updates: int = -1) -> None:
        """
        Main learner loop.

        Args:
            max_updates: Maximum number of updates (-1 for unlimited)
        """
        self._log(f"DDQN Learner started on {self.device}")
        self._log(f"  LR: {self.learning_rate} → {self.lr_end}, Tau: {self.tau}")
        self._log(f"  Accumulate grads: {self.accumulate_grads}")
        self._log(f"  Gradient mode: shared_memory (non-blocking)")
        if self._weights_loaded_version is not None:
            self._log(f"  Loaded weights.pt v{self._weights_loaded_version}")
        self._log(f"  Current state: v{self.weight_version}, update={self.update_count}, timesteps={self.total_timesteps_collected:,}")
        if hasattr(self, "_csv_had_data") and not self._csv_had_data:
            self._log(f"  WARNING: CSV file exists but appears empty - previous data may have been lost")

        while max_updates < 0 or self.update_count < max_updates:
            try:
                # Collect gradients from shared memory (NON-BLOCKING, can't deadlock!)
                gradient_packets = self._collect_gradients()

                if not gradient_packets:
                    # No gradients available, wait briefly and retry
                    time.sleep(0.01)
                    continue

                # Apply gradients (will average if multiple)
                self.apply_gradients(gradient_packets)

                # Enhanced logging every 10 updates
                if self.update_count % 10 == 0:
                    self._log_status()

            except Exception as e:
                self._log(f"Learner error: {e}")
                continue

        # Final save
        self.save_weights()

    def _collect_gradients(self) -> list[GradientPacket]:
        """
        Collect gradients from shared memory pool (NON-BLOCKING).
        
        This method polls all worker buffers and collects any ready packets.
        It CANNOT deadlock because there's no blocking I/O.
        """
        gradient_packets: list[GradientPacket] = []
        
        # Wait briefly for gradients to accumulate
        wait_start = time.time()
        while len(gradient_packets) < self.accumulate_grads:
            # Poll all worker buffers (non-blocking, zero-copy)
            new_packets = self.gradient_pool.read_all_available()
            gradient_packets.extend(new_packets)
            
            # Timeout after 5 seconds of waiting
            if time.time() - wait_start > 5.0:
                break
            
            # Brief sleep to avoid busy-waiting
            if not new_packets:
                time.sleep(0.001)
        
        return gradient_packets

    def _log_status(self) -> None:
        """Log current training status."""
        current_lr = self.optimizer.param_groups[0]["lr"]
        ready_count = self.gradient_pool.count_total_ready()
        
        self._log(
            f"Update {self.update_count}: "
            f"LR={current_lr:.6f}, "
            f"loss={self.last_loss:.4f}, "
            f"q_mean={self.last_q_mean:.2f}, "
            f"grads_received={self.gradients_received}, "
            f"timesteps={self.total_timesteps_collected:,}, "
            f"shm_ready={ready_count}"
        )
        self._log(f"Training complete. Total updates: {self.update_count}")


def run_ddqn_learner(
    weights_path: Path,
    save_dir: Path,
    shm_dir: Path,
    num_workers: int,
    ui_queue: Optional[mp.Queue] = None,
    restore_snapshot: bool = False,
    snapshot_path: Optional[Path] = None,
    **kwargs,
) -> None:
    """Entry point for learner process."""
    from datetime import datetime
    
    # Set up crash log directory
    crash_log_dir = save_dir / "crash_logs"
    crash_log_dir.mkdir(parents=True, exist_ok=True)
    stack_trace_file = crash_log_dir / "learner_stack.log"
    
    def dump_stack_trace(signum, frame):
        """Dump stack trace when receiving SIGUSR1."""
        try:
            with open(stack_trace_file, "a") as f:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"Stack trace dump at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"LEARNER (PID: {os.getpid()})\n")
                f.write(f"Signal: {signum}\n")
                f.write(f"{'=' * 80}\n\n")
                
                # Dump all thread stacks
                import threading
                for thread_id, frame_obj in sys._current_frames().items():
                    thread_name = "Unknown"
                    for t in threading.enumerate():
                        if t.ident == thread_id:
                            thread_name = t.name
                            break
                    f.write(f"\nThread {thread_id} ({thread_name}):\n")
                    f.write(''.join(traceback.format_stack(frame_obj)))
                
                f.write(f"\n{'=' * 80}\n")
                f.flush()
        except Exception as e:
            print(f"[LEARNER] Failed to dump stack trace: {e}", file=sys.stderr, flush=True)
    
    # Register signal handler (SIGUSR1)
    signal.signal(signal.SIGUSR1, dump_stack_trace)
    
    try:
        learner = DDQNLearner(
            weights_path=weights_path,
            save_dir=save_dir,
            shm_dir=shm_dir,
            num_workers=num_workers,
            ui_queue=ui_queue,
            **kwargs,
        )

        # Optionally restore from snapshot
        if restore_snapshot:
            learner.restore_snapshot(snapshot_path)

        learner.run()
    except Exception as e:
        # Log crash to file
        crash_log_path = crash_log_dir / "learner_crash.log"
        with open(crash_log_path, "w") as f:
            f.write(f"LEARNER CRASHED\n")
            f.write(f"Error: {e}\n")
            f.write(f"Type: {type(e).__name__}\n")
            f.write(f"\nFull traceback:\n")
            f.write(traceback.format_exc())
        print(f"[LEARNER] CRASHED: {e}", file=sys.stderr, flush=True)
        raise
