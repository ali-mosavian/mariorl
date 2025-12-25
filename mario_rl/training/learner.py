"""
Learner process for distributed training.

Pulls experiences from shared buffer, trains network, saves weights for workers.
"""

import os
import sys
import time
import multiprocessing as mp

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from typing import Any
from pathlib import Path
from typing import Optional
from dataclasses import field
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch import nn

from mario_rl.agent.neural import DuelingDDQNNet
from mario_rl.agent.replay import ExperienceBatch
from mario_rl.training.shared_buffer import SharedReplayBuffer


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ExperienceBatchTensor:
    """Batch of experiences as PyTorch tensors."""

    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    state_next: torch.Tensor
    done: torch.Tensor
    actions: torch.Tensor

    def __iter__(self):
        return iter(
            [
                self.state,
                self.action,
                self.reward,
                self.state_next,
                self.done,
                self.actions,
            ]
        )


def to_tensors(batch: ExperienceBatch, device: str) -> ExperienceBatchTensor:
    return ExperienceBatchTensor(
        state=torch.from_numpy(batch.state).to(device),
        action=torch.from_numpy(batch.action).to(device),
        reward=torch.from_numpy(batch.reward).to(device),
        state_next=torch.from_numpy(batch.state_next).to(device),
        done=torch.from_numpy(batch.done).to(device),
        actions=torch.from_numpy(batch.actions).to(device),
    )


def calc_dqn_td(
    net_target: nn.Module,
    net_online: nn.Module,
    gamma: float,
    exp: ExperienceBatchTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate Double DQN TD error.
    Uses online network for action selection, target network for value estimation.
    """
    s, a, r, s_, d, _ = exp

    mask = 1 - d.float()
    idx = torch.arange(0, s.shape[0], device=s.device)

    q = net_online(s)[idx, a]
    with torch.no_grad():
        an = net_online(s_).argmax(dim=-1)
        q_n = r + mask * gamma * net_target(s_)[idx, an]

    return q, q_n


@dataclass
class Learner:
    """Learner that trains from shared replay buffer and saves weights."""

    # Required fields (no defaults)
    shared_buffer: SharedReplayBuffer
    weights_path: Path
    save_dir: Path

    # Configuration fields with defaults
    batch_size: int = 64
    gamma: float = 0.9
    lr: float = 1e-4
    sync_every: int = 1000
    save_every: int = 100
    burnin: int = 1000
    device: Optional[str] = None
    ui_queue: Optional[mp.Queue] = None
    max_steps: int = -1

    # Private fields initialized in __post_init__
    net: Any = field(init=False, repr=False)
    optimizer: Any = field(init=False, repr=False)
    train_step: int = field(init=False, default=0)
    total_loss: float = field(init=False, default=0.0)
    experiences_pulled: int = field(init=False, default=0)
    last_q_mean: float = field(init=False, default=0.0)
    last_q_max: float = field(init=False, default=0.0)
    last_td_error: float = field(init=False, default=0.0)
    last_grad_norm: float = field(init=False, default=0.0)
    last_reward_mean: float = field(init=False, default=0.0)
    steps_per_sec: float = field(init=False, default=0.0)
    _step_times: list = field(init=False, repr=False, default_factory=list)
    _last_time: float = field(init=False, repr=False, default=0.0)
    _last_loss: float = field(init=False, repr=False, default=0.0)
    _last_wait_print: int = field(init=False, repr=False, default=0)

    def __post_init__(self):
        """Initialize network and optimizer after dataclass fields are set."""
        # Set device
        if self.device is None:
            self.device = best_device()

        # Create network
        state_dim = (4, 64, 64, 1)
        action_dim = 12  # COMPLEX_MOVEMENT has 12 actions
        self.net = DuelingDDQNNet(input_shape=state_dim, num_actions=action_dim, hidden_dim=512)
        self.net = self.net.to(self.device)

        self.optimizer = torch.optim.AdamW(self.net.online.parameters(), lr=self.lr)

        # Initialize stats
        self.train_step = 0
        self.total_loss = 0.0
        self.experiences_pulled = 0

        # Initialize real-time metrics
        self.last_q_mean = 0.0
        self.last_q_max = 0.0
        self.last_td_error = 0.0
        self.last_grad_norm = 0.0
        self.last_reward_mean = 0.0
        self.steps_per_sec = 0.0
        self._step_times = []
        self._last_time = time.time()
        self._last_loss = 0.0
        self._last_wait_print = 0

        # Save initial weights for workers to load
        self.save_weights()

    def sync_target(self):
        """Copy online network weights to target network."""
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save_weights(self):
        """Save online network weights for workers to load."""
        torch.save(self.net.online.state_dict(), self.weights_path)

    def train_step_fn(self) -> Optional[float]:
        """Perform one training step. Returns loss or None if not enough data."""
        # Pull new experiences from workers
        pulled = self.shared_buffer.pull_all()
        self.experiences_pulled += pulled

        # Check if we have enough data
        if len(self.shared_buffer) < self.burnin:
            return None

        # Sample batch
        try:
            batch, indices = self.shared_buffer.sample(self.batch_size)
        except ValueError:
            return None

        # Track reward statistics from batch
        self.last_reward_mean = float(batch.reward.mean())

        # Convert to tensors
        exp = to_tensors(batch, self.device)

        # Calculate TD error
        self.net.train()
        q, q_target = calc_dqn_td(self.net.target, self.net.online, self.gamma, exp)
        td_error = q_target - q

        # Track Q-value statistics
        with torch.no_grad():
            self.last_q_mean = float(q.mean())
            self.last_q_max = float(q.max())
            self.last_td_error = float(td_error.abs().mean())

        # Update network
        loss = torch.nn.functional.mse_loss(q, q_target)
        self.optimizer.zero_grad()
        loss.backward()

        # Calculate gradient norm before clipping
        total_norm = 0.0
        for p in self.net.online.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        self.last_grad_norm = total_norm**0.5

        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(self.net.online.parameters(), max_norm=10.0)

        self.optimizer.step()

        # Update priorities
        self.shared_buffer.update_priorities(indices, td_error.abs().detach().cpu().numpy())

        self.train_step += 1
        loss_val = loss.item()
        self.total_loss += loss_val

        # Calculate steps per second
        now = time.time()
        self._step_times.append(now - self._last_time)
        self._last_time = now
        # Keep last 100 step times for moving average
        if len(self._step_times) > 100:
            self._step_times.pop(0)
        if self._step_times:
            avg_step_time = sum(self._step_times) / len(self._step_times)
            self.steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0

        # Sync target network
        if self.train_step % self.sync_every == 0:
            self.sync_target()

        # Save weights for workers
        if self.train_step % self.save_every == 0:
            self.save_weights()

        return loss_val

    def _log(self, text: str):
        """Log message to UI or stdout."""
        if self.ui_queue is not None:
            from mario_rl.training.training_ui import send_learner_log

            send_learner_log(self.ui_queue, text)
        else:
            print(f"[LEARNER] {text}", flush=True)

    def _send_status(self, status: str = "training"):
        """Send status update to UI."""
        avg_loss = self.total_loss / max(1, self.train_step)

        # Get queue throughput
        msgs_per_sec, kb_per_sec = self.shared_buffer.get_throughput()

        if self.ui_queue is not None:
            from mario_rl.training.training_ui import send_learner_status

            send_learner_status(
                self.ui_queue,
                step=self.train_step,
                loss=self._last_loss,
                avg_loss=avg_loss,
                buf_size=len(self.shared_buffer),
                pulled=self.experiences_pulled,
                max_steps=self.max_steps,
                status=status,
                q_mean=self.last_q_mean,
                q_max=self.last_q_max,
                td_error=self.last_td_error,
                grad_norm=self.last_grad_norm,
                reward_mean=self.last_reward_mean,
                steps_per_sec=self.steps_per_sec,
                queue_msgs_per_sec=msgs_per_sec,
                queue_kb_per_sec=kb_per_sec,
            )
        elif status == "training" and self.train_step % 100 == 0:
            # Print to stdout when not using UI (every 100 steps)
            print(
                f"[LEARNER] Step {self.train_step:6d} | "
                f"loss={self._last_loss:7.2f} avg={avg_loss:7.2f} | "
                f"Q={self.last_q_mean:6.1f}/{self.last_q_max:6.1f} | "
                f"TD={self.last_td_error:5.2f} | "
                f"∇={self.last_grad_norm:8.1f} | "
                f"r̄={self.last_reward_mean:6.1f} | "
                f"buf={len(self.shared_buffer):6d} | "
                f"{self.steps_per_sec:.1f} sps | "
                f"queue: {msgs_per_sec:.0f} msg/s, {kb_per_sec:.0f} KB/s",
                flush=True,
            )

    def run(self, max_steps: int = -1):
        """
        Run learner continuously or for max_steps.
        max_steps=-1 means run forever.
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if max_steps > 0:
            self.max_steps = max_steps

        # Save initial random weights
        self.save_weights()
        self._log(f"Initialized on {self.device} | batch={self.batch_size}, γ={self.gamma}")
        self._send_status("initializing")

        step = 0

        while max_steps < 0 or step < max_steps:
            loss = self.train_step_fn()

            if loss is not None:
                step += 1
                self._last_loss = loss

                # Send status update
                self._send_status("training")

                # Note target network syncs
                if self.train_step % self.sync_every == 0:
                    self._log(f"🔄 Synced target network at step {self.train_step}")

                # Note weight saves
                if self.train_step % self.save_every == 0:
                    self._log(f"💾 Saved weights at step {self.train_step}")
            else:
                # Not enough data yet, wait a bit
                time.sleep(0.1)
                # Only print waiting status every second (10 iterations at 0.1s each)
                buf_size = len(self.shared_buffer)
                if buf_size - self._last_wait_print >= 100:
                    self._last_wait_print = buf_size
                    if self.ui_queue is None:
                        print(
                            f"[LEARNER] Waiting for experiences... {buf_size}/{self.burnin}",
                            flush=True,
                        )
                    else:
                        self._send_status("waiting")

        # Final save
        self.save_weights()
        self._send_status("finished")
        self._log(f"✅ Training complete! Final weights saved after {self.train_step} steps")


def run_learner(
    shared_buffer: SharedReplayBuffer,
    weights_path: Path,
    save_dir: Path,
    max_steps: int = -1,
    ui_queue: Optional[mp.Queue] = None,
    **kwargs,
):
    """Entry point for learner process."""
    learner = Learner(
        shared_buffer=shared_buffer,
        weights_path=weights_path,
        save_dir=save_dir,
        ui_queue=ui_queue,
        max_steps=max_steps,
        **kwargs,
    )
    learner.run(max_steps=max_steps)


if __name__ == "__main__":
    # Test learner standalone
    save_dir = Path("checkpoints/test_learner")
    weights_path = save_dir / "weights.pt"

    buffer = SharedReplayBuffer(max_len=10000)

    learner = Learner(
        shared_buffer=buffer,
        weights_path=weights_path,
        save_dir=save_dir,
        burnin=10,  # Low for testing
    )

    # Add some fake experiences for testing
    for _i in range(100):
        buffer.push(
            state=np.random.rand(4, 64, 64, 1).astype(np.float32),
            action=0,
            reward=1.0,
            next_state=np.random.rand(4, 64, 64, 1).astype(np.float32),
            done=False,
            actions=list(range(12)),
        )

    learner.run(max_steps=50)
