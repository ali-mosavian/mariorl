"""
PPO Learner for distributed training.

Pulls rollouts from shared buffer, computes PPO loss, and updates the network.
"""

import os
import csv
import time
import multiprocessing as mp

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from typing import Any
from pathlib import Path
from typing import Optional
from dataclasses import field
from dataclasses import dataclass

import torch
import numpy as np
from torch import nn

from mario_rl.agent.ppo_net import ActorCritic
from mario_rl.training.rollout_buffer import RolloutBatch
from mario_rl.training.rollout_buffer import RolloutBuffer


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class PPOLearner:
    """
    PPO Learner that trains the actor-critic network.

    Features:
    - Clipped surrogate objective
    - Value function clipping
    - Entropy bonus with scheduling
    - Learning rate annealing
    - Gradient clipping
    """

    # Required fields
    weights_path: Path
    save_dir: Path
    rollout_queue: mp.Queue  # Queue to receive rollouts from workers

    # Hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float = 0.2  # Value function clip range (None to disable)
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    learning_rate: float = 2.5e-4
    n_epochs: int = 4
    batch_size: int = 64
    target_kl: Optional[float] = 0.03  # Early stopping on KL divergence

    # Scheduling
    lr_schedule: str = "linear"  # "linear", "constant"
    ent_schedule: str = "linear"  # "linear", "constant"
    total_timesteps: int = 10_000_000

    # Other
    save_every: int = 100
    device: Optional[str] = None
    ui_queue: Optional[mp.Queue] = None

    # Private fields
    net: Any = field(init=False, repr=False)
    optimizer: Any = field(init=False, repr=False)
    train_step: int = field(init=False, default=0)
    total_timesteps_collected: int = field(init=False, default=0)
    _metrics_csv: Path = field(init=False, repr=False)

    # Tracking
    last_policy_loss: float = field(init=False, default=0.0)
    last_value_loss: float = field(init=False, default=0.0)
    last_entropy: float = field(init=False, default=0.0)
    last_kl: float = field(init=False, default=0.0)
    last_clip_fraction: float = field(init=False, default=0.0)
    steps_per_sec: float = field(init=False, default=0.0)
    _last_time: float = field(init=False, default=0.0)

    def __post_init__(self):
        """Initialize network and optimizer."""
        if self.device is None:
            self.device = best_device()

        # Create network
        state_dim = (4, 64, 64, 1)
        action_dim = 12  # COMPLEX_MOVEMENT
        self.net = ActorCritic(
            input_shape=state_dim,
            num_actions=action_dim,
            feature_dim=512,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate,
            eps=1e-5,
        )

        # Initialize tracking
        self.train_step = 0
        self.total_timesteps_collected = 0
        self._last_time = time.time()

        # Load existing weights or save initial
        if self.weights_path.exists():
            try:
                checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=True)
                self.net.load_state_dict(checkpoint)
                print(f"✅ Resumed from checkpoint: {self.weights_path}")
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint: {e}")
                self.save_weights()
        else:
            self.save_weights()

        # Initialize CSV logging
        self._metrics_csv = self.save_dir / "ppo_metrics.csv"
        if not self._metrics_csv.exists():
            with open(self._metrics_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "step",
                        "timesteps",
                        "policy_loss",
                        "value_loss",
                        "entropy",
                        "kl",
                        "clip_fraction",
                        "lr",
                        "ent_coef",
                    ]
                )

    def save_weights(self) -> None:
        """Save network weights for workers to load."""
        torch.save(self.net.state_dict(), self.weights_path)

    def get_scheduled_lr(self) -> float:
        """Get learning rate based on schedule."""
        if self.lr_schedule == "constant":
            return self.learning_rate

        # Linear decay
        progress = self.total_timesteps_collected / self.total_timesteps
        return self.learning_rate * (1.0 - progress)

    def get_scheduled_ent_coef(self) -> float:
        """Get entropy coefficient based on schedule."""
        if self.ent_schedule == "constant":
            return self.ent_coef

        # Linear decay from ent_coef to 0.001
        progress = self.total_timesteps_collected / self.total_timesteps
        return max(0.001, self.ent_coef * (1.0 - progress))

    def update_lr(self) -> None:
        """Update learning rate in optimizer."""
        lr = self.get_scheduled_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train_on_rollouts(self, rollouts: list) -> dict:
        """
        Train on a list of rollouts.

        Args:
            rollouts: List of rollout dicts from workers

        Returns:
            Dictionary of training metrics
        """
        if not rollouts:
            return {}

        # Combine all rollouts into one buffer
        all_states = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_values = []
        all_log_probs = []

        for rollout in rollouts:
            all_states.append(rollout["states"])
            all_actions.append(rollout["actions"])
            all_rewards.append(rollout["rewards"])
            all_dones.append(rollout["dones"])
            all_values.append(rollout["values"])
            all_log_probs.append(rollout["log_probs"])

        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        rewards = np.concatenate(all_rewards, axis=0)
        dones = np.concatenate(all_dones, axis=0)
        values = np.concatenate(all_values, axis=0)
        log_probs = np.concatenate(all_log_probs, axis=0)

        # Use last rollout's bootstrap values
        last_value = rollouts[-1]["last_value"]
        last_done = rollouts[-1]["last_done"]

        # Create buffer and compute GAE
        buffer = RolloutBuffer(
            buffer_size=len(states),
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # Add all data
        for i in range(len(states)):
            buffer.add(
                state=states[i],
                action=actions[i],
                reward=rewards[i],
                done=dones[i],
                value=values[i],
                log_prob=log_probs[i],
            )

        # Compute advantages
        buffer.compute_gae(last_value, last_done)

        # Update timestep count
        self.total_timesteps_collected += len(states)

        # Update learning rate
        self.update_lr()
        current_lr = self.optimizer.param_groups[0]["lr"]
        current_ent_coef = self.get_scheduled_ent_coef()

        # Training loop
        policy_losses = []
        value_losses = []
        entropies = []
        kls = []
        clip_fractions = []

        for _epoch in range(self.n_epochs):
            for batch in buffer.get_batches(self.batch_size, self.device or "cpu"):
                metrics = self._train_step(batch, current_ent_coef)

                policy_losses.append(metrics["policy_loss"])
                value_losses.append(metrics["value_loss"])
                entropies.append(metrics["entropy"])
                kls.append(metrics["kl"])
                clip_fractions.append(metrics["clip_fraction"])

                # Early stopping on KL divergence
                if self.target_kl is not None and metrics["kl"] > self.target_kl:
                    break

        self.train_step += 1

        # Update metrics
        self.last_policy_loss = np.mean(policy_losses)
        self.last_value_loss = np.mean(value_losses)
        self.last_entropy = np.mean(entropies)
        self.last_kl = np.mean(kls)
        self.last_clip_fraction = np.mean(clip_fractions)

        # Calculate speed
        now = time.time()
        elapsed = now - self._last_time
        self.steps_per_sec = len(states) / elapsed if elapsed > 0 else 0
        self._last_time = now

        # Save weights periodically
        if self.train_step % self.save_every == 0:
            self.save_weights()
            self._log_metrics(current_lr, current_ent_coef)

        # Send UI update
        self._send_ui_status(current_lr, current_ent_coef)

        return {
            "policy_loss": self.last_policy_loss,
            "value_loss": self.last_value_loss,
            "entropy": self.last_entropy,
            "kl": self.last_kl,
            "clip_fraction": self.last_clip_fraction,
        }

    def _train_step(self, batch: RolloutBatch, ent_coef: float) -> dict:
        """Perform one gradient update on a minibatch."""
        self.optimizer.zero_grad()

        # Get current policy outputs
        _, new_log_probs, entropy, new_values = self.net.get_action_and_value(batch.states, batch.actions)

        # Policy loss (clipped surrogate)
        log_ratio = new_log_probs - batch.old_log_probs
        ratio = torch.exp(log_ratio)

        # Approximate KL divergence
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()

        surr1 = ratio * batch.advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch.advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss (with optional clipping)
        if self.clip_range_vf is not None:
            values_clipped = batch.old_values + torch.clamp(
                new_values - batch.old_values,
                -self.clip_range_vf,
                self.clip_range_vf,
            )
            value_loss_unclipped = (new_values - batch.returns) ** 2
            value_loss_clipped = (values_clipped - batch.returns) ** 2
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
        else:
            value_loss = 0.5 * ((new_values - batch.returns) ** 2).mean()

        # Entropy loss (negative because we want to maximize entropy)
        entropy_loss = -entropy.mean()

        # Total loss
        loss = policy_loss + self.vf_coef * value_loss + ent_coef * entropy_loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

    def _log_metrics(self, lr: float, ent_coef: float) -> None:
        """Log metrics to CSV."""
        with open(self._metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    time.time(),
                    self.train_step,
                    self.total_timesteps_collected,
                    self.last_policy_loss,
                    self.last_value_loss,
                    self.last_entropy,
                    self.last_kl,
                    self.last_clip_fraction,
                    lr,
                    ent_coef,
                ]
            )

    def _send_ui_status(self, lr: float, ent_coef: float) -> None:
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
                    "step": self.train_step,
                    "timesteps": self.total_timesteps_collected,
                    "policy_loss": self.last_policy_loss,
                    "value_loss": self.last_value_loss,
                    "entropy": self.last_entropy,
                    "clip_fraction": self.last_clip_fraction,
                    "kl": self.last_kl,
                    "lr": lr,
                    "ent_coef": ent_coef,
                    "steps_per_sec": self.steps_per_sec,
                    "elapsed_time": time.time() - self._last_time,
                },
            )
            self.ui_queue.put_nowait(msg)
        except Exception:
            pass

    def run(self, max_updates: int = -1) -> None:
        """
        Main training loop.

        Args:
            max_updates: Maximum number of updates (-1 for unlimited)
        """
        print(f"PPO Learner started on {self.device}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Entropy coef: {self.ent_coef}")
        print(f"  Clip range: {self.clip_range}")

        update_count = 0
        while max_updates < 0 or update_count < max_updates:
            # Wait for rollouts
            try:
                rollouts = []
                # Collect available rollouts (non-blocking after first)
                rollout = self.rollout_queue.get(timeout=1.0)
                rollouts.append(rollout)

                # Get any additional available rollouts
                while True:
                    try:
                        rollout = self.rollout_queue.get_nowait()
                        rollouts.append(rollout)
                    except Exception:
                        break

                if rollouts:
                    self.train_on_rollouts(rollouts)
                    update_count += 1

                    if update_count % 10 == 0:
                        print(
                            f"Update {update_count} | "
                            f"Steps: {self.total_timesteps_collected:,} | "
                            f"π_loss: {self.last_policy_loss:.4f} | "
                            f"v_loss: {self.last_value_loss:.4f} | "
                            f"H: {self.last_entropy:.4f} | "
                            f"KL: {self.last_kl:.4f}"
                        )

            except Exception as e:
                if "Empty" not in str(type(e).__name__):
                    print(f"Learner error: {e}")
                continue

        # Final save
        self.save_weights()
        print(f"Training complete. Total updates: {update_count}")


def run_ppo_learner(
    weights_path: Path,
    save_dir: Path,
    rollout_queue: mp.Queue,
    ui_queue: Optional[mp.Queue] = None,
    **kwargs,
) -> None:
    """Entry point for learner process."""
    learner = PPOLearner(
        weights_path=weights_path,
        save_dir=save_dir,
        rollout_queue=rollout_queue,
        ui_queue=ui_queue,
        **kwargs,
    )
    learner.run()
