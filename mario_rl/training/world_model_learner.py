"""
World Model Learner for distributed training.

Alternates between:
1. Training the world model (encoder, decoder, dynamics, reward predictor)
2. Training the Q-network on frozen latent representations

This enables level-agnostic learning through abstract latent representations.
"""

import os
import csv
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

from mario_rl.agent.replay import ExperienceBatch
from mario_rl.agent.world_model import LatentDDQN
from mario_rl.agent.world_model import WorldModelLoss
from mario_rl.agent.world_model import MarioWorldModel
from mario_rl.agent.world_model import WorldModelMetrics
from mario_rl.training.shared_buffer import SequenceBatch
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
    # Convert states from uint8 (0-255) to float32 (0-1)
    state = torch.from_numpy(batch.state).float().div(255.0).to(device)
    state_next = torch.from_numpy(batch.state_next).float().div(255.0).to(device)

    return ExperienceBatchTensor(
        state=state,
        action=torch.from_numpy(batch.action).to(device),
        reward=torch.from_numpy(batch.reward).to(device),
        state_next=state_next,
        done=torch.from_numpy(batch.done).to(device),
        actions=torch.from_numpy(batch.actions).to(device),
    )


@dataclass
class SequenceBatchTensor:
    """Batch of sequential experiences as PyTorch tensors."""

    states: torch.Tensor  # (batch, seq_len, F, H, W, C) float [0,1]
    actions: torch.Tensor  # (batch, seq_len) long
    rewards: torch.Tensor  # (batch, seq_len) float
    dones: torch.Tensor  # (batch, seq_len) bool


def to_sequence_tensors(batch: SequenceBatch, device: str) -> SequenceBatchTensor:
    """Convert SequenceBatch to GPU tensors."""
    return SequenceBatchTensor(
        states=torch.from_numpy(batch.states).float().div(255.0).to(device),
        actions=torch.from_numpy(batch.actions).long().to(device),
        rewards=torch.from_numpy(batch.rewards).float().to(device),
        dones=torch.from_numpy(batch.dones).bool().to(device),
    )


@dataclass
class WorldModelLearner:
    """
    Learner that trains a world model and Q-network with alternating schedule.

    Training phases:
    1. World Model Phase: Train encoder, decoder, dynamics, reward predictor
    2. Q-Network Phase: Train Q-network on frozen latent representations
    """

    # Required fields (no defaults)
    shared_buffer: SharedReplayBuffer
    weights_path: Path
    save_dir: Path

    # World model configuration
    latent_dim: int = 128
    wm_hidden_dim: int = 256
    wm_lr: float = 1e-4
    wm_steps: int = 500  # Steps per world model phase
    seq_len: int = 8  # Sequence length for dynamics training
    beta_kl: float = 0.1
    beta_dynamics: float = 0.1  # Lower to prevent dynamics loss explosion early
    beta_ssim: float = 0.1

    # Q-network configuration
    q_hidden_dim: int = 256
    q_lr: float = 1e-4
    q_steps: int = 500  # Steps per Q-network phase
    gamma: float = 0.9
    sync_every: int = 1000

    # General configuration
    batch_size: int = 64
    save_every: int = 100
    burnin: int = 1000
    device: Optional[str] = None
    ui_queue: Optional[mp.Queue] = None
    max_steps: int = -1

    # Private fields initialized in __post_init__
    world_model: Any = field(init=False, repr=False)
    q_network: Any = field(init=False, repr=False)
    wm_optimizer: Any = field(init=False, repr=False)
    q_optimizer: Any = field(init=False, repr=False)
    wm_loss_fn: Any = field(init=False, repr=False)

    train_step: int = field(init=False, default=0)
    wm_train_step: int = field(init=False, default=0)
    q_train_step: int = field(init=False, default=0)
    experiences_pulled: int = field(init=False, default=0)

    # World model metrics
    last_wm_metrics: Optional[WorldModelMetrics] = field(init=False, default=None)

    # Q-network metrics
    last_q_mean: float = field(init=False, default=0.0)
    last_q_max: float = field(init=False, default=0.0)
    last_td_error: float = field(init=False, default=0.0)
    last_q_loss: float = field(init=False, default=0.0)
    last_grad_norm: float = field(init=False, default=0.0)
    last_reward_mean: float = field(init=False, default=0.0)

    # Performance tracking
    steps_per_sec: float = field(init=False, default=0.0)
    _step_times: list = field(init=False, repr=False, default_factory=list)
    _last_time: float = field(init=False, repr=False, default=0.0)
    _last_wait_print: int = field(init=False, repr=False, default=0)

    def __post_init__(self):
        """Initialize networks and optimizers after dataclass fields are set."""
        # Set device
        if self.device is None:
            self.device = best_device()

        # Frame shape: (frames, height, width, channels)
        state_dim = (4, 64, 64, 1)
        action_dim = 12  # COMPLEX_MOVEMENT has 12 actions

        # Create world model
        self.world_model = MarioWorldModel(
            frame_shape=state_dim,
            num_actions=action_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.wm_hidden_dim,
        ).to(self.device)

        # Create Q-network (operates on latent space)
        self.q_network = LatentDDQN(
            latent_dim=self.latent_dim,
            num_actions=action_dim,
            hidden_dim=self.q_hidden_dim,
        ).to(self.device)

        # Optimizers
        self.wm_optimizer = torch.optim.AdamW(self.world_model.parameters(), lr=self.wm_lr)
        self.q_optimizer = torch.optim.AdamW(self.q_network.online.parameters(), lr=self.q_lr)

        # Loss function
        self.wm_loss_fn = WorldModelLoss(
            beta_kl=self.beta_kl,
            beta_dynamics=self.beta_dynamics,
            beta_ssim=self.beta_ssim,
        )

        # Initialize counters
        self.train_step = 0
        self.wm_train_step = 0
        self.q_train_step = 0
        self.experiences_pulled = 0
        self._last_time = time.time()

        # Ensure save directory exists before saving initial weights
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV logging
        self.log_file = self.save_dir / "training.csv"
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "step",
                        "phase",
                        "wm_step",
                        "q_step",
                        "buffer_size",
                        "wm_recon_mse",
                        "wm_ssim",
                        "wm_dynamics_loss",
                        "wm_reward_loss",
                        "wm_total_loss",
                        "q_loss",
                        "q_mean",
                        "q_max",
                        "td_error",
                        "steps_per_sec",
                    ]
                )

        # Try to resume from checkpoint
        if not self.load_checkpoint():
            # No checkpoint found, save initial weights
            self.save_weights()

    @property
    def current_phase(self) -> str:
        """Get current training phase."""
        cycle_length = self.wm_steps + self.q_steps
        cycle_position = self.train_step % cycle_length
        return "world_model" if cycle_position < self.wm_steps else "q_network"

    @property
    def phase_progress(self) -> tuple[int, int]:
        """Get (current_step_in_phase, total_steps_in_phase)."""
        cycle_length = self.wm_steps + self.q_steps
        cycle_position = self.train_step % cycle_length

        if cycle_position < self.wm_steps:
            return cycle_position, self.wm_steps
        else:
            return cycle_position - self.wm_steps, self.q_steps

    def sync_target(self):
        """Copy online Q-network weights to target Q-network."""
        self.q_network.sync_target()

    def save_weights(self):
        """Save all network weights for workers to load."""
        torch.save(
            {
                "world_model": self.world_model.state_dict(),
                "q_network": self.q_network.state_dict(),
                "encoder": self.world_model.encoder.state_dict(),
            },
            self.weights_path,
        )

    def save_checkpoint(self):
        """Save full training state for resumption."""
        checkpoint_path = self.save_dir / "checkpoint.pt"
        torch.save(
            {
                # Model states
                "world_model": self.world_model.state_dict(),
                "q_network": self.q_network.state_dict(),
                # Optimizer states
                "wm_optimizer": self.wm_optimizer.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                # Training progress
                "train_step": self.train_step,
                "wm_train_step": self.wm_train_step,
                "q_train_step": self.q_train_step,
                "experiences_pulled": self.experiences_pulled,
                # Config
                "latent_dim": self.latent_dim,
                "wm_hidden_dim": self.wm_hidden_dim,
                "q_hidden_dim": self.q_hidden_dim,
            },
            checkpoint_path,
        )

    def load_checkpoint(self) -> bool:
        """Load training state from checkpoint. Returns True if successful."""
        checkpoint_path = self.save_dir / "checkpoint.pt"
        if not checkpoint_path.exists():
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # Verify architecture matches
            if checkpoint.get("latent_dim") != self.latent_dim:
                self._log("⚠️ Checkpoint latent_dim mismatch, skipping load")
                return False

            # Load model states
            self.world_model.load_state_dict(checkpoint["world_model"])
            self.q_network.load_state_dict(checkpoint["q_network"])

            # Load optimizer states
            self.wm_optimizer.load_state_dict(checkpoint["wm_optimizer"])
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])

            # Restore training progress
            self.train_step = checkpoint["train_step"]
            self.wm_train_step = checkpoint["wm_train_step"]
            self.q_train_step = checkpoint["q_train_step"]
            self.experiences_pulled = checkpoint["experiences_pulled"]

            self._log(
                f"✅ Resumed from checkpoint at step {self.train_step} "
                f"(WM: {self.wm_train_step}, Q: {self.q_train_step})"
            )
            return True

        except Exception as e:
            self._log(f"⚠️ Failed to load checkpoint: {e}")
            return False

    def train_world_model_step(self, batch: ExperienceBatchTensor) -> tuple[float, WorldModelMetrics]:
        """
        Perform one world model training step.

        Returns loss and metrics.
        """
        self.world_model.train()
        s, a, r, s_, d, _ = batch

        # Forward pass
        output = self.world_model(s, a, s_)

        # Get target latent encoding for dynamics loss
        with torch.no_grad():
            z_next_mu, z_next_logvar = self.world_model.encoder(s_)

        # Compute loss
        loss, metrics = self.wm_loss_fn(
            frames=s,
            frames_next=s_,
            rewards=r,
            output=output,
            z_next_target=(z_next_mu, z_next_logvar),
        )

        # Backward pass
        self.wm_optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=10.0)

        self.wm_optimizer.step()

        self.wm_train_step += 1
        self.last_wm_metrics = metrics

        return loss.item(), metrics

    def train_world_model_sequence(self, seq_batch: SequenceBatchTensor) -> tuple[float, WorldModelMetrics]:
        """
        Train world model on sequences with GRU hidden state propagation.

        This is more effective for dynamics learning because the GRU can
        accumulate context across multiple timesteps.

        Args:
            seq_batch: Batch of sequences with shape (batch, seq_len, ...)

        Returns:
            Total loss and metrics from the last timestep
        """
        self.world_model.train()
        batch_size, seq_len = seq_batch.states.shape[:2]

        # Use tensor for loss accumulation to enable backprop
        total_loss: torch.Tensor = torch.tensor(0.0, device=self.device)
        total_recon_loss = 0.0
        total_dynamics_loss = 0.0
        total_reward_loss = 0.0

        # Initialize GRU hidden state
        hidden: torch.Tensor | None = None

        # Process sequence step by step
        for t in range(seq_len - 1):
            # Get current and next frame
            s_t = seq_batch.states[:, t]  # (batch, F, H, W, C)
            s_next = seq_batch.states[:, t + 1]
            a_t = seq_batch.actions[:, t]
            r_t = seq_batch.rewards[:, t]

            # Encode current state
            z_mu, z_logvar = self.world_model.encoder(s_t)
            z_t = z_mu + torch.randn_like(z_mu) * torch.exp(0.5 * z_logvar)

            # Encode next state (target for dynamics)
            with torch.no_grad():
                z_next_mu_target, z_next_logvar_target = self.world_model.encoder(s_next)

            # Predict next latent with dynamics model (propagate hidden state!)
            z_next_mu_pred, z_next_logvar_pred, z_next_sample, h_next = self.world_model.dynamics(z_t, a_t, hidden)
            # Detach hidden for next step to prevent gradient explosion through time
            hidden = h_next.detach()

            # Decode predicted next latent to frame
            frame_next_pred = self.world_model.decoder(z_next_sample)

            # Predict reward
            reward_pred = self.world_model.reward_pred(z_t)

            # Reconstruction loss (MSE + SSIM)
            recon_mse = nn.functional.mse_loss(frame_next_pred, s_next)
            recon_loss = recon_mse

            # Dynamics loss (KL divergence between predicted and target distributions)
            var_pred = z_next_logvar_pred.exp().clamp(min=1e-6)
            var_target = z_next_logvar_target.exp().clamp(min=1e-6)
            dynamics_kl = 0.5 * (
                z_next_logvar_target
                - z_next_logvar_pred
                + var_pred / var_target
                + (z_next_mu_target - z_next_mu_pred).pow(2) / var_target
                - 1
            ).mean().clamp(min=0, max=100)

            # Reward prediction loss
            reward_loss = nn.functional.mse_loss(reward_pred.squeeze(-1), r_t)

            # Combined loss for this timestep
            step_loss = recon_loss + self.beta_dynamics * dynamics_kl + 0.1 * reward_loss

            total_loss += step_loss
            total_recon_loss += recon_mse.item()
            total_dynamics_loss += dynamics_kl.item()
            total_reward_loss += reward_loss.item()

        # Average over timesteps
        num_steps = seq_len - 1
        avg_loss = total_loss / num_steps

        # Backward pass
        self.wm_optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=10.0)
        self.wm_optimizer.step()

        self.wm_train_step += 1

        # Create metrics (use averaged values)
        from mario_rl.agent.world_model import ssim

        # Compute final SSIM for metrics
        with torch.no_grad():
            final_ssim = (
                ssim(
                    frame_next_pred.reshape(-1, 1, 64, 64),
                    s_next.reshape(-1, 1, 64, 64),
                )
                .mean()
                .item()
            )

        metrics = WorldModelMetrics(
            recon_mse=total_recon_loss / num_steps,
            pred_mse=total_recon_loss / num_steps,  # Same for now
            ssim=final_ssim,
            dynamics_loss=total_dynamics_loss / num_steps,
            reward_loss=total_reward_loss / num_steps,
            kl_loss=0.0,  # Not using VAE KL here
            total_loss=avg_loss.item(),
        )
        self.last_wm_metrics = metrics

        return avg_loss.item(), metrics

    def train_q_network_step(self, batch: ExperienceBatchTensor) -> tuple[float, np.ndarray]:
        """
        Perform one Q-network training step on frozen latent representations.

        Returns:
            loss: The training loss value
            td_errors: TD errors for priority updates
        """
        self.q_network.train()
        s, a, r, s_, d, _ = batch

        # Encode states using frozen world model encoder
        with torch.no_grad():
            z = self.world_model.encode(s, deterministic=True)
            z_ = self.world_model.encode(s_, deterministic=True)

        mask = 1 - d.float()
        idx = torch.arange(0, s.shape[0], device=s.device)

        # Q-values for current state-action pairs
        q = self.q_network.online(z)[idx, a]

        # Double DQN: use online network to select action, target to evaluate
        with torch.no_grad():
            a_next = self.q_network.online(z_).argmax(dim=-1)
            q_target = r + mask * self.gamma * self.q_network.target(z_)[idx, a_next]

        # TD error
        td_error = q_target - q

        # Track metrics
        self.last_q_mean = float(q.mean())
        self.last_q_max = float(q.max())
        self.last_td_error = float(td_error.abs().mean())
        self.last_reward_mean = float(r.mean())

        # Compute loss and update
        loss = torch.nn.functional.mse_loss(q, q_target)
        self.q_optimizer.zero_grad()
        loss.backward()

        # Track gradient norm
        total_norm = 0.0
        for p in self.q_network.online.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        self.last_grad_norm = total_norm**0.5

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.online.parameters(), max_norm=10.0)

        self.q_optimizer.step()

        self.q_train_step += 1
        self.last_q_loss = loss.item()

        return loss.item(), td_error.abs().detach().cpu().numpy()

    def train_step_fn(self) -> float | None:
        """Perform one training step. Returns loss or None if not enough data."""
        # Pull new experiences from workers
        pulled = self.shared_buffer.pull_all()
        self.experiences_pulled += pulled

        # Check if we have enough data
        if len(self.shared_buffer) < self.burnin:
            return None

        # Device is guaranteed to be set in __post_init__
        assert self.device is not None

        # Determine which phase we're in
        phase = self.current_phase

        if phase == "world_model":
            # Use sequence sampling for world model training
            # This allows the GRU to learn temporal dynamics properly
            try:
                seq_batch, indices = self.shared_buffer.sample_sequences(
                    batch_size=self.batch_size,
                    seq_len=self.seq_len,
                )
            except ValueError:
                # Fall back to regular sampling if not enough sequences
                try:
                    batch, indices = self.shared_buffer.sample(self.batch_size)
                except ValueError:
                    return None
                exp = to_tensors(batch, self.device)
                loss, _ = self.train_world_model_step(exp)
                return loss

            # Convert sequences to tensors
            seq_exp = to_sequence_tensors(seq_batch, self.device)
            loss, _ = self.train_world_model_sequence(seq_exp)
        else:
            # Use random sampling for Q-network (standard RL approach)
            try:
                batch, indices = self.shared_buffer.sample(self.batch_size)
            except ValueError:
                return None
            exp = to_tensors(batch, self.device)
            loss, td_errors = self.train_q_network_step(exp)
            # Update priorities based on TD error
            self.shared_buffer.update_priorities(indices, np.abs(td_errors))

        self.train_step += 1

        # Update steps per second
        now = time.time()
        self._step_times.append(now - self._last_time)
        self._last_time = now
        if len(self._step_times) > 100:
            self._step_times.pop(0)
        if self._step_times:
            avg_step_time = sum(self._step_times) / len(self._step_times)
            self.steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0

        # Sync Q target network periodically
        if self.q_train_step > 0 and self.q_train_step % self.sync_every == 0:
            self.sync_target()

        # Log metrics to CSV every 100 steps
        if self.train_step % 100 == 0:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                wm_metrics = self.last_wm_metrics
                writer.writerow(
                    [
                        self.train_step,
                        self.current_phase,
                        self.wm_train_step,
                        self.q_train_step,
                        len(self.shared_buffer),
                        wm_metrics.recon_mse if wm_metrics else 0,
                        wm_metrics.ssim if wm_metrics else 0,
                        wm_metrics.dynamics_loss if wm_metrics else 0,
                        wm_metrics.reward_loss if wm_metrics else 0,
                        wm_metrics.total_loss if wm_metrics else 0,
                        self.last_q_loss,
                        self.last_q_mean,
                        self.last_q_max,
                        self.last_td_error,
                        self.steps_per_sec,
                    ]
                )

        # Save weights periodically
        if self.train_step % self.save_every == 0:
            self.save_weights()
            # Save full checkpoint every 1000 steps for resumption
            if self.train_step % 1000 == 0:
                self.save_checkpoint()

        return loss

    def _log(self, text: str):
        """Log message to UI or stdout."""
        if self.ui_queue is not None:
            from mario_rl.training.training_ui import send_learner_log

            send_learner_log(self.ui_queue, text)
        else:
            print(f"[WM-LEARNER] {text}", flush=True)

    def _send_status(self, status: str = "training"):
        """Send status update to UI."""
        # Get queue throughput
        msgs_per_sec, kb_per_sec = self.shared_buffer.get_throughput()
        phase_step, phase_total = self.phase_progress

        if self.ui_queue is not None:
            from mario_rl.training.training_ui import send_world_model_status

            send_world_model_status(
                self.ui_queue,
                step=self.train_step,
                wm_step=self.wm_train_step,
                q_step=self.q_train_step,
                buf_size=len(self.shared_buffer),
                pulled=self.experiences_pulled,
                max_steps=self.max_steps,
                status=status,
                phase=self.current_phase,
                phase_step=phase_step,
                phase_total=phase_total,
                # World model metrics
                wm_metrics=self.last_wm_metrics,
                # Q-network metrics
                q_mean=self.last_q_mean,
                q_max=self.last_q_max,
                td_error=self.last_td_error,
                q_loss=self.last_q_loss,
                grad_norm=self.last_grad_norm,
                reward_mean=self.last_reward_mean,
                steps_per_sec=self.steps_per_sec,
                queue_msgs_per_sec=msgs_per_sec,
                queue_kb_per_sec=kb_per_sec,
            )
        elif status == "training" and self.train_step % 100 == 0:
            # Print to stdout when not using UI
            phase = self.current_phase
            if phase == "world_model" and self.last_wm_metrics:
                m = self.last_wm_metrics
                print(
                    f"[WM-LEARNER] Step {self.train_step:6d} | "
                    f"Phase: {phase} ({phase_step}/{phase_total}) | "
                    f"MSE={m.recon_mse:.4f} SSIM={m.ssim:.3f} "
                    f"Dyn={m.dynamics_loss:.4f} Rew={m.reward_loss:.4f} "
                    f"KL={m.kl_loss:.4f} | "
                    f"buf={len(self.shared_buffer):6d} | "
                    f"{self.steps_per_sec:.1f} sps",
                    flush=True,
                )
            else:
                print(
                    f"[WM-LEARNER] Step {self.train_step:6d} | "
                    f"Phase: {phase} ({phase_step}/{phase_total}) | "
                    f"Q={self.last_q_mean:6.1f}/{self.last_q_max:6.1f} | "
                    f"TD={self.last_td_error:5.2f} Loss={self.last_q_loss:.4f} | "
                    f"∇={self.last_grad_norm:8.1f} | "
                    f"buf={len(self.shared_buffer):6d} | "
                    f"{self.steps_per_sec:.1f} sps",
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

        # Save initial weights
        self.save_weights()
        self._log(
            f"Initialized on {self.device} | latent_dim={self.latent_dim}, "
            f"wm_steps={self.wm_steps}, q_steps={self.q_steps}"
        )
        self._send_status("initializing")

        step = 0

        while max_steps < 0 or step < max_steps:
            loss = self.train_step_fn()

            if loss is not None:
                step += 1

                # Send status update
                self._send_status("training")

                # Note phase transitions
                phase_step, phase_total = self.phase_progress
                if phase_step == 0 and self.train_step > 1:
                    self._log(f"🔄 Switched to {self.current_phase} phase")

                # Note Q target syncs
                if (
                    self.current_phase == "q_network"
                    and self.q_train_step > 0
                    and self.q_train_step % self.sync_every == 0
                ):
                    self._log(f"🔄 Synced Q target network at Q-step {self.q_train_step}")

                # Note weight saves
                if self.train_step % self.save_every == 0:
                    self._log(f"💾 Saved weights at step {self.train_step}")
            else:
                # Not enough data yet, wait a bit
                time.sleep(0.1)
                buf_size = len(self.shared_buffer)
                if buf_size - self._last_wait_print >= 100:
                    self._last_wait_print = buf_size
                    if self.ui_queue is None:
                        print(
                            f"[WM-LEARNER] Waiting for experiences... {buf_size}/{self.burnin}",
                            flush=True,
                        )
                    else:
                        self._send_status("waiting")

        # Final save
        self.save_weights()
        self._send_status("finished")
        self._log(f"✅ Training complete! Final weights saved after {self.train_step} steps")


def run_world_model_learner(
    shared_buffer: SharedReplayBuffer,
    weights_path: Path,
    save_dir: Path,
    max_steps: int = -1,
    ui_queue: Optional[mp.Queue] = None,
    **kwargs,
):
    """Entry point for world model learner process."""
    learner = WorldModelLearner(
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
    save_dir = Path("checkpoints/test_wm_learner")
    weights_path = save_dir / "weights.pt"
    save_dir.mkdir(parents=True, exist_ok=True)

    buffer = SharedReplayBuffer(max_len=10000)

    learner = WorldModelLearner(
        shared_buffer=buffer,
        weights_path=weights_path,
        save_dir=save_dir,
        burnin=10,  # Low for testing
        wm_steps=5,
        q_steps=5,
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
