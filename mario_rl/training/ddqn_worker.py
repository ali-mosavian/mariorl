"""
Distributed DDQN Worker with local gradient computation.

Workers collect experiences in a local replay buffer, compute DQN loss locally,
and send only gradients to the learner. This is similar to Gorila DQN.

Features
========
- Prioritized Experience Replay (PER) with SumTree
- N-step returns for better credit assignment
- Double DQN for reduced overestimation
- Importance sampling weights for unbiased gradients
"""

import os
import csv
import time
import multiprocessing as mp

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from typing import Any
from typing import Dict
from pathlib import Path
from typing import Optional
from dataclasses import field
from dataclasses import dataclass

import torch
import numpy as np
from torch import nn

from mario_rl.buffers import NStepBuffer
from mario_rl.core.types import Transition
from mario_rl.core.timing import TimingStats
from mario_rl.agent.ddqn_net import DoubleDQN
from mario_rl.core.config import WorkerConfig
from mario_rl.core.device import detect_device
from mario_rl.core.episode import EpisodeState
from mario_rl.core.config import SnapshotConfig
from mario_rl.core.metrics import MetricsTracker
from mario_rl.core.ui_reporter import UIReporter
from mario_rl.core.weight_sync import WeightSync
from mario_rl.core.exploration import EpsilonGreedy
from mario_rl.environment.factory import create_env
from mario_rl.buffers import PrioritizedReplayBuffer
from mario_rl.training.snapshot import SnapshotManager
from mario_rl.training.ddqn_status import DDQNStatusCollector
from mario_rl.core.reward_normalizer import RewardNormalizer
from mario_rl.training.shared_gradient_tensor import SharedGradientTensor
from mario_rl.training.shared_gradient_tensor import attach_tensor_buffer


@dataclass
class DDQNWorker:
    """
    DDQN Worker that computes gradients locally and sends them to the learner.

    Each worker:
    1. Collects experiences using epsilon-greedy
    2. Stores in local PER buffer
    3. Samples batch with priority weighting
    4. Computes Double DQN loss with importance sampling
    5. Updates priorities based on TD errors
    6. Sends gradients to learner
    7. Periodically syncs weights from learner
    """

    # Required fields
    config: WorkerConfig
    weights_path: Path
    gradient_tensor_path: Path  # Path to SharedGradientTensor file
    ui_queue: Optional[mp.Queue] = None
    heartbeats: Any = None  # SharedHeartbeats for monitoring (attached in worker process)
    
    # Gradient tensor - created after model init in __post_init__
    gradient_tensor: SharedGradientTensor = field(init=False, repr=False)

    # Components - initialized in __post_init__ (8)
    env: Any = field(init=False, repr=False)
    base_env: Any = field(init=False, repr=False)
    net: DoubleDQN = field(init=False, repr=False)
    buffer: PrioritizedReplayBuffer = field(init=False, repr=False)
    n_step_buffer: NStepBuffer = field(init=False, repr=False)
    snapshots: Optional[SnapshotManager] = field(init=False, repr=False)
    action_dim: int = field(init=False)
    _fstack: Any = field(init=False, repr=False)

    # State tracking - composed components (7)
    metrics: MetricsTracker = field(init=False)
    episode: EpisodeState = field(init=False)
    weights: WeightSync = field(init=False)
    exploration: EpsilonGreedy = field(init=False)
    timing: TimingStats = field(init=False)
    status_collector: DDQNStatusCollector = field(init=False)
    ui: UIReporter = field(init=False)

    # Remaining state (1)
    _current_state: Optional[np.ndarray] = field(init=False, default=None)

    # Reward normalization
    _reward_normalizer: Optional[RewardNormalizer] = field(init=False, default=None)

    # CSV logging
    _episodes_csv: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize environment, network, and buffer."""
        
        # Get device
        device = self.config.device or detect_device()

        # Create environment
        self.env, self.base_env, self._fstack = create_env(
            level=self.config.level,
            render_frames=self.config.render_frames,
        )
        self.action_dim = self.env.action_space.n

        # Create network
        state_dim = (4, 64, 64)
        self.net = DoubleDQN(
            input_shape=state_dim,
            num_actions=self.action_dim,
        ).to(device)

        # Attach to SharedGradientTensor (must be after model creation)
        self.gradient_tensor = attach_tensor_buffer(
            worker_id=self.config.worker_id,
            model=self.net.online,
            shm_path=self.gradient_tensor_path,
            num_slots=8,  # 8 slots per worker for good async buffering
        )

        # Create buffers using config
        buf = self.config.buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity=buf.capacity,
            obs_shape=state_dim,
            alpha=buf.alpha,
            beta_start=buf.beta_start,
            beta_end=buf.beta_end,
        )
        self.n_step_buffer = NStepBuffer(n_step=buf.n_step, gamma=buf.gamma)
        self.n_step_gamma = buf.gamma**buf.n_step

        # Create snapshot manager
        snap = self.config.snapshot
        if snap.enabled:
            snapshot_config = SnapshotConfig(
                enabled=True,
                slots=snap.slots,
                interval=snap.interval,
                max_restores_without_progress=snap.max_restores_without_progress,
            )
            self.snapshots = SnapshotManager(
                config=snapshot_config,
                base_env=self.base_env,
                fstack=self._fstack,
            )
        else:
            self.snapshots = None

        # Initialize state trackers
        self.metrics = MetricsTracker()
        self.metrics.total_steps = self.config.initial_steps  # Resume support
        self.episode = EpisodeState()
        self.timing = TimingStats()
        
        # Heartbeat tracking for inner loops
        self._last_heartbeat = time.time()

        # Create exploration policy from config
        exp = self.config.exploration
        self.exploration = EpsilonGreedy(
            start=exp.epsilon_start,
            end=exp.epsilon_end,
            decay_steps=exp.decay_steps,
        )

        # Create weight sync from config
        self.weights = WeightSync(
            path=self.weights_path,
            device=device,
            interval=self.config.weight_sync_interval,
        )

        # Load initial weights
        self.weights.load(self.net)
        self.metrics.weight_sync_count = self.weights.count

        # Create status collector (algorithm-specific data gathering)
        self.status_collector = DDQNStatusCollector(
            worker_id=self.config.worker_id,
            metrics=self.metrics,
            episode=self.episode,
            weights=self.weights,
            exploration=self.exploration,
            timing=self.timing,
            buffer=self.buffer,
            snapshots=self.snapshots,
            get_level=lambda: self.base_env.current_level,
            batch_size=self.config.buffer.batch_size,
        )

        # Create UI reporter (generic sender)
        self.ui = UIReporter(worker_id=self.config.worker_id, queue=self.ui_queue)

        # Diagnostic tracking
        self._grads_sent = 0

        # Initialize reward normalizer if using running normalization
        if self.config.reward_norm == "running":
            self._reward_normalizer = RewardNormalizer(clip=self.config.reward_clip)

        # Initialize CSV logging for episode metrics - NEVER truncate existing data
        save_dir = self.weights_path.parent
        self._episodes_csv = save_dir / f"ddqn_worker_{self.config.worker_id}_episodes.csv"
        self._csv_had_data = False
        csv_headers = [
            "timestamp",
            "episode",
            "steps",
            "reward",
            "x_pos",
            "best_x",
            "best_x_ever",
            "deaths",
            "flags",
            "epsilon",
            "weight_version",
            "weight_sync_count",
            "gradients_sent",
            "steps_per_sec",
            "rolling_avg_reward",
            "avg_speed",
            "avg_x_at_death",
            "entropy",
            "buffer_size",
            "buffer_fill_pct",
            "per_beta",
            "snapshot_restores",
            "current_level",
            "game_time",
            "flag_get",
        ]
        if not self._episodes_csv.exists():
            with open(self._episodes_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(csv_headers)
        else:
            # Check if file has data and load last episode number for continuity
            csv_size = self._episodes_csv.stat().st_size
            self._csv_had_data = csv_size > 200  # Headers are ~150 bytes
            if self._csv_had_data:
                # Read last episode number from CSV for graph continuity
                try:
                    with open(self._episodes_csv, "r") as f:
                        lines = f.readlines()
                        if len(lines) > 1:  # Has data beyond header
                            last_line = lines[-1].strip()
                            if last_line:
                                last_episode = int(last_line.split(",")[1])
                                self.metrics.episode_count = last_episode
                except Exception:
                    pass  # Keep default episode_count = 0

    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        """Convert state from (4, 64, 64, 1) to (4, 64, 64)."""
        if state.ndim == 4 and state.shape[-1] == 1:
            state = np.squeeze(state, axis=-1)
        return state

    def _log_episode(self, info: dict) -> None:
        """Log episode metrics to CSV."""
        buffer_fill_pct = len(self.buffer) / self.config.buffer.capacity * 100

        with open(self._episodes_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(),
                self.metrics.episode_count,
                self.episode.length,
                self.episode.reward,
                info.get("x_pos", 0),
                self.episode.best_x,
                self.metrics.best_x_ever,
                self.metrics.deaths,
                self.metrics.flags,
                self.exploration.get_epsilon(self.metrics.total_steps),
                self.weights.version,
                self.weights.count,
                self.metrics.gradients_sent,
                self.timing.steps_per_sec,
                self.metrics.avg_reward,
                self.metrics.avg_speed,
                self.metrics.avg_x_at_death,
                self.metrics.avg_entropy,
                len(self.buffer),
                buffer_fill_pct,
                self.buffer.current_beta,
                self.snapshots.restore_count if self.snapshots else 0,
                self.base_env.current_level,
                info.get("time", 0),
                info.get("flag_get", False),
            ])

    @torch.no_grad()
    def _get_action(self, state: np.ndarray) -> int:
        """Get action using epsilon-greedy policy."""
        if self.exploration.should_explore(self.metrics.total_steps):
            return int(np.random.randint(0, self.action_dim))

        state = self._preprocess_state(state)
        device = self.config.device or detect_device()
        state_tensor = torch.from_numpy(np.expand_dims(state, 0)).float().to(device)
        q_values = self.net.online(state_tensor)

        # Compute entropy
        probs = torch.softmax(q_values / 0.5, dim=1)
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=1).item()
        self.timing.last_entropy = entropy

        return int(q_values.argmax(dim=1).item())

    def _maybe_heartbeat(self) -> None:
        """Send heartbeat if enough time has passed (called from inner loops)."""
        current_time = time.time()
        if current_time - self._last_heartbeat < 5.0:  # Check every 5 seconds
            return
            
        self._last_heartbeat = current_time
        
        # Update shared memory heartbeat (for monitor thread)
        if self.heartbeats is not None:
            try:
                self.heartbeats.update(self.config.worker_id, current_time)
            except Exception:
                pass
        
        # Also send to UI queue (for text UI heartbeat display)
        if self.ui_queue is not None:
            try:
                from mario_rl.training.training_ui import UIMessage, MessageType
                self.ui_queue.put_nowait(UIMessage(
                    msg_type=MessageType.WORKER_HEARTBEAT,
                    source_id=self.config.worker_id,
                    data={
                        "timestamp": current_time,
                        "episodes": self.metrics.episode_count,
                        "steps": self.metrics.total_steps,
                    },
                ))
            except Exception:
                pass  # Queue full or other error - don't block

    def collect_steps(self, num_steps: int) -> int:
        """Collect experiences for num_steps and store in local PER buffer."""
        episodes_completed = 0

        # Get current state
        if self._current_state is None:
            state, _ = self.env.reset()
            self._current_state = state
            if self.snapshots:
                self.snapshots.reset()
        else:
            state = self._current_state

        for step_idx in range(num_steps):
            # Send heartbeat every 16 steps to avoid stall detection during long collect
            if step_idx % 16 == 0:
                self._maybe_heartbeat()
            
            action = self._get_action(state)

            # Track entropy
            if self.metrics.total_steps % 10 == 0:
                self.metrics.add_entropy(self.timing.last_entropy)

            # Step environment with error recovery
            try:
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.timing.record_action()
            except Exception as e:
                # Log error and reset environment
                self.ui.log(f"ERROR during env.step: {e}. Resetting environment.")
                try:
                    # Try to reset environment
                    state, _ = self.env.reset()
                    self._current_state = state
                    # Flush n-step buffer to prevent corruption
                    self.n_step_buffer.reset()
                    # Skip this step and continue
                    continue
                except Exception as reset_error:
                    # If reset also fails, this is critical
                    self.ui.log(f"CRITICAL: Environment reset failed: {reset_error}")
                    raise

            is_dead = info.get("is_dead", False) or info.get("is_dying", False)

            # Normalize reward to prevent Q-value explosion
            raw_reward = reward
            if self.config.reward_norm == "running" and self._reward_normalizer is not None:
                # Running normalization: (r - mean) / std
                normalized_reward = self._reward_normalizer.normalize(reward)
            elif self.config.reward_norm == "scale":
                # Fixed scaling
                normalized_reward = reward * self.config.reward_scale
                if self.config.reward_clip > 0:
                    normalized_reward = float(np.clip(normalized_reward, -self.config.reward_clip, self.config.reward_clip))
            else:
                # No normalization
                normalized_reward = reward

            # Process states
            state_processed = self._preprocess_state(state)
            next_state_processed = self._preprocess_state(next_state)

            # Add to buffers using normalized reward for training
            transition = Transition(
                state=state_processed,
                action=action,
                reward=normalized_reward,
                next_state=next_state_processed,
                done=done,
            )
            n_step_transition = self.n_step_buffer.add(transition)
            if n_step_transition is not None:
                self.buffer.add(n_step_transition)

            # Update tracking with RAW reward (for logging/analysis)
            x_pos = info.get("x_pos", 0)
            self.episode.step(raw_reward, x_pos)
            self.metrics.total_steps += 1
            self.metrics.update_best_x(x_pos)

            if info.get("flag_get", False):
                self.metrics.flags += 1

            # Update PER beta
            progress = min(1.0, self.metrics.total_steps / self.config.exploration.decay_steps)
            self.buffer.update_beta(progress)

            # Send UI status
            if self.metrics.total_steps % 50 == 0:
                self._send_ui_status(info)

            # Save snapshot
            if self.snapshots:
                self.snapshots.maybe_save(state, info)

            # Try restore on death
            if is_dead and self.snapshots:
                game_time = info.get("time", 0)
                checkpoint_time = game_time // self.config.snapshot.interval
                if checkpoint_time > 3:
                    try:
                        restored_state, restored = self.snapshots.try_restore(info, self.episode.best_x)
                        if restored:
                            # Verify restore succeeded and state is valid
                            if restored_state.size > 0:
                                self.metrics.add_death(x_pos)
                                self.n_step_buffer.reset()
                                state = restored_state
                                self._current_state = state
                                continue
                    except Exception:
                        # Restore failed - let episode end normally
                        pass

            if done:
                # Flush N-step buffer
                for trans in self.n_step_buffer.flush():
                    self.buffer.add(trans)

                # Track stats
                self.metrics.episode_count += 1
                episodes_completed += 1
                self.metrics.add_reward(self.episode.reward)

                # Calculate speed
                game_time = info.get("time", 0)
                time_elapsed = self.episode.start_time - game_time
                if time_elapsed > 0:
                    speed = x_pos / time_elapsed
                    self.metrics.add_speed(speed)

                # Track death/flag
                if info.get("flag_get", False):
                    self.metrics.add_flag(game_time)
                elif is_dead:
                    self.metrics.add_death(x_pos)

                # Log episode to CSV
                self._log_episode(info)

                self._send_ui_status(info)

                # Reset for new episode
                state, _ = self.env.reset()
                self._current_state = state
                if self.snapshots:
                    self.snapshots.reset()
                self.episode.reset()
            else:
                state = next_state

        self._current_state = state
        return episodes_completed

    def compute_and_send_gradients(self) -> Dict[str, float]:
        """Sample from PER buffer, compute loss, and send gradients."""
        if len(self.buffer) < self.config.buffer.batch_size:
            return {}

        # Sample batch (returns PERBatch dataclass)
        batch = self.buffer.sample(self.config.buffer.batch_size)

        # Convert to tensors
        device = self.config.device or detect_device()
        states_t = torch.from_numpy(batch.states).to(device)
        actions_t = torch.from_numpy(batch.actions).to(device)
        rewards_t = torch.from_numpy(batch.rewards).to(device)
        next_states_t = torch.from_numpy(batch.next_states).to(device)
        dones_t = torch.from_numpy(batch.dones).to(device)
        weights_t = torch.from_numpy(batch.weights).to(device)

        # Compute Double DQN loss
        self.net.train()
        current_q = self.net.online(states_t)
        current_q_selected = current_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.net.online(next_states_t)
            best_actions = next_q_online.argmax(dim=1)
            next_q_target = self.net.target(next_states_t)
            next_q_selected = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.n_step_gamma * next_q_selected * (1.0 - dones_t)

        td_errors = (current_q_selected - target_q).detach()
        element_wise_loss = torch.nn.functional.huber_loss(current_q_selected, target_q, reduction="none", delta=1.0)
        td_loss = (weights_t * element_wise_loss).mean()

        # Entropy regularization: encourage exploration by penalizing low entropy
        # Convert Q-values to policy using softmax, then compute entropy
        policy = torch.nn.functional.softmax(current_q, dim=1)
        log_policy = torch.nn.functional.log_softmax(current_q, dim=1)
        entropy = -(policy * log_policy).sum(dim=1).mean()

        # Total loss: TD loss - entropy bonus (we want to maximize entropy)
        loss = td_loss - self.config.entropy_coef * entropy

        # Backward
        self.net.online.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.online.parameters(), self.config.max_grad_norm)

        # Update priorities
        self.buffer.update_priorities(batch.indices, td_errors.abs().cpu().numpy())

        # Extract gradients (no serialization - zero-copy via shared memory)
        grads = {
            name: param.grad
            for name, param in self.net.online.named_parameters()
            if param.grad is not None
        }

        # Metrics (stored locally for now, TODO: send via separate channel if needed)
        metrics = {
            "loss": loss.item(),
            "td_loss": td_loss.item(),
            "q_mean": current_q_selected.mean().item(),
            "q_max": current_q_selected.max().item(),
            "td_error": td_errors.abs().mean().item(),
            "per_beta": self.buffer.current_beta,
            "entropy": entropy.item(),  # Policy entropy from Q-values
            "avg_reward": self.metrics.avg_reward,
            "avg_speed": self.metrics.avg_speed,
            "avg_time_to_flag": self.metrics.avg_time_to_flag,
            "total_deaths": self.metrics.deaths,
            "total_flags": self.metrics.flags,
            "best_x_ever": self.metrics.best_x_ever,
        }

        # Write gradients to shared memory (zero-copy, non-blocking)
        success = self.gradient_tensor.write(
            grads=grads,
            version=self.weights.version,
            worker_id=self.config.worker_id,
            timesteps=self.config.buffer.batch_size,
            episodes=self.metrics.episode_count,
            loss=metrics["loss"],
            q_mean=metrics["q_mean"],
            td_error=metrics["td_error"],
        )
        if success:
            self.metrics.gradients_sent += 1
            self._grads_sent += 1

        # Log occasionally to UI
        if self.metrics.total_steps % 500 == 0:
            self.ui.log(
                f"Grad: loss={metrics['loss']:.4f}, "
                f"q={metrics['q_mean']:.2f}, td={metrics['td_error']:.4f}"
            )

        return metrics

    def _send_ui_status(self, info: dict) -> None:
        """Send status to UI queue using collector pattern."""
        status = self.status_collector.collect(info)
        self.ui.send_status(status)

    def get_diagnostics(self) -> dict:
        """Get diagnostic information about worker state."""
        current_epsilon = self.exploration.get_epsilon(self.metrics.total_steps)
        buffer_fill_pct = len(self.buffer) / self.config.buffer.capacity * 100

        return {
            "worker_id": self.config.worker_id,
            "total_steps": self.metrics.total_steps,
            "episodes": self.metrics.episode_count,
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.config.buffer.capacity,
            "buffer_fill_pct": buffer_fill_pct,
            "epsilon": current_epsilon,
            "best_x": self.metrics.best_x_ever,
            "weight_sync_count": self.metrics.weight_sync_count,
            "grads_sent": self._grads_sent,
            "can_train": len(self.buffer) >= self.config.buffer.batch_size,
        }

    def run(self) -> None:
        """Main training loop."""
        device = self.config.device or detect_device()
        per_mode = "PER" if self.config.buffer.alpha > 0 else "uniform"
        self.ui.log(
            f"Worker {self.config.worker_id} started (level={self.config.level}, "
            f"device={device}, ε_end={self.config.exploration.epsilon_end:.4f}, {per_mode})"
        )
        if self._csv_had_data:
            self.ui.log(f"Worker {self.config.worker_id} resuming from episode {self.metrics.episode_count}")
        elif self._episodes_csv.exists():
            self.ui.log(f"Worker {self.config.worker_id} WARNING: CSV exists but appears empty")

        loop_count = 0
        
        while True:
            loop_count += 1
            
            # Send heartbeat at start of each outer loop iteration
            self._maybe_heartbeat()

            if loop_count % 100 == 1:
                # Get current epsilon
                current_epsilon = self.exploration.get_epsilon(self.metrics.total_steps)
                buffer_fill_pct = len(self.buffer) / self.config.buffer.capacity * 100

                self.ui.log(
                    f"Loop {loop_count}: "
                    f"buf={len(self.buffer)}/{self.config.buffer.capacity} "
                    f"({buffer_fill_pct:.1f}%), "
                    f"steps={self.metrics.total_steps:,}, "
                    f"eps={self.metrics.episode_count}, "
                    f"ε={current_epsilon:.4f}, "
                    f"best_x={self.metrics.best_x_ever}, "
                    f"grads={self._grads_sent}"
                )

            # Sync weights if needed
            if self.weights.maybe_sync(self.net):
                self.metrics.weight_sync_count = self.weights.count
                self.ui.log(f"Synced weights v{self.weights.version}")

            self.collect_steps(self.config.steps_per_collection)
            self.timing.update_speed(self.config.steps_per_collection)

            if len(self.buffer) >= self.config.buffer.batch_size:
                for _ in range(self.config.train_steps):
                    self.compute_and_send_gradients()
                    self._maybe_heartbeat()  # Heartbeat after each gradient

            # Periodic diagnostic dump
            if loop_count % 500 == 0:
                diag = self.get_diagnostics()
                self.ui.log(
                    f"DIAG: steps={diag['total_steps']:,}, "
                    f"eps={diag['episodes']}, "
                    f"buf={diag['buffer_size']}/{diag['buffer_capacity']} "
                    f"({diag['buffer_fill_pct']:.1f}%), "
                    f"ε={diag['epsilon']:.4f}, "
                    f"best_x={diag['best_x']}, "
                    f"can_train={diag['can_train']}, "
                    f"wgt_sync={diag['weight_sync_count']}, "
                    f"grads={diag['grads_sent']}"
                )


def run_ddqn_worker(
    config: WorkerConfig,
    weights_path: Path,
    gradient_tensor_path: Path,
    ui_queue: Optional[mp.Queue] = None,
    heartbeat_path: Optional[Path] = None,
    crash_log_dir: Optional[Path] = None,
) -> None:
    """Entry point for worker process."""
    import sys
    import signal
    import traceback
    from datetime import datetime
    from mario_rl.training.shared_gradients import SharedHeartbeats
    
    # Attach to shared heartbeats if path provided
    heartbeats = None
    if heartbeat_path is not None:
        try:
            heartbeats = SharedHeartbeats.__new__(SharedHeartbeats)
            heartbeats.shm_path = heartbeat_path
            heartbeats.num_workers = 32  # Max workers - actual count doesn't matter for update()
            heartbeats.buffer_size = 32 * 8
            heartbeats._attach()
        except Exception:
            heartbeats = None
    
    # Setup crash log file
    crash_log_file = None
    stack_trace_file = None
    if crash_log_dir is not None:
        crash_log_dir.mkdir(parents=True, exist_ok=True)
        crash_log_file = crash_log_dir / f"worker_{config.worker_id}_crashes.log"
        stack_trace_file = crash_log_dir / f"worker_{config.worker_id}_stack.log"
    
    # Setup signal handler to dump stack trace
    def dump_stack_trace(signum, frame):
        """Dump stack trace when receiving SIGUSR1."""
        if stack_trace_file is not None:
            try:
                with open(stack_trace_file, "a") as f:
                    f.write(f"\n{'=' * 80}\n")
                    f.write(f"Stack trace dump at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Worker {config.worker_id} (PID: {os.getpid()})\n")
                    f.write(f"Signal: {signum}\n")
                    f.write(f"{'=' * 80}\n\n")
                    
                    # Dump all thread stacks
                    import threading
                    for thread_id, frame_obj in sys._current_frames().items():
                        f.write(f"\nThread {thread_id}:\n")
                        f.write(''.join(traceback.format_stack(frame_obj)))
                    
                    f.write(f"\n{'=' * 80}\n")
                    f.flush()
            except Exception as e:
                print(f"[W{config.worker_id}] Failed to dump stack trace: {e}", file=sys.stderr, flush=True)
    
    # Register signal handler (SIGUSR1)
    signal.signal(signal.SIGUSR1, dump_stack_trace)
    
    # Send startup heartbeat via shared memory
    if heartbeats is not None:
        try:
            heartbeats.update(config.worker_id)
        except Exception:
            pass
    
    try:
        worker = DDQNWorker(
            config=config,
            weights_path=weights_path,
            gradient_tensor_path=gradient_tensor_path,
            ui_queue=ui_queue,
            heartbeats=heartbeats,
        )
        
        # Send ready heartbeat
        if heartbeats is not None:
            try:
                heartbeats.update(config.worker_id)
            except Exception:
                pass
        
        worker.run()
    except KeyboardInterrupt:
        # Graceful shutdown
        print(f"[W{config.worker_id}] Interrupted", file=sys.stderr, flush=True)
    except Exception as e:
        # Log crash with full traceback
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_msg = f"CRASH: {e}\n{traceback.format_exc()}"
        
        # Write to crash log file
        if crash_log_file is not None:
            try:
                with open(crash_log_file, "a") as f:
                    f.write(f"\n{'=' * 80}\n")
                    f.write(f"Crash at {timestamp}\n")
                    f.write(f"{'=' * 80}\n")
                    f.write(f"Error: {e}\n")
                    f.write(f"Type: {type(e).__name__}\n\n")
                    f.write("Traceback:\n")
                    f.write(traceback.format_exc())
                    f.write("\n")
                    f.flush()
            except Exception as log_err:
                print(f"[W{config.worker_id}] Failed to write crash log: {log_err}", file=sys.stderr, flush=True)
        
        if ui_queue is not None:
            try:
                from mario_rl.training.training_ui import UIMessage
                from mario_rl.training.training_ui import MessageType

                ui_queue.put_nowait(
                    UIMessage(
                        msg_type=MessageType.WORKER_LOG,
                        source_id=config.worker_id,
                        data={"text": error_msg},
                    )
                )
            except Exception:
                pass

        print(f"[W{config.worker_id}] {error_msg}", file=sys.stderr, flush=True)
        raise
