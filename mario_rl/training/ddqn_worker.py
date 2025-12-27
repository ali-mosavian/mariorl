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

from mario_rl.buffers import NStepBuffer
from mario_rl.core.config import LevelType
from mario_rl.agent.ddqn_net import DoubleDQN
from mario_rl.core.config import SnapshotConfig
from mario_rl.environment.factory import create_env
from mario_rl.buffers import PrioritizedReplayBuffer
from mario_rl.training.snapshot import SnapshotManager


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
    worker_id: int
    weights_path: Path
    gradient_queue: mp.Queue

    # Configuration
    level: LevelType = (1, 1)
    n_step: int = 3
    gamma: float = 0.99
    render_frames: bool = False
    weight_sync_interval: float = 5.0

    # Local buffer settings (PER)
    local_buffer_size: int = 10_000
    batch_size: int = 32
    steps_per_collection: int = 64
    train_steps: int = 4

    # PER hyperparameters
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0

    # Exploration
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 100_000

    # Gradient clipping
    max_grad_norm: float = 10.0

    # Device
    device: Optional[str] = None

    # UI
    ui_queue: Optional[mp.Queue] = None

    # Snapshots
    use_snapshots: bool = True
    snapshot_slots: int = 10
    snapshot_interval: int = 5
    max_restores_without_progress: int = 3

    # Private fields (initialized in __post_init__)
    env: Any = field(init=False, repr=False)
    base_env: Any = field(init=False, repr=False)
    net: Any = field(init=False, repr=False)
    buffer: PrioritizedReplayBuffer = field(init=False, repr=False)
    n_step_buffer: NStepBuffer = field(init=False, repr=False)
    snapshots: Optional[SnapshotManager] = field(init=False, repr=False)
    action_dim: int = field(init=False)
    _fstack: Any = field(init=False, repr=False)

    # Tracking
    episode_count: int = field(init=False, default=0)
    total_steps: int = field(init=False, default=0)
    episode_reward: float = field(init=False, default=0.0)
    episode_length: int = field(init=False, default=0)
    best_x: int = field(init=False, default=0)
    best_x_ever: int = field(init=False, default=0)
    flags: int = field(init=False, default=0)
    deaths: int = field(init=False, default=0)
    reward_history: List[float] = field(init=False, default_factory=list)
    last_weight_sync: float = field(init=False, default=0.0)
    weight_version: int = field(init=False, default=0)
    weight_sync_count: int = field(init=False, default=0)
    gradients_sent: int = field(init=False, default=0)
    steps_per_sec: float = field(init=False, default=0.0)
    _last_time: float = field(init=False, default=0.0)
    current_epsilon: float = field(init=False, default=1.0)

    # Additional metrics
    x_at_death_history: List[int] = field(init=False, default_factory=list)
    time_to_flag_history: List[int] = field(init=False, default_factory=list)
    speed_history: List[float] = field(init=False, default_factory=list)
    episode_start_time: int = field(init=False, default=400)

    # Entropy tracking
    entropy_history: List[float] = field(init=False, default_factory=list)
    last_entropy: float = field(init=False, default=0.0)

    # Debug
    _last_action_time: float = field(init=False, default=0.0)
    _current_state: Optional[np.ndarray] = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Initialize environment, network, and buffer."""
        # Auto-detect device
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Create environment
        self.env, self.base_env, self._fstack = create_env(
            level=self.level,
            render_frames=self.render_frames,
        )
        self.action_dim = self.env.action_space.n

        # Create network
        state_dim = (4, 64, 64)
        self.net = DoubleDQN(
            input_shape=state_dim,
            num_actions=self.action_dim,
        ).to(self.device)

        # Create buffers
        self.buffer = PrioritizedReplayBuffer(
            capacity=self.local_buffer_size,
            obs_shape=state_dim,
            alpha=self.per_alpha,
            beta_start=self.per_beta_start,
            beta_end=self.per_beta_end,
        )
        self.n_step_buffer = NStepBuffer(n_step=self.n_step, gamma=self.gamma)
        self.n_step_gamma = self.gamma**self.n_step

        # Create snapshot manager
        if self.use_snapshots:
            snapshot_config = SnapshotConfig(
                enabled=True,
                slots=self.snapshot_slots,
                interval=self.snapshot_interval,
                max_restores_without_progress=self.max_restores_without_progress,
            )
            self.snapshots = SnapshotManager(
                config=snapshot_config,
                base_env=self.base_env,
                fstack=self._fstack,
            )
        else:
            self.snapshots = None

        # Initialize timing
        self._last_time = time.time()
        self._last_action_time = time.time()

        # Load initial weights
        self._load_weights()

    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        """Convert state from (4, 64, 64, 1) to (4, 64, 64)."""
        if state.ndim == 4 and state.shape[-1] == 1:
            state = np.squeeze(state, axis=-1)
        return state

    def _load_weights(self) -> bool:
        """Load latest weights from disk."""
        if not self.weights_path.exists():
            return False

        for attempt in range(3):
            try:
                checkpoint = torch.load(
                    self.weights_path,
                    map_location=self.device,
                    weights_only=True,
                )
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self.net.load_state_dict(checkpoint["state_dict"])
                    self.weight_version = checkpoint.get("version", 0)
                else:
                    self.net.load_state_dict(checkpoint)
                self.net.sync_target()
                self.last_weight_sync = time.time()
                self.weight_sync_count += 1
                return True
            except Exception:
                if attempt < 2:
                    time.sleep(0.1)
                continue
        return False

    def _maybe_sync_weights(self) -> None:
        """Sync weights if enough time has passed."""
        if time.time() - self.last_weight_sync >= self.weight_sync_interval:
            self._load_weights()

    def _get_epsilon(self) -> float:
        """Get current epsilon based on decay schedule."""
        progress = min(1.0, self.total_steps / self.eps_decay_steps)
        return self.eps_start + (self.eps_end - self.eps_start) * progress

    @torch.no_grad()
    def _get_action(self, state: np.ndarray) -> int:
        """Get action using epsilon-greedy policy."""
        self.current_epsilon = self._get_epsilon()

        if np.random.random() < self.current_epsilon:
            return int(np.random.randint(0, self.action_dim))

        state = self._preprocess_state(state)
        state_tensor = torch.from_numpy(np.expand_dims(state, 0)).float().to(self.device)
        q_values = self.net.online(state_tensor)

        # Compute entropy
        probs = torch.softmax(q_values / 0.5, dim=1)
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=1).item()
        self.last_entropy = entropy

        return int(q_values.argmax(dim=1).item())

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

        for _ in range(num_steps):
            action = self._get_action(state)

            # Track entropy
            if self.total_steps % 10 == 0:
                self.entropy_history.append(self.last_entropy)
                if len(self.entropy_history) > 100:
                    self.entropy_history.pop(0)

            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self._last_action_time = time.time()

            is_dead = info.get("is_dead", False) or info.get("is_dying", False)

            # Process states
            state_processed = self._preprocess_state(state)
            next_state_processed = self._preprocess_state(next_state)

            # Add to buffers
            n_step_transition = self.n_step_buffer.add(state_processed, action, reward, next_state_processed, done)
            if n_step_transition is not None:
                self.buffer.add(*n_step_transition)

            # Update tracking
            self.episode_reward += reward
            self.episode_length += 1
            self.total_steps += 1

            x_pos = info.get("x_pos", 0)
            if x_pos > self.best_x:
                self.best_x = x_pos
            if x_pos > self.best_x_ever:
                self.best_x_ever = x_pos

            if info.get("flag_get", False):
                self.flags += 1

            # Update PER beta
            progress = min(1.0, self.total_steps / self.eps_decay_steps)
            self.buffer.update_beta(progress)

            # Send UI status
            if self.total_steps % 50 == 0:
                self._send_ui_status(info)

            # Save snapshot
            if self.snapshots:
                self.snapshots.maybe_save(state, info)

            # Try restore on death
            if is_dead and self.snapshots:
                game_time = info.get("time", 0)
                checkpoint_time = game_time // self.snapshot_interval
                if checkpoint_time > 3:
                    restored_state, restored = self.snapshots.try_restore(info, self.best_x)
                    if restored:
                        self.deaths += 1
                        self.x_at_death_history.append(x_pos)
                        if len(self.x_at_death_history) > 20:
                            self.x_at_death_history.pop(0)
                        self.n_step_buffer.reset()
                        state = restored_state
                        self._current_state = state
                        continue

            if done:
                # Flush N-step buffer
                for transition in self.n_step_buffer.flush():
                    self.buffer.add(*transition)

                # Track stats
                self.episode_count += 1
                episodes_completed += 1
                self.reward_history.append(self.episode_reward)
                if len(self.reward_history) > 100:
                    self.reward_history.pop(0)

                # Calculate speed
                game_time = info.get("time", 0)
                time_elapsed = self.episode_start_time - game_time
                if time_elapsed > 0:
                    speed = x_pos / time_elapsed
                    self.speed_history.append(speed)
                    if len(self.speed_history) > 20:
                        self.speed_history.pop(0)

                # Track death/flag
                if info.get("flag_get", False):
                    self.time_to_flag_history.append(game_time)
                    if len(self.time_to_flag_history) > 20:
                        self.time_to_flag_history.pop(0)
                elif is_dead:
                    self.deaths += 1
                    self.x_at_death_history.append(x_pos)
                    if len(self.x_at_death_history) > 20:
                        self.x_at_death_history.pop(0)

                self._send_ui_status(info)

                # Reset for new episode
                state, _ = self.env.reset()
                self._current_state = state
                if self.snapshots:
                    self.snapshots.reset()
                self.episode_reward = 0.0
                self.episode_length = 0
                self.best_x = 0
            else:
                state = next_state

        self._current_state = state
        return episodes_completed

    def compute_and_send_gradients(self) -> Dict[str, float]:
        """Sample from PER buffer, compute loss, and send gradients."""
        if len(self.buffer) < self.batch_size:
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(self.batch_size)

        # Convert to tensors
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)
        weights_t = torch.from_numpy(weights).to(self.device)

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
        loss = (weights_t * element_wise_loss).mean()

        # Backward
        self.net.online.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.online.parameters(), self.max_grad_norm)

        # Update priorities
        self.buffer.update_priorities(indices, td_errors.abs().cpu().numpy())

        # Extract gradients
        grads = {
            name: param.grad.cpu().clone()
            for name, param in self.net.online.named_parameters()
            if param.grad is not None
        }

        # Metrics
        rolling_avg_reward = np.mean(self.reward_history) if self.reward_history else 0.0
        avg_speed = np.mean(self.speed_history) if self.speed_history else 0.0
        avg_entropy = np.mean(self.entropy_history) if self.entropy_history else 0.0
        metrics = {
            "loss": loss.item(),
            "q_mean": current_q_selected.mean().item(),
            "q_max": current_q_selected.max().item(),
            "td_error": td_errors.abs().mean().item(),
            "per_beta": self.buffer.current_beta,
            "entropy": avg_entropy,
            "avg_reward": rolling_avg_reward,
            "avg_speed": avg_speed,
            "total_deaths": self.deaths,
            "total_flags": self.flags,
            "best_x_ever": self.best_x_ever,
        }

        # Send gradients
        gradient_packet = {
            "grads": grads,
            "timesteps": self.batch_size,
            "episodes": self.episode_count,
            "worker_id": self.worker_id,
            "weight_version": self.weight_version,
            "metrics": metrics,
        }

        try:
            self.gradient_queue.put(gradient_packet, timeout=0.5)
            self.gradients_sent += 1
        except Exception:
            pass

        return metrics

    def _send_ui_status(self, info: dict) -> None:
        """Send status to UI queue."""
        if self.ui_queue is None:
            return

        try:
            from mario_rl.training.training_ui import UIMessage
            from mario_rl.training.training_ui import MessageType

            rolling_avg = np.mean(self.reward_history) if self.reward_history else 0.0
            level_str = self.base_env.current_level
            avg_speed = np.mean(self.speed_history) if self.speed_history else 0.0
            avg_x_at_death = np.mean(self.x_at_death_history) if self.x_at_death_history else 0.0
            avg_time_to_flag = np.mean(self.time_to_flag_history) if self.time_to_flag_history else 0.0
            avg_entropy = np.mean(self.entropy_history) if self.entropy_history else 0.0

            snapshot_restores = self.snapshots.restore_count if self.snapshots else 0
            restores_without_progress = self.snapshots.restores_without_progress if self.snapshots else 0
            max_restores = self.snapshots.max_restores if self.snapshots else 3

            msg = UIMessage(
                msg_type=MessageType.WORKER_STATUS,
                source_id=self.worker_id,
                data={
                    "episode": self.episode_count,
                    "step": self.episode_length,
                    "reward": self.episode_reward,
                    "x_pos": info.get("x_pos", 0),
                    "game_time": info.get("time", 0),
                    "best_x": self.best_x,
                    "best_x_ever": self.best_x_ever,
                    "deaths": self.deaths,
                    "flags": self.flags,
                    "epsilon": self.current_epsilon,
                    "experiences": self.total_steps,
                    "q_mean": 0.0,
                    "q_max": 0.0,
                    "weight_sync_count": self.weight_sync_count,
                    "gradients_sent": self.gradients_sent,
                    "steps_per_sec": self.steps_per_sec,
                    "snapshot_restores": snapshot_restores,
                    "restores_without_progress": restores_without_progress,
                    "max_restores": max_restores,
                    "current_level": level_str,
                    "last_weight_sync": self.last_weight_sync,
                    "rolling_avg_reward": rolling_avg,
                    "first_flag_time": avg_time_to_flag,
                    "per_beta": self.buffer.current_beta,
                    "avg_speed": avg_speed,
                    "avg_x_at_death": avg_x_at_death,
                    "avg_time_to_flag": avg_time_to_flag,
                    "entropy": avg_entropy,
                    "last_action_time": self._last_action_time,
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
                    msg_type=MessageType.WORKER_LOG,
                    source_id=self.worker_id,
                    data={"text": text},
                )
                self.ui_queue.put_nowait(msg)
            except Exception:
                print(f"[W{self.worker_id}] {text}")
        else:
            print(f"[W{self.worker_id}] {text}")

    def run(self) -> None:
        """Main training loop."""
        self._log(
            f"Worker {self.worker_id} started (level={self.level}, "
            f"device={self.device}, ε_end={self.eps_end:.4f}, PER)"
        )

        loop_count = 0
        while True:
            loop_count += 1

            if loop_count % 100 == 1:
                import sys

                print(
                    f"[W{self.worker_id}] Loop {loop_count}: buf={len(self.buffer)}, steps={self.total_steps}",
                    file=sys.stderr,
                    flush=True,
                )

            self._maybe_sync_weights()
            self.collect_steps(self.steps_per_collection)

            now = time.time()
            elapsed = now - self._last_time
            self.steps_per_sec = self.steps_per_collection / elapsed if elapsed > 0 else 0
            self._last_time = now

            if len(self.buffer) >= self.batch_size:
                for _ in range(self.train_steps):
                    self.compute_and_send_gradients()


def run_ddqn_worker(
    worker_id: int,
    weights_path: Path,
    gradient_queue: mp.Queue,
    level: LevelType = (1, 1),
    ui_queue: Optional[mp.Queue] = None,
    **kwargs: Any,
) -> None:
    """Entry point for worker process."""
    try:
        worker = DDQNWorker(
            worker_id=worker_id,
            weights_path=weights_path,
            gradient_queue=gradient_queue,
            level=level,
            ui_queue=ui_queue,
            **kwargs,
        )
        worker.run()
    except Exception as e:
        import sys
        import traceback

        if ui_queue is not None:
            try:
                from mario_rl.training.training_ui import UIMessage
                from mario_rl.training.training_ui import MessageType

                ui_queue.put_nowait(
                    UIMessage(
                        msg_type=MessageType.WORKER_LOG,
                        source_id=worker_id,
                        data={"text": f"CRASH: {e}\n{traceback.format_exc()}"},
                    )
                )
            except Exception:
                pass

        print(f"[W{worker_id}] CRASH: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        raise
