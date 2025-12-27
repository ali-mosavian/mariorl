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
from mario_rl.core.config import LevelType
from mario_rl.core.types import Transition
from mario_rl.core.timing import TimingStats
from mario_rl.agent.ddqn_net import DoubleDQN
from mario_rl.core.device import detect_device
from mario_rl.core.episode import EpisodeState
from mario_rl.core.config import SnapshotConfig
from mario_rl.core.metrics import MetricsTracker
from mario_rl.core.weight_sync import WeightSync
from mario_rl.core.exploration import EpsilonGreedy
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

    # Components (initialized in __post_init__)
    env: Any = field(init=False, repr=False)
    base_env: Any = field(init=False, repr=False)
    net: DoubleDQN = field(init=False, repr=False)
    buffer: PrioritizedReplayBuffer = field(init=False, repr=False)
    n_step_buffer: NStepBuffer = field(init=False, repr=False)
    snapshots: Optional[SnapshotManager] = field(init=False, repr=False)
    action_dim: int = field(init=False)
    _fstack: Any = field(init=False, repr=False)

    # State tracking (composed components)
    metrics: MetricsTracker = field(init=False)
    episode: EpisodeState = field(init=False)
    weights: WeightSync = field(init=False)
    exploration: EpsilonGreedy = field(init=False)
    timing: TimingStats = field(init=False)

    # Remaining state
    _current_state: Optional[np.ndarray] = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Initialize environment, network, and buffer."""
        # Auto-detect device
        if self.device is None:
            self.device = detect_device()

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

        # Initialize state trackers
        self.metrics = MetricsTracker()
        self.episode = EpisodeState()
        self.timing = TimingStats()
        self.exploration = EpsilonGreedy(
            start=self.eps_start,
            end=self.eps_end,
            decay_steps=self.eps_decay_steps,
        )
        self.weights = WeightSync(
            path=self.weights_path,
            device=self.device,
            interval=self.weight_sync_interval,
        )

        # Load initial weights
        self.weights.load(self.net)
        self.metrics.weight_sync_count = self.weights.count

    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        """Convert state from (4, 64, 64, 1) to (4, 64, 64)."""
        if state.ndim == 4 and state.shape[-1] == 1:
            state = np.squeeze(state, axis=-1)
        return state

    @torch.no_grad()
    def _get_action(self, state: np.ndarray) -> int:
        """Get action using epsilon-greedy policy."""
        # Check if should explore
        if self.exploration.should_explore(self.metrics.total_steps):
            return int(np.random.randint(0, self.action_dim))

        state = self._preprocess_state(state)
        state_tensor = torch.from_numpy(np.expand_dims(state, 0)).float().to(self.device)
        q_values = self.net.online(state_tensor)

        # Compute entropy
        probs = torch.softmax(q_values / 0.5, dim=1)
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=1).item()
        self.timing.last_entropy = entropy

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
            if self.metrics.total_steps % 10 == 0:
                self.metrics.add_entropy(self.timing.last_entropy)

            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.timing.record_action()

            is_dead = info.get("is_dead", False) or info.get("is_dying", False)

            # Process states
            state_processed = self._preprocess_state(state)
            next_state_processed = self._preprocess_state(next_state)

            # Add to buffers using Transition
            transition = Transition(
                state=state_processed,
                action=action,
                reward=reward,
                next_state=next_state_processed,
                done=done,
            )
            n_step_transition = self.n_step_buffer.add(transition)
            if n_step_transition is not None:
                self.buffer.add(n_step_transition)

            # Update tracking
            x_pos = info.get("x_pos", 0)
            self.episode.step(reward, x_pos)
            self.metrics.total_steps += 1
            self.metrics.update_best_x(x_pos)

            if info.get("flag_get", False):
                self.metrics.flags += 1

            # Update PER beta
            progress = min(1.0, self.metrics.total_steps / self.eps_decay_steps)
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
                checkpoint_time = game_time // self.snapshot_interval
                if checkpoint_time > 3:
                    restored_state, restored = self.snapshots.try_restore(info, self.episode.best_x)
                    if restored:
                        self.metrics.add_death(x_pos)
                        self.n_step_buffer.reset()
                        state = restored_state
                        self._current_state = state
                        continue

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
        if len(self.buffer) < self.batch_size:
            return {}

        # Sample batch (returns PERBatch dataclass)
        batch = self.buffer.sample(self.batch_size)

        # Convert to tensors
        states_t = torch.from_numpy(batch.states).to(self.device)
        actions_t = torch.from_numpy(batch.actions).to(self.device)
        rewards_t = torch.from_numpy(batch.rewards).to(self.device)
        next_states_t = torch.from_numpy(batch.next_states).to(self.device)
        dones_t = torch.from_numpy(batch.dones).to(self.device)
        weights_t = torch.from_numpy(batch.weights).to(self.device)

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
        self.buffer.update_priorities(batch.indices, td_errors.abs().cpu().numpy())

        # Extract gradients
        grads = {
            name: param.grad.cpu().clone()
            for name, param in self.net.online.named_parameters()
            if param.grad is not None
        }

        # Metrics
        metrics = {
            "loss": loss.item(),
            "q_mean": current_q_selected.mean().item(),
            "q_max": current_q_selected.max().item(),
            "td_error": td_errors.abs().mean().item(),
            "per_beta": self.buffer.current_beta,
            "entropy": self.metrics.avg_entropy,
            "avg_reward": self.metrics.avg_reward,
            "avg_speed": self.metrics.avg_speed,
            "total_deaths": self.metrics.deaths,
            "total_flags": self.metrics.flags,
            "best_x_ever": self.metrics.best_x_ever,
        }

        # Send gradients
        gradient_packet = {
            "grads": grads,
            "timesteps": self.batch_size,
            "episodes": self.metrics.episode_count,
            "worker_id": self.worker_id,
            "weight_version": self.weights.version,
            "metrics": metrics,
        }

        try:
            self.gradient_queue.put(gradient_packet, timeout=0.5)
            self.metrics.gradients_sent += 1
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

            level_str = self.base_env.current_level
            snapshot_restores = self.snapshots.restore_count if self.snapshots else 0
            restores_without_progress = self.snapshots.restores_without_progress if self.snapshots else 0
            max_restores = self.snapshots.max_restores if self.snapshots else 3
            current_epsilon = self.exploration.get_epsilon(self.metrics.total_steps)

            msg = UIMessage(
                msg_type=MessageType.WORKER_STATUS,
                source_id=self.worker_id,
                data={
                    "episode": self.metrics.episode_count,
                    "step": self.episode.length,
                    "reward": self.episode.reward,
                    "x_pos": info.get("x_pos", 0),
                    "game_time": info.get("time", 0),
                    "best_x": self.episode.best_x,
                    "best_x_ever": self.metrics.best_x_ever,
                    "deaths": self.metrics.deaths,
                    "flags": self.metrics.flags,
                    "epsilon": current_epsilon,
                    "experiences": self.metrics.total_steps,
                    "q_mean": 0.0,
                    "q_max": 0.0,
                    "weight_sync_count": self.weights.count,
                    "gradients_sent": self.metrics.gradients_sent,
                    "steps_per_sec": self.timing.steps_per_sec,
                    "snapshot_restores": snapshot_restores,
                    "restores_without_progress": restores_without_progress,
                    "max_restores": max_restores,
                    "current_level": level_str,
                    "last_weight_sync": self.weights.last_sync,
                    "rolling_avg_reward": self.metrics.avg_reward,
                    "first_flag_time": self.metrics.avg_time_to_flag,
                    "per_beta": self.buffer.current_beta,
                    "avg_speed": self.metrics.avg_speed,
                    "avg_x_at_death": self.metrics.avg_x_at_death,
                    "avg_time_to_flag": self.metrics.avg_time_to_flag,
                    "entropy": self.metrics.avg_entropy,
                    "last_action_time": self.timing.last_action_time,
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
                    f"[W{self.worker_id}] Loop {loop_count}: "
                    f"buf={len(self.buffer)}, steps={self.metrics.total_steps}",
                    file=sys.stderr,
                    flush=True,
                )

            # Sync weights if needed
            if self.weights.maybe_sync(self.net):
                self.metrics.weight_sync_count = self.weights.count

            self.collect_steps(self.steps_per_collection)
            self.timing.update_speed(self.steps_per_collection)

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
