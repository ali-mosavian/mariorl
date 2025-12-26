"""
Distributed DDQN Worker with local gradient computation.

Workers collect experiences in a local replay buffer, compute DQN loss locally,
and send only gradients to the learner. This is similar to Gorila DQN.

Architecture
============

                    ┌─────────────────────────┐
                    │    LEARNER (GPU/MPS)    │
                    │  ┌───────────────────┐  │
                    │  │  Global Network   │  │
                    │  │    (weights θ)    │  │
                    │  └───────────────────┘  │
                    │           │             │
                    │  ┌────────▼────────┐    │
                    │  │    Optimizer    │    │
                    │  │ (gradient step) │    │
                    │  └────────┬────────┘    │
                    │           │             │
                    │  ┌────────▼────────┐    │
                    │  │   weights.pt    │    │
                    └──┴────────┬────────┴────┘
                                │
           ┌────────────────────┼────────────────────┐
           │                    │                    │
           ▼                    ▼                    ▼
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │  WORKER 0   │      │  WORKER 1   │      │  WORKER N   │
    │ ┌─────────┐ │      │ ┌─────────┐ │      │ ┌─────────┐ │
    │ │  Local  │ │      │ │  Local  │ │      │ │  Local  │ │
    │ │ Network │ │      │ │ Network │ │      │ │ Network │ │
    │ │ + Buffer│ │      │ │ + Buffer│ │      │ │ + Buffer│ │
    │ └────┬────┘ │      │ └────┬────┘ │      │ └────┬────┘ │
    │      │      │      │      │      │      │      │      │
    │ 1.Collect   │      │ 1.Collect   │      │ 1.Collect   │
    │   N steps   │      │   N steps   │      │   N steps   │
    │      │      │      │      │      │      │      │      │
    │ 2.Sample    │      │ 2.Sample    │      │ 2.Sample    │
    │   batch     │      │   batch     │      │   batch     │
    │      │      │      │      │      │      │      │      │
    │ 3.Compute   │      │ 3.Compute   │      │ 3.Compute   │
    │   DQN loss  │      │   DQN loss  │      │   DQN loss  │
    │      │      │      │      │      │      │      │      │
    │ 4.Backward  │      │ 4.Backward  │      │ 4.Backward  │
    │   (grads)   │      │   (grads)   │      │   (grads)   │
    │      │      │      │      │      │      │      │      │
    └──────┼──────┘      └──────┼──────┘      └──────┼──────┘
           │                    │                    │
           │    GRADIENT QUEUE (small ~2MB each)     │
           └────────────────────┼────────────────────┘
                                │
                                ▼
                          ┌──────────┐
                          │ LEARNER  │
                          │ applies  │
                          │ gradients│
                          └──────────┘

Data Flow
=========

Standard DQN:   Worker → [experiences ~8MB] → Learner (compute loss + backward)
Distributed:    Worker (compute loss + backward) → [gradients ~2MB] → Learner

Benefits:
- ~4x less data through IPC
- Distributed computation across worker CPUs/GPUs
- Reduced learner bottleneck
- Each worker maintains diverse local buffer
"""

import os
import time
import multiprocessing as mp

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from typing import Any
from typing import Dict
from typing import List
from pathlib import Path
from typing import Tuple
from typing import Literal
from typing import Optional
from dataclasses import field
from dataclasses import dataclass

import torch
import numpy as np
from torch import nn
from gymnasium.spaces import Box
from nes_py.wrappers import JoypadSpace
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import TransformObservation
from gymnasium.wrappers import FrameStackObservation
from gym_super_mario_bros import actions as smb_actions

from mario_rl.agent.ddqn_net import DoubleDQN
from mario_rl.environment.wrappers import SkipFrame
from mario_rl.environment.wrappers import ResizeObservation
from mario_rl.environment.mariogym import SuperMarioBrosMultiLevel

LevelType = Literal["sequential", "random"] | tuple[Literal[1, 2, 3, 4, 5, 6, 7, 8], Literal[1, 2, 3, 4]]


def create_env(level: LevelType = (1, 1), render_frames: bool = False):
    """Create wrapped Mario environment."""
    if render_frames:
        try:
            from pyglet.window import key
            import nes_py._image_viewer as _iv

            _iv.key = key
        except Exception:
            pass

    base_env = SuperMarioBrosMultiLevel(level=level)
    env = JoypadSpace(base_env, actions=smb_actions.COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4, render_frames=render_frames)
    env = GrayscaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=64)
    env = TransformObservation(
        env,
        func=lambda x: x / 255.0,
        observation_space=Box(low=0.0, high=1.0, shape=(64, 64, 1), dtype=np.float32),
    )
    env = FrameStackObservation(env, stack_size=4)
    return env, base_env


class LocalReplayBuffer:
    """
    Small local replay buffer for each worker.

    Workers store experiences locally and sample for gradient computation.
    This enables diverse experiences across workers.
    """

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...]):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.pos = 0
        self.size = 0

        # Pre-allocate storage
        self.states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size


@dataclass
class DDQNWorker:
    """
    DDQN Worker that computes gradients locally and sends them to the learner.

    Each worker:
    1. Collects experiences using epsilon-greedy
    2. Stores in local replay buffer
    3. Samples batch and computes Double DQN loss
    4. Sends gradients to learner
    5. Periodically syncs weights from learner
    """

    # Required fields
    worker_id: int
    weights_path: Path
    gradient_queue: mp.Queue

    # Configuration
    level: LevelType = (1, 1)
    n_step: int = 3  # N-step returns
    gamma: float = 0.99
    render_frames: bool = False
    weight_sync_interval: float = 5.0  # Seconds between weight syncs

    # Local buffer settings
    local_buffer_size: int = 10_000  # Each worker has smaller buffer
    batch_size: int = 32
    steps_per_collection: int = 64  # Steps to collect before training
    train_steps: int = 4  # Training steps per collection

    # Exploration (different per worker)
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 100_000

    # Gradient clipping
    max_grad_norm: float = 10.0

    # Device
    device: Optional[str] = None

    # UI
    ui_queue: Optional[mp.Queue] = None

    # Private fields
    env: Any = field(init=False, repr=False)
    base_env: Any = field(init=False, repr=False)
    net: Any = field(init=False, repr=False)
    buffer: Any = field(init=False, repr=False)
    action_dim: int = field(init=False)

    # N-step buffer
    n_step_buffer: List = field(init=False, default_factory=list)

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

    def __post_init__(self):
        """Initialize environment, network, and buffer."""
        # Auto-detect best device
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Create environment
        self.env, self.base_env = create_env(
            level=self.level,
            render_frames=self.render_frames,
        )
        self.action_dim = self.env.action_space.n

        # Create network
        state_dim = (4, 64, 64)
        self.net = DoubleDQN(
            input_shape=state_dim,
            num_actions=self.action_dim,
            feature_dim=512,
            hidden_dim=256,
            dropout=0.1,
        ).to(self.device)

        # Create local replay buffer
        self.buffer = LocalReplayBuffer(
            capacity=self.local_buffer_size,
            obs_shape=state_dim,
        )

        # Compute n-step gamma
        self.n_step_gamma = self.gamma**self.n_step

        # Initialize tracking
        self.episode_count = 0
        self.total_steps = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.best_x = 0
        self.best_x_ever = 0
        self.flags = 0
        self.deaths = 0
        self.reward_history = []
        self.last_weight_sync = 0.0
        self.weight_version = 0
        self.weight_sync_count = 0
        self.gradients_sent = 0
        self._last_time = time.time()
        self.n_step_buffer = []
        self.current_epsilon = self.eps_start

        # Load initial weights
        self._load_weights()

    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        """Convert state from (4, 64, 64, 1) to (4, 64, 64)."""
        if state.ndim == 4 and state.shape[-1] == 1:
            state = np.squeeze(state, axis=-1)
        return state

    def _load_weights(self) -> bool:
        """Load latest weights from disk with retry on failure."""
        if not self.weights_path.exists():
            return False

        # Retry a few times in case learner is writing
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
                # Sync target network with online
                self.net.sync_target()
                self.last_weight_sync = time.time()
                self.weight_sync_count += 1
                return True
            except Exception:
                if attempt < 2:
                    time.sleep(0.1)  # Brief delay before retry
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
        return int(q_values.argmax(dim=1).item())

    def _compute_n_step_return(self) -> tuple | None:
        """Compute n-step return from buffer."""
        if len(self.n_step_buffer) < self.n_step:
            return None

        n_step_reward = 0.0
        for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma**i) * r
            if d:
                return (
                    self.n_step_buffer[0][0],
                    self.n_step_buffer[0][1],
                    n_step_reward,
                    self.n_step_buffer[i][3],
                    True,
                )

        return (
            self.n_step_buffer[0][0],
            self.n_step_buffer[0][1],
            n_step_reward,
            self.n_step_buffer[-1][3],
            self.n_step_buffer[-1][4],
        )

    def _flush_n_step_buffer(self) -> List[tuple]:
        """Flush remaining transitions at episode end."""
        transitions = []
        while len(self.n_step_buffer) > 0:
            n_step_reward = 0.0
            last_idx = len(self.n_step_buffer) - 1

            for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma**i) * r
                if d:
                    last_idx = i
                    break

            transitions.append(
                (
                    self.n_step_buffer[0][0],
                    self.n_step_buffer[0][1],
                    n_step_reward,
                    self.n_step_buffer[last_idx][3],
                    self.n_step_buffer[last_idx][4],
                )
            )
            self.n_step_buffer.pop(0)

        return transitions

    def collect_steps(self, num_steps: int) -> int:
        """
        Collect experiences for num_steps and store in local buffer.

        Returns number of episodes completed.
        """
        state, _ = self.env.reset()
        episodes_completed = 0

        for _ in range(num_steps):
            # Get action
            action = self._get_action(state)

            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Preprocess states
            state_processed = self._preprocess_state(state)
            next_state_processed = self._preprocess_state(next_state)

            # Add to n-step buffer
            self.n_step_buffer.append((state_processed.copy(), action, reward, next_state_processed.copy(), done))

            # Compute and store n-step transition
            n_step_transition = self._compute_n_step_return()
            if n_step_transition is not None:
                self.buffer.add(*n_step_transition)
                self.n_step_buffer.pop(0)

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

            # Send UI status periodically
            if self.total_steps % 50 == 0:
                self._send_ui_status(info)

            if done:
                # Flush remaining n-step transitions
                for transition in self._flush_n_step_buffer():
                    self.buffer.add(*transition)

                # Track episode stats
                self.episode_count += 1
                episodes_completed += 1
                self.reward_history.append(self.episode_reward)
                if len(self.reward_history) > 100:
                    self.reward_history.pop(0)

                if info.get("is_dead", False) or info.get("is_dying", False):
                    self.deaths += 1

                self._send_ui_status(info)

                # Reset
                state, _ = self.env.reset()
                self.episode_reward = 0.0
                self.episode_length = 0
                self.best_x = 0
            else:
                state = next_state

        return episodes_completed

    def compute_and_send_gradients(self) -> Dict[str, float]:
        """
        Sample from local buffer, compute DQN loss, and send gradients to learner.

        Returns metrics dictionary.
        """
        if len(self.buffer) < self.batch_size:
            return {}

        # Sample batch from local buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Convert to tensors
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        # Compute Double DQN loss
        self.net.train()

        # Current Q-values
        current_q = self.net.online(states_t)
        current_q_selected = current_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_q_online = self.net.online(next_states_t)
            best_actions = next_q_online.argmax(dim=1)
            next_q_target = self.net.target(next_states_t)
            next_q_selected = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.n_step_gamma * next_q_selected * (1.0 - dones_t)

        # Huber loss
        loss = torch.nn.functional.huber_loss(current_q_selected, target_q, delta=1.0)

        # Backward
        self.net.online.zero_grad()
        loss.backward()

        # Clip gradients
        nn.utils.clip_grad_norm_(self.net.online.parameters(), self.max_grad_norm)

        # Extract gradients (move to CPU for IPC)
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
            "td_error": (current_q_selected - target_q).abs().mean().item(),
        }

        # Send gradients to learner
        gradient_packet = {
            "grads": grads,
            "timesteps": self.batch_size,
            "episodes": self.episode_count,
            "worker_id": self.worker_id,
            "weight_version": self.weight_version,
            "metrics": metrics,
        }

        try:
            self.gradient_queue.put(gradient_packet, timeout=5.0)
            self.gradients_sent += 1
        except Exception as e:
            print(f"Worker {self.worker_id}: Failed to send gradients: {e}")

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
                    "snapshot_restores": 0,
                    "current_level": level_str,
                    "last_weight_sync": self.last_weight_sync,
                    "rolling_avg_reward": rolling_avg,
                    "first_flag_time": 0.0,
                },
            )
            self.ui_queue.put_nowait(msg)
        except Exception:
            pass

    def run(self) -> None:
        """Main worker loop."""
        import sys

        print(f"Worker {self.worker_id} started (level={self.level}, device={self.device}, ε_end={self.eps_end:.4f})")
        sys.stdout.flush()

        while True:
            # Maybe sync weights from learner
            self._maybe_sync_weights()

            # Collect experiences
            self.collect_steps(self.steps_per_collection)

            # Calculate speed
            now = time.time()
            elapsed = now - self._last_time
            self.steps_per_sec = self.steps_per_collection / elapsed if elapsed > 0 else 0
            self._last_time = now

            # Compute gradients and send to learner
            if len(self.buffer) >= self.batch_size:
                for _ in range(self.train_steps):
                    self.compute_and_send_gradients()

            # Log progress occasionally
            if self.episode_count % 10 == 0 and self.episode_count > 0:
                avg_reward = np.mean(self.reward_history[-10:]) if self.reward_history else 0
                print(
                    f"W{self.worker_id} | "
                    f"Ep: {self.episode_count} | "
                    f"Steps: {self.total_steps:,} | "
                    f"Best X: {self.best_x_ever} | "
                    f"Avg R: {avg_reward:.1f} | "
                    f"ε: {self.current_epsilon:.3f} | "
                    f"Grads: {self.gradients_sent} | "
                    f"Flags: {self.flags}"
                )


def run_ddqn_worker(
    worker_id: int,
    weights_path: Path,
    gradient_queue: mp.Queue,
    level: LevelType = (1, 1),
    ui_queue: Optional[mp.Queue] = None,
    **kwargs,
) -> None:
    """Entry point for worker process."""
    worker = DDQNWorker(
        worker_id=worker_id,
        weights_path=weights_path,
        gradient_queue=gradient_queue,
        level=level,
        ui_queue=ui_queue,
        **kwargs,
    )
    worker.run()
