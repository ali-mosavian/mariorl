"""
PPO Worker for distributed training.

Collects rollouts from the environment and sends them to the learner.
"""

import os
import time
import multiprocessing as mp

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from typing import Any
from typing import List
from pathlib import Path
from typing import Literal
from typing import Optional
from dataclasses import field
from dataclasses import dataclass

import torch
import numpy as np
from gymnasium.spaces import Box
from nes_py.wrappers import JoypadSpace
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import TransformObservation
from gymnasium.wrappers import FrameStackObservation
from gym_super_mario_bros import actions as smb_actions

from mario_rl.agent.ppo_net import ActorCritic
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


@dataclass
class PPOWorker:
    """
    Worker that collects rollouts for PPO training.

    Runs the environment, computes values and log_probs using the current policy,
    and sends complete rollouts to the learner.
    """

    # Required fields
    worker_id: int
    weights_path: Path
    rollout_queue: mp.Queue

    # Configuration
    level: LevelType = (1, 1)
    n_steps: int = 128  # Steps per rollout
    render_frames: bool = False
    weight_sync_interval: float = 5.0  # Seconds between weight syncs
    device: Optional[str] = None
    ui_queue: Optional[mp.Queue] = None

    # Private fields
    env: Any = field(init=False, repr=False)
    base_env: Any = field(init=False, repr=False)
    net: Any = field(init=False, repr=False)
    action_dim: int = field(init=False)

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
    steps_per_sec: float = field(init=False, default=0.0)
    _last_time: float = field(init=False, default=0.0)

    def __post_init__(self):
        """Initialize environment and network."""
        # Create environment
        self.env, self.base_env = create_env(
            level=self.level,
            render_frames=self.render_frames,
        )
        self.action_dim = self.env.action_space.n

        # Set device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create network (for inference only)
        state_dim = (4, 64, 64, 1)
        self.net = ActorCritic(
            input_shape=state_dim,
            num_actions=self.action_dim,
            feature_dim=512,
        ).to(self.device)
        self.net.eval()

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
        self._last_time = time.time()

        # Load initial weights
        self._load_weights()

    def _load_weights(self) -> bool:
        """Load latest weights from disk."""
        if not self.weights_path.exists():
            return False

        try:
            checkpoint = torch.load(
                self.weights_path,
                map_location=self.device,
                weights_only=True,
            )
            self.net.load_state_dict(checkpoint)
            self.last_weight_sync = time.time()
            return True
        except Exception as e:
            print(f"Worker {self.worker_id}: Failed to load weights: {e}")
            return False

    def _maybe_sync_weights(self) -> None:
        """Sync weights if enough time has passed."""
        if time.time() - self.last_weight_sync >= self.weight_sync_interval:
            self._load_weights()

    @torch.no_grad()
    def _get_action_and_value(self, state: np.ndarray) -> tuple[int, float, float]:
        """Get action, log_prob, and value for a state."""
        state_tensor = torch.from_numpy(np.expand_dims(state, 0)).to(self.device)
        action, log_prob, _, value = self.net.get_action_and_value(state_tensor)
        return (
            action.item(),
            log_prob.item(),
            value.item(),
        )

    @torch.no_grad()
    def _get_value(self, state: np.ndarray) -> float:
        """Get value for a state (for bootstrapping)."""
        state_tensor = torch.from_numpy(np.expand_dims(state, 0)).to(self.device)
        return float(self.net.get_value(state_tensor).item())

    def collect_rollout(self) -> dict:
        """
        Collect a rollout of n_steps transitions.

        Returns:
            Dictionary with rollout data
        """
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []

        state, _ = self.env.reset()
        self.episode_reward = 0.0
        self.episode_length = 0
        self.best_x = 0

        for _step in range(self.n_steps):
            # Get action and value
            action, log_prob, value = self._get_action_and_value(state)

            # Store transition
            states.append(state.copy())
            actions.append(action)
            values.append(value)
            log_probs.append(log_prob)

            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            rewards.append(reward)
            dones.append(done)

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

            if done:
                # Episode ended
                self.episode_count += 1
                self.reward_history.append(self.episode_reward)
                if len(self.reward_history) > 100:
                    self.reward_history.pop(0)

                if info.get("is_dead", False) or info.get("is_dying", False):
                    self.deaths += 1

                # Send UI status
                self._send_ui_status(info)

                # Reset for next episode
                state, _ = self.env.reset()
                self.episode_reward = 0.0
                self.episode_length = 0
                self.best_x = 0
            else:
                state = next_state

        # Get bootstrap value for last state
        last_value = self._get_value(state)
        last_done = dones[-1] if dones else False

        # Calculate speed
        now = time.time()
        elapsed = now - self._last_time
        self.steps_per_sec = self.n_steps / elapsed if elapsed > 0 else 0
        self._last_time = now

        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "dones": np.array(dones),
            "values": np.array(values),
            "log_probs": np.array(log_probs),
            "last_value": last_value,
            "last_done": last_done,
        }

    def _send_ui_status(self, info: dict) -> None:
        """Send status to UI queue."""
        if self.ui_queue is None:
            return

        try:
            from mario_rl.training.training_ui import UIMessage
            from mario_rl.training.training_ui import MessageType

            rolling_avg = np.mean(self.reward_history) if self.reward_history else 0.0

            # Get level display string
            if isinstance(self.level, tuple):
                level_str = f"{self.level[0]}-{self.level[1]}"
            else:
                level_str = str(self.level)

            msg = UIMessage(
                msg_type=MessageType.WORKER_STATUS,
                source_id=self.worker_id,
                data={
                    "episode": self.episode_count,
                    "step": self.episode_length,
                    "reward": self.episode_reward,
                    "x_pos": info.get("x_pos", 0),
                    "best_x": self.best_x,
                    "best_x_ever": self.best_x_ever,
                    "deaths": self.deaths,
                    "flags": self.flags,
                    "epsilon": 0.0,  # PPO doesn't use epsilon
                    "experiences": self.total_steps,
                    "q_mean": 0.0,
                    "q_max": 0.0,
                    "weight_sync_count": 0,
                    "steps_per_sec": self.steps_per_sec,
                    "snapshot_restores": 0,
                    "current_level": level_str,
                    "rolling_avg_reward": rolling_avg,
                    "first_flag_time": 0.0,
                },
            )
            self.ui_queue.put_nowait(msg)
        except Exception:
            pass

    def run(self) -> None:
        """Main worker loop."""
        print(f"Worker {self.worker_id} started (level={self.level})")

        while True:
            # Maybe sync weights
            self._maybe_sync_weights()

            # Collect rollout
            rollout = self.collect_rollout()

            # Send to learner
            try:
                self.rollout_queue.put(rollout, timeout=1.0)
            except Exception:
                print(f"Worker {self.worker_id}: Queue full, dropping rollout")
                continue

            # Print progress occasionally
            if self.episode_count % 10 == 0 and self.episode_count > 0:
                avg_reward = np.mean(self.reward_history[-10:]) if self.reward_history else 0
                print(
                    f"W{self.worker_id} | "
                    f"Ep: {self.episode_count} | "
                    f"Steps: {self.total_steps:,} | "
                    f"Best X: {self.best_x_ever} | "
                    f"Avg R: {avg_reward:.1f} | "
                    f"Flags: {self.flags}"
                )


def run_ppo_worker(
    worker_id: int,
    weights_path: Path,
    rollout_queue: mp.Queue,
    level: LevelType = (1, 1),
    ui_queue: Optional[mp.Queue] = None,
    **kwargs,
) -> None:
    """Entry point for worker process."""
    worker = PPOWorker(
        worker_id=worker_id,
        weights_path=weights_path,
        rollout_queue=rollout_queue,
        level=level,
        ui_queue=ui_queue,
        **kwargs,
    )
    worker.run()
