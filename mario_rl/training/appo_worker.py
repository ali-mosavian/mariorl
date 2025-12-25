"""
APPO (Asynchronous PPO) Worker for distributed training.

Combines PPO's stability with A3C's efficiency by computing gradients locally
and sending only gradients (not full rollouts) to the learner.

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
    │ └────┬────┘ │      │ └────┬────┘ │      │ └────┬────┘ │
    │      │      │      │      │      │      │      │      │
    │ 1.Collect   │      │ 1.Collect   │      │ 1.Collect   │
    │   rollout   │      │   rollout   │      │   rollout   │
    │      │      │      │      │      │      │      │      │
    │ 2.Compute   │      │ 2.Compute   │      │ 2.Compute   │
    │   PPO loss  │      │   PPO loss  │      │   PPO loss  │
    │      │      │      │      │      │      │      │      │
    │ 3.Backward  │      │ 3.Backward  │      │ 3.Backward  │
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

Current PPO:  Worker → [rollout 8MB] → Learner (compute loss + backward)
APPO:         Worker (compute loss + backward) → [gradients 2MB] → Learner

Benefits:
- ~4x less data through IPC
- Distributed computation across CPU cores
- PPO stability (clipped objective)
"""

import os
import time
import multiprocessing as mp

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from typing import Any
from typing import Dict
from typing import List
from pathlib import Path
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
class APPOWorker:
    """
    APPO Worker that computes gradients locally and sends them to the learner.

    This is more efficient than sending full rollouts because:
    - Gradients are ~2 MB vs ~8 MB for rollouts
    - Computation is distributed across worker CPUs
    - Reduces learner bottleneck
    """

    # Required fields
    worker_id: int
    weights_path: Path
    gradient_queue: mp.Queue

    # Configuration
    level: LevelType = (1, 1)
    n_steps: int = 128  # Steps per rollout
    n_epochs: int = 4  # PPO epochs per rollout
    minibatch_size: int = 32  # Minibatch size for PPO updates
    render_frames: bool = False
    weight_sync_interval: float = 5.0  # Seconds between weight syncs
    reward_scale: float = 0.1  # Scale rewards for stable training
    device: Optional[str] = None  # Auto-detect: MPS on Mac, CUDA on NVIDIA, else CPU
    ui_queue: Optional[mp.Queue] = None

    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

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
    weight_version: int = field(init=False, default=0)
    steps_per_sec: float = field(init=False, default=0.0)
    _last_time: float = field(init=False, default=0.0)

    def __post_init__(self):
        """Initialize environment and network."""
        # Auto-detect best device (MPS on Mac, CUDA on NVIDIA, else CPU)
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

        # Create network on detected device
        state_dim = (4, 64, 64)
        self.net = ActorCritic(
            input_shape=state_dim,
            num_actions=self.action_dim,
            feature_dim=512,
        ).to(self.device)

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
        self._last_time = time.time()

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
        state = self._preprocess_state(state)
        state_tensor = torch.from_numpy(np.expand_dims(state, 0)).float().to(self.device)
        action, log_prob, _, value = self.net.get_action_and_value(state_tensor)
        return (
            action.item(),
            log_prob.item(),
            value.item(),
        )

    @torch.no_grad()
    def _get_value(self, state: np.ndarray) -> float:
        """Get value for a state (for bootstrapping)."""
        state = self._preprocess_state(state)
        state_tensor = torch.from_numpy(np.expand_dims(state, 0)).float().to(self.device)
        return float(self.net.get_value(state_tensor).item())

    def collect_rollout(self) -> Dict[str, Any]:
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
            states.append(self._preprocess_state(state).copy())
            actions.append(action)
            values.append(value)
            log_probs.append(log_prob)

            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Scale reward
            scaled_reward = reward * self.reward_scale

            rewards.append(scaled_reward)
            dones.append(done)

            # Update tracking
            self.episode_reward += reward  # Track unscaled for logging
            self.episode_length += 1
            self.total_steps += 1

            x_pos = info.get("x_pos", 0)
            if x_pos > self.best_x:
                self.best_x = x_pos
            if x_pos > self.best_x_ever:
                self.best_x_ever = x_pos

            if info.get("flag_get", False):
                self.flags += 1

            # Send real-time UI status every 50 steps
            if self.total_steps % 50 == 0:
                self._send_ui_status(info)

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
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
            "values": np.array(values, dtype=np.float32),
            "log_probs": np.array(log_probs, dtype=np.float32),
            "last_value": last_value,
            "last_done": last_done,
        }

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: Any,
        last_done: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.

        Returns:
            advantages: GAE advantages
            returns: Target values (advantages + values)
        """
        n_steps = len(rewards)
        advantages = np.zeros(n_steps, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def compute_ppo_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO loss (clipped surrogate + value + entropy).

        Returns:
            loss: Total loss
            metrics: Dictionary of loss components
        """
        # Get current policy outputs
        _, new_log_probs, entropy, new_values = self.net.get_action_and_value(states, actions)

        # Policy loss (clipped surrogate)
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        # Approximate KL divergence
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()

        # Normalize advantages
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        surr1 = ratio * advantages_normalized
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages_normalized
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss (with clipping)
        values_clipped = old_values + torch.clamp(
            new_values - old_values,
            -self.clip_range,
            self.clip_range,
        )
        value_loss_unclipped = (new_values - returns) ** 2
        value_loss_clipped = (values_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

        # Entropy loss
        entropy_loss = -entropy.mean()

        # Total loss
        loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

        return loss, metrics

    def compute_and_send_gradients(self, rollout: Dict[str, Any]) -> None:
        """
        Compute PPO loss locally and send gradients to learner.

        This is the key difference from standard PPO:
        - Standard PPO: Send rollout (~8 MB) to learner
        - APPO: Compute loss here, send gradients (~2 MB)
        """
        # Convert rollout to tensors
        states = torch.from_numpy(rollout["states"]).to(self.device)
        actions = torch.from_numpy(rollout["actions"]).to(self.device)
        old_log_probs = torch.from_numpy(rollout["log_probs"]).to(self.device)
        old_values = torch.from_numpy(rollout["values"]).to(self.device)

        # Compute GAE
        advantages_np, returns_np = self.compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            rollout["last_value"],
            rollout["last_done"],
        )
        advantages = torch.from_numpy(advantages_np).to(self.device)
        returns = torch.from_numpy(returns_np).to(self.device)

        # Training loop - multiple epochs over the rollout
        all_metrics: List[Dict[str, float]] = []

        for _epoch in range(self.n_epochs):
            # Shuffle indices for minibatches
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), self.minibatch_size):
                end = start + self.minibatch_size
                batch_indices = indices[start:end]

                # Get minibatch
                mb_states = states[batch_indices]
                mb_actions = actions[batch_indices]
                mb_old_log_probs = old_log_probs[batch_indices]
                mb_advantages = advantages[batch_indices]
                mb_returns = returns[batch_indices]
                mb_old_values = old_values[batch_indices]

                # Compute loss
                loss, metrics = self.compute_ppo_loss(
                    mb_states,
                    mb_actions,
                    mb_old_log_probs,
                    mb_advantages,
                    mb_returns,
                    mb_old_values,
                )

                # Backward pass
                self.net.zero_grad()
                loss.backward()

                # Clip gradients locally
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)

                all_metrics.append(metrics)

        # Average metrics across all minibatches
        avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}

        # Extract gradients (move to CPU for IPC)
        grads = {
            name: param.grad.cpu().clone() for name, param in self.net.named_parameters() if param.grad is not None
        }

        # Send gradients to learner
        gradient_packet = {
            "grads": grads,
            "timesteps": len(states),
            "worker_id": self.worker_id,
            "weight_version": self.weight_version,
            "metrics": avg_metrics,
        }

        try:
            self.gradient_queue.put(gradient_packet, timeout=5.0)
        except Exception as e:
            print(f"Worker {self.worker_id}: Failed to send gradients: {e}")

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
                    "epsilon": 0.0,
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
        import sys

        print(f"Worker {self.worker_id} started (level={self.level}, device={self.device})")
        sys.stdout.flush()

        while True:
            # Maybe sync weights from learner
            self._maybe_sync_weights()

            # Collect rollout
            rollout = self.collect_rollout()

            # Compute PPO loss locally and send gradients
            self.compute_and_send_gradients(rollout)

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


def run_appo_worker(
    worker_id: int,
    weights_path: Path,
    gradient_queue: mp.Queue,
    level: LevelType = (1, 1),
    ui_queue: Optional[mp.Queue] = None,
    **kwargs,
) -> None:
    """Entry point for worker process."""
    worker = APPOWorker(
        worker_id=worker_id,
        weights_path=weights_path,
        gradient_queue=gradient_queue,
        level=level,
        ui_queue=ui_queue,
        **kwargs,
    )
    worker.run()
