"""Full Training Worker with environment and buffer.

Combines:
- Environment interaction (via EnvRunner)
- Local replay buffer
- Epsilon-greedy exploration
- Gradient computation (via Learner)
- Weight synchronization from file
- Optional: MetricLogger for CSV/ZMQ metrics
- Optional: LevelTracker for per-level stats
"""

import time
from typing import Any, Protocol
from pathlib import Path
from dataclasses import field
from dataclasses import dataclass

import torch
import numpy as np
from torch import Tensor

from mario_rl.core.types import Transition
from mario_rl.core.env_runner import EnvRunner
from mario_rl.core.replay_buffer import ReplayBuffer
from mario_rl.learners.base import Learner


class MetricsLogger(Protocol):
    """Protocol for metrics logger (avoids hard dependency)."""
    
    def count(self, name: str, n: int = 1) -> None: ...
    def gauge(self, name: str, value: float) -> None: ...
    def observe(self, name: str, value: float) -> None: ...
    def text(self, name: str, value: str) -> None: ...
    def flush(self) -> None: ...
    def save_state(self) -> dict[str, Any]: ...
    def load_state(self, state: dict[str, Any]) -> None: ...


@dataclass
class TrainingWorker:
    """Full training worker with environment and buffer.

    Runs the complete training loop:
    1. Collect experience from environment
    2. Store in local replay buffer
    3. Sample batches and compute gradients
    4. Sync weights from coordinator's file
    """

    env: Any  # Gymnasium-like environment
    learner: Learner
    buffer_capacity: int = 10_000
    batch_size: int = 32
    n_step: int = 1
    gamma: float = 0.99
    alpha: float = 0.0  # PER alpha (0 = uniform)

    # Exploration
    epsilon: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 1_000_000

    # Tracking
    total_steps: int = 0
    weight_version: int = 0
    total_episodes: int = 0

    # Optional metrics (can be None)
    logger: MetricsLogger | None = None
    flush_every: int = 100  # Flush metrics every N steps

    # Internal state
    _buffer: ReplayBuffer = field(init=False, repr=False)
    _env_runner: EnvRunner = field(init=False, repr=False)
    _last_weights_mtime: float = field(init=False, default=0.0)
    _steps_since_flush: int = field(init=False, default=0)
    _action_counts: np.ndarray = field(init=False, repr=False)  # Action distribution tracking

    def __post_init__(self) -> None:
        """Initialize buffer and env runner."""
        # Get obs shape from env and preprocess
        obs, _ = self.env.reset()
        obs = self._preprocess_state(obs)
        obs_shape = obs.shape

        # Create buffer
        self._buffer = ReplayBuffer(
            capacity=self.buffer_capacity,
            obs_shape=obs_shape,
            n_step=self.n_step,
            gamma=self.gamma,
            alpha=self.alpha,
        )

        # Create env runner
        self._env_runner = EnvRunner(
            env=self.env,
            action_fn=self._get_action,
        )

        # Initialize action counts (for distribution tracking)
        self._action_counts = np.zeros(self.model.num_actions, dtype=np.int64)

    @property
    def model(self):
        """Access the underlying model."""
        return self.learner.model

    @property
    def device(self) -> torch.device:
        """Get device the model is on."""
        return next(self.model.parameters()).device

    @property
    def buffer(self) -> ReplayBuffer:
        """Access the replay buffer."""
        return self._buffer

    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        """Preprocess state: squeeze extra channel dimension if present.

        Converts (4, 64, 64, 1) -> (4, 64, 64) for frame-stacked grayscale.
        """
        if state.ndim == 4 and state.shape[-1] == 1:
            state = np.squeeze(state, axis=-1)
        return state

    def _get_action(self, state: np.ndarray) -> int:
        """Get action using epsilon-greedy policy."""
        eps = self.epsilon_at(self.total_steps)

        if np.random.random() < eps:
            action = int(np.random.randint(0, self.model.num_actions))
        else:
            state = self._preprocess_state(state)
            with torch.no_grad():
                state_t = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
                q_values = self.model(state_t)
                action = int(q_values.argmax(dim=1).item())

        # Track action distribution
        self._action_counts[action] += 1
        return action

    def epsilon_at(self, steps: int) -> float:
        """Get epsilon for given step count."""
        progress = min(1.0, steps / self.epsilon_decay_steps)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def collect(self, num_steps: int) -> dict[str, Any]:
        """Collect experience from environment.

        Args:
            num_steps: Number of steps to collect

        Returns:
            Collection info dict
        """
        collect_start = time.time()
        
        transitions, info = self._env_runner.collect_with_info(num_steps)

        # Add to buffer with preprocessing
        for t in transitions:
            # Preprocess observations to squeeze extra channel dim
            preprocessed = Transition(
                state=self._preprocess_state(t.state),
                action=t.action,
                reward=t.reward,
                next_state=self._preprocess_state(t.next_state),
                done=t.done,
            )
            self._buffer.add(preprocessed)

        self.total_steps += num_steps
        self._steps_since_flush += num_steps

        # Update PER beta based on training progress
        # Beta anneals from beta_start (0.4) to beta_end (1.0) to correct for sampling bias
        progress = min(1.0, self.total_steps / self.epsilon_decay_steps)
        self._buffer.update_beta(progress)

        # Calculate steps per second (actual collection duration)
        collect_end = time.time()
        elapsed = collect_end - collect_start
        steps_per_sec = num_steps / max(elapsed, 0.001)

        # Track episodes completed
        episodes_completed = info.get("episodes_completed", 0)
        self.total_episodes += episodes_completed

        # Update metrics if logger provided
        if self.logger is not None:
            self.logger.count("steps", n=num_steps)
            self.logger.count("episodes", n=episodes_completed)
            self.logger.gauge("epsilon", self.epsilon_at(self.total_steps))
            self.logger.gauge("buffer_size", len(self._buffer))
            self.logger.gauge("steps_per_sec", steps_per_sec)
            self.logger.gauge("per_beta", self._buffer._current_beta)  # Log PER beta for dashboard

            # Log action distribution metrics
            total_actions = self._action_counts.sum()
            if total_actions > 0:
                # Compute action entropy (entropy of actual taken actions)
                action_probs = self._action_counts / total_actions
                # Avoid log(0) by filtering zero probabilities
                nonzero = action_probs > 0
                action_entropy = -np.sum(action_probs[nonzero] * np.log(action_probs[nonzero]))
                # Normalize by max entropy (log of num_actions) for 0-1 scale
                max_entropy = np.log(len(self._action_counts))
                normalized_entropy = action_entropy / max_entropy if max_entropy > 0 else 0.0
                self.logger.gauge("action_entropy", float(normalized_entropy))

                # Log action distribution as percentages (for CSV)
                pct_str = ",".join(f"{p * 100:.1f}" for p in action_probs)
                self.logger.text("action_dist", pct_str)

            # Track episode rewards
            episode_rewards = info.get("episode_rewards", [])
            for reward in episode_rewards:
                self.logger.observe("reward", reward)  # Rolling average
            # Store last episode's reward as gauge for UI display
            if episode_rewards:
                self.logger.gauge("episode_reward", episode_rewards[-1])

            # Flush periodically
            if self._steps_since_flush >= self.flush_every:
                self.logger.flush()
                self._steps_since_flush = 0

        return info

    def can_train(self) -> bool:
        """Check if buffer has enough data for training."""
        return self._buffer.can_sample(self.batch_size)

    def train_step(self) -> tuple[dict[str, Tensor], dict[str, Any]]:
        """Compute gradients from a sampled batch.

        Returns:
            (gradients, metrics) tuple

        Raises:
            ValueError: If not enough data in buffer
        """
        if not self.can_train():
            raise ValueError(
                f"Not enough data in buffer: {len(self._buffer)} < {self.batch_size}"
            )

        # Sample batch and move to device
        batch = self._buffer.sample(self.batch_size, device=str(self.device))

        # Zero gradients
        self.model.zero_grad()

        # Compute loss with importance sampling weights for PER bias correction
        loss, metrics = self.learner.compute_loss(
            states=batch.states,
            actions=batch.actions,
            rewards=batch.rewards,
            next_states=batch.next_states,
            dones=batch.dones,
            weights=batch.weights,  # Pass PER importance sampling weights
        )

        # Backprop
        loss.backward()

        # Collect gradients
        gradients = {
            name: param.grad.detach().clone()
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }

        # Update priorities if using PER - use actual TD errors from loss computation
        if batch.indices is not None and self.alpha > 0 and "td_error" in metrics:
            # Recompute TD errors for each sample (need per-sample errors, not mean)
            with torch.no_grad():
                current_q = self.model(batch.states)
                current_q_selected = current_q.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
                
                next_q_online = self.model(batch.next_states)
                best_actions = next_q_online.argmax(dim=1)
                next_q_target = self.model(batch.next_states, network="target") if hasattr(self.model, "target") else next_q_online
                next_q_selected = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                
                n_step_gamma = self.gamma ** self.n_step
                target_q = batch.rewards + n_step_gamma * next_q_selected * (1.0 - batch.dones.float())
                td_errors = (current_q_selected - target_q).abs().cpu().numpy()
            
            self._buffer.update_priorities(batch.indices, td_errors)

        # Track training metrics
        if self.logger is not None:
            if "loss" in metrics:
                self.logger.observe("loss", float(metrics["loss"]))
            if "q_mean" in metrics:
                self.logger.observe("q_mean", float(metrics["q_mean"]))
            if "td_error" in metrics:
                self.logger.observe("td_error", float(metrics["td_error"]))
            if "entropy" in metrics:
                self.logger.observe("entropy", float(metrics["entropy"]))

        return gradients, metrics

    def sync_weights(self, weights_path: Path) -> bool:
        """Sync weights from file if changed.

        Args:
            weights_path: Path to weights file

        Returns:
            True if weights were synced, False otherwise
        """
        if not weights_path.exists():
            return False

        # Check if file has changed
        mtime = weights_path.stat().st_mtime
        if mtime <= self._last_weights_mtime:
            return False

        # Load and apply weights
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
            self._last_weights_mtime = mtime
            self.weight_version += 1
            return True
        except Exception:
            return False

    def run_cycle(
        self,
        collect_steps: int,
        train_steps: int,
    ) -> dict[str, Any]:
        """Run a full collect → train cycle.

        Args:
            collect_steps: Steps to collect
            train_steps: Number of gradient computations

        Returns:
            Cycle result with gradients and info
        """
        # Collect
        collection_info = self.collect(collect_steps)

        # Train (if we have enough data)
        all_grads: list[dict[str, Tensor]] = []
        all_metrics: list[dict[str, Any]] = []

        if self.can_train():
            for _ in range(train_steps):
                grads, metrics = self.train_step()
                all_grads.append(grads)
                all_metrics.append(metrics)

        # Average gradients
        averaged_grads = {}
        if all_grads:
            for name in all_grads[0]:
                stacked = torch.stack([g[name] for g in all_grads])
                averaged_grads[name] = stacked.mean(dim=0)

        return {
            "gradients": averaged_grads,
            "collection_info": collection_info,
            "train_metrics": all_metrics,
            "steps": collect_steps,
            "weight_version": self.weight_version,
        }
