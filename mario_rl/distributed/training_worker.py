"""Full Training Worker with environment and buffer.

Combines:
- Environment interaction (via EnvRunner)
- Local replay buffer
- Epsilon-greedy exploration
- Gradient computation (via Learner)
- Weight synchronization from file
- Optional: MetricLogger for CSV/ZMQ metrics
- Optional: LevelTracker for per-level stats
- Optional: MCTS exploration for enhanced learning
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

# MCTS imports (optional)
from mario_rl.mcts import MCTSExplorer, MCTSConfig
from mario_rl.mcts.protocols import WorldModelAdapter


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

    # MCTS exploration (optional)
    mcts_enabled: bool = False
    mcts_num_simulations: int = 50
    mcts_max_depth: int = 20
    mcts_stuck_threshold: int = 500  # Steps without x progress to trigger MCTS
    mcts_periodic_interval: int = 10  # Use MCTS every N episodes
    mcts_sequence_length: int = 15  # Execute this many actions from best MCTS rollout

    # Internal state
    _buffer: ReplayBuffer = field(init=False, repr=False)
    _env_runner: EnvRunner = field(init=False, repr=False)
    _last_weights_mtime: float = field(init=False, default=0.0)
    _steps_since_flush: int = field(init=False, default=0)
    _action_window: list = field(init=False, repr=False, default_factory=list)  # Rolling window of recent actions
    _action_window_size: int = 10000  # Track last 10k actions for recent distribution

    # MCTS state
    _mcts_explorer: MCTSExplorer | None = field(init=False, default=None)
    _steps_without_x_progress: int = field(init=False, default=0)
    _last_best_x: int = field(init=False, default=0)
    _mcts_episodes_since_last: int = field(init=False, default=0)

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

        # Action window is initialized by default_factory in field definition

        # Initialize MCTS if enabled (adapter is injected via learner)
        if self.mcts_enabled and self.learner.mcts_adapter is not None:
            adapter = self.learner.mcts_adapter

            # Check if adapter supports world model (Dreamer)
            world_model = adapter if isinstance(adapter, WorldModelAdapter) else None

            # Use Mario-specific rollout policy for MCTS
            # This biases rollouts toward RIGHT (70%) to make them meaningful
            # Without this, random rollouts can't distinguish good vs bad actions
            from mario_rl.mcts.adapters import MarioRolloutAdapter
            rollout_adapter = MarioRolloutAdapter(num_actions=self.model.num_actions)

            self._mcts_explorer = MCTSExplorer(
                config=MCTSConfig(
                    num_simulations=self.mcts_num_simulations,
                    max_rollout_depth=self.mcts_max_depth,
                    rollout_policy="policy",  # Use rollout adapter
                    value_source="rollout",   # Use actual rollout returns
                    sequence_length=self.mcts_sequence_length,  # Return action sequences
                ),
                policy=rollout_adapter,  # RIGHT-biased rollouts
                value_fn=rollout_adapter,  # Zero value (use rollout)
                num_actions=self.model.num_actions,
                world_model=world_model,  # Dreamer can use imagined rollouts
            )

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

        # Track action distribution (rolling window for recent distribution)
        self._action_window.append(action)
        if len(self._action_window) > self._action_window_size:
            self._action_window.pop(0)
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

            # Log action distribution metrics (from rolling window of recent actions)
            if len(self._action_window) > 100:  # Need enough samples for meaningful distribution
                # Count actions in rolling window
                action_counts = np.zeros(self.model.num_actions, dtype=np.int64)
                for a in self._action_window:
                    action_counts[a] += 1
                
                total_actions = action_counts.sum()
                action_probs = action_counts / total_actions
                
                # Compute action entropy (entropy of actual taken actions)
                # Avoid log(0) by filtering zero probabilities
                nonzero = action_probs > 0
                action_entropy = -np.sum(action_probs[nonzero] * np.log(action_probs[nonzero]))
                # Normalize by max entropy (log of num_actions) for 0-1 scale
                max_entropy = np.log(self.model.num_actions)
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

    def _get_frame_stack_obs(self) -> np.ndarray:
        """Get current observation from env (preprocessed)."""
        # Access current observation through env runner
        if hasattr(self._env_runner, "_current_obs") and self._env_runner._current_obs is not None:
            return self._preprocess_state(self._env_runner._current_obs)
        # Fallback: reset and get observation
        obs, _ = self.env.reset()
        return self._preprocess_state(obs)

    def should_use_mcts(self) -> bool:
        """Check if MCTS should be used this cycle."""
        if not self.mcts_enabled or self._mcts_explorer is None:
            return False

        # Check stuck trigger
        if self._steps_without_x_progress >= self.mcts_stuck_threshold:
            return True

        # Check periodic trigger
        if self._mcts_episodes_since_last >= self.mcts_periodic_interval:
            return True

        return False

    def collect_with_mcts(self, num_steps: int) -> dict[str, Any]:
        """Collect experience using MCTS exploration for entire episode.

        Uses pure MCTS tree search (no policy network) to explore and
        find optimal action SEQUENCES. Like the old MCTS implementation,
        this runs MCTS to find a good sequence, executes the entire sequence,
        then runs MCTS again when the sequence is exhausted.

        Args:
            num_steps: Target number of steps to collect

        Returns:
            Collection info dict
        """
        if self._mcts_explorer is None:
            return self.collect(num_steps)

        collect_start = time.time()
        transitions_collected = 0
        episodes_completed = 0
        step_infos: list[dict] = []
        episode_end_infos: list[dict] = []
        mcts_runs = 0
        actions_from_mcts = 0  # Track how many actions came from MCTS sequences

        # Get current observation
        obs = self._get_frame_stack_obs()
        done = False
        
        # Action sequence from MCTS (like old implementation)
        action_sequence: list[int] = []

        # Run until episode ends or num_steps collected
        while not done and transitions_collected < num_steps:
            # If no actions in sequence, run MCTS to get new sequence
            if len(action_sequence) == 0:
                # Clear visited states for fresh MCTS from current position
                self._mcts_explorer.clear_visited_states()

                # Run MCTS to find best action SEQUENCE (like old MCTS)
                result = self._mcts_explorer.explore(
                    env=self.env,
                    root_obs=obs,
                    get_obs_fn=None,
                )
                mcts_runs += 1

                # Add MCTS exploration transitions to buffer
                for t in result.transitions:
                    self._buffer.add(t)
                transitions_collected += len(result.transitions)

                # Get the best action sequence from MCTS
                action_sequence = list(result.best_sequence)
                
                # Fallback: if no sequence, use best_action
                if not action_sequence:
                    action_sequence = [result.best_action]

            # Pop and execute next action from sequence
            action = action_sequence.pop(0)
            actions_from_mcts += 1

            # Execute action in real environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Add the real transition to buffer
            from mario_rl.core.types import Transition
            real_transition = Transition(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs if not done else obs,
                done=done,
            )
            self._buffer.add(real_transition)
            transitions_collected += 1

            # Update observation for next iteration
            obs = next_obs
            step_infos.append(info)

            # Track x progress
            x_pos = info.get("x_pos", 0)
            if x_pos > self._last_best_x:
                self._last_best_x = x_pos
                self._steps_without_x_progress = 0
            else:
                self._steps_without_x_progress += 1

            # If we die, clear the action sequence (old sequence is now invalid)
            if done:
                action_sequence = []

        # Episode ended
        if done:
            episodes_completed = 1
            self.total_episodes += 1
            episode_end_infos.append(info)
            # Reset env for next cycle
            self.env.reset()

        # Update counters
        self.total_steps += transitions_collected
        self._steps_since_flush += transitions_collected
        self._steps_without_x_progress = 0
        self._mcts_episodes_since_last = 0

        # Calculate stats
        collect_end = time.time()
        elapsed = collect_end - collect_start
        steps_per_sec = transitions_collected / max(elapsed, 0.001)

        # Log metrics
        if self.logger is not None:
            self.logger.count("steps", n=transitions_collected)
            self.logger.count("episodes", n=episodes_completed)
            self.logger.count("mcts_explorations", n=mcts_runs)
            self.logger.gauge("mcts_transitions", transitions_collected)
            self.logger.gauge("mcts_runs_per_episode", mcts_runs)
            self.logger.gauge("mcts_actions_executed", actions_from_mcts)
            self.logger.gauge("epsilon", self.epsilon_at(self.total_steps))
            self.logger.gauge("buffer_size", len(self._buffer))
            self.logger.gauge("steps_per_sec", steps_per_sec)

            if self._steps_since_flush >= self.flush_every:
                self.logger.flush()
                self._steps_since_flush = 0

        return {
            "transitions": transitions_collected,
            "episodes_completed": episodes_completed,
            "step_infos": step_infos,
            "episode_end_infos": episode_end_infos,
            "mcts_used": True,
            "mcts_runs": mcts_runs,
            "mcts_actions_executed": actions_from_mcts,
            "mcts_runs": mcts_runs,
        }

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
            # Common metrics
            if "loss" in metrics:
                self.logger.observe("loss", float(metrics["loss"]))
            if "entropy" in metrics:
                self.logger.observe("entropy", float(metrics["entropy"]))
            
            # DDQN-specific metrics
            if "q_mean" in metrics:
                self.logger.observe("q_mean", float(metrics["q_mean"]))
            if "td_error" in metrics:
                self.logger.observe("td_error", float(metrics["td_error"]))
            if "q_max" in metrics:
                self.logger.gauge("q_max", float(metrics["q_max"]))
            
            # Dreamer-specific metrics
            if "wm_loss" in metrics:
                self.logger.observe("wm_loss", float(metrics["wm_loss"]))
            if "recon_loss" in metrics:
                self.logger.observe("recon_loss", float(metrics["recon_loss"]))
            if "ssim" in metrics:
                self.logger.gauge("ssim", float(metrics["ssim"]))
            if "kl_loss" in metrics:
                self.logger.observe("kl_loss", float(metrics["kl_loss"]))
            if "dynamics_loss" in metrics:
                self.logger.observe("dynamics_loss", float(metrics["dynamics_loss"]))
            if "reward_loss" in metrics:
                self.logger.observe("reward_loss", float(metrics["reward_loss"]))
            if "behavior_loss" in metrics:
                self.logger.observe("behavior_loss", float(metrics["behavior_loss"]))
            if "actor_loss" in metrics:
                self.logger.observe("actor_loss", float(metrics["actor_loss"]))
            if "critic_loss" in metrics:
                self.logger.observe("critic_loss", float(metrics["critic_loss"]))
            if "value_mean" in metrics:
                self.logger.observe("value_mean", float(metrics["value_mean"]))
            if "return_mean" in metrics:
                self.logger.observe("return_mean", float(metrics["return_mean"]))

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
        # Collect (hybrid: MCTS or epsilon-greedy)
        if self.should_use_mcts():
            collection_info = self.collect_with_mcts(collect_steps)
        else:
            collection_info = self.collect(collect_steps)
            # Track episodes for periodic MCTS trigger
            self._mcts_episodes_since_last += collection_info.get("episodes_completed", 0)

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
