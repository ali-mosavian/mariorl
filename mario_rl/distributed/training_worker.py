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
- MuZero trajectory collection with MCTS policy/value targets
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
from mario_rl.core.replay_buffer import MuZeroReplayBuffer
from mario_rl.learners.base import Learner
from mario_rl.learners.muzero import MuZeroTrajectoryCollector

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

    # MuZero-specific configuration
    muzero_unroll_steps: int = 5  # K steps to unroll during training
    muzero_td_steps: int = 10  # n-step returns for value targets

    # Action history configuration
    action_history_len: int = 4  # Track and use action history (default: 4)
    danger_prediction_bins: int = 16  # Predict danger for auxiliary task (default: 16)

    # Internal state
    _buffer: ReplayBuffer = field(init=False, repr=False)
    _current_action_history: np.ndarray | None = field(init=False, repr=False, default=None)
    _env_runner: EnvRunner = field(init=False, repr=False)
    _last_weights_mtime: float = field(init=False, default=0.0)
    _steps_since_flush: int = field(init=False, default=0)
    _action_window: list = field(init=False, repr=False, default_factory=list)  # Rolling window of recent actions
    _action_window_size: int = 10000  # Track last 10k actions for recent distribution
    # Death tracking for danger prediction
    _death_positions: dict = field(init=False, repr=False, default_factory=dict)  # {(world, stage): [(x, count), ...]}
    _max_deaths_per_level: int = 500
    _death_merge_threshold: int = 20

    # MCTS state
    _mcts_explorer: MCTSExplorer | None = field(init=False, default=None)
    _steps_without_x_progress: int = field(init=False, default=0)
    _last_best_x: int = field(init=False, default=0)
    _mcts_episodes_since_last: int = field(init=False, default=0)

    # MuZero trajectory collection state
    _muzero_buffer: MuZeroReplayBuffer | None = field(init=False, default=None)
    _muzero_collector: MuZeroTrajectoryCollector | None = field(init=False, default=None)
    _muzero_last_policy: np.ndarray | None = field(init=False, default=None)
    _muzero_last_value: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        """Initialize buffer and env runner."""
        # Get obs shape from env and preprocess
        obs, info = self.env.reset()
        obs = self._preprocess_state(obs)
        obs_shape = obs.shape

        # Initialize action history tracking if enabled
        action_history_shape = None
        if self.action_history_len > 0:
            # Get action history from info if available
            if "action_history" in info:
                self._current_action_history = info["action_history"].copy()
                action_history_shape = self._current_action_history.shape
            else:
                # Create zero action history (history_len, num_actions)
                num_actions = self.model.num_actions
                self._current_action_history = np.zeros(
                    (self.action_history_len, num_actions), dtype=np.float32
                )
                action_history_shape = self._current_action_history.shape

        # Create buffer
        self._buffer = ReplayBuffer(
            capacity=self.buffer_capacity,
            obs_shape=obs_shape,
            n_step=self.n_step,
            gamma=self.gamma,
            alpha=self.alpha,
            action_history_shape=action_history_shape,
            danger_target_bins=self.danger_prediction_bins,
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

        # Initialize MuZero trajectory buffer and collector if using MuZero
        if self._use_muzero_mcts():
            self._muzero_buffer = MuZeroReplayBuffer(
                capacity=self.buffer_capacity,
                obs_shape=obs_shape,
                num_actions=self.model.num_actions,
                unroll_steps=self.muzero_unroll_steps,
                gamma=self.gamma,
                td_steps=self.muzero_td_steps,
            )
            self._muzero_collector = MuZeroTrajectoryCollector(
                unroll_steps=self.muzero_unroll_steps,
                num_actions=self.model.num_actions,
                td_steps=self.muzero_td_steps,
                discount=self.gamma,
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

    def _record_death(self, level: tuple, x_pos: int) -> None:
        """Record a death position for danger prediction supervision.
        
        Args:
            level: (world, stage) tuple
            x_pos: X position where death occurred
        """
        if level not in self._death_positions:
            self._death_positions[level] = []
        
        deaths = self._death_positions[level]
        
        # Check if there's already a death nearby - if so, increment count
        for i, (dx, count) in enumerate(deaths):
            if abs(dx - x_pos) < self._death_merge_threshold:
                deaths[i] = (dx, count + 1)
                return
        
        # New death location
        deaths.append((x_pos, 1))
        
        # Limit memory size - keep most frequent deaths
        if len(deaths) > self._max_deaths_per_level:
            deaths.sort(key=lambda d: d[1], reverse=True)
            self._death_positions[level] = deaths[:self._max_deaths_per_level]

    def _compute_danger_target(self, level: tuple, current_x: int) -> np.ndarray:
        """Compute danger target vector from recorded death positions.
        
        Creates a vector where each bin represents danger level at a relative
        distance from the current position. This is used as supervision for
        the auxiliary danger prediction head.
        
        Args:
            level: (world, stage) tuple
            current_x: Mario's current X position
            
        Returns:
            danger_target: Array of shape (danger_prediction_bins,) with values in [0, 1]
        """
        if self.danger_prediction_bins == 0:
            return None
            
        deaths = self._death_positions.get(level, [])
        danger_vector = np.zeros(self.danger_prediction_bins, dtype=np.float32)
        
        if not deaths:
            return danger_vector
        
        # Bin layout: 25% behind, 75% ahead (same as RelativeDeathMemory wrapper)
        max_lookahead = 256
        max_lookbehind = 64
        behind_bins = self.danger_prediction_bins // 4
        ahead_bins = self.danger_prediction_bins - behind_bins
        behind_bin_size = max_lookbehind / max(behind_bins, 1)
        ahead_bin_size = max_lookahead / max(ahead_bins, 1)
        
        for death_x, count in deaths:
            relative_x = death_x - current_x
            
            if -max_lookbehind <= relative_x < 0:
                # Death is behind Mario
                bin_idx = int((-relative_x) / behind_bin_size)
                bin_idx = min(bin_idx, behind_bins - 1)
                danger_vector[behind_bins - 1 - bin_idx] += count
                
            elif 0 <= relative_x < max_lookahead:
                # Death is ahead of Mario
                bin_idx = int(relative_x / ahead_bin_size)
                bin_idx = min(bin_idx, ahead_bins - 1)
                danger_vector[behind_bins + bin_idx] += count
        
        # Normalize to [0, 1]
        max_val = danger_vector.max()
        if max_val > 0:
            danger_vector = danger_vector / max_val
        
        return danger_vector

    def _get_action(self, state: np.ndarray, info: dict | None = None) -> int:
        """Get action using epsilon-greedy policy or MuZero MCTS.
        
        Args:
            state: Current observation
            info: Optional info dict from env (contains action_history if enabled)
        """
        # Check if we have a MuZero adapter (latent-space MCTS)
        if self._use_muzero_mcts():
            # MuZero uses its own exploration via MCTS + Dirichlet noise
            # Also stores MCTS policy/value targets for training
            state = self._preprocess_state(state)
            action, policy, value = self.learner.mcts_adapter.get_action_with_targets(state)
            # Store for trajectory collection
            self._muzero_last_policy = policy
            self._muzero_last_value = value
        else:
            # Standard epsilon-greedy for DDQN/Dreamer
            eps = self.epsilon_at(self.total_steps)

            if np.random.random() < eps:
                action = int(np.random.randint(0, self.model.num_actions))
            else:
                state = self._preprocess_state(state)
                with torch.no_grad():
                    state_t = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
                    
                    # Prepare auxiliary inputs
                    action_hist_t = None
                    
                    if self._current_action_history is not None and hasattr(self.model, 'action_history_len'):
                        action_hist_t = torch.from_numpy(
                            self._current_action_history
                        ).unsqueeze(0).float().to(self.device)
                    
                    q_values = self.model(state_t, action_history=action_hist_t)
                    action = int(q_values.argmax(dim=1).item())

        # Track action distribution (rolling window for recent distribution)
        self._action_window.append(action)
        if len(self._action_window) > self._action_window_size:
            self._action_window.pop(0)
        return action

    def _use_muzero_mcts(self) -> bool:
        """Check if we should use MuZero's latent-space MCTS for action selection."""
        if not hasattr(self.learner, "mcts_adapter") or self.learner.mcts_adapter is None:
            return False
        # Check if it's a MuZero adapter (has run latent MCTS)
        adapter = self.learner.mcts_adapter
        return hasattr(adapter, "model") and hasattr(adapter.model, "initial_inference")

    def _collect_with_action_history(self, num_steps: int, collect_start: float) -> dict[str, Any]:
        """Collect experience while tracking action history and danger targets.
        
        This method is used when action_history_len > 0 or danger_prediction_bins > 0
        to properly track and store auxiliary inputs and targets in transitions.
        """
        episodes_completed = 0
        episode_rewards: list[float] = []
        step_infos: list[dict] = []
        current_episode_reward = 0.0

        # Get current state
        if not hasattr(self, '_collect_state') or self._collect_state is None:
            obs, info = self.env.reset()
            self._collect_state = obs
            self._current_level = (info.get("world", 1), info.get("stage", 1))
            if "action_history" in info:
                self._current_action_history = info["action_history"].copy()

        obs = self._collect_state
        
        for _ in range(num_steps):
            # Get current auxiliary inputs before taking action
            current_hist = self._current_action_history.copy() if self._current_action_history is not None else None
            
            # Get current level and x position for danger target
            current_level = getattr(self, '_current_level', (1, 1))
            current_x = getattr(self, '_current_x', 0)
            
            # Compute danger target from recorded death positions
            danger_target = self._compute_danger_target(current_level, current_x)
            
            # Get action
            action = self._get_action(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            current_episode_reward += reward
            step_infos.append(info)
            
            # Update current position tracking
            self._current_level = (info.get("world", 1), info.get("stage", 1))
            self._current_x = info.get("x_pos", 0)
            
            # Record death for danger prediction supervision
            if done and reward < 0:
                self._record_death(self._current_level, info.get("x_pos", 0))
            
            # Get next action history from info
            next_hist = None
            if "action_history" in info:
                self._current_action_history = info["action_history"].copy()
                next_hist = self._current_action_history.copy()
            
            # Create transition with auxiliary inputs and danger target
            # CRITICAL: Use actual_death flag if present (from snapshot wrapper)
            # This ensures TD targets treat deaths as terminal even if episode continues
            effective_done = info.get("actual_death", done)
            transition = Transition(
                state=self._preprocess_state(obs),
                action=action,
                reward=reward,
                next_state=self._preprocess_state(next_obs),
                done=effective_done,
                action_history=current_hist,
                next_action_history=next_hist,
                danger_target=danger_target,
            )
            self._buffer.add(transition)
            
            # Track action distribution
            self._action_window.append(action)
            if len(self._action_window) > self._action_window_size:
                self._action_window.pop(0)
            
            if done:
                episodes_completed += 1
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                
                # Reset environment
                obs, info = self.env.reset()
                self._current_level = (info.get("world", 1), info.get("stage", 1))
                self._current_x = info.get("x_pos", 40)
                if "action_history" in info:
                    self._current_action_history = info["action_history"].copy()
            else:
                obs = next_obs
            
            self._collect_state = obs

        # Update tracking
        self.total_steps += num_steps
        self._steps_since_flush += num_steps
        self.total_episodes += episodes_completed
        
        # Update PER beta
        progress = min(1.0, self.total_steps / self.epsilon_decay_steps)
        self._buffer.update_beta(progress)
        
        # Calculate timing
        collect_end = time.time()
        elapsed = collect_end - collect_start
        steps_per_sec = num_steps / max(elapsed, 0.001)
        
        # Log metrics if logger provided
        if self.logger is not None:
            self.logger.count("steps", n=num_steps)
            self.logger.count("episodes", n=episodes_completed)
            self.logger.gauge("epsilon", self.epsilon_at(self.total_steps))
            self.logger.gauge("buffer_size", len(self._buffer))
            self.logger.gauge("steps_per_sec", steps_per_sec)
            self.logger.gauge("per_beta", self._buffer._current_beta)

            # Log action distribution
            if len(self._action_window) > 100:
                action_counts = np.zeros(self.model.num_actions, dtype=np.int64)
                for a in self._action_window:
                    action_counts[a] += 1
                total_actions = action_counts.sum()
                action_probs = action_counts / total_actions
                nonzero = action_probs > 0
                action_entropy = -np.sum(action_probs[nonzero] * np.log(action_probs[nonzero]))
                max_entropy = np.log(self.model.num_actions)
                normalized_entropy = action_entropy / max_entropy if max_entropy > 0 else 0.0
                self.logger.gauge("action_entropy", float(normalized_entropy))
                pct_str = ",".join(f"{p * 100:.1f}" for p in action_probs)
                self.logger.text("action_dist", pct_str)

            for reward in episode_rewards:
                self.logger.observe("reward", reward)
            if episode_rewards:
                self.logger.gauge("episode_reward", episode_rewards[-1])

            if self._steps_since_flush >= self.flush_every:
                self.logger.flush()
                self._steps_since_flush = 0
        
        return {
            "steps": num_steps,
            "episodes_completed": episodes_completed,
            "episode_rewards": episode_rewards,
            "step_infos": step_infos,
        }

    def epsilon_at(self, steps: int) -> float:
        """Get epsilon for given step count."""
        progress = min(1.0, steps / self.epsilon_decay_steps)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def collect_muzero(self, num_steps: int) -> dict[str, Any]:
        """Collect experience for MuZero with MCTS policy/value targets.

        This method handles MuZero's special requirements:
        - Runs MCTS at each step to get policy and value targets
        - Collects trajectory segments for K-step unrolling
        - Stores trajectories in the MuZero replay buffer

        Args:
            num_steps: Number of environment steps to collect

        Returns:
            Collection info dict
        """
        if self._muzero_buffer is None or self._muzero_collector is None:
            raise RuntimeError("MuZero buffer/collector not initialized")

        collect_start = time.time()
        steps_collected = 0
        episodes_completed = 0
        episode_rewards: list[float] = []
        current_episode_reward = 0.0
        trajectories_stored = 0

        # Get initial observation
        obs, _ = self.env.reset()
        obs = self._preprocess_state(obs)

        # Run initial MCTS to get first policy/value
        action, policy, value = self.learner.mcts_adapter.get_action_with_targets(obs)
        self._muzero_collector.start_episode(obs, policy, value)

        while steps_collected < num_steps:
            # Take action in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            next_obs = self._preprocess_state(next_obs)
            current_episode_reward += reward

            # Track action distribution
            self._action_window.append(action)
            if len(self._action_window) > self._action_window_size:
                self._action_window.pop(0)

            # Get MCTS policy/value for next state (if not done)
            if not done:
                next_action, next_policy, next_value = self.learner.mcts_adapter.get_action_with_targets(next_obs)
            else:
                # For terminal states, use uniform policy and zero value
                next_policy = np.ones(self.model.num_actions) / self.model.num_actions
                next_value = 0.0
                next_action = 0  # Won't be used

            # Add step to trajectory collector
            self._muzero_collector.add_step(
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                next_policy=next_policy,
                next_value=next_value,
            )

            steps_collected += 1

            if done:
                # Extract and store trajectory segments
                trajectories = self._muzero_collector.get_trajectories()
                for traj in trajectories:
                    self._muzero_buffer.add(**traj)
                    trajectories_stored += 1

                # Track episode
                episodes_completed += 1
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0

                # Reset for next episode
                obs, _ = self.env.reset()
                obs = self._preprocess_state(obs)
                action, policy, value = self.learner.mcts_adapter.get_action_with_targets(obs)
                self._muzero_collector.start_episode(obs, policy, value)
            else:
                # Continue to next step
                obs = next_obs
                action = next_action

        # Store partial trajectories from incomplete episode
        trajectories = self._muzero_collector.get_trajectories()
        for traj in trajectories:
            self._muzero_buffer.add(**traj)
            trajectories_stored += 1

        self.total_steps += steps_collected
        self.total_episodes += episodes_completed
        self._steps_since_flush += steps_collected

        # Calculate stats
        collect_end = time.time()
        elapsed = collect_end - collect_start
        steps_per_sec = steps_collected / max(elapsed, 0.001)

        # Update metrics if logger provided
        if self.logger is not None:
            self.logger.count("steps", n=steps_collected)
            self.logger.count("episodes", n=episodes_completed)
            self.logger.gauge("buffer_size", len(self._muzero_buffer))
            self.logger.gauge("steps_per_sec", steps_per_sec)
            self.logger.gauge("trajectories_stored", trajectories_stored)

            for reward in episode_rewards:
                self.logger.observe("reward", reward)
            if episode_rewards:
                self.logger.gauge("episode_reward", episode_rewards[-1])

            if self._steps_since_flush >= self.flush_every:
                self.logger.flush()
                self._steps_since_flush = 0

        return {
            "steps": steps_collected,
            "episodes_completed": episodes_completed,
            "episode_rewards": episode_rewards,
            "trajectories_stored": trajectories_stored,
        }

    def collect(self, num_steps: int) -> dict[str, Any]:
        """Collect experience from environment.

        Args:
            num_steps: Number of steps to collect

        Returns:
            Collection info dict
        """
        collect_start = time.time()
        
        # If action history or death memory is enabled, do manual collection to track them
        if self.action_history_len > 0 or self.danger_prediction_bins > 0:
            return self._collect_with_action_history(num_steps, collect_start)
        
        # Otherwise use standard EnvRunner collection
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

        # If periodic_interval is 1, ALWAYS use MCTS (every episode)
        if self.mcts_periodic_interval == 1:
            return True

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
        
        # Aggregate MCTS stats across all runs
        total_rollouts = 0
        total_expansions = 0
        total_tree_depth = 0
        total_tree_size = 0

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
                
                # Aggregate MCTS stats
                stats = result.stats
                total_rollouts += stats.get("rollouts_done", 0)
                total_expansions += stats.get("expansions", 0)
                total_tree_depth += stats.get("tree_depth", 0)
                total_tree_size += stats.get("tree_size", 0)

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
            # Use actual_death flag for TD targets even if episode continues
            effective_done = info.get("actual_death", done)

            # Add the real transition to buffer
            from mario_rl.core.types import Transition
            real_transition = Transition(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs if not done else obs,
                done=effective_done,
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

        # Compute average MCTS stats
        avg_rollouts = total_rollouts / max(mcts_runs, 1)
        avg_expansions = total_expansions / max(mcts_runs, 1)
        avg_tree_depth = total_tree_depth / max(mcts_runs, 1)
        avg_tree_size = total_tree_size / max(mcts_runs, 1)

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
            # New MCTS metrics
            self.logger.gauge("mcts_avg_rollouts", avg_rollouts)
            self.logger.gauge("mcts_avg_expansions", avg_expansions)
            self.logger.gauge("mcts_avg_tree_depth", avg_tree_depth)
            self.logger.gauge("mcts_avg_tree_size", avg_tree_size)

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
            # MCTS quality metrics
            "mcts_avg_rollouts": avg_rollouts,
            "mcts_avg_expansions": avg_expansions,
            "mcts_avg_tree_depth": avg_tree_depth,
            "mcts_avg_tree_size": avg_tree_size,
        }

    def can_train(self) -> bool:
        """Check if buffer has enough data for training."""
        # Use MuZero trajectory buffer if applicable
        if self._use_muzero_mcts() and self._muzero_buffer is not None:
            return self._muzero_buffer.can_sample(self.batch_size)
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

        # Zero gradients
        self.model.zero_grad()

        # Use MuZero trajectory training if applicable
        if self._use_muzero_mcts() and self._muzero_buffer is not None:
            return self._train_step_muzero()

        # Sample batch and move to device
        batch = self._buffer.sample(self.batch_size, device=str(self.device))

        # Compute loss with importance sampling weights for PER bias correction
        loss, metrics = self.learner.compute_loss(
            states=batch.states,
            actions=batch.actions,
            rewards=batch.rewards,
            next_states=batch.next_states,
            dones=batch.dones,
            weights=batch.weights,  # Pass PER importance sampling weights
            action_histories=batch.action_histories,
            next_action_histories=batch.next_action_histories,
            danger_targets=batch.danger_targets,
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

            # MuZero-specific metrics
            if "policy_loss" in metrics:
                self.logger.observe("policy_loss", float(metrics["policy_loss"]))
            if "value_loss" in metrics:
                self.logger.observe("value_loss", float(metrics["value_loss"]))
            if "consistency_loss" in metrics:
                self.logger.observe("consistency_loss", float(metrics["consistency_loss"]))
            if "contrastive_loss" in metrics:
                self.logger.observe("contrastive_loss", float(metrics["contrastive_loss"]))
            if "value_pred_mean" in metrics:
                self.logger.observe("value_pred_mean", float(metrics["value_pred_mean"]))
            if "value_target_mean" in metrics:
                self.logger.observe("value_target_mean", float(metrics["value_target_mean"]))

        return gradients, metrics

    def _train_step_muzero(self) -> tuple[dict[str, Tensor], dict[str, Any]]:
        """Compute gradients from MuZero trajectory batch.

        Uses compute_trajectory_loss() with proper MCTS policy/value targets.

        Returns:
            (gradients, metrics) tuple
        """
        # Sample trajectory batch from MuZero buffer
        batch = self._muzero_buffer.sample(self.batch_size, device=str(self.device))

        # Compute loss using trajectory-based training
        loss, metrics = self.learner.compute_trajectory_loss(
            s=batch.obs,
            actions=batch.actions,
            rewards=batch.rewards,
            target_policies=batch.target_policies,
            target_values=batch.target_values,
            next_states=batch.next_obs,
            dones=batch.dones,
            weights=batch.weights,
        )

        # Backprop
        loss.backward()

        # Collect gradients
        gradients = {
            name: param.grad.detach().clone()
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }

        # Update priorities if using PER
        if batch.indices is not None and hasattr(self._muzero_buffer, 'alpha') and self._muzero_buffer.alpha > 0:
            # Use total loss as priority (simplified)
            td_errors = np.full(len(batch.indices), metrics.get("loss", 1.0))
            self._muzero_buffer.update_priorities(batch.indices, td_errors)

        # Log MuZero-specific metrics
        if self.logger is not None:
            if "loss" in metrics:
                self.logger.observe("loss", float(metrics["loss"]))
            if "policy_loss" in metrics:
                self.logger.observe("policy_loss", float(metrics["policy_loss"]))
            if "value_loss" in metrics:
                self.logger.observe("value_loss", float(metrics["value_loss"]))
            if "reward_loss" in metrics:
                self.logger.observe("reward_loss", float(metrics["reward_loss"]))
            if "consistency_loss" in metrics:
                self.logger.observe("consistency_loss", float(metrics["consistency_loss"]))
            if "contrastive_loss" in metrics:
                self.logger.observe("contrastive_loss", float(metrics["contrastive_loss"]))
            if "value_pred_mean" in metrics:
                self.logger.observe("value_pred_mean", float(metrics["value_pred_mean"]))
            if "value_target_mean" in metrics:
                self.logger.observe("value_target_mean", float(metrics["value_target_mean"]))

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
        # Collect - use MuZero-specific collection if using MuZero
        if self._use_muzero_mcts():
            collection_info = self.collect_muzero(collect_steps)
        elif self.should_use_mcts():
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
