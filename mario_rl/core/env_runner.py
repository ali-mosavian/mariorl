"""Environment Runner for step collection.

Wraps environment interaction into a clean interface that:
- Collects steps from environment
- Processes rewards (scaling, clipping)
- Handles episode boundaries
- Returns transitions for training

This is a GENERIC runner - game-specific metrics extraction should
be done via MetricCollectors, not in this class.
"""

from typing import Any
from typing import Callable
from dataclasses import field
from dataclasses import dataclass

import numpy as np

from mario_rl.core.types import Transition


@dataclass(frozen=True, slots=True)
class CollectionInfo:
    """Information about a collection run."""

    steps: int
    total_reward: float
    episodes_completed: int
    episode_rewards: list[float] = field(default_factory=list)

    # Raw step infos from environment (for collectors to process)
    step_infos: list[dict[str, Any]] = field(default_factory=list)
    episode_end_infos: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EnvRunner:
    """Runs environment and collects transitions.

    Handles:
    - Step collection with action selection
    - Reward processing (scaling, clipping)
    - Episode boundary handling (reset)
    - Optional callbacks for step/episode events

    Game-specific metrics should be extracted by MetricCollectors
    that receive the step_infos, not by this runner.
    """

    env: Any  # Gymnasium-like environment
    action_fn: Callable[[np.ndarray], int]  # state -> action

    # Reward processing
    reward_scale: float = 1.0
    reward_clip: float = 0.0  # 0 = no clipping

    # Callbacks (for collectors)
    on_step: Callable[[dict[str, Any]], None] | None = None
    on_episode_end: Callable[[dict[str, Any]], None] | None = None

    # State tracking
    _current_state: np.ndarray | None = field(init=False, default=None, repr=False)

    def collect(self, num_steps: int) -> list[Transition]:
        """Collect transitions for num_steps.

        Args:
            num_steps: Number of environment steps to take

        Returns:
            List of transitions
        """
        transitions, _ = self._collect_impl(num_steps)
        return transitions

    def collect_with_info(self, num_steps: int) -> tuple[list[Transition], dict[str, Any]]:
        """Collect transitions and return collection info.

        Args:
            num_steps: Number of environment steps to take

        Returns:
            (transitions, info) where info contains stats and raw infos
        """
        transitions, info = self._collect_impl(num_steps)
        return transitions, {
            "steps": info.steps,
            "total_reward": info.total_reward,
            "episodes_completed": info.episodes_completed,
            "episode_rewards": info.episode_rewards,
            "step_infos": info.step_infos,
            "episode_end_infos": info.episode_end_infos,
        }

    def _collect_impl(self, num_steps: int) -> tuple[list[Transition], CollectionInfo]:
        """Internal collection implementation."""
        if num_steps <= 0:
            return [], CollectionInfo(steps=0, total_reward=0.0, episodes_completed=0)

        # Initialize if needed
        if self._current_state is None:
            state, _ = self.env.reset()
            self._current_state = state
        else:
            state = self._current_state

        transitions: list[Transition] = []
        total_reward = 0.0
        episodes_completed = 0
        episode_rewards: list[float] = []
        episode_reward = 0.0
        step_infos: list[dict[str, Any]] = []
        episode_end_infos: list[dict[str, Any]] = []

        for _ in range(num_steps):
            # Get action
            action = self.action_fn(state)

            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Process reward
            processed_reward = self._process_reward(reward)
            total_reward += reward  # Track raw reward
            episode_reward += reward

            # Create transition
            transition = Transition(
                state=state.copy(),
                action=action,
                reward=processed_reward,
                next_state=next_state.copy(),
                done=done,
            )
            transitions.append(transition)

            # Store step info for collectors
            step_infos.append(info)

            # Callback for step (collectors can observe)
            if self.on_step is not None:
                self.on_step(info)

            if done:
                episodes_completed += 1
                episode_rewards.append(episode_reward)
                episode_reward = 0.0

                # Store episode end info for collectors
                episode_end_infos.append(info)

                # Callback for episode end
                if self.on_episode_end is not None:
                    self.on_episode_end(info)

                # Reset for next episode
                next_state, _ = self.env.reset()

            # Update state for next step
            state = next_state
            self._current_state = state

        return transitions, CollectionInfo(
            steps=num_steps,
            total_reward=total_reward,
            episodes_completed=episodes_completed,
            episode_rewards=episode_rewards,
            step_infos=step_infos,
            episode_end_infos=episode_end_infos,
        )

    def _process_reward(self, reward: float) -> float:
        """Apply reward processing (scaling, clipping)."""
        # Scale
        reward = reward * self.reward_scale

        # Clip
        if self.reward_clip > 0:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        return reward

    def reset(self) -> None:
        """Reset runner state (forces env reset on next collect)."""
        self._current_state = None
