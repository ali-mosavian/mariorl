"""
Snapshot wrapper for Mario environment.

This wrapper adds intelligent save/restore functionality based on
death hotspots. It intercepts step() calls and handles:
- Saving snapshots at strategic positions (before hotspots, time-based)
- Restoring from snapshots on death
- Tracking save/restore statistics
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Tuple

import numpy as np

from mario_rl.training.snapshot_handler import SnapshotHandler
from mario_rl.training.snapshot_handler import SnapshotResult
from mario_rl.training.snapshot_state_machine import SnapshotAction
from mario_rl.training.snapshot_state_machine import SnapshotStateMachine
from mario_rl.training.snapshot import SnapshotManager
from mario_rl.core.config import SnapshotConfig


@dataclass
class SnapshotMarioEnvironment:
    """
    Mario environment with intelligent snapshot save/restore.

    Wraps a MarioEnvironment and adds:
    - Death hotspot-aware snapshot saving
    - Automatic restore on death
    - Progress tracking to avoid infinite restore loops

    Usage:
        from mario_rl.environment.factory import create_mario_env
        from mario_rl.environment.snapshot_wrapper import SnapshotMarioEnvironment

        base_env = create_mario_env(level=(1, 1))
        env = SnapshotMarioEnvironment(
            env=base_env,
            hotspot_path=Path("death_hotspots.json"),
        )

        obs, info = env.reset()
        for _ in range(1000):
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            # Snapshots are handled automatically!

    Attributes:
        env: The underlying MarioEnvironment
        hotspot_path: Path to death hotspots JSON file
        checkpoint_interval: Game ticks between time-based checkpoints
        max_restores_without_progress: Max restores before giving up
    """

    env: Any  # MarioEnvironment
    hotspot_path: Path | None = None
    checkpoint_interval: int = 500
    max_restores_without_progress: int = 3
    hotspot_approach_distance: int = 100
    hotspot_save_offset: int = 50
    enabled: bool = True

    # Internal components
    _handler: SnapshotHandler = field(init=False, repr=False)
    _current_obs: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize the snapshot handler."""
        if self.enabled:
            # Create snapshot config
            config = SnapshotConfig(
                enabled=True,
                interval=self.checkpoint_interval // 2,
                slots=10,
                max_restores_without_progress=self.max_restores_without_progress,
            )

            # Create snapshot manager with references to inner components
            manager = SnapshotManager(
                config=config,
                base_env=self.env.base_env,
                fstack=self.env.fstack,
            )

            # Create state machine
            state_machine = SnapshotStateMachine(
                hotspot_approach_distance=self.hotspot_approach_distance,
                hotspot_save_offset=self.hotspot_save_offset,
                checkpoint_interval=self.checkpoint_interval,
                max_restores_without_progress=self.max_restores_without_progress,
            )

            # Create handler
            self._handler = SnapshotHandler(
                state_machine=state_machine,
                snapshot_manager=manager,
                hotspot_path=self.hotspot_path,
            )

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def base_env(self) -> Any:
        """Access to base NES environment."""
        return self.env.base_env

    @property
    def fstack(self) -> Any:
        """Access to frame stack wrapper."""
        return self.env.fstack

    @property
    def snapshot_stats(self) -> dict[str, int]:
        """Get current snapshot statistics."""
        if self.enabled:
            return self._handler.get_stats()
        return {
            "snapshot_saves": 0,
            "snapshot_restores": 0,
            "saves_this_episode": 0,
            "restores_this_episode": 0,
        }

    @property
    def total_saves(self) -> int:
        """Total snapshots saved."""
        return self._handler.total_saves if self.enabled else 0

    @property
    def total_restores(self) -> int:
        """Total snapshots restored."""
        return self._handler.total_restores if self.enabled else 0

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset environment and snapshot handler."""
        obs, info = self.env.reset(**kwargs)
        self._current_obs = obs

        if self.enabled:
            self._handler.reset_episode()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step environment with snapshot handling.

        If a death occurs and a snapshot is available, the environment
        will be restored and the episode will continue (done=False).
        If too many restores fail, the episode ends (done=True).

        Returns:
            Standard Gym tuple: (obs, reward, terminated, truncated, info)
            Additional info keys:
            - snapshot_action: The action taken (save/restore/none)
            - snapshot_saved: True if a save occurred
            - snapshot_restored: True if a restore occurred
        """
        # Take the step
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if not self.enabled:
            self._current_obs = obs
            return obs, reward, terminated, truncated, info

        # Process through snapshot handler
        result = self._handler.process_step(
            obs=self._current_obs if self._current_obs is not None else obs,
            info=info,
            done=done,
        )

        # Add snapshot info to info dict
        info["snapshot_action"] = result.action_taken.name
        info["snapshot_saved"] = result.action_taken == SnapshotAction.SAVE_SNAPSHOT
        info["snapshot_restored"] = result.observation is not None

        # Handle actions
        if result.observation is not None:
            # Restored from snapshot - return restored obs, episode continues
            self._current_obs = result.observation
            # Important: Reset the done flag on the base env
            self.env.base_env.env._done = False
            # Record the death position for hotspot learning (even though we restored)
            info["death_position"] = info.get("x_pos", 0)
            return result.observation, reward, False, False, info

        if result.episode_ended:
            # Handler decided to end episode (too many failed restores)
            self._current_obs = obs
            return obs, reward, True, False, info

        # Normal step
        self._current_obs = obs
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the environment."""
        self.env.close()


def create_snapshot_mario_env(
    level: Any = (1, 1),
    render_frames: bool = False,
    hotspot_path: Path | None = None,
    checkpoint_interval: int = 500,
    max_restores_without_progress: int = 3,
    enabled: bool = True,
) -> SnapshotMarioEnvironment:
    """
    Factory function to create a Mario environment with snapshot support.

    Args:
        level: Level specification - tuple (world, stage) or "random"/"sequential"
        render_frames: Whether to render frames for visualization
        hotspot_path: Path to death hotspots JSON file
        checkpoint_interval: Game ticks between time-based checkpoints
        max_restores_without_progress: Max restores before giving up
        enabled: Whether snapshot functionality is enabled

    Returns:
        SnapshotMarioEnvironment with full snapshot support
    """
    from mario_rl.environment.factory import create_mario_env

    base_env = create_mario_env(level=level, render_frames=render_frames)

    return SnapshotMarioEnvironment(
        env=base_env,
        hotspot_path=hotspot_path,
        checkpoint_interval=checkpoint_interval,
        max_restores_without_progress=max_restores_without_progress,
        enabled=enabled,
    )
