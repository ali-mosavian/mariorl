"""
Snapshot Handler that integrates the state machine with environment control.

This module provides a high-level handler that:
1. Uses SnapshotStateMachine for decision making
2. Manages actual save/restore operations via SnapshotManager
3. Integrates with death hotspots for intelligent checkpointing
4. Provides a simple step() interface for integration with training loops
"""

from __future__ import annotations

from typing import Any
from pathlib import Path
from dataclasses import field
from dataclasses import dataclass

import numpy as np

from mario_rl.training.snapshot import SnapshotManager
from mario_rl.metrics.levels import DeathHotspotAggregate
from mario_rl.training.snapshot_state_machine import SnapshotAction
from mario_rl.training.snapshot_state_machine import SnapshotContext
from mario_rl.training.snapshot_state_machine import SnapshotStateMachine


@dataclass
class SnapshotResult:
    """Result of processing a step through the snapshot handler."""

    action_taken: SnapshotAction
    observation: np.ndarray | None = None  # Set when restore happened
    episode_ended: bool = False  # True if handler ended the episode
    save_x: int | None = None  # X position where save occurred (if any)
    restore_x: int | None = None  # X position where restore occurred (if any)


@dataclass
class SnapshotHandler:
    """High-level snapshot handler integrating state machine and save/restore.

    This handler processes each step through the state machine and executes
    the appropriate actions (save/restore) based on the machine's decisions.

    Usage:
        handler = SnapshotHandler(
            state_machine=SnapshotStateMachine(),
            snapshot_manager=snapshot_manager,
            hotspot_path=Path("death_hotspots.json"),
        )

        # In training loop:
        for step in range(num_steps):
            action = agent.act(obs)
            next_obs, reward, done, truncated, info = env.step(action)

            result = handler.process_step(
                obs=obs,
                info=info,
                done=done,
            )

            if result.observation is not None:
                # Restored from snapshot
                obs = result.observation
                continue

            if result.episode_ended:
                # Handler decided to end episode
                obs, _ = env.reset()
                continue

            obs = next_obs

    Attributes:
        state_machine: The state machine for decision making.
        snapshot_manager: Manager for actual save/restore operations.
        hotspot_path: Path to death hotspots JSON file.
        hotspot_reload_interval: Seconds between hotspot reloads.
    """

    state_machine: SnapshotStateMachine
    snapshot_manager: SnapshotManager
    hotspot_path: Path | None = None
    hotspot_reload_interval: float = 5.0

    # Internal state
    _hotspots: DeathHotspotAggregate | None = field(default=None, init=False)
    _last_hotspot_load: float = field(default=0.0, init=False)
    _best_x: int = field(default=0, init=False)
    _current_level: str = field(default="1-1", init=False)

    # Statistics (for metrics/UI)
    total_saves: int = field(default=0, init=False)
    total_restores: int = field(default=0, init=False)
    saves_this_episode: int = field(default=0, init=False)
    restores_this_episode: int = field(default=0, init=False)

    def process_step(
        self,
        obs: np.ndarray,
        info: dict[str, Any],
        done: bool,
    ) -> SnapshotResult:
        """Process a step and handle snapshotting/restoring.

        Args:
            obs: Current observation (needed for saving)
            info: Environment info dict with game state
            done: Whether the episode is done

        Returns:
            SnapshotResult indicating what action was taken
        """
        import time

        # Reload hotspots periodically
        now = time.time()
        if (
            self.hotspot_path is not None
            and now - self._last_hotspot_load > self.hotspot_reload_interval
            and self.hotspot_path.exists()
        ):
            try:
                self._hotspots = DeathHotspotAggregate.load(self.hotspot_path)
                self._last_hotspot_load = now
            except Exception:
                pass  # File may be being written

        # Extract game state
        x_pos = info.get("x_pos", 0)
        game_time = info.get("time", 0)
        world = info.get("world", 1)
        stage = info.get("stage", 1)
        level_id = f"{world}-{stage}"
        flag_get = info.get("flag_get", False)
        # is_timeout from env = died because timer ran out (not skill failure)
        is_timeout = info.get("is_timeout", False)
        # Skill-based death = episode ended without flag and wasn't a timeout
        is_dead = done and not flag_get and not is_timeout

        # Track best x and level
        self._best_x = max(self._best_x, x_pos)
        self._current_level = level_id

        # Get hotspot positions for this level
        hotspot_positions: tuple[int, ...] = ()
        suggested_restore_x: int | None = None

        if self._hotspots is not None:
            positions = self._hotspots.suggest_snapshot_positions(level_id)
            hotspot_positions = tuple(positions)

            if is_dead:
                suggested_restore_x = self._hotspots.suggest_restore_position(level_id, x_pos)

        # Build context for state machine
        ctx = SnapshotContext(
            x_pos=x_pos,
            level_id=level_id,
            game_time=game_time,
            is_dead=is_dead,
            is_timeout=is_timeout,
            flag_get=flag_get,
            best_x=self._best_x,
            hotspot_positions=hotspot_positions,
            snapshot_available=self._has_snapshots(),
            suggested_restore_x=suggested_restore_x,
        )

        # Run state machine until we get an action or stabilize
        # Some transitions (like DEAD -> EVALUATE -> RESTORE) need multiple steps
        action = SnapshotAction.NONE
        max_transitions = 5  # Prevent infinite loops
        for _ in range(max_transitions):
            _, new_action = self.state_machine.transition(ctx)
            if new_action != SnapshotAction.NONE:
                action = new_action
                break
            # If state machine returned NONE and we're in a stable state, stop
            from mario_rl.training.snapshot_state_machine import SnapshotState

            stable_states = {
                SnapshotState.RUNNING,
                SnapshotState.APPROACHING_HOTSPOT,
            }
            if self.state_machine.state in stable_states:
                break

        # Execute action
        match action:
            case SnapshotAction.SAVE_SNAPSHOT:
                success = self._save_snapshot(obs, info)
                self.state_machine.notify_save_complete()
                if success:
                    self.total_saves += 1
                    self.saves_this_episode += 1
                    return SnapshotResult(action_taken=action, save_x=x_pos)
                return SnapshotResult(action_taken=action)

            case SnapshotAction.RESTORE_SNAPSHOT:
                restored_obs = self._restore_snapshot(info, suggested_restore_x)
                if restored_obs is not None:
                    self.state_machine.notify_restore_complete(x_pos)
                    self.total_restores += 1
                    self.restores_this_episode += 1
                    return SnapshotResult(
                        action_taken=action,
                        observation=restored_obs,
                        restore_x=x_pos,
                    )
                # Restore failed, end episode
                return SnapshotResult(
                    action_taken=SnapshotAction.END_EPISODE,
                    episode_ended=True,
                )

            case SnapshotAction.END_EPISODE:
                self.reset_episode()
                return SnapshotResult(
                    action_taken=action,
                    episode_ended=True,
                )

            case _:
                return SnapshotResult(action_taken=action)

    def reset_episode(self) -> None:
        """Reset handler state for a new episode."""
        self.state_machine.reset()
        self.snapshot_manager.reset()
        self._best_x = 0
        self.saves_this_episode = 0
        self.restores_this_episode = 0

    def get_stats(self) -> dict[str, int]:
        """Get current snapshot statistics.

        Returns:
            Dict with save/restore counts (for metrics logging)
        """
        return {
            "snapshot_saves": self.total_saves,
            "snapshot_restores": self.total_restores,
            "saves_this_episode": self.saves_this_episode,
            "restores_this_episode": self.restores_this_episode,
        }

    def _has_snapshots(self) -> bool:
        """Check if any snapshots are available."""
        return bool(self.snapshot_manager._slot_to_state)

    def _save_snapshot(self, obs: np.ndarray, info: dict[str, Any]) -> bool:
        """Save a snapshot at current position."""
        return self.snapshot_manager.maybe_save(obs, info)

    def _restore_snapshot(
        self,
        info: dict[str, Any],
        suggested_x: int | None,
    ) -> np.ndarray | None:
        """Attempt to restore from a snapshot.

        Args:
            info: Current environment info
            suggested_x: Suggested restore position from hotspots

        Returns:
            Restored observation if successful, None otherwise
        """
        restored_state, success = self.snapshot_manager.try_restore(info, self._best_x)

        if success and restored_state.size > 0:
            return restored_state

        return None


def create_snapshot_handler(
    base_env: Any,
    frame_stack: Any,
    hotspot_path: Path | None = None,
    checkpoint_interval: int = 500,
    max_restores_without_progress: int = 3,
    hotspot_approach_distance: int = 100,
    hotspot_save_offset: int = 50,
) -> SnapshotHandler:
    """Factory function to create a SnapshotHandler with all components.

    Args:
        base_env: The base Mario environment (with dump_state/load_state)
        frame_stack: The frame stacking wrapper (with obs_queue)
        hotspot_path: Path to death hotspots JSON (optional)
        checkpoint_interval: Game ticks between time-based checkpoints
        max_restores_without_progress: Max restores before giving up
        hotspot_approach_distance: Pixels before hotspot to start tracking
        hotspot_save_offset: Pixels before hotspot to save snapshot

    Returns:
        Configured SnapshotHandler instance
    """
    from mario_rl.core.config import SnapshotConfig

    # Create snapshot config
    config = SnapshotConfig(
        enabled=True,
        interval=checkpoint_interval // 2,  # Save every interval/2 game ticks
        slots=10,
        max_restores_without_progress=max_restores_without_progress,
    )

    # Create manager
    manager = SnapshotManager(
        config=config,
        base_env=base_env,
        fstack=frame_stack,
    )

    # Create state machine with matching config
    state_machine = SnapshotStateMachine(
        hotspot_approach_distance=hotspot_approach_distance,
        hotspot_save_offset=hotspot_save_offset,
        checkpoint_interval=checkpoint_interval,
        max_restores_without_progress=max_restores_without_progress,
    )

    return SnapshotHandler(
        state_machine=state_machine,
        snapshot_manager=manager,
        hotspot_path=hotspot_path,
    )
